"""Step-type check: does an adapter still choose the same kind of action as the base model at real points of a run?

  python train/step_check.py --check D:/data/replay_v1/check.jsonl --base-url http://127.0.0.1:8000/v1 \\
      --model gemma-4-e4b-it-qat-w4a16-ct --model edit_v1 --out /content/step_check

`check.jsonl` (train/build_replay_data.py) holds turns of the base model's own runs on tasks the adapter never trained on, with the exact prompt token
ids. Each prompt is sent to /v1/completions as token ids (no re-rendering) with the baseline sampling, and the tool named in the generated call is
compared with the tool the base model chose in the recorded run. Reported per recorded tool and per model: agreement, and the distribution of tools
actually generated. The base model is included as a model on purpose: it re-samples the same prefixes, so its own agreement shows the ceiling that
sampling noise allows; an adapter is judged against that, not against 100%.

Gate (decided before looking at any adapter): for every recorded tool except edit_file, the adapter's agreement must be within 10 points of the
base model's, and its share of `no call` (no tool call produced) must not exceed the base model's by more than 5 points. edit_file is reported but not
gated: changing how edits go is the point of the adapter.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CALL_RE = re.compile(r"<\|tool_call>call:([A-Za-z_]\w*)")


def classify(text: str) -> str:
    m = CALL_RE.search(text or "")
    return m.group(1) if m else "no_call"


def complete(base_url, model, prompt_ids, temperature, top_p, max_tokens, timeout=900, top_k=None):
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
            "skip_special_tokens": False}
    if top_k:
        body["top_k"] = top_k
    req = urllib.request.Request(f"{base_url}/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    ch = out["choices"][0]
    return ch["text"], ch.get("finish_reason"), out["usage"]["completion_tokens"]


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["recorded"]].append(r)
    table = {}
    for tool, rs in sorted(by.items()):
        table[tool] = {"n": len(rs), "agree_%": round(100 * sum(r["generated"] == tool for r in rs) / len(rs), 1),
                       "generated": dict(Counter(r["generated"] for r in rs).most_common()),
                       "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs))}
    n = len(rows)
    return {"n": n, "no_call_%": round(100 * sum(r["generated"] == "no_call" for r in rows) / n, 1),
            "request_error_%": round(100 * sum(r["generated"] == "request_error" for r in rows) / n, 1), "by_recorded_tool": table}


def gate(base, other, tol=10.0, no_call_tol=5.0, error_tol=2.0):
    """List of failures of `other` against `base` (both are summarize() outputs).

    A run in which requests failed proves nothing: with every request failing, the base agrees 0% and no adapter can fall 10 points below it,
    so the comparison would pass. Request errors above `error_tol` percent in either model therefore fail the gate (the entry starts with INVALID).
    """
    if base["request_error_%"] > error_tol:
        return [f"INVALID: the base model had {base['request_error_%']}% request errors, so nothing was compared (is the server up?)"]
    fails = []
    if other["request_error_%"] > error_tol:
        fails.append(f"INVALID: {other['request_error_%']}% request errors (adapter not loadable at request time, or the server died)")
    for tool, b in base["by_recorded_tool"].items():
        if tool == "edit_file" or tool not in other["by_recorded_tool"] or b["n"] < 10:
            continue
        o = other["by_recorded_tool"][tool]
        if o["agree_%"] < b["agree_%"] - tol:
            fails.append(f"{tool}: agreement {o['agree_%']}% vs base {b['agree_%']}%")
    if other["no_call_%"] > base["no_call_%"] + no_call_tol:
        fails.append(f"no_call {other['no_call_%']}% vs base {base['no_call_%']}%")
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", action="append", required=True, help="first one is the reference (the base model)")
    ap.add_argument("--n", type=int, default=0, help="use at most this many check turns (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    turns = [json.loads(l) for l in open(a.check, encoding="utf-8")]
    if a.n:
        turns = turns[:a.n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model in a.model:
        def one(t):
            try:
                text, finish, ntok = complete(a.base_url, model, t["prompt_ids"], a.temperature, a.top_p, a.max_tokens)
                return {"instance_id": t["instance_id"], "turn": t["turn"], "recorded": t["tool"], "generated": classify(text), "finish": finish,
                        "tokens": ntok, "thought": "<|channel>thought" in text}
            except Exception as e:  # noqa: BLE001
                return {"instance_id": t["instance_id"], "turn": t["turn"], "recorded": t["tool"], "generated": "request_error", "tokens": 0,
                        "error": str(e)[:200]}
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, turns))
        summaries[model] = summarize(rows)
        with open(out / f"{model}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(json.dumps({model: summaries[model]}, indent=1), flush=True)
    ref = a.model[0]
    for model in a.model[1:]:
        fails = gate(summaries[ref], summaries[model])
        print(f"GATE {model}: " + ("PASS" if not fails else "FAIL - " + "; ".join(fails)))
        summaries[model]["gate_failures"] = fails
    json.dump(summaries, open(out / "summary.json", "w"), indent=1)
    return 1 if any(f.startswith("INVALID") for m in a.model[1:] for f in summaries[m]["gate_failures"]) else 0


if __name__ == "__main__":
    sys.exit(main())
