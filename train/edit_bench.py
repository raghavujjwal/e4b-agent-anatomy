"""Offline edit benchmark: replay validation prefixes against served models and check the generated edit with the real tool code.

  python train/edit_bench.py --data D:/data/edit_sft --base-url http://127.0.0.1:8000/v1 \\
      --model gemma-4-e4b-it-qat-w4a16-ct --model edit_v1 --n 300 --out /content/bench

For each validation `edit_file` sample the server gets the same messages the agent would have at that step (system prompt, issue, the
read_file call and result) and the tool schemas, with thinking on and the baseline sampling. The model's tool call is applied to the file
window it was shown with `apply_replacement` from agent/vendor/tools.py (the harness's three-tier matcher) and compared with the gold edit
applied the same way. Reported per model:
  call_ok      a parsed edit_file call with only the schema's keys
  applies      the edit applies to the window (exactly one match)
  exact_tier   it applies at the strict tier, i.e. old_string was copied exactly (no whitespace/regex rescue)
  reproduces   the edited window equals the gold-edited window
Limits: the window is not the whole file, so uniqueness is checked inside the window only (slightly optimistic); one sample per prefix;
the gold edit is one valid fix, not the only one, so `reproduces` can undercount valid alternatives while `applies` and `exact_tier` do not.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent" / "vendor"))
from tools import apply_replacement  # noqa: E402  (stdlib only)

EDIT_KEYS = {"filepath", "old_string", "new_string", "allow_multiple"}


def window_of(messages):
    """The file text the model was shown: the content field of the last tool result."""
    for m in reversed(messages):
        if m["role"] == "tool":
            return json.loads(m["content"]).get("content", "")
    return ""


def gold_call(messages):
    return json.loads(messages[-1]["tool_calls"][0]["function"]["arguments"])


def ask(base_url, model, messages, tools, temperature, top_p, max_tokens, timeout=600):
    body = {"model": model, "messages": messages[:-1], "tools": tools, "tool_choice": "auto", "temperature": temperature,
            "top_p": top_p, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def grade(rec, out):
    msgs = [dict(m) for m in rec["messages"]]
    window, gold = window_of(msgs), gold_call(msgs)
    expected, _, _ = apply_replacement(window, gold["old_string"], gold["new_string"], False)
    ch = out["choices"][0]
    msg = ch["message"]
    res = {"id": rec["id"], "finish": ch["finish_reason"], "completion_tokens": out["usage"]["completion_tokens"],
           "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or "")}
    calls = msg.get("tool_calls") or []
    if not calls:
        return {**res, "status": "no_tool_call", "text": (msg.get("content") or "")[:300]}
    fn = calls[0]["function"]
    res["tool"] = fn["name"]
    if fn["name"] != "edit_file":
        return {**res, "status": "wrong_tool"}
    try:
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    except ValueError:
        return {**res, "status": "bad_json"}
    if not {"filepath", "old_string", "new_string"} <= set(args) or set(args) - EDIT_KEYS:
        return {**res, "status": "bad_keys", "keys": sorted(args)}
    res["call_ok"] = True
    res["same_file"] = args["filepath"] == gold["filepath"]
    res["old_equals_gold"] = args["old_string"] == gold["old_string"]
    new_text, n, tier = apply_replacement(window, args["old_string"], args["new_string"], bool(args.get("allow_multiple")))
    if new_text is None:
        return {**res, "status": "ambiguous" if n > 1 else "not_found", "tier": tier}
    return {**res, "status": "applied", "tier": tier, "reproduces": new_text == expected and res["same_file"]}


def run_model(a, model, recs, tools):
    def one(rec):
        try:
            return grade(rec, ask(a.base_url, model, rec["messages"], tools, a.temperature, a.top_p, a.max_tokens))
        except Exception as e:  # noqa: BLE001
            return {"id": rec["id"], "status": "request_error", "error": str(e)[:200]}
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, recs))
    print(f"{model}: {len(rows)} requests in {time.time() - t0:.0f}s", flush=True)
    return rows


def summarize(rows):
    n = len(rows)
    c = Counter(r["status"] for r in rows)
    applied = [r for r in rows if r["status"] == "applied"]
    pct = lambda k: round(100 * k / n, 1)
    return {"n": n, "status": dict(c),
            "call_ok_%": pct(sum(bool(r.get("call_ok")) for r in rows)),
            "applies_%": pct(len(applied)),
            "exact_tier_%": pct(sum(r["tier"] == "exact" for r in applied)),
            "reproduces_%": pct(sum(bool(r.get("reproduces")) for r in applied)),
            "old_string_equals_gold_%": pct(sum(bool(r.get("old_equals_gold")) for r in rows)),
            "finish_length": sum(r.get("finish") == "length" for r in rows),
            "mean_completion_tokens": round(sum(r.get("completion_tokens", 0) for r in rows) / n),
            "mean_reasoning_chars": round(sum(r.get("reasoning_chars", 0) for r in rows) / n)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", action="append", required=True, help="served model or LoRA name; repeat to compare on the same samples")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.2, help="0.2 = the baseline submission; v2rec uses 1.0")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    data = Path(a.data)
    tools = json.load(open(data / "tools_edit_file.json", encoding="utf-8"))
    recs = [json.loads(l) for l in open(data / "validation.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r["format"] == "edit_file"]
    random.Random(a.seed).shuffle(recs)
    recs = recs[:a.n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allrows, summary = {}, {}
    for model in a.model:
        rows = run_model(a, model, recs, tools)
        allrows[model] = rows
        summary[model] = summarize(rows)
        with open(out / f"{model}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(json.dumps({model: summary[model]}, indent=1), flush=True)
    if len(a.model) == 2:
        x, y = (dict((r["id"], r) for r in allrows[m]) for m in a.model)
        both = [i for i in x if i in y]
        gain = sum(bool(y[i].get("reproduces")) and not x[i].get("reproduces") for i in both)
        loss = sum(bool(x[i].get("reproduces")) and not y[i].get("reproduces") for i in both)
        print(f"paired reproduces: {a.model[1]} only {gain}, {a.model[0]} only {loss} (of {len(both)})")
        summary["paired_reproduces"] = {"second_only": gain, "first_only": loss, "n": len(both)}
    json.dump({"args": vars(a), "summary": summary}, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
