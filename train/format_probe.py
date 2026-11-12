"""Format probe (report only; docs/PLAN_E4B_SFT.md, note of 2026-11-07): how often does a model write an unreadable tool call at the points where the base model did?

  python train/format_probe.py --rejected check_rejected.jsonl --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it-qat-w4a16-ct --model pair_dpo --out /content/probe

check_rejected.jsonl (train/build_replay_data.py): the prefixes before an edit_file / write_file call that the framework refused without running it. Each model gets `--samples`
generations per prefix at the shipped sampling. A generation is classed by how its first tool call is written:
  ok               one call that the strict reader (train/s4_gates.py `well_formed`) accepts
  python_kwargs    `old_string="..."`: the framework refused every call written this way in the control runs
  missing_argument | unclosed     refused too (a mandatory argument is missing / the call never ends)
  unreadable       the strict reader cannot read the arguments in some other way; the server's tool parser sometimes can (13 of 38 such edit calls were executed)
  no_call | several_calls | undeclared | bad_type     the rest
`rejected_%` counts the first three. No threshold: the probe was defined after the S4 result was known, so it cannot count as a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edit_pairs as E  # noqa: E402
import s4_gates as G  # noqa: E402

DEFINITELY_REFUSED = ("python_kwargs", "missing_argument", "unclosed")


def classify(text: str) -> tuple[str, str]:
    """(class, why) of one generation."""
    ok, why = G.well_formed(text)
    if ok:
        return "ok", ""
    if why.startswith("0 tool calls"):
        return "no_call", why
    if why.endswith(" tool calls"):
        return "several_calls", why
    if why == "call not closed":
        return "unclosed", why
    if why.startswith("missing "):
        return "missing_argument", why
    if why.startswith("undeclared"):
        return "undeclared", why
    if why == "arguments do not parse":
        return ("python_kwargs" if E.KWARG_STYLE.search(text[text.index(E.CALL_OPEN):]) else "unreadable"), why
    return "bad_type", why


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    c = Counter(r["class"] for r in rows)
    by_prefix = defaultdict(list)
    for r in rows:
        by_prefix[r["prefix"]].append(r["class"] in DEFINITELY_REFUSED)
    pct = lambda k: round(100 * k / n, 1) if n else 0.0                                      # noqa: E731
    return {"n": n, "prefixes": len(by_prefix), "ok_%": pct(c["ok"]), "rejected_%": pct(sum(c[k] for k in DEFINITELY_REFUSED)),
            "prefixes_with_a_rejected_%": round(100 * sum(any(v) for v in by_prefix.values()) / max(len(by_prefix), 1), 1),
            "mean_tokens": round(sum(r["tokens"] for r in rows) / n) if n else 0, "length_hits_%": pct(sum(r["finish"] == "length" for r in rows)),
            "request_errors": sum(r["finish"] == "error" for r in rows), "classes": dict(c.most_common())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rejected", required=True, help="check_rejected.jsonl of train/build_replay_data.py")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", action="append", required=True, help="first one is the reference (the base model)")
    ap.add_argument("--samples", type=int, default=3, help="generations per prefix")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prefixes = [json.loads(l) for l in open(a.rejected, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for model in a.model:
        def one(t):
            g = G.generate(a.base_url, model, t["prompt_ids"], a)
            cls, why = ("request_error", g.get("error", "")) if g["finish"] == "error" else classify(g["text"])
            return {"prefix": f'{t.get("run", "")}|{t["instance_id"]}|{t["turn"]}', "recorded": t["tool"], "class": cls, "why": why, "finish": g["finish"], "tokens": g["tokens"],
                    "text": g["text"][:20000]}
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, [t for t in prefixes for _ in range(a.samples)]))
        with open(out / f"{model}_probe.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        result[model] = summarize(rows)
        print(model, json.dumps(result[model]), flush=True)
    json.dump(result, open(out / "probe_summary.json", "w"), indent=1)
    ref = a.model[0]
    for m in a.model[1:]:
        print(f"{m}: rejected-style generations {result[m]['rejected_%']}% (base {result[ref]['rejected_%']}%), "
              f"exactly one well-formed call {result[m]['ok_%']}% (base {result[ref]['ok_%']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
