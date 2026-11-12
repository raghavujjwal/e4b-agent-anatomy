"""Where do runs lose the task: finding the file, editing it, or the fix itself?

  python mine/localization_funnel.py --tasks D:/data/kaggle_comp/tasks.jsonl --runs D:/data/e4b_runs/base1 ... --out D:/data/analysis/funnel.json

For every task run: the files the gold patch changes (non-test files), the files the agent read (read_file, or named in a run_command), the files it
tried to edit, and whether an edit applied. Funnel per run set:
  runs -> gold file seen -> edit attempted on a gold file -> a gold-file edit applied -> resolved
Also: the call index of the first sight of a gold file and of the first edit attempt, split by resolved / not resolved.
"""
from __future__ import annotations
import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_analysis import parse_calls, result_info  # noqa: E402

TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*$|_test\.py$|(^|/)conftest\.py$")


def gold_files(patch: str):
    files = set(re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)) | set(re.findall(r"^--- a/(\S+)", patch, re.M))
    files.discard("/dev/null")
    return {f for f in files if not TEST_RE.search(f)}


def norm(p):
    return (p or "").lstrip("./").replace("\\", "/")


def run_one(msgs, gold):
    results = {m["tool_call_id"]: m.get("content", "") for m in msgs if m.get("role") == "tool" and m.get("tool_call_id")}
    calls = [c for m in msgs if m.get("role") == "assistant" for c in parse_calls(m)]
    first_seen = first_edit = None
    attempted, applied, seen_any = set(), set(), set()
    for i, c in enumerate(calls, 1):
        a = c["args"] if isinstance(c["args"], dict) else {}
        if c["name"] == "read_file":
            f = norm(a.get("filepath"))
            seen_any.add(f)
            if f in gold and first_seen is None:
                first_seen = i
        elif c["name"] == "run_command":
            cmd = str(a.get("command", ""))
            for g in gold:
                if g in cmd and first_seen is None:
                    first_seen = i
                if g in cmd:
                    seen_any.add(g)
        elif c["name"] in ("edit_file", "write_file"):
            f = norm(a.get("filepath"))
            if first_edit is None:
                first_edit = i
            if f in gold:
                attempted.add(f)
                if result_info(results.get(c["id"], ""))[0] == "ok":
                    applied.add(f)
    return {"calls": len(calls), "seen": bool(seen_any & gold), "first_seen": first_seen, "first_edit": first_edit,
            "attempted_gold": bool(attempted), "applied_gold": bool(applied)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    gold = {}
    for line in open(a.tasks, encoding="utf-8"):
        t = json.loads(line)
        gold[t["instance_id"]] = gold_files(t["patch"])
    rows = []
    for run in a.runs:
        res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(f"{run}.jsonl", encoding="utf-8")}
        for line in open(f"{run}_trace.jsonl", encoding="utf-8"):
            tr = json.loads(line)
            iid = tr["instance_id"]
            convs = [e for e in tr["trace"] if e.get("event") == "conversation"]
            if not convs or iid not in gold or not gold[iid]:
                continue
            r = run_one(convs[-1]["messages"], gold[iid])
            r.update(run=Path(run).name, task=iid, resolved=bool(res.get(iid, {}).get("resolved")), multi=len(gold[iid]) > 1)
            rows.append(r)
    n = len(rows)
    pct = lambda k, d=None: round(100 * k / (d or n), 1)
    sel = lambda f: [r for r in rows if f(r)]
    out = {"task_runs": n,
           "funnel": {"gold file seen": [len(sel(lambda r: r["seen"])), pct(len(sel(lambda r: r["seen"])))],
                      "edit attempted on a gold file": [len(sel(lambda r: r["attempted_gold"])), pct(len(sel(lambda r: r["attempted_gold"])))],
                      "gold-file edit applied": [len(sel(lambda r: r["applied_gold"])), pct(len(sel(lambda r: r["applied_gold"])))],
                      "resolved": [len(sel(lambda r: r["resolved"])), pct(len(sel(lambda r: r["resolved"])))]}}
    for label, f in (("single-file gold", lambda r: not r["multi"]), ("multi-file gold", lambda r: r["multi"])):
        s = sel(f)
        if s:
            out[label] = {"task_runs": len(s), "seen_%": pct(sum(r["seen"] for r in s), len(s)), "attempted_%": pct(sum(r["attempted_gold"] for r in s), len(s)),
                          "applied_%": pct(sum(r["applied_gold"] for r in s), len(s)), "resolved_%": pct(sum(r["resolved"] for r in s), len(s))}
    # conditional rates
    seen = sel(lambda r: r["seen"])
    att = sel(lambda r: r["attempted_gold"])
    app = sel(lambda r: r["applied_gold"])
    out["conditional_%"] = {"attempt | seen": pct(len(att), len(seen) or 1), "applied | attempted": pct(len(app), len(att) or 1),
                            "resolved | applied": pct(sum(r["resolved"] for r in app), len(app) or 1),
                            "resolved | seen but no applied gold edit": pct(sum(r["resolved"] for r in seen if not r["applied_gold"]), len([r for r in seen if not r["applied_gold"]]) or 1)}
    med = lambda xs: statistics.median(xs) if xs else None
    for label, group in (("resolved", sel(lambda r: r["resolved"])), ("not resolved", sel(lambda r: not r["resolved"]))):
        out[f"timing ({label})"] = {"median call of first gold sight": med([r["first_seen"] for r in group if r["first_seen"]]),
                                    "median call of first edit attempt": med([r["first_edit"] for r in group if r["first_edit"]]),
                                    "median calls": med([r["calls"] for r in group])}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": out, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
