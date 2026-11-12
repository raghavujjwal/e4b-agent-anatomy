"""Score a run of the ver1 arm against the rules in agent/submission_e4b_ver1/README.md (and any run, for comparison).

  python mine/ver_metrics.py --runs D:/data/e4b_runs/ver1_1 D:/data/e4b_runs/aid2_1 D:/data/e4b_runs/base1

Per task run:
  repro_before_edit   a run_command before the first edit that runs a script (python/python3/pytest on a path under /tmp, or a heredoc/redirect that writes one)
  rerun_after_edit    after the first edit that applied: a run_command that runs python or pytest
  first_edit_call     index of the first edit_file / edit_retry call
Run level: resolved, median calls, outcome classes (tool_budget / empty patch), stop reasons.
Rules (README): repro before the first edit >= 60%, rerun after the first applied edit >= 50%; median calls <= 20, tool_budget stops <= 12 of 66, empty patches <= 6; resolved >= 11.
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

RUNS_SCRIPT = re.compile(r"(python3?|pytest)\b[^|;&]*", re.I)
WRITES_TMP = re.compile(r"(>\s*/tmp/\S+\.py|cat\s*>\s*/tmp/|tee\s+/tmp/|/tmp/\S+\.py)")


def is_edit(c):
    if c["name"] == "edit_file":
        return True
    a = c["args"] if isinstance(c["args"], dict) else {}
    return c["name"] == "run_skill_script" and str(a.get("file_path", "")).endswith("edit_retry.py")


def command_of(c):
    a = c["args"] if isinstance(c["args"], dict) else {}
    return str(a.get("command", "")) if c["name"] == "run_command" else ""


def run_row(msgs):
    results = {m["tool_call_id"]: m.get("content", "") for m in msgs if m.get("role") == "tool" and m.get("tool_call_id")}
    calls = [c for m in msgs if m.get("role") == "assistant" for c in parse_calls(m)]
    first = next((i for i, c in enumerate(calls) if is_edit(c)), None)
    first_ok = next((i for i, c in enumerate(calls) if is_edit(c) and (result_info(results.get(c["id"], ""))[0] == "ok" or "OK: edit applied" in str(results.get(c["id"], "")))), None)
    before = calls[:first] if first is not None else calls
    repro = any((RUNS_SCRIPT.search(command_of(c)) and "/tmp/" in command_of(c)) or (WRITES_TMP.search(command_of(c)) and RUNS_SCRIPT.search(command_of(c))) for c in before)
    after = calls[first_ok + 1:] if first_ok is not None else []
    rerun = any(RUNS_SCRIPT.search(command_of(c)) for c in after)
    return {"calls": len(calls), "first_edit_call": first, "repro_before_edit": bool(repro), "rerun_after_edit": bool(rerun), "applied_edit": first_ok is not None}


def report(rows, res):
    n = len(rows)
    pct = lambda k, d: round(100 * k / max(d, 1), 1)
    applied = [r for r in rows if r["applied_edit"]]
    classes = Counter(r.get("outcome_class") for r in res)
    out = {"task_runs": n, "resolved": sum(bool(r.get("resolved")) for r in res), "median_calls": statistics.median(r["calls"] for r in rows) if rows else None,
           "repro_before_edit_%": pct(sum(r["repro_before_edit"] for r in rows), n), "rerun_after_applied_edit_%": pct(sum(r["rerun_after_edit"] for r in applied), len(applied)),
           "tool_budget_stops": classes.get("tool_budget", 0), "empty_patches": sum(v for k, v in classes.items() if (k or "").startswith("empty")),
           "outcome_classes": dict(classes), "stop_reasons": dict(Counter(r.get("stop_reason") for r in res))}
    out["rules"] = {"repro before edit >= 60%": out["repro_before_edit_%"] >= 60, "rerun after edit >= 50%": out["rerun_after_applied_edit_%"] >= 50,
                    "median calls <= 20": (out["median_calls"] or 99) <= 20, "tool_budget stops <= 12": out["tool_budget_stops"] <= 12,
                    "empty patches <= 6": out["empty_patches"] <= 6, "resolved >= 11": out["resolved"] >= 11}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args(argv)
    for run in a.runs:
        res = [json.loads(l) for l in open(f"{run}.jsonl", encoding="utf-8")]
        rows = []
        for line in open(f"{run}_trace.jsonl", encoding="utf-8"):
            tr = json.loads(line)
            convs = [e for e in tr["trace"] if e.get("event") == "conversation"]
            if convs:
                rows.append(run_row(convs[-1]["messages"]))
        print(Path(run).name, json.dumps(report(rows, res), indent=1))


if __name__ == "__main__":
    sys.exit(main())
