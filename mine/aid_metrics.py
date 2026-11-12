"""Score a run of the aid arm against the rules fixed in agent/submission_e4b_aid/README.md.

  python mine/aid_metrics.py --tasks D:/data/kaggle_comp/tasks.jsonl --runs D:/data/e4b_runs/aid1 --out D:/data/analysis/aid1.json

Per task run (from <run>.jsonl and <run>_trace.jsonl):
  find_called / find_in_first3   run_skill_script with skill repo_aid and script find_files.py, anywhere / among the first 3 calls
  gold_listed                    a gold (non-test) file appears in the output of find_files
  seen                           the gold file was opened with read_file or named in a run_command (the funnel's definition, so it compares with the baseline's 56.7% / 21% / 74%)
  nf_edits / retry_follow        edit_file calls that failed with "old_string not found"; of those, how many had edit_retry called within the next 2 calls
  retry_calls / retry_ok         edit_retry calls, and those whose output starts with "OK: edit applied"
Rules (README): compliance >= 80% (and in the first 3 calls); single-file gold seen >= 65% overall and >= 40% on issues under 300 characters; retry follow >= 50% and retry_ok >= 40% of retry calls;
10 or more resolved of 66 to run two more seeds.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization_funnel import gold_files, run_one  # noqa: E402
from trace_analysis import parse_calls, result_info, edit_outcome  # noqa: E402


def skill_of(call):
    a = call["args"] if isinstance(call["args"], dict) else {}
    if call["name"] != "run_skill_script" or a.get("skill_name") != "repo_aid":
        return None
    fp = str(a.get("file_path", ""))
    return "find" if fp.endswith("find_files.py") else "retry" if fp.endswith("edit_retry.py") else None


def stdout_of(content):
    """Text the skill printed: top-level stdout on success, details.stdout on a failed run (the harness moves it there when the exit code is not 0)."""
    try:
        d = json.loads(content) if isinstance(content, str) else content
        if isinstance(d, dict) and isinstance(d.get("result"), str):
            d = json.loads(d["result"])
    except ValueError:
        return str(content)
    if not isinstance(d, dict):
        return str(content)
    det = d.get("details") if isinstance(d.get("details"), dict) else {}
    return str(d.get("stdout", "")) + str(det.get("stdout", "")) + str(d.get("error_message", ""))


def aid_row(msgs, gold):
    results = {m["tool_call_id"]: m.get("content", "") for m in msgs if m.get("role") == "tool" and m.get("tool_call_id")}
    calls = [c for m in msgs if m.get("role") == "assistant" for c in parse_calls(m)]
    kinds = [skill_of(c) for c in calls]
    row = run_one(msgs, gold)
    row["find_called"] = "find" in kinds
    row["find_in_first3"] = "find" in kinds[:3]
    row["gold_listed"] = any(g in stdout_of(results.get(c["id"], "")) for c, k in zip(calls, kinds) if k == "find" for g in gold)
    nf = [i for i, c in enumerate(calls) if c["name"] == "edit_file" and edit_outcome(*result_info(results.get(c["id"], ""))) == "old_string not found"]
    row["nf_edits"] = len(nf)
    row["retry_follow"] = sum(1 for i in nf if "retry" in kinds[i + 1:i + 3])
    retries = [c for c, k in zip(calls, kinds) if k == "retry"]
    row["retry_calls"] = len(retries)
    row["retry_ok"] = sum(1 for c in retries if stdout_of(results.get(c["id"], "")).lstrip().startswith("OK: edit applied"))
    row["retry_why"] = [stdout_of(results.get(c["id"], "")).lstrip()[:48] for c in retries if not stdout_of(results.get(c["id"], "")).lstrip().startswith("OK: edit applied")]
    return row


def report(rows):
    n = len(rows)
    pct = lambda k, d: round(100 * k / max(d, 1), 1)
    single = [r for r in rows if not r["multi"]]
    short = [r for r in single if r["short"]]
    long_ = [r for r in single if not r["short"]]
    nf, follow = sum(r["nf_edits"] for r in rows), sum(r["retry_follow"] for r in rows)
    rc, rok = sum(r["retry_calls"] for r in rows), sum(r["retry_ok"] for r in rows)
    out = {"task_runs": n, "resolved": sum(r["resolved"] for r in rows),
           "find_called_%": pct(sum(r["find_called"] for r in rows), n), "find_in_first3_%": pct(sum(r["find_in_first3"] for r in rows), n),
           "single_file": {"runs": len(single), "gold_listed_%": pct(sum(r["gold_listed"] for r in single), len(single)), "seen_%": pct(sum(r["seen"] for r in single), len(single)),
                           "short_runs": len(short), "short_seen_%": pct(sum(r["seen"] for r in short), len(short)), "short_listed_%": pct(sum(r["gold_listed"] for r in short), len(short)),
                           "long_runs": len(long_), "long_seen_%": pct(sum(r["seen"] for r in long_), len(long_)),
                           "resolved_%": pct(sum(r["resolved"] for r in single), len(single))},
           "retry": {"not_found_edits": nf, "followed_by_retry_%": pct(follow, nf), "retry_calls": rc, "retry_ok_%": pct(rok, rc)}}
    s = out["single_file"]
    rules = {"compliance >= 80% and in first 3 calls": out["find_called_%"] >= 80 and out["find_in_first3_%"] >= 80,
             "seen >= 65% (single-file)": s["seen_%"] >= 65, "short seen >= 40%": s["short_seen_%"] >= 40,
             "retry follow >= 50%": out["retry"]["followed_by_retry_%"] >= 50, "retry ok >= 40%": out["retry"]["retry_ok_%"] >= 40,
             "resolved >= 10 of 66": out["resolved"] >= 10}
    out["rules"] = rules
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    tasks = {}
    for line in open(a.tasks, encoding="utf-8"):
        t = json.loads(line)
        tasks[t["instance_id"]] = t
    rows = []
    for run in a.runs:
        res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(f"{run}.jsonl", encoding="utf-8")}
        for line in open(f"{run}_trace.jsonl", encoding="utf-8"):
            tr = json.loads(line)
            iid = tr["instance_id"]
            convs = [e for e in tr["trace"] if e.get("event") == "conversation"]
            gold = gold_files(tasks[iid]["patch"]) if iid in tasks else set()
            if not convs or not gold:
                continue
            r = aid_row(convs[-1]["messages"], gold)
            r.update(run=Path(run).name, task=iid, resolved=bool(res.get(iid, {}).get("resolved")), multi=len(gold) > 1, short=len(tasks[iid]["problem_statement"]) < 300)
            rows.append(r)
    rep = report(rows)
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"report": rep, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
