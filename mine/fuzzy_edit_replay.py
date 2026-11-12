"""Would a forgiving edit_file have rescued the failed edits in saved runs?  (no model, no GPU)

  python mine/fuzzy_edit_replay.py --runs D:/data/e4b_runs/base1 ... --tasks D:/data/kaggle_comp/tasks.jsonl --out D:/data/analysis/fuzzy_edit.json

For every edit_file call that failed with "old_string not found" and whose file the model had read, a forgiving tool would look for ONE place in the file the model was shown:
  escape     old_string with backslash sequences turned into the real characters occurs exactly once
  space      the lines match after collapsing runs of blanks and stripping line ends, exactly one window
  similar    the best window of the same number of lines has similarity >= MIN_SIM and beats the second best window by >= MARGIN
and apply the edit there. We cannot run the tool on a live file, so what is measured is the text it would have replaced:
  same_region  the model's own next successful edit_file on that file replaced (almost) the same text (the window and that edit's old_string overlap: one contains the other, or similarity >= 0.8)
  other_region it succeeded later but somewhere else: the forgiving tool would have edited a different place than the model finally did
  no_later_ok  the model never got an edit onto that file applied
Also reported: the share of failures on files the gold patch changes, and for runs that ended without a patch or unresolved how many had a rescuable first failure.
Limits: file text is what read_file showed (earlier edits of the run are not replayed), so a window may be stale; "same_region" is a proxy for correct, not proof.
"""
from __future__ import annotations
import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_analysis import parse_calls, result_info, edit_outcome, file_texts, unescape, squash  # noqa: E402

MIN_SIM, MARGIN = 0.9, 0.05


def windows(text, k):
    tl = text.split("\n")
    return [(i, "\n".join(tl[i:i + k])) for i in range(0, max(len(tl) - k + 1, 1))]


def find_place(old, texts):
    """(strategy, window text) a forgiving tool would edit, or (None, None)."""
    needle = unescape(old)
    if needle != old and sum(t.count(needle) for t in texts) == 1:
        return "escape", needle
    k = max(len(old.split("\n")), 1)
    sq = squash(old)
    for t in texts:
        ws = [w for _i, w in windows(t, k) if squash(w) == sq]
        if len(set(ws)) == 1 and len(ws) == 1:
            return "space", ws[0]
    best, second = (0.0, None), 0.0
    for t in texts:
        for _i, w in windows(t, k):
            sm = difflib.SequenceMatcher(None, old, w, autojunk=False)
            if sm.quick_ratio() <= max(second, 0.6):
                continue
            r = sm.ratio()
            if r > best[0]:
                best, second = (r, w), max(second, best[0]) if best[1] != w else second
            elif r > second and w != best[1]:
                second = r
    if best[1] is not None and best[0] >= MIN_SIM and best[0] - second >= MARGIN:
        return "similar", best[1]
    return None, None


def same_region(win, later_old):
    if win in later_old or later_old in win:
        return True
    return difflib.SequenceMatcher(None, win, later_old, autojunk=False).ratio() >= 0.8


def gold_files(patch):
    return set(re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)) | set(re.findall(r"^--- a/(\S+)", patch, re.M))


def analyse(runs, tasks):
    rows, per_run = [], []
    for run in runs:
        name = Path(run).name
        res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(f"{run}.jsonl", encoding="utf-8")}
        for line in open(f"{run}_trace.jsonl", encoding="utf-8"):
            tr = json.loads(line)
            iid = tr["instance_id"]
            convs = [e for e in tr["trace"] if e.get("event") == "conversation"]
            if not convs or iid not in res:
                continue
            msgs = convs[-1]["messages"]
            results = {m["tool_call_id"]: m.get("content", "") for m in msgs if m.get("role") == "tool" and m.get("tool_call_id")}
            calls = [c for m in msgs if m.get("role") == "assistant" for c in parse_calls(m)]
            shown = file_texts(msgs, {c["id"]: c for c in calls})
            gold = gold_files(tasks[iid]["patch"]) if iid in tasks else set()
            edits = [(n, c) for n, c in enumerate(calls) if c["name"] == "edit_file" and isinstance(c["args"], dict)]
            first_rescue = None
            for n, c in edits:
                a = c["args"]
                outcome = edit_outcome(*result_info(results.get(c["id"], "")))
                if outcome != "old_string not found":
                    continue
                path, old = a.get("filepath"), str(a.get("old_string", ""))
                texts = shown.get(path, [])
                row = {"run": name, "task": iid, "resolved": bool(res[iid].get("resolved")), "file": path, "gold_file": path in gold or any(path.endswith(g) or g.endswith(path or "?") for g in gold),
                       "read": bool(texts), "strategy": None, "later": None}
                if texts:
                    strat, win = find_place(old, texts)
                    row["strategy"] = strat
                    if strat:
                        later = [str(c2["args"].get("old_string", "")) for n2, c2 in edits if n2 > n and c2["args"].get("filepath") == path
                                 and edit_outcome(*result_info(results.get(c2["id"], ""))) == "ok"]
                        row["later"] = "no_later_ok" if not later else ("same_region" if any(same_region(win, o) for o in later) else "other_region")
                        if first_rescue is None:
                            first_rescue = row
                rows.append(row)
            per_run.append({"run": name, "task": iid, "resolved": bool(res[iid].get("resolved")), "outcome_class": res[iid].get("outcome_class"), "rescuable_failure": first_rescue is not None,
                            "n_fail_nf": sum(1 for r in rows if r["run"] == name and r["task"] == iid)})
    return rows, per_run


def report(rows, per_run):
    n = len(rows)
    out = {"not_found_edits": n, "file_not_read": sum(not r["read"] for r in rows)}
    resc = [r for r in rows if r["strategy"]]
    out["rescued_by_forgiving_tool"] = len(resc)
    out["rescued_%"] = round(100 * len(resc) / max(n, 1), 1)
    out["by_strategy"] = dict(Counter(r["strategy"] for r in resc))
    out["rescued_vs_later_success"] = dict(Counter(r["later"] for r in resc))
    out["rescued_on_gold_file_%"] = round(100 * sum(r["gold_file"] for r in resc) / max(len(resc), 1), 1)
    out["all_failures_on_gold_file_%"] = round(100 * sum(r["gold_file"] for r in rows) / max(n, 1), 1)
    out["wrong_place_risk_%_of_rescued_with_later_ok"] = round(100 * sum(r["later"] == "other_region" for r in resc) / max(sum(r["later"] in ("same_region", "other_region") for r in resc), 1), 1)
    for label, sel in (("unresolved_runs", [p for p in per_run if not p["resolved"]]), ("resolved_runs", [p for p in per_run if p["resolved"]])):
        out[label] = {"runs": len(sel), "with_a_not_found_edit": sum(p["n_fail_nf"] > 0 for p in sel), "with_rescuable_failure": sum(p["rescuable_failure"] for p in sel)}
    out["unresolved_empty_patch_runs"] = {"runs": sum(1 for p in per_run if (p["outcome_class"] or "").startswith("empty")),
                                           "with_rescuable_failure": sum(1 for p in per_run if (p["outcome_class"] or "").startswith("empty") and p["rescuable_failure"])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tasks = {}
    for line in open(a.tasks, encoding="utf-8"):
        r = json.loads(line)
        tasks[r["instance_id"]] = r
    rows, per_run = analyse(a.runs, tasks)
    rep = report(rows, per_run)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"report": rep, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    sys.exit(main())
