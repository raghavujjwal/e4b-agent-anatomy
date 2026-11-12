"""What went wrong in saved agent runs: edit failures, task killers on the scorer, empty patches.

  python mine/trace_analysis.py --runs D:/data/e4b_runs/base1 D:/data/e4b_runs/base2 ... --out D:/data/analysis/trace_report.json

Reads <run>.jsonl (one result per task) and <run>_trace.jsonl (events + the final conversation). Tool calls and their results are matched by
tool_call_id, so compaction cannot misalign them. Reports:
  edit_file     outcome of every call (ok / old_string not found / ambiguous / other error / malformed arguments), failure rate by features
                of old_string (triple quotes, backslashes, multi-line, long), identical consecutive repeats, per-task failed-edit counts
  killers       calls that the scorer would answer by ending the task (tool name not declared, run_skill_script argument of the wrong type),
                which a gentler local harness may have turned into an error message the model could recover from
  empty_patch   for tasks that ended with no patch: what the last calls were, whether any edit was attempted / applied, whether submit_patch ran
"""
from __future__ import annotations
import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DECLARED = {"read_file", "write_file", "edit_file", "run_command", "get_status", "submit_patch"}      # the baseline bundle; --declared overrides
EDIT_KEYS = {"filepath", "old_string", "new_string", "allow_multiple"}


def parse_calls(msg):
    tcs = msg.get("tool_calls")
    if isinstance(tcs, str):
        try:
            tcs = ast.literal_eval(tcs)
        except (ValueError, SyntaxError):
            try:
                tcs = json.loads(tcs)
            except ValueError:
                return []
    out = []
    for tc in tcs or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            args = None
        out.append({"id": tc.get("id"), "name": fn.get("name"), "args": args, "raw": raw})
    return out


def result_info(content):
    """(status, error_type, message) from a tool result string."""
    try:
        d = json.loads(content) if isinstance(content, str) else content
    except ValueError:
        return "unparsed", None, (content or "")[:120]
    if isinstance(d, dict) and "result" in d and isinstance(d["result"], str):       # ADK wrapper from older harness versions
        return result_info(d["result"])
    if not isinstance(d, dict):
        return "unparsed", None, str(d)[:120]
    return d.get("status", "?"), d.get("error_type"), str(d.get("error_message") or d.get("error") or "")[:160]


def edit_outcome(status, etype, msg):
    if status == "ok":
        return "ok"
    m = msg.lower()
    if "not found" in m:
        return "old_string not found"
    if "multiple" in m or "ambiguous" in m or "matches" in m:
        return "ambiguous"
    if "mandatory" in m or "missing" in m or "required" in m or "not present" in m:
        return "missing arguments"
    return f"other: {etype or ''} {msg[:50]}".strip()


def features(old):
    return {"triple_quote": '"""' in old or "'''" in old, "backslash": "\\" in old, "multiline": "\n" in old,
            "long(>400)": len(old) > 400, "short(<40)": len(old) < 40}


def file_texts(msgs, calls_by_id):
    """filepath -> list of file texts the model was shown by read_file in this conversation (decoded, so with real newlines and quotes)."""
    shown = defaultdict(list)
    for m in msgs:
        if m.get("role") != "tool":
            continue
        c = calls_by_id.get(m.get("tool_call_id"))
        if not c or c["name"] != "read_file" or not isinstance(c["args"], dict):
            continue
        try:
            d = json.loads(m["content"])
            if isinstance(d, dict) and isinstance(d.get("result"), str):
                d = json.loads(d["result"])
            if d.get("status") == "ok":
                shown[c["args"].get("filepath")].append(d.get("content", ""))
        except (ValueError, AttributeError):
            pass
    return shown


UNESC = [("\\n", "\n"), ("\\t", "\t"), ('\\"', '"'), ("\\'", "'"), ("\\\\", "\\")]


def unescape(t):
    for a, b in UNESC:
        t = t.replace(a, b)
    return t


def squash(t):
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.strip().split("\n"))


def near(old, text):
    """best similarity of old against windows of the same number of lines inside text"""
    import difflib
    k = max(len(old.split("\n")), 1)
    tl = text.split("\n")
    best = 0.0
    for i in range(0, max(len(tl) - k + 1, 1)):
        sm = difflib.SequenceMatcher(None, old, "\n".join(tl[i:i + k]), autojunk=False)
        if sm.quick_ratio() > max(best, 0.6):
            best = max(best, sm.ratio())
    return best


def why_not_found(old, texts):
    if not texts:
        return "file not read in this conversation"
    if any(old in t for t in texts):
        return "old_string IS in the file as shown (changed by an earlier edit, or window/uniqueness issue)"
    if any(unescape(old) in t for t in texts):
        return "escape sequences copied (backslash-n, backslash-quote, ...)"
    if any(squash(old) in squash(t) for t in texts):
        return "whitespace/indentation differs only"
    r = max(near(old, t) for t in texts)
    return "near miss (similarity >= 0.9)" if r >= 0.9 else "partial (0.7-0.9)" if r >= 0.7 else "far from anything in the file (<0.7)"



def analyse(runs):
    edits, kill, empty = [], [], []
    per_task_fail = []
    by_run = {}
    for run in runs:
        name = Path(run).name
        res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(f"{run}.jsonl", encoding="utf-8")}
        for line in open(f"{run}_trace.jsonl", encoding="utf-8"):
            tr = json.loads(line)
            iid = tr["instance_id"]
            convs = [e for e in tr["trace"] if e.get("event") == "conversation"]
            if not convs:
                continue
            msgs = convs[-1]["messages"]
            results = {m["tool_call_id"]: m.get("content", "") for m in msgs if m.get("role") == "tool" and m.get("tool_call_id")}
            calls = [c for m in msgs if m.get("role") == "assistant" for c in parse_calls(m)]
            shown = file_texts(msgs, {c["id"]: c for c in calls})
            r = res.get(iid, {})
            prev_key, n_fail, n_ok = None, 0, 0
            for c in calls:
                st = result_info(results.get(c["id"], ""))
                if c["name"] not in DECLARED:
                    kill.append({"run": name, "task": iid, "kind": "undeclared tool", "name": c["name"]})
                if c["name"] == "run_skill_script" and isinstance(c["args"], dict) and isinstance(c["args"].get("file_path"), (list, dict)):
                    kill.append({"run": name, "task": iid, "kind": "run_skill_script file_path not a string", "name": c["name"]})
                if c["name"] != "edit_file":
                    continue
                a = c["args"]
                if not isinstance(a, dict):
                    edits.append({"run": name, "task": iid, "outcome": "arguments not JSON", "feat": {}, "repeat": False})
                    n_fail += 1
                    continue
                odd = sorted(set(a) - EDIT_KEYS)
                outcome = edit_outcome(*st)
                if odd and outcome != "ok":
                    outcome = "odd argument keys: " + ",".join(odd)[:40]
                key = json.dumps(a, sort_keys=True)
                edits.append({"run": name, "task": iid, "outcome": outcome, "feat": features(str(a.get("old_string", ""))), "repeat": key == prev_key,
                              "why": why_not_found(str(a.get("old_string", "")), shown.get(a.get("filepath"), [])) if outcome == "old_string not found" else None})
                prev_key = key
                n_fail, n_ok = n_fail + (outcome != "ok"), n_ok + (outcome == "ok")
            per_task_fail.append({"run": name, "task": iid, "failed": n_fail, "ok": n_ok, "resolved": bool(r.get("resolved"))})
            if r.get("outcome_class", "").startswith("empty"):
                names = [c["name"] for c in calls]
                e_att = [c for c in calls if c["name"] in ("edit_file", "write_file")]
                empty.append({"run": name, "task": iid, "calls": len(calls), "edits_attempted": len(e_att),
                              "edits_ok": sum(1 for c in e_att if result_info(results.get(c["id"], ""))[0] == "ok"),
                              "skill_calls": names.count("run_skill_script"), "submit_called": "submit_patch" in names, "last3": names[-3:],
                              "stop": r.get("stop_reason")})
        by_run[name] = len(res)
    return edits, kill, empty, per_task_fail


def report(edits, kill, empty, per_task):
    n = len(edits)
    out = {"edit_calls": n, "outcomes": dict(Counter(e["outcome"] for e in edits).most_common())}
    if n:
        fail = [e for e in edits if e["outcome"] != "ok"]
        out["failure_rate_%"] = round(100 * len(fail) / n, 1)
        feat = {}
        for k in ("triple_quote", "backslash", "multiline", "long(>400)", "short(<40)"):
            with_f = [e for e in edits if e["feat"].get(k)]
            wo = [e for e in edits if e["feat"] and not e["feat"].get(k)]
            feat[k] = {"calls_with": len(with_f), "fail_%_with": round(100 * sum(e["outcome"] != "ok" for e in with_f) / max(len(with_f), 1), 1),
                       "fail_%_without": round(100 * sum(e["outcome"] != "ok" for e in wo) / max(len(wo), 1), 1)}
        out["failure_by_old_string_feature"] = feat
        out["why_old_string_not_found"] = dict(Counter(e["why"] for e in edits if e.get("why")).most_common())
        out["identical_consecutive_repeats"] = sum(e["repeat"] for e in edits)
        out["repeats_that_failed"] = sum(e["repeat"] and e["outcome"] != "ok" for e in edits)
        buckets = defaultdict(lambda: [0, 0])
        for t in per_task:
            b = min(t["failed"], 4)
            buckets[b][0] += 1
            buckets[b][1] += t["resolved"]
        out["tasks_by_failed_edits"] = {("4+" if b == 4 else str(b)): {"tasks": v[0], "resolved": v[1]} for b, v in sorted(buckets.items())}
    out["scorer_task_killers"] = dict(Counter((k["kind"], k["name"]) .__str__() for k in kill))
    out["tasks_with_killer"] = len({(k["run"], k["task"]) for k in kill})
    if empty:
        out["empty_patch_tasks"] = len(empty)
        out["empty_patch_breakdown"] = {
            "no edit attempted": sum(e["edits_attempted"] == 0 and e["skill_calls"] == 0 for e in empty),
            "edit attempted, none applied": sum(e["edits_attempted"] > 0 and e["edits_ok"] == 0 for e in empty),
            "some edit applied but patch empty": sum(e["edits_ok"] > 0 for e in empty),
            "skill called": sum(e["skill_calls"] > 0 for e in empty),
            "submit_patch called": sum(e["submit_called"] for e in empty),
            "median calls": sorted(e["calls"] for e in empty)[len(empty) // 2]}
        out["empty_last_call_names"] = dict(Counter(e["last3"][-1] if e["last3"] else None for e in empty).most_common(6))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--declared", nargs="+", help="tool names the bundle declares (default: the baseline's six)")
    a = ap.parse_args(argv)
    if a.declared:
        DECLARED.clear()
        DECLARED.update(a.declared)
    edits, kill, empty, per_task = analyse(a.runs)
    rep = report(edits, kill, empty, per_task)
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"report": rep, "empty": empty, "kill": kill}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
