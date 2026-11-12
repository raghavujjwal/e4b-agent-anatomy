"""The two control runs (ctl_v2rec1/2, corrected harness) and the S4 check files built from them. Analysis helper.
usage: python -I -X utf8 analyze_ctl.py <folder with ctl_v2rec{1,2}[_trace].jsonl> <check folder>
"""
import collections
import json
import statistics as st
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import s4_gates as G  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
SRC, CHK = sys.argv[1], sys.argv[2]
RUNS = ("ctl_v2rec1", "ctl_v2rec2")
out = {}

# ---- 1. the edits whose status could not be read
unk = collections.Counter()
examples = collections.defaultdict(list)
edits = collections.Counter()
per_episode = []
for run in RUNS:
    for line in open(f"{SRC}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        ev = tr["trace"]
        chains = B.chains_of(tr)
        n_comp = sum(e.get("event") == "compaction" for e in ev)
        n_nudge = sum(e.get("event") == "nudge" for e in ev)
        n_budget = sum(e.get("event") == "budget_stop" for e in ev)
        n_err = sum(e.get("event") == "model_error" for e in ev)
        fails = collections.Counter()
        for ci, turns in enumerate(chains):
            for k, (prompt, gen, finish, _n, suffix) in enumerate(turns):
                text = tok.decode(gen, skip_special_tokens=False)
                tool = B.tool_of(text)
                if tool not in B.EDIT_TOOLS:
                    continue
                s = B.edit_status(tok, turns, k, tool)
                edits[s or "unknown"] += 1
                if s == "error":
                    c = G.first_call(text)
                    if c and c[1] is not None:
                        fails[(c[0], c[1].get("filepath"), c[1].get("old_string"))] += 1
                if s is None:
                    if k + 1 >= len(turns):
                        cat = "last turn of its run"
                        why = [e.get("event") for e in ev if e.get("event") in ("budget_stop", "model_error", "final_response_skip_summarization")][-1:]
                        cat += f" (events at the end: {why})"
                    else:
                        nxt = turns[k + 1]
                        nsuf = tok.decode(nxt[4], skip_special_tokens=False)
                        if f"response:{tool}" in nsuf:
                            cat = "response present but status not parsed"
                        elif nxt[0] and len(nxt[0]) and prompt and nxt[0][:len(prompt)] != prompt:
                            cat = "next prompt does not extend this prompt (compaction or rewritten history); response not in the new suffix"
                        else:
                            cat = "next suffix has no response for this tool"
                    unk[cat] += 1
                    if len(examples[cat]) < 2:
                        examples[cat].append((run, tr["instance_id"], ci, k, len(chains[ci]), (tok.decode(turns[k + 1][4], skip_special_tokens=False)[:300] if k + 1 < len(turns) else "")))
        per_episode.append({"run": run, "instance_id": tr["instance_id"], "runs_of_loop": len(chains), "compactions": n_comp, "nudges": n_nudge, "budget_stop": n_budget,
                            "model_errors": n_err, "max_identical_failed_edit": max(fails.values(), default=0)})
print("== edits by status:", dict(edits))
print("== unknown status, by cause:")
for c, n in unk.most_common():
    print(f"  {n:>3}  {c}")
    for ex in examples[c]:
        print("       e.g.", ex[:5], repr(ex[5][:200]))
out["edits"] = dict(edits)
out["unknown_by_cause"] = dict(unk)

# ---- 2. the control's own reference numbers (for A1; NOT the A1 control, which is rerun with the candidate)
rows = []
for run in RUNS:
    rows += [dict(json.loads(l), run=run) for l in open(f"{SRC}/{run}.jsonl", encoding="utf-8") if l.strip()]
n = len(rows)
def share(pred):
    return round(100 * sum(1 for r in rows if pred(r)) / n, 1)
calls = [r["tool_calls"] for r in rows]
print(f"\n== control reference, {n} episodes (2 seeds x 66 tasks)")
print("resolved", sum(bool(r["resolved"]) for r in rows), "| empty patch", share(lambda r: r["outcome_class"] == "empty_patch_without_error"), "% | stop reasons", dict(collections.Counter(r.get("stop_reason") for r in rows)))
print("tool calls: mean %.1f median %.0f  | at the 24-call budget: %.1f%% | completion tokens per episode: mean %.0f" % (st.mean(calls), st.median(calls), share(lambda r: r["tool_calls"] >= 24), st.mean(r["completion_tokens"] for r in rows)))
ok, err = edits["ok"], edits["error"]
print("edit/write calls: ok %d, error %d, unknown %d -> failure rate among known %.1f%%" % (ok, err, edits["unknown"], 100 * err / max(1, ok + err)))
pe = per_episode
print("episodes with >=1 compaction: %.1f%% | with >=1 nudge: %.1f%% | with >1 run of the loop: %.1f%% | budget_stop events: %d | model errors: %d" % (
    100 * sum(e["compactions"] > 0 for e in pe) / n, 100 * sum(e["nudges"] > 0 for e in pe) / n, 100 * sum(e["runs_of_loop"] > 1 for e in pe) / n,
    sum(e["budget_stop"] for e in pe), sum(e["model_errors"] for e in pe)))
print("episodes with >=4 identical FAILING edits (loop, A1 definition limited to edits): %.1f%%" % (100 * sum(e["max_identical_failed_edit"] >= 4 for e in pe) / n))
out["reference"] = {"episodes": n, "resolved": sum(bool(r["resolved"]) for r in rows), "calls_mean": round(st.mean(calls), 2), "edit_failure_pct": round(100 * err / max(1, ok + err), 1)}


# ---- 3. what the check prefixes look like
def features(prompt_ids):
    t = tok.decode(prompt_ids, skip_special_tokens=False)
    first_user, first_model = t.find("<|turn>user"), t.find("<|turn>model")
    post_compaction = first_model != -1 and (first_user == -1 or first_model < first_user)
    users = t.count("<|turn>user")
    return {"len": len(prompt_ids), "post_compaction": post_compaction, "nudge": users >= (1 if post_compaction else 2), "thoughts_in_history": t.count("<|channel>thought") > 0}


def describe(name, rows_):
    feats = [features(r["prompt_ids"]) for r in rows_]
    L = sorted(f["len"] for f in feats)
    print(f"\n== {name}: {len(rows_)} prefixes from {len({r['instance_id'] for r in rows_})} tasks | tokens median {L[len(L) // 2]}, p90 {L[int(.9 * len(L))]}, max {L[-1]}")
    print("   after a compaction: %d | after a nudge: %d | thoughts in the history: %d | first turn of a run: %d" % (
        sum(f["post_compaction"] for f in feats), sum(f["nudge"] for f in feats), sum(f["thoughts_in_history"] for f in feats), sum(1 for r in rows_ if r.get("turn_in_run") == 0)))
    return feats


check = [json.loads(l) for l in open(f"{CHK}/check.jsonl", encoding="utf-8")]
print("\nG3 prefixes by recorded tool:", dict(collections.Counter(r["tool"] for r in check)))
f3 = describe("G3 check.jsonl", check)
for tool in ("read_file", "run_command", "edit_file", "submit_patch"):
    sub = [f for r, f in zip(check, f3) if r["tool"] == tool]
    print(f"   {tool:<13} n={len(sub):>3} after compaction {sum(f['post_compaction'] for f in sub):>3} after nudge {sum(f['nudge'] for f in sub):>3} thoughts in history {sum(f['thoughts_in_history'] for f in sub):>3}")
fe = [json.loads(l) for l in open(f"{CHK}/check_failed_edit.jsonl", encoding="utf-8")]
f4 = describe("G4 check_failed_edit.jsonl", fe)
by_task = collections.Counter(r["instance_id"] for r in fe)
print("   prefixes per task: top 5", by_task.most_common(5), "| tasks", len(by_task))
rec = collections.Counter()
for r in fe:
    nx = r["recorded_next"]
    text = "I cannot continue." if not nx or nx["args"] is None else f"<|tool_call>call:{nx['name']}{{" + ",".join(f"{k}:{v}" for k, v in sorted(nx["args"].items())) + "}<tool_call|>"
    rec[G.next_kind(text, r["failed_call"])] += 1
print("   what the recorded run did next:", dict(rec), "-> repeat %.1f%%, re-read %.1f%%" % (100 * rec["repeat"] / len(fe), 100 * rec["reread"] / len(fe)))
ok_wf = bad_wf = 0
why = collections.Counter()
for name in ("turns_train.jsonl", "turns_holdout.jsonl"):
    for line in open(f"{CHK}/{name}", encoding="utf-8"):
        r = json.loads(line)
        good, reason = G.well_formed(tok.decode(r["gen_ids"], skip_special_tokens=False))
        ok_wf += good
        bad_wf += not good
        if not good:
            why[reason] += 1
print(f"\n== G5 on the recorded kept generations: {ok_wf} of {ok_wf + bad_wf} well formed; misses {dict(why)}")
json.dump(out, open(f"{SRC}/../ctl_analysis.json", "w"), indent=1)
