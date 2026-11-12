"""Report on saved agent runs: notebook 03's table, the overlap and paired comparisons of docs/EVAL_PROTOCOL.md (rules 11, 12), the episodes a slower scorer
clock would cut, the stage split of the pipeline arms, and the ledger rows of rule 14.

  python -I -X utf8 mine/lab_report.py RUN.jsonl [RUN.jsonl ...] [--ledger docs/run_ledger.csv --date 2026-11-08 --arm RUN=folder ...]

A RUN is a results file of agent/run_eval.py; its trace is the sibling <name>_trace.jsonl. The first run is the reference of the paired comparisons.
Per-task and per-repository lines are printed only when every task is in a readable pool (dev66, mined tune). For any other task the report stops at
aggregate counts and reads no gold patch (docs/EVAL_PROTOCOL.md rules 3 and 13).
A run may have been stopped and carried on later (run_eval.py skips the tasks already in the file). The episodes that were running when the server went away are
written as finished all the same; they are found by their trace (cut_episodes), left out of their run and named, unless --keep-cut.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_lab_pack import side  # noqa: E402  (the split rule is written down once)

COLUMNS = ("date_utc,run_id,arm,adapter_checkpoint,pool,seed,server_build,n_tasks,resolved,empty_patch,mean_calls,edit_fail_rate,gold_file_open_rate,"
           "repeat_after_failure_rate,completion_tokens_median,time_cap_hits,colab_units_used,notes").split(",")
DEV66_REPOS = {"fastapi/fastapi", "psf/requests", "encode/httpx"}
TIME_CAP_S = 330                       # max_time_minutes 5.5 in the e4b submissions' eval_config
# (seconds per tool call, decoded tokens per second) at which the scorer might run an E4B episode: assumed speeds (a tool call 6-12 s, 62-100 tok/s)
# and the grader clock's 25.5 tok/s; the lab's own wall clock on a fast GPU is about ten times faster
SCORER_CLOCKS = ((6, 100), (12, 100), (12, 62), (12, 25.5))
STAGES = ("investigate", "fix")        # agent names of the pipeline arms (mine/make_pipeline_arms.py)
INVESTIGATE_MAX_CALLS = 8              # the largest call limit a rung gives `investigate` (S1: 6, S2 and S3: 8; mine/make_pipeline_arms.py INV_BUDGET*)
ARM_SIDE = re.compile(r"\bHTTP 4\d\d\b")   # a request the server refused: the arm's own doing (cut_episodes)
EDIT_OK = re.compile(r'^\{"status": "ok"')
OPENED = re.compile(r'^\{"status": "ok", "filepath": "([^"]+)"')


def read_jsonl(path) -> tuple:
    """(objects, unreadable lines). A run that was stopped and carried on later can hold a line that was cut while it was written: counted, not fatal."""
    rows, broken = [], 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                broken += 1
    return rows, broken


def load_run(path) -> dict:
    """A task that is in a file twice keeps its LAST line and its last trace. That pairs the right two in both ways it can happen: a run started again while the
    old process was still alive (two lines, two traces), and a session that died after a trace reached Drive but before its result line did (the task is run
    again: one line, two traces). `no_trace` = result lines whose trace is missing although the run has a trace file."""
    path = Path(path)
    raw, broken = read_jsonl(path)
    last = {r["instance_id"]: r for r in raw}
    rows = [r for r in raw if last[r["instance_id"]] is r]
    trace, tp = {}, path.with_name(path.stem + "_trace.jsonl")
    if tp.exists():
        traces, cut = read_jsonl(tp)
        broken += cut
        for t in traces:
            trace[t["instance_id"]] = t["trace"]
    return {"name": path.stem, "rows": rows, "trace": trace, "duplicates": len(raw) - len(rows), "broken": broken,
            "no_trace": sum(r["instance_id"] not in trace for r in rows) if tp.exists() else 0}


def model_errors(run) -> dict:
    """instance_id -> error text of the episodes in which a model call failed for good: the trace events `model_error` (a root agent) and `fatal` (a stage of a
    tree), and a row of outcome class `model_error` (the episode raised)."""
    out = {r["instance_id"]: str(r.get("error") or "") for r in run["rows"] if r.get("outcome_class") == "model_error"}
    for i, events in run["trace"].items():
        for e in events:
            if e.get("event") in ("model_error", "fatal"):
                out.setdefault(i, str(e.get("error") or ""))
    return out


def cut_episodes(run) -> dict:
    """The episodes a failing SERVER ended: it was stopped, or the notebook cell was interrupted, while they ran (connection refused or reset, a timeout, HTTP 5xx
    after the retries). run_eval.py still writes such an episode as finished and skips it when the run is carried on, so the result file holds it, but it is not a
    result of the arm. An HTTP 4xx is different: the server refused the arm's own request (context too long), the scorer would too, and the episode counts."""
    return {i: err for i, err in model_errors(run).items() if not ARM_SIDE.search(err)}


def pool_of(rows, usable) -> str:
    """dev66 | tune | other. 'other' = at least one task is outside the readable pools, so nothing below the aggregates may be shown."""
    if rows and all(r["repo"] in DEV66_REPOS for r in rows):
        return "dev66"
    if rows and all(r["instance_id"] in usable and side(r["repo"]) == "tune" for r in rows):
        return "tune"
    return "other"


def gold_files(patch) -> set:
    """The files a unified diff touches: the competition's patches carry only ---/+++ lines, the mined ones have diff --git lines too."""
    return set(re.findall(r"^(?:\+\+\+ b/|--- a/)(\S+)", patch or "", re.M)) | set(re.findall(r"^diff --git a/\S+ b/(\S+)", patch or "", re.M))


def edit_stats(trace) -> tuple:
    """(edit_file calls, failed, failed ones followed by another edit_file of the episode, of those the number whose next edit failed too)."""
    calls = failed = follow = again = 0
    for events in trace.values():
        bad = [EDIT_OK.match(e.get("result_head", "")) is None for e in events if e.get("tool") == "edit_file"]
        calls += len(bad)
        failed += sum(bad)
        for prev, nxt in zip(bad, bad[1:]):
            if prev:
                follow += 1
                again += nxt
    return calls, failed, follow, again


def opened_gold(events, gold) -> bool:
    seen = set()
    for e in events:
        if e.get("tool") in ("read_file", "edit_file", "write_file"):
            m = OPENED.match(e.get("result_head", ""))
            if m:
                seen.add(m.group(1))
    return bool(seen & gold)


def reasoning_share(trace):
    """Share of the assistant messages that carry reasoning_content (the fullest conversation record of each agent in each episode), or None without a trace."""
    n = k = 0
    for events in trace.values():
        fullest = {}
        for e in events:
            if e.get("event") == "conversation" and isinstance(e.get("messages"), list):
                if len(e["messages"]) >= len(fullest.get(e.get("agent"), [])):
                    fullest[e.get("agent")] = e["messages"]
        for messages in fullest.values():
            for m in messages:
                if m.get("role") == "assistant":
                    n += 1
                    k += bool((m.get("reasoning_content") or "").strip())
    return k / n if n else None


def modelled_seconds(row, per_call, tok_per_s) -> float:
    """Calls x seconds per call + completion tokens / decode speed: a LOWER bound of the scorer's time (no prompt processing, compaction or retries)."""
    return row["tool_calls"] * per_call + (row.get("completion_tokens") or 0) / tok_per_s


def over_cap(rows, per_call, tok_per_s, cap=TIME_CAP_S) -> tuple:
    """(episodes, resolved episodes) whose modelled scorer time passes the cap. A resolved episode over the cap would have been cut before it finished."""
    late = [r for r in rows if modelled_seconds(r, per_call, tok_per_s) > cap]
    return len(late), sum(bool(r["resolved"]) for r in late)


def stage_split(trace):
    """For pipeline runs (agents `investigate` and `fix`): median counted calls of each stage, the episodes whose note came out empty, those in which
    `fix` made no edit_file call (the investigation used the budget up, or the note was no use), those in which `investigate` went past the largest call
    limit any rung gives it (it ignored the prompt, and the shared budget is what it took from `fix`), and those with a nudge. A nudge starts a second pass
    of the root, i.e. `investigate` again, as on the scorer (agent/vendor/runner.py reproduces that pass: tests/fixtures/scorer_requests_pipe_s2_nudge.json);
    what the model makes of a second pass was never measured before these runs. None for a flat run or without a trace."""
    if not any(e.get("agent") == STAGES[0] for events in trace.values() for e in events):
        return None
    free = ("submit_patch", "get_status")
    calls = {s: [] for s in STAGES}
    empty = no_edit = long_investigation = nudged = 0
    for events in trace.values():
        for s in STAGES:
            calls[s].append(sum(1 for e in events if e.get("agent") == s and e.get("tool") and e["tool"] not in free))
        long_investigation += calls[STAGES[0]][-1] > INVESTIGATE_MAX_CALLS
        nudged += any(e.get("event") == "nudge" for e in events)
        note = [e for e in events if e.get("event") == "output_key_set" and e.get("key") == "findings"]
        empty += not note or not note[-1].get("chars")
        no_edit += not any(e.get("agent") == STAGES[1] and e.get("tool") == "edit_file" for e in events)
    return {"calls": {s: st.median(v) for s, v in calls.items()}, "empty_note": empty, "fix_without_edit": no_edit,
            "investigate_over_limit": long_investigation, "nudged": nudged, "episodes": len(trace)}


def run_metrics(run, tasks, readable) -> dict:
    rows, trace = run["rows"], run["trace"]
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    calls, failed, follow, again = edit_stats(trace)
    m = {"n": len(rows), "resolved": sum(bool(r["resolved"]) for r in rows), "empty": sum(not r.get("patch_bytes") for r in rows),
         "mean_calls": st.mean(r["tool_calls"] for r in rows), "tokens_median": st.median(toks) if toks else None,
         "median_calls": st.median(r["tool_calls"] for r in rows), "median_s": st.median(r["agent_seconds"] for r in rows),
         "max_s": max(r["agent_seconds"] for r in rows),
         # completion tokens per second of the episode (model time and tool time together): what the session's GPU gave this run, comparable between runs of one arm
         "tok_per_s": st.median([r["completion_tokens"] / r["agent_seconds"] for r in rows if r.get("completion_tokens") and r["agent_seconds"]] or [0]),
         "time_cap_hits": sum(r["agent_seconds"] >= TIME_CAP_S or r.get("stop_reason") in ("time", "timeout") for r in rows),
         "call_budget_stops": sum(r.get("stop_reason") == "budget" for r in rows), "classes": Counter(r["outcome_class"] for r in rows),
         "edit_calls": calls, "edit_fail": failed / calls if calls else None, "repeat_after_fail": again / follow if follow else None,
         "reasoning": reasoning_share(trace), "gold_open": None, "stages": stage_split(trace),
         "over_cap": {sc: over_cap(rows, *sc) for sc in SCORER_CLOCKS}}
    if readable and trace:                      # reads the gold patches, so only for the readable pools
        known = [(r["instance_id"], gold_files(tasks[r["instance_id"]].get("patch"))) for r in rows if r["instance_id"] in tasks]
        m["gold_open"] = sum(opened_gold(trace.get(i, []), g) for i, g in known) / len(known) if known else None
    return m


def paired(a, b, reps=10000, seed=0) -> dict:
    """b against a on the tasks both ran: tasks only one resolved, the difference in tasks with a bootstrap-over-tasks 95% interval (rule 12), and the exact sign-test p."""
    ra = {r["instance_id"]: bool(r["resolved"]) for r in a["rows"]}
    rb = {r["instance_id"]: bool(r["resolved"]) for r in b["rows"]}
    d = [int(rb[i]) - int(ra[i]) for i in sorted(ra.keys() & rb.keys())]
    only_a, only_b = sum(x == -1 for x in d), sum(x == 1 for x in d)
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) for _ in range(reps))
    k = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(k, i) for i in range(min(only_a, only_b) + 1)) / 2 ** k) if k else 1.0
    return {"n": len(d), "only_a": only_a, "only_b": only_b, "diff": sum(d), "lo": boots[int(0.025 * reps)], "hi": boots[int(0.975 * reps) - 1], "p": p}


def tail(ps, k) -> float:
    """P(at least k successes) of independent trials with success probabilities ps."""
    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            new[i] += q * (1 - p)
            new[i + 1] += q * p
        dist = new
    return sum(dist[k:])


def overlap(runs) -> dict:
    ids = sorted(set.intersection(*[{r["instance_id"] for r in run["rows"]} for run in runs]))
    won = [{r["instance_id"]: bool(r["resolved"]) for r in run["rows"]} for run in runs]
    per_task = {i: [w[i] for w in won] for i in ids}
    ps = [sum(w[i] for i in ids) / len(ids) for w in won]
    return {"n": len(ids), "per_task": per_task, "by_count": Counter(sum(v) for v in per_task.values()),
            "union": sum(any(v) for v in per_task.values()), "everyone": sum(all(v) for v in per_task.values()),
            "expected_union": len(ids) * (1 - math.prod(1 - p for p in ps)), "expected_two_or_more": len(ids) * tail(ps, 2)}


def short_names(names) -> list:
    pre = os.path.commonprefix(names)
    pre = pre[: pre.rfind("_") + 1] if len(names) > 1 else ""
    return [n[len(pre):] or n for n in names]


def ledger_row(run, m, pool, a) -> dict:
    cls = ", ".join(f"{k} {v}" for k, v in sorted(m["classes"].items(), key=lambda kv: -kv[1]))
    notes = [cls, f"call-budget stops {m['call_budget_stops']}"]
    if m["reasoning"] is not None:
        notes.append(f"reasoning in {100 * m['reasoning']:.0f}% of assistant messages")
    if m["repeat_after_fail"] is not None:
        notes.append("repeat_after_failure = P(next edit fails | edit failed), a proxy: the compact trace keeps no arguments")
    if m["stages"]:
        s = m["stages"]
        notes.append(f"pipeline: median calls investigate {s['calls'][STAGES[0]]:g} / fix {s['calls'][STAGES[1]]:g}, empty notes {s['empty_note']}, "
                     f"fix without an edit call {s['fix_without_edit']}, investigate over {INVESTIGATE_MAX_CALLS} calls {s['investigate_over_limit']}, "
                     f"episodes with a nudge {s['nudged']}")
    slow = m["over_cap"][(12, 62)]
    notes.append(f"modelled scorer clock (12 s per call, 62 tok/s, lower bound): {slow[0]} episodes over {TIME_CAP_S} s, {slow[1]} of them resolved")
    if run.get("ended"):
        notes.append(f"{len(run['ended'])} episode(s) ended by a failing server (run stopped or server down) "
                     f"{'kept in the counts' if getattr(a, 'keep_cut', False) else 'left out of n_tasks and of every count'}")
    if run.get("refused"):
        notes.append(f"{run['refused']} episode(s) with a request the server refused (HTTP 4xx), counted")
    if a.notes:
        notes.append(a.notes)
    f = lambda x, nd=3: "" if x is None else f"{x:.{nd}f}"                                  # noqa: E731
    seed = re.search(r"(\d+)$", run["name"])
    return {"date_utc": a.date, "run_id": run["name"], "arm": dict(x.split("=", 1) for x in a.arm or []).get(run["name"], ""), "adapter_checkpoint": a.adapter,
            "pool": pool, "seed": seed.group(1) if seed else "", "server_build": a.server_build, "n_tasks": m["n"], "resolved": m["resolved"],
            "empty_patch": m["empty"], "mean_calls": f(m["mean_calls"], 2), "edit_fail_rate": f(m["edit_fail"]), "gold_file_open_rate": f(m["gold_open"]),
            "repeat_after_failure_rate": f(m["repeat_after_fail"]), "completion_tokens_median": f(m["tokens_median"], 0), "time_cap_hits": m["time_cap_hits"],
            "colab_units_used": a.units or "", "notes": "; ".join(notes)}


def append_ledger(path, rows) -> tuple:
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    if lines and lines[0].split(",") != COLUMNS:
        raise SystemExit(f"{path}: the header differs from the columns of docs/EVAL_PROTOCOL.md rule 14")
    have = {r["run_id"] for r in csv.DictReader(lines)}
    new = [r for r in rows if r["run_id"] not in have]
    with path.open("a", encoding="utf-8", newline="") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        if not lines:
            csv.writer(fh, lineterminator="\n").writerow(COLUMNS)
        csv.DictWriter(fh, COLUMNS, lineterminator="\n").writerows(new)
    return len(new), len(rows) - len(new)


def report(runs, tasks, usable, a, out=print) -> list:
    pools = [pool_of(r["rows"], usable) for r in runs]                  # before anything is left out: the pool is a property of the file
    readable = all(p != "other" for p in pools)
    keep_cut = getattr(a, "keep_cut", False)
    runs = [dict(r) for r in runs]
    for r in runs:
        errors = model_errors(r)
        r["ended"] = {i: e for i, e in errors.items() if not ARM_SIDE.search(e)}
        r["refused"] = len(errors) - len(r["ended"])
        if not keep_cut:
            r["rows"] = [x for x in r["rows"] if x["instance_id"] not in r["ended"]]
            r["trace"] = {i: t for i, t in r["trace"].items() if i not in r["ended"]}
    metrics = [run_metrics(r, tasks, readable) for r in runs]
    names = short_names([r["name"] for r in runs])
    out(f"{'run':16s} {'resolved':>8s} {'empty':>5s} {'calls med/mean':>14s} {'agent s med':>11s} {'tokens med':>10s} {'edit fail':>9s} {'gold open':>9s}  outcome classes")
    for n, m in zip(names, metrics):
        pc = lambda x: "-" if x is None else f"{100 * x:.0f}%"                                 # noqa: E731
        out(f"{n:16s} {m['resolved']:>5d}/{m['n']:<2d} {m['empty']:>5d} {m['median_calls']:>7.0f}/{m['mean_calls']:<6.1f} {m['median_s']:>11.0f} "
            f"{(m['tokens_median'] or 0):>10.0f} {pc(m['edit_fail']):>9s} {pc(m['gold_open']):>9s}  {dict(m['classes'])}")
    if any(r["ended"] or r["refused"] or r.get("duplicates") or r.get("broken") or r.get("no_trace") for r in runs):
        out(f"\nepisodes a failing server ended (the run was stopped or the server died while they ran: not results; "
            f"{'KEPT in every count (--keep-cut)' if keep_cut else 'left out of every count of their run, and a pair is compared on the tasks both runs have'}), "
            f"episodes with a request the server refused (HTTP 4xx: the arm's own doing, counted as they are), task lines found twice (the last is used), "
            f"unreadable lines, result lines without a trace (counted; whether the server ended them cannot be checked)")
        for n, r in zip(names, runs):
            ids = "   " + ", ".join(f"{i} ({e[:70]})" for i, e in sorted(r["ended"].items())) if readable and r["ended"] else ""
            out(f"  {n:14s} ended by the server {len(r['ended'])}   refused {r['refused']}   twice {r.get('duplicates', 0)}   unreadable {r.get('broken', 0)}   "
                f"without a trace {r.get('no_trace', 0)}{ids}")
    out(f"\nthe lab's own clock (real seconds on the session's GPU; an episode is cut at {TIME_CAP_S} s, so runs made on different cards compare only while none is near it): "
        f"completion tokens per episode second (median) / longest episode / episodes at the cap")
    out("  " + "   ".join(f"{n}: {m['tok_per_s']:.0f} tok/s / {m['max_s']:.0f} s / {m['time_cap_hits']}" for n, m in zip(names, metrics)))
    out(f"\nmodelled scorer clock, episodes over {TIME_CAP_S} s: all/resolved (tool calls x seconds per call + completion tokens / tokens per second; a lower bound, "
        f"no prompt processing, compaction or retries; the lab itself runs about ten times faster than the scorer)")
    for per_call, speed in SCORER_CLOCKS:
        out(f"  {per_call:>2} s per call, {speed:>5g} tok/s   " + "  ".join(
            f"{n}: {m['over_cap'][(per_call, speed)][0]}/{m['over_cap'][(per_call, speed)][1]}" for n, m in zip(names, metrics)))
    if any(m["stages"] for m in metrics):
        out(f"\npipeline stages: median counted calls investigate / fix, episodes with an empty note, in which fix made no edit_file call, in which investigate used more than "
            f"{INVESTIGATE_MAX_CALLS} calls (past every rung's limit), with a nudge (a second pass of the root, which starts again at investigate)")
        for n, m in zip(names, metrics):
            if m["stages"]:
                s = m["stages"]
                out(f"  {n:14s} {s['calls'][STAGES[0]]:g} / {s['calls'][STAGES[1]]:g}   empty notes {s['empty_note']}/{s['episodes']}   "
                    f"fix without an edit call {s['fix_without_edit']}/{s['episodes']}   investigate over limit {s['investigate_over_limit']}/{s['episodes']}   "
                    f"nudged {s['nudged']}/{s['episodes']}")
    if not readable:
        out("\nAggregates only: at least one task is outside the readable pools (docs/EVAL_PROTOCOL.md rules 3 and 13), so no task, repository or overlap lines.")
    else:
        repos = sorted({r["repo"] for r in runs[0]["rows"]})
        out("\nper repository (resolved/tasks)  " + "  ".join(f"{n:>8s}" for n in names))
        for repo in repos:
            cells = []
            for run in runs:
                sub = [r for r in run["rows"] if r["repo"] == repo]
                cells.append(f"{sum(bool(r['resolved']) for r in sub)}/{len(sub)}")
            out(f"  {repo:30s}" + "  ".join(f"{c:>8s}" for c in cells))
        if len(runs) > 1:
            out(f"\npaired against {names[0]} (rule 12; difference = tasks, bootstrap over tasks 95% interval, exact sign test on the tasks only one run resolved)")
            for n, run in zip(names[1:], runs[1:]):
                p = paired(runs[0], run)
                out(f"  {n:14s} only {names[0]} {p['only_a']:>2d} | only {n} {p['only_b']:>2d} | difference {p['diff']:+d} [{p['lo']:+d}, {p['hi']:+d}] | p = {p['p']:.2f}")
            o = overlap(runs)
            out(f"\noverlap on {o['n']} tasks: resolved by none {o['by_count'][0]}, " + ", ".join(f"by {k} of {len(runs)}: {o['by_count'][k]}" for k in range(1, len(runs) + 1)))
            out(f"  union {o['union']} (pass@{len(runs)}), by every run {o['everyone']}; independent draws at the same rates would give union {o['expected_union']:.1f} "
                f"and {o['expected_two_or_more']:.1f} tasks resolved twice or more")
            out("  tasks resolved at least once:  " + " ".join(f"{n:>8s}" for n in names))
            for i, v in o["per_task"].items():
                if any(v):
                    out(f"    {i:28s}   " + " ".join(f"{'x' if x else '.':>8s}" for x in v))
    rows = [ledger_row(r, m, p, a) for r, m, p in zip(runs, metrics, pools)]
    if a.ledger:
        added, kept = append_ledger(a.ledger, rows)
        out(f"\nledger {a.ledger}: {added} row(s) added, {kept} already there")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="results files of agent/run_eval.py (the first is the reference)")
    ap.add_argument("--tasks", action="append", help="tasks.jsonl with gold patches, repeatable (default: the mined tasks)")
    ap.add_argument("--usable", default=str(ROOT / "mine" / "usable.json"))
    ap.add_argument("--ledger", help="append the rows of rule 14 to this csv (a run_id already there is left alone)")
    ap.add_argument("--date", default="", help="date_utc of the runs")
    ap.add_argument("--arm", action="append", help="RUN=folder under agent/ (repeatable)")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--server-build", default="unrecorded", help="vllm --version, launch command, chat_template sha256 (rule 15)")
    ap.add_argument("--units", default="", help="Colab units of these runs, if known per run")
    ap.add_argument("--notes", default="")
    ap.add_argument("--keep-cut", action="store_true", help="count the episodes a failing server ended too (default: they are left out of their run and named)")
    a = ap.parse_args(argv)
    tasks = {}
    for p in a.tasks or ["D:/data/mined/data/tasks.jsonl"]:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                tasks[t["instance_id"]] = t
    usable = set(json.loads(Path(a.usable).read_text(encoding="utf-8"))["usable"])
    report([load_run(p) for p in a.runs], tasks, usable, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
