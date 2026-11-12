"""mine/lab_report.py: the numbers that go into docs/run_ledger.csv, the paired comparison, and the guard that keeps the unseen pools at aggregate counts."""
import argparse
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mine"))
import lab_report as R  # noqa: E402
from make_lab_pack import side  # noqa: E402


def repo_on(which):
    return next(f"owner/repo{i}" for i in range(100) if side(f"owner/repo{i}") == which)


def run(name, results, repo, trace=None):
    rows = [{"instance_id": i, "repo": repo, "resolved": ok, "outcome_class": "resolved" if ok else "unresolved_nonempty_patch", "tool_calls": 3, "llm_turns": 4,
             "patch_bytes": 10, "agent_seconds": 20.0, "completion_tokens": 1000, "stop_reason": "submitted"} for i, ok in results]
    return {"name": name, "rows": rows, "trace": trace or {}}


def args(**kw):
    base = dict(ledger=None, date="2026-11-08", arm=None, adapter="none", server_build="x", units="", notes="")
    return argparse.Namespace(**{**base, **kw})


def edit(ok):
    return {"agent": "swe_agent", "tool": "edit_file", "args_keys": ["filepath"], "result_head": '{"status": "ok", "filepath": "a.py"' if ok else '{"status": "error", "error_type": "FileEditError"'}


def read(path):
    return {"agent": "swe_agent", "tool": "read_file", "args_keys": ["filepath"], "result_head": '{"status": "ok", "filepath": "%s", "content": "x' % path}


def test_gold_files_reads_the_competition_style_and_the_mined_style():
    plain = "--- a/fastapi/sse.py\n+++ b/fastapi/sse.py\n@@ -1 +1 @@\n-x\n+y\n"
    mined = "diff --git a/src/click/core.py b/src/click/core.py\nindex 1..2 100644\n--- a/src/click/core.py\n+++ b/src/click/core.py\n"
    new = "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n"
    assert R.gold_files(plain) == {"fastapi/sse.py"}
    assert R.gold_files(mined) == {"src/click/core.py"}
    assert R.gold_files(new) == {"new.py"}
    assert R.gold_files(None) == set()


def test_edit_stats_counts_failures_and_what_follows_one():
    trace = {"t1": [edit(False), edit(False), edit(True), edit(False)],         # fail, fail, ok, fail: 3 of 4 failed; after the 1st fail the next failed, after the 3rd nothing follows
             "t2": [read("a.py"), edit(True)]}
    calls, failed, follow, again = R.edit_stats(trace)
    assert (calls, failed, follow, again) == (5, 3, 2, 1)


def test_opened_gold_needs_a_successful_open_of_a_gold_file():
    assert R.opened_gold([read("src/a.py")], {"src/a.py"})
    assert not R.opened_gold([read("src/b.py")], {"src/a.py"})
    assert not R.opened_gold([{"tool": "read_file", "result_head": '{"status": "error", "error_message": "src/a.py not found"'}], {"src/a.py"})


def test_paired_difference_and_the_sign_test():
    a = run("a", [(f"t{i}", i == 0) for i in range(10)], "r")
    b = run("b", [(f"t{i}", True) for i in range(10)], "r")
    p = R.paired(a, b)
    assert (p["only_a"], p["only_b"], p["diff"]) == (0, 9, 9)
    assert p["p"] == pytest.approx(2 / 2 ** 9)                 # nine discordant pairs, all for b
    assert p["lo"] <= p["diff"] <= p["hi"]
    same = R.paired(a, a)
    assert (same["only_a"], same["only_b"], same["diff"], same["p"]) == (0, 0, 0, 1.0)
    split = R.paired(run("x", [("t0", True), ("t1", False), ("t2", False)], "r"), run("y", [("t0", False), ("t1", True), ("t2", False)], "r"))
    assert (split["only_a"], split["only_b"], split["p"]) == (1, 1, 1.0)


def test_overlap_counts_tasks_by_how_many_runs_resolved_them():
    a = run("a", [("t0", True), ("t1", True), ("t2", False), ("t3", False)], "r")
    b = run("b", [("t0", True), ("t1", False), ("t2", True), ("t3", False)], "r")
    o = R.overlap([a, b])
    assert o["by_count"] == {2: 1, 1: 2, 0: 1} and o["union"] == 3 and o["everyone"] == 1
    assert o["expected_union"] == pytest.approx(4 * (1 - 0.5 * 0.5)) and o["expected_two_or_more"] == pytest.approx(4 * 0.25)


def test_run_metrics_and_the_ledger_row():
    tr = {"t0": [read("src/a.py"), edit(False), edit(True)], "t1": [edit(False)]}
    r = run("lab_tune_x2", [("t0", True), ("t1", False)], "r", trace=tr)
    r["rows"][1]["patch_bytes"] = 0
    tasks = {"t0": {"patch": "--- a/src/a.py\n+++ b/src/a.py\n"}, "t1": {"patch": "--- a/src/b.py\n+++ b/src/b.py\n"}}
    m = R.run_metrics(r, tasks, readable=True)
    assert (m["n"], m["resolved"], m["empty"]) == (2, 1, 1)
    assert m["edit_fail"] == pytest.approx(2 / 3) and m["repeat_after_fail"] == 0.0 and m["gold_open"] == 0.5
    row = R.ledger_row(r, m, "tune", args(arm=["lab_tune_x2=submission_e4b"], notes="a note"))
    assert list(row) == R.COLUMNS
    assert row["run_id"] == "lab_tune_x2" and row["arm"] == "submission_e4b" and row["seed"] == "2" and row["pool"] == "tune"
    assert row["edit_fail_rate"] == "0.667" and row["gold_file_open_rate"] == "0.500" and row["completion_tokens_median"] == "1000" and "a note" in row["notes"]


def test_the_unseen_pools_never_get_more_than_aggregates():
    unseen = repo_on("unseen")
    runs = [run("a", [("zz_unseen_1", True), ("zz_unseen_2", False)], unseen), run("b", [("zz_unseen_1", False), ("zz_unseen_2", False)], unseen)]
    lines = []
    R.report(runs, tasks={}, usable={"zz_unseen_1", "zz_unseen_2"}, a=args(), out=lines.append)       # no tasks: it must not need a gold patch either
    text = "\n".join(lines)
    assert "Aggregates only" in text
    assert "zz_unseen" not in text and unseen not in text
    assert "paired against" not in text and "overlap on" not in text and "per repository" not in text and "tasks resolved at least once" not in text
    assert R.pool_of(runs[0]["rows"], {"zz_unseen_1", "zz_unseen_2"}) == "other"
    assert R.pool_of(runs[0]["rows"], set()) == "other"                  # an id the usable list does not know is not readable either


def test_a_mix_of_readable_and_unreadable_tasks_is_unreadable():
    tune, unseen = repo_on("tune"), repo_on("unseen")
    rows = run("a", [("t_tune", True)], tune)["rows"] + run("a", [("t_unseen", True)], unseen)["rows"]
    assert R.pool_of(rows, {"t_tune", "t_unseen"}) == "other"
    assert R.pool_of(rows[:1], {"t_tune"}) == "tune"
    assert R.pool_of(run("a", [("fastapi_1", True)], "fastapi/fastapi")["rows"], set()) == "dev66"


def test_the_readable_pools_print_the_per_task_lines():
    tune = repo_on("tune")
    runs = [run("lab_a", [("t0", True), ("t1", False)], tune), run("lab_b", [("t0", True), ("t1", True)], tune)]
    lines = []
    R.report(runs, tasks={}, usable={"t0", "t1"}, a=args(), out=lines.append)
    text = "\n".join(lines)
    assert "t0" in text and "t1" in text and "paired against" in text and tune in text


def test_the_ledger_is_appended_once_and_keeps_its_header(tmp_path):
    ledger = tmp_path / "run_ledger.csv"
    ledger.write_text(",".join(R.COLUMNS) + "\n", encoding="utf-8")
    rows = [R.ledger_row(run("lab_tune_a1", [("t0", True)], "r"), R.run_metrics(run("lab_tune_a1", [("t0", True)], "r"), {}, False), "tune", args(notes="n, with a comma"))]
    assert R.append_ledger(ledger, rows) == (1, 0)
    assert R.append_ledger(ledger, rows) == (0, 1)
    got = list(csv.DictReader(ledger.read_text(encoding="utf-8").splitlines()))
    assert len(got) == 1 and got[0]["notes"].endswith("n, with a comma") and got[0]["n_tasks"] == "1"
    bad = tmp_path / "other.csv"
    bad.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        R.append_ledger(bad, rows)
    nl = tmp_path / "nonl.csv"
    nl.write_text(",".join(R.COLUMNS), encoding="utf-8")                  # a header without a final newline must not swallow the first row
    R.append_ledger(nl, rows)
    assert len(list(csv.DictReader(nl.read_text(encoding="utf-8").splitlines()))) == 1


def test_the_repo_ledger_header_is_the_one_the_reporter_writes():
    assert (ROOT / "docs" / "run_ledger.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == R.COLUMNS


def test_the_modelled_scorer_clock_counts_the_episodes_it_would_cut():
    rows = [{"tool_calls": 10, "completion_tokens": 7000, "resolved": True}, {"tool_calls": 20, "completion_tokens": 9000, "resolved": True},
            {"tool_calls": 24, "completion_tokens": 12000, "resolved": False}, {"tool_calls": 5, "completion_tokens": None, "resolved": True}]
    assert R.modelled_seconds(rows[0], 12, 62) == pytest.approx(120 + 7000 / 62)
    assert R.modelled_seconds(rows[3], 12, 62) == 60                   # no token count: only the calls (a lower bound)
    assert R.over_cap(rows, 6, 100) == (0, 0)
    assert R.over_cap(rows, 12, 100) == (1, 0)                         # 20 calls and 9000 tokens come to exactly 330 s: not over
    assert R.over_cap(rows, 12, 62) == (2, 1)
    assert R.over_cap(rows, 12, 25.5) == (3, 2)


def test_a_pipeline_run_gets_its_stage_split():
    ev = lambda agent, tool=None, **kw: {"agent": agent, "tool": tool, **kw}                                       # noqa: E731
    note = lambda chars: {"agent": "investigate", "event": "output_key_set", "key": "findings", "chars": chars}    # noqa: E731
    trace = {"t1": [ev("investigate", "run_command"), ev("investigate", "read_file"), ev("investigate", "get_status"), note(300),
                    ev("fix", "read_file"), ev("fix", "edit_file"), ev("fix", "submit_patch")],
             "t2": [ev("investigate", "run_command"), note(0), ev("fix", "run_command")],
             "t3": [ev("investigate", "read_file")]}
    s = R.stage_split(trace)
    assert s["calls"] == {"investigate": 1, "fix": 1} and (s["empty_note"], s["fix_without_edit"], s["episodes"]) == (2, 2, 3)
    assert (s["investigate_over_limit"], s["nudged"]) == (0, 0)
    assert R.stage_split({"t": [ev("swe_agent", "read_file")]}) is None and R.stage_split({}) is None
    # an investigation that ignored every rung's limit (9 counted calls; the free ones do not count) and a second pass of the root after a nudge
    long = [ev("investigate", "read_file")] * 9 + [ev("investigate", "get_status"), note(120), ev("fix", "edit_file"), {"event": "nudge", "n": 1}]
    exact = [ev("investigate", "read_file")] * R.INVESTIGATE_MAX_CALLS + [note(80), ev("fix", "edit_file")]
    s = R.stage_split({"long": long, "exact": exact})
    assert (s["investigate_over_limit"], s["nudged"], s["episodes"]) == (1, 1, 2)


def test_reasoning_share_counts_every_agent_of_a_pipeline():
    conv = lambda agent, msgs: {"agent": agent, "event": "conversation", "messages": msgs}                         # noqa: E731
    yes, no = {"role": "assistant", "reasoning_content": "x"}, {"role": "assistant", "reasoning_content": ""}
    assert R.reasoning_share({"t": [conv("investigate", [yes, no]), conv("fix", [yes]), conv("fix", [yes, yes, no])]}) == pytest.approx(3 / 5)
    assert R.reasoning_share({"t": [conv("swe_agent", [yes]), conv("swe_agent", [yes, no, no])]}) == pytest.approx(1 / 3)      # a flat run: the fullest record
    assert R.reasoning_share({}) is None


def test_the_report_prints_the_modelled_clock_and_the_stages_without_task_ids():
    tune, unseen = repo_on("tune"), repo_on("unseen")
    for repo, usable in ((tune, {"t0", "t1"}), (unseen, {"zz_unseen_0", "zz_unseen_1"})):
        ids = sorted(usable)
        runs = [run("lab_a", [(i, True) for i in ids], repo), run("lab_b", [(i, False) for i in ids], repo)]
        lines = []
        R.report(runs, tasks={}, usable=usable, a=args(), out=lines.append)
        text = "\n".join(lines)
        assert "modelled scorer clock, episodes over 330 s" in text and "12 s per call" in text
        if repo == unseen:
            assert "zz_unseen" not in text and unseen not in text


def test_the_report_prints_the_pipeline_counts_and_the_ledger_note_carries_them():
    tune = repo_on("tune")
    ev = lambda agent, tool=None, **kw: {"agent": agent, "tool": tool, **kw}                                       # noqa: E731
    note = {"agent": "investigate", "event": "output_key_set", "key": "findings", "chars": 200}
    trace = {"t0": [ev("investigate", "read_file"), note, ev("fix", "edit_file"), {"event": "nudge", "n": 0},
                    ev("investigate", "read_file"), note, ev("fix", "submit_patch")],
             "t1": [ev("investigate", "read_file")] * 9 + [note, ev("fix", "edit_file")]}
    runs = [run("lab_a", [("t0", True), ("t1", False)], tune, trace), run("lab_b", [("t0", False), ("t1", False)], tune)]
    lines = []
    R.report(runs, tasks={}, usable={"t0", "t1"}, a=args(), out=lines.append)
    text = "\n".join(lines)
    assert "pipeline stages" in text and "investigate over limit 1/2" in text and "nudged 1/2" in text
    row = R.ledger_row(runs[0], R.run_metrics(runs[0], {}, True), "tune", args())
    assert "investigate over 8 calls 1" in row["notes"] and "episodes with a nudge 1" in row["notes"]


def test_an_episode_a_failing_server_ended_is_left_out_of_its_run_and_named():
    """A stop of the notebook cell takes the server away from the episodes that are running; run_eval.py writes them as finished and skips them later."""
    tune = repo_on("tune")
    dead = {"agent": "swe_agent", "event": "model_error", "error": "URLError: <urlopen error [Errno 111] Connection refused>"}
    full = {"agent": "swe_agent", "event": "model_error", "error": "RuntimeError: HTTP 400: maximum context length is 32768 tokens"}
    fatal = {"event": "fatal", "error": "investigate: RemoteDisconnected: Remote end closed connection without response"}
    trace = {"t0": [read("a.py")], "t1": [read("a.py"), dead], "t2": [read("a.py"), full], "t3": [fatal]}
    a_run = run("lab_a", [("t0", True), ("t1", True), ("t2", False), ("t3", False)], tune, trace)
    b_run = run("lab_b", [("t0", False), ("t1", True), ("t2", False), ("t3", True)], tune)
    assert set(R.model_errors(a_run)) == {"t1", "t2", "t3"}
    assert set(R.cut_episodes(a_run)) == {"t1", "t3"}                    # the refused request (HTTP 400) is the arm's own doing and stays
    lines = []
    rows = R.report([a_run, b_run], tasks={}, usable={"t0", "t1", "t2", "t3"}, a=args(), out=lines.append)
    text = "\n".join(lines)
    assert "ended by the server 2   refused 1" in text and "t1 (URLError" in text and "t3 (investigate: RemoteDisconnected" in text
    assert (rows[0]["n_tasks"], rows[0]["resolved"]) == (2, 1) and (rows[1]["n_tasks"], rows[1]["resolved"]) == (4, 2)
    assert "2 episode(s) ended by a failing server" in rows[0]["notes"] and "1 episode(s) with a request the server refused" in rows[0]["notes"]
    assert "difference -1" in text                                       # the pair is compared on t0 and t2, the tasks both runs have
    assert len(a_run["rows"]) == 4                                       # the caller's run is not changed
    lines = []
    rows = R.report([a_run, b_run], tasks={}, usable={"t0", "t1", "t2", "t3"}, a=args(keep_cut=True), out=lines.append)
    text = "\n".join(lines)
    assert rows[0]["n_tasks"] == 4 and "KEPT in every count" in text and "difference +0" in text and "kept in the counts" in rows[0]["notes"]
    # an unseen pool: the count is printed, the task ids are not
    unseen = repo_on("unseen")
    u_run = run("lab_u", [("zz_unseen_0", True), ("zz_unseen_1", False)], unseen, {"zz_unseen_1": [dead]})
    lines = []
    R.report([u_run], tasks={}, usable={"zz_unseen_0", "zz_unseen_1"}, a=args(), out=lines.append)
    text = "\n".join(lines)
    assert "ended by the server 1" in text and "zz_unseen" not in text


def test_a_clean_run_prints_no_line_about_server_errors_and_shows_the_lab_clock():
    tune = repo_on("tune")
    lines = []
    R.report([run("lab_a", [("t0", True), ("t1", False)], tune)], tasks={}, usable={"t0", "t1"}, a=args(), out=lines.append)
    text = "\n".join(lines)
    assert "ended by the server" not in text
    assert "the lab's own clock" in text and "lab_a: 50 tok/s / 20 s / 0" in text        # 1,000 completion tokens in 20 s


def test_a_results_file_with_a_cut_line_a_task_written_twice_or_a_missing_trace_still_loads(tmp_path):
    rows = run("x", [("t0", True), ("t1", False)], repo_on("tune"))["rows"]
    p = tmp_path / "lab_x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows + [dict(rows[0], resolved=False)]) + '\n{"instance_id": "t2", "reso', encoding="utf-8")
    (tmp_path / "lab_x_trace.jsonl").write_text(json.dumps({"instance_id": "t0", "trace": []}) + "\n"
                                                + json.dumps({"instance_id": "t0", "trace": [read("a.py")]}) + '\n{"instance_id": "t1", "tra', encoding="utf-8")
    got = R.load_run(p)
    assert [r["instance_id"] for r in got["rows"]] == ["t1", "t0"] and got["rows"][1]["resolved"] is False     # the LAST line of a task is the one kept...
    assert set(got["trace"]) == {"t0"} and len(got["trace"]["t0"]) == 1                                         # ...with its last trace
    assert (got["duplicates"], got["broken"], got["no_trace"]) == (1, 2, 1)                                     # t1 has a result line and no trace
    lines = []
    R.report([got], tasks={}, usable={"t0", "t1"}, a=args(), out=lines.append)
    assert "twice 1   unreadable 2   without a trace 1" in "\n".join(lines)
    (tmp_path / "lab_x_trace.jsonl").unlink()
    assert R.load_run(p)["no_trace"] == 0                                                                       # a run without a trace file is not flagged
