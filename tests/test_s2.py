"""train/s2_*.py: the pieces of the S2 training-set builder that decide what a sample may contain.

The condenser rules and the replay helpers are plain functions and run anywhere; the render test needs the competition model's tokenizer
files (as in test_served.py) and is skipped where they are absent. The WSL replay itself is exercised by the pilot, not here.
"""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
pytest.importorskip("pandas")
import s2_condense as K  # noqa: E402
import s2_replay as R  # noqa: E402


@pytest.mark.parametrize("cmd,patched,expected", [
    ("python -m pytest tests/test_union.py -v", ["tests/test_union.py"], True),
    ("python -m pytest tests/ -q", ["tests/test_a.py"], True),                               # a directory holding a patched test
    ("python -m pytest tests/test_other.py::TestX::test_y -v", ["tests/test_union.py"], False),
    ("python -m pytest test_union.py", ["tests/test_union.py"], True),                       # same file name
    ("python -m pytest /workspace/repo__x__1.0/tests/test_union.py", ["tests/test_union.py"], True),
    ("python -c 'from pkg import f; print(f(1))'", ["tests/test_union.py"], False),
    ("python -m pytest tests/unit -x", ["tests/integration/test_b.py"], False),
])
def test_runs_patched_test(cmd, patched, expected):
    assert K.runs_patched_test(cmd, patched) is expected


def test_tidy_command_and_output_drop_the_teacher_paths():
    cmd = "cd /workspace/foo__bar__1.0 && python -m pytest /workspace/foo__bar__1.0/tests/x.py -v"
    assert K.tidy_command(cmd) == "python -m pytest tests/x.py -v"
    out = ("rootdir: /workspace/foo__bar__1.0\n/opt/conda/envs/testbed/lib/python3.10/site-packages/pytest_asyncio/plugin.py:1: W\n"
           "1 passed\n[The command completed with exit code 0.]\n[Current working directory: /workspace/foo__bar__1.0]\n"
           "[Python interpreter: /opt/conda/envs/testbed/bin/python]\n[Command finished with exit code 0]")
    got = K.tidy_output(out)
    assert got == "rootdir: /workspace\n/usr/local/lib/python3.10/site-packages/pytest_asyncio/plugin.py:1: W\n1 passed"


def test_pytest_banners_and_progress_markers_are_rewritten_at_80_columns():
    wide = ("=" * 245 + " test session starts " + "=" * 245 + "\nplatform linux -- Python 3.9.21, pytest-8.3.5\n"
            "tests/test_a.py ...." + " " * 440 + "[ 50%]\n" + "tests/test_b.py::TestX::test_y PASSED" + " " * 430 + "[100%]\n"
            + "=" * 491 + "\n" + "_" * 240 + " test_z " + "_" * 240 + "\n" + "=" * 230 + " 3 passed in 0.12s " + "=" * 230)
    got = K.pytest_at_80_columns(wide).splitlines()
    assert got[0] == "=" * 29 + " test session starts " + "=" * 30 and len(got[0]) == 80       # what pytest prints at 80 columns
    assert got[1] == "platform linux -- Python 3.9.21, pytest-8.3.5"
    assert got[2] == "tests/test_a.py ...." + " [ 50%]".rjust(79 - len("tests/test_a.py ....")) and len(got[2]) == 79
    assert got[3].endswith(" PASSED" + " [100%]".rjust(79 - len("tests/test_b.py::TestX::test_y PASSED"))) and len(got[3]) == 79
    assert got[4] == "=" * 80
    assert got[5] == "_" * 36 + " test_z " + "_" * 36 and len(got[5]) == 80                  # (80 - 6 - 2) // 2 = 36 on each side: exactly full, no extra character
    assert got[6] == "=" * 30 + " 3 passed in 0.12s " + "=" * 30 + "=" and len(got[6]) == 80
    assert K.pytest_at_80_columns("\n".join(got)) == "\n".join(got)                           # nothing left to change


def test_a_progress_marker_after_a_long_prefix_gets_a_single_space_like_pytest():
    prefix = "tests/very/long/path/test_something_with_a_long_name.py::TestClass::test_method_x PASSED"
    assert len(prefix) > 79 - 7
    assert K.pytest_at_80_columns(prefix + " " * 300 + "[ 99%]") == prefix + " [ 99%]"


def test_output_that_is_already_80_columns_is_not_touched():
    marker = "tests/test_a.py .." + " [100%]".rjust(79 - len("tests/test_a.py .."))              # pytest right-aligns the marker to column 79
    out = ("============================= test session starts ==============================\n"
           f"collected 2 items\n\n{marker}\n\n"
           "============================== 2 passed in 0.03s ===============================")
    assert len(marker) == 79 and K.pytest_at_80_columns(out) == out


def test_a_multi_line_command_echo_is_removed_from_the_output():
    cmd = 'cd /workspace/pre-commit__hooks__1.0 && python3 -c "\nimport os\n\nprint(os.getcwd())\n"'
    echo = 'cd /workspace/pre-commit__hooks__1.0 && python3 -c "\nimport os\nprint(os.getcwd())\n"\n'          # the terminal drops the blank line
    out = echo + "/workspace/pre-commit__hooks__1.0\n[The command completed with exit code 0.]"
    assert K.strip_command_echo(cmd, out) == "/workspace/pre-commit__hooks__1.0\n[The command completed with exit code 0.]"


def test_output_without_an_echo_and_single_line_commands_are_left_alone():
    assert K.strip_command_echo("python -m pytest tests/x.py -q", "1 passed in 0.1s") == "1 passed in 0.1s"
    assert K.strip_command_echo('python -c "\nprint(1)\n"', "1\n") == "1\n"                       # starts with something else
    assert K.strip_command_echo('python -c "\nprint(1)\n"', 'python -c "\nprint(2)\n"\n2') == 'python -c "\nprint(2)\n"\n2'   # a different script


@pytest.mark.parametrize("out,failed", [
    ("3 passed in 0.52s", False),
    ("collected 2 items\n\ntests/test_a.py::test_error_handling PASSED\n1 warning, 0 failed", False),
    ("  /usr/local/lib/python3.9/site-packages/x.py:3: PytestUnraisableExceptionWarning: boom", False),
    ("Ran 2 tests in 0.001s\n\nOK", False),
    ("Traceback (most recent call last):\n  File \"x\", line 1\nValueError: nope", True),
    ("Testing:\nError: Error tokenizing 'select'", True),
    ("FAILED tests/test_a.py::test_b - assert 1 == 2", True),
    ("2 failed, 3 passed in 0.4s", True),
    ("Case 3: ✗ wrong", True),
    ("pkg.errors.CustomError: bad input", True),
])
def test_shows_failure(out, failed):
    assert K.shows_failure(out) is failed


def test_eval_pool_repositories_are_excluded_by_name_and_by_fork_name():
    assert K.is_eval_repo("fastapi/fastapi") and K.is_eval_repo("Textualize/rich")
    assert K.is_eval_repo("someone/requests")             # a fork of an evaluation repository
    assert not K.is_eval_repo("tox-dev/tox")


def test_find_span_needs_a_unique_match():
    text = "a\nb\nc\nb\n"
    assert R.find_span(text, "a\nb") == (1, 2)
    assert R.find_span(text, "b") is None                  # twice
    assert R.find_span(text, "zzz") is None


def test_read_window_for_a_long_file_starts_above_the_hit_and_covers_the_edit():
    text = "\n".join(f"line {i}" for i in range(1, 501))
    args, why = R.read_args(text, [(300, 304)], hit_line=290)
    assert why is None and args["start_line"] == 270 and args["end_line"] >= 307
    assert args["end_line"] - args["start_line"] + 1 <= R.CAP


def test_short_files_are_read_whole():
    args, why = R.read_args("x = 1\ny = 2\n", [(1, 1)], hit_line=None)
    assert args == {} and why is None


def test_a_window_over_the_cap_is_refused():
    text = "\n".join(f"line {i}" for i in range(1, 801))
    args, why = R.read_args(text, [(100, 100), (400, 400)], hit_line=100)
    assert args is None and why == "window_too_big"


def _shard(path, rows, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")


def test_resume_counts_a_kept_record_and_a_rule_drop_as_settled_but_not_an_accident(tmp_path):
    _shard(tmp_path / "replayed.shard0.jsonl",
           [{"instance_id": "a", "messages": []}, {"instance_id": "b", "dropped": "no_targeted_search_term"},
            {"instance_id": "c", "dropped": "fetch:fetch failed (6): Could not resolve host"},
            {"instance_id": "d", "dropped": "exception:RuntimeError:repo_too_large"},
            {"instance_id": "e", "dropped": "fetch:fetch failed (63): Maximum file size exceeded"},
            {"instance_id": "f", "dropped": "child_failed"}],
           tail='{"instance_id": "g", "dro')                   # the line a killed run was writing
    assert R.settled(tmp_path) == {"a", "b", "c", "d", "e"}                       # first pass: a failed download is settled (slow or big repository)
    assert R.settled(tmp_path, retry_fetch=True) == {"a", "b", "d", "e"}          # second pass tries it again; the size cap (63) never changes


def test_merge_never_lets_a_drop_replace_a_kept_record(tmp_path):
    _shard(tmp_path / "replayed.shard0.jsonl", [{"instance_id": "a", "messages": [1]}, {"instance_id": "c", "dropped": "child_failed"}])
    _shard(tmp_path / "replayed.shard1.jsonl", [{"instance_id": "c", "messages": [2]}, {"instance_id": "a", "dropped": "exception:OSError"}])
    assert R.merge(tmp_path) == 0
    merged = {r["instance_id"]: r for r in map(json.loads, (tmp_path / "replayed.jsonl").read_text(encoding="utf-8").splitlines())}
    assert set(merged) == {"a", "c"} and merged["a"]["messages"] == [1] and merged["c"]["messages"] == [2]


def test_resume_closes_a_cut_off_line_and_skips_what_is_settled(tmp_path):
    jobs, out = tmp_path / "jobs.jsonl", tmp_path / "replayed.shard0.jsonl"
    jobs.write_text(json.dumps({"instance_id": "a"}) + "\n", encoding="utf-8")
    _shard(out, [{"instance_id": "a", "messages": []}], tail='{"instance_id": "b", "dro')
    assert R.main(["--jobs", str(jobs), "--out", str(out), "--resume", "--cache", str(tmp_path / "c"), "--tmp", str(tmp_path / "t")]) == 0
    assert out.read_bytes().endswith(b"\n") and out.read_text(encoding="utf-8").count("\n") == 2


def test_a_shard_is_chosen_by_position_in_the_whole_list_before_settled_jobs_are_dropped(tmp_path, capsys):
    """Shards that start at different moments see different settled sets; the split must not depend on that or two shards could take the same job."""
    jobs, out = tmp_path / "jobs.jsonl", tmp_path / "replayed.shard0.jsonl"
    jobs.write_text("".join(json.dumps({"instance_id": f"i{k}"}) + "\n" for k in range(10)), encoding="utf-8")
    _shard(tmp_path / "replayed.shard7.jsonl", [{"instance_id": "i0", "messages": []}, {"instance_id": "i1", "messages": []},
                                                {"instance_id": "i3", "messages": []}])
    R.main(["--jobs", str(jobs), "--out", str(out), "--resume", "--shard", "0/3", "--cache", str(tmp_path / "c"), "--tmp", str(tmp_path / "t")])
    # shard 0 of 3 holds positions 0, 3, 6, 9 (four jobs); i0 and i3 are settled, i1 belongs to another shard
    assert "3 instances settled, 2 of this shard's 4 skipped" in capsys.readouterr().out


# ---- render ---------------------------------------------------------------------------------------------------------------
TOK_DIR = None
try:
    import served as SV
    TOK_DIR = Path(os.environ.get("E4B_TOKENIZER_DIR") or SV.DEFAULT_TOKENIZER_DIR)
except Exception:          # transformers missing
    pass
need_tok = pytest.mark.skipif(TOK_DIR is None or not (TOK_DIR / "chat_template.jinja").exists(), reason="E4B tokenizer files not present")
REQS = json.loads((ROOT / "tests" / "fixtures" / "scorer_requests_v2rec.json").read_text(encoding="utf-8"))["requests"]


def _fixture_conversation(closing):
    """The recorded scorer conversation (without its system message) plus a closing text. Its scripted 'verification' is the command `false`, which the renderer rightly
    refuses (it checks nothing), so it is replaced by a test run that names the edited module."""
    msgs = copy.deepcopy(REQS[-1]["messages"])[1:]
    for m in msgs:
        m.pop("reasoning", None)
        m.pop("reasoning_content", None)
        for tc in m.get("tool_calls") or []:
            if tc["function"]["name"] == "run_command" and json.loads(tc["function"]["arguments"])["command"] == "false":
                tc["function"]["arguments"] = json.dumps({"command": "python -m pytest tests/test_mod.py"})
    return msgs + [{"role": "assistant", "content": closing}]


@need_tok
def test_render_trains_on_the_calls_and_the_closing_text_only():
    pytest.importorskip("transformers")
    import s2_render as M
    tok = SV.load_tokenizer()
    sys_text, tools = M.system_text(ROOT / "agent" / "submission_e4b_v2rec")
    msgs = _fixture_conversation("I changed pkg/mod.py and submitted the patch.")
    rec = {"instance_id": "x__y-1", "repo": "x/y", "messages": msgs}
    for think in (True, False):
        out, bad = M.render_one(tok, rec, sys_text, tools, think)
        assert bad == "" and out is not None
        trained = tok.decode([i for i, m in zip(out["input_ids"], out["loss_mask"]) if m])
        n_calls = sum(1 for m in msgs if m.get("tool_calls"))
        assert trained.count("<|tool_call>") == n_calls and trained.count("<|tool_response>") == n_calls
        assert trained.endswith("I changed pkg/mod.py and submitted the patch.<turn|>")
        assert "response:" not in trained                  # tool results are context, never targets
        assert ("<|think|>" in out["text"]) is think


def _tool(stdout):
    return [{"role": "tool", "tool_call_id": "a", "content": json.dumps({"status": "ok", "stdout": stdout})}]


def test_audit_rejects_a_teacher_path_and_an_evaluation_pool_repository():
    import s2_render as M
    assert M.audit_messages(_tool("fine")) == ""
    assert "teacher path" in M.audit_messages(_tool("/opt/conda/envs/testbed/bin/python"))
    assert "evaluation-pool" in M.audit_messages(_tool("see https://github.com/fastapi/fastapi/issues/1"))
    assert "evaluation-pool" in M.audit_messages(_tool("/workspace/scrapy/spiders.py"))
    assert M.audit_messages(_tool("import urllib3\nfrom fastapi import FastAPI")) == ""      # a library used by the sample's own repository is fine
    assert M.audit_messages([{"role": "tool", "tool_call_id": "a", "content": "not json"}]) == "tool result is not JSON"


def test_render_and_condense_agree_on_the_evaluation_repositories():
    import s2_render as M
    assert {r.lower() for r in K.EVAL_REPOS} == set(M.EVAL_REPOS)


@need_tok
def test_render_leaves_out_a_sample_over_the_token_limit():
    import s2_render as M
    tok = SV.load_tokenizer()
    sys_text, tools = M.system_text(ROOT / "agent" / "submission_e4b_v2rec")
    rec = {"instance_id": "x__y-1", "repo": "x/y", "messages": _fixture_conversation("done.")}
    out, bad = M.render_one(tok, rec, sys_text, tools, False, max_tokens=500)
    assert out is None and bad == "over 500 tokens"
    out, bad = M.render_one(tok, rec, sys_text, tools, False)
    assert bad == "" and out["n_tokens"] <= M.MAX_TOKENS


@pytest.mark.parametrize("lic,dropped", [(None, True), ("None", True), ("", True), ("MIT License", False), ("Mozilla Public License 2.0", False)])
def test_unlicensed_repositories_are_left_out_only_when_asked(lic, dropped):
    import s2_render as M
    rec = {"instance_id": "x__y-1", "repo": "x/y", "license": lic, "messages": []}
    if dropped:
        assert M.render_one(None, rec, "", [], True, drop_unlicensed=True) == (None, "repository has no declared licence")
    else:
        with pytest.raises(Exception):                           # past the licence check the render needs a real tokenizer: it was not refused for its licence
            M.render_one(None, rec, "", [], True, drop_unlicensed=True)
    with pytest.raises(Exception):                               # without the option nothing is refused for its licence, whatever it is
        M.render_one(None, dict(rec, license=None), "", [], True)


def test_render_refuses_a_sample_from_an_evaluation_repository():
    import s2_render as M
    out, bad = M.render_one(None, {"instance_id": "x", "repo": "Textualize/rich", "messages": []}, "", [], True)
    assert out is None and bad == "evaluation-pool repository"
