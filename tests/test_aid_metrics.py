"""mine/aid_metrics.py on a made-up conversation: skill calls, the gold file listed and opened, a failed edit followed by edit_retry."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mine"))
import aid_metrics as am  # noqa: E402


def call(i, name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def result(i, payload):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(payload)}


def test_aid_row_counts_the_skill_use():
    msgs = [
        call(1, "run_skill_script", {"skill_name": "repo_aid", "file_path": "scripts/find_files.py", "args": {"query": "x"}}),
        result(1, {"status": "ok", "stdout": "Files ...:\n1. pkg/mod.py   [f]\n2. pkg/other.py\n"}),
        call(2, "read_file", {"filepath": "pkg/mod.py"}),
        result(2, {"status": "ok", "content": "x = 1\n"}),
        call(3, "edit_file", {"filepath": "pkg/mod.py", "old_string": "x = 2", "new_string": "x = 3"}),
        result(3, {"status": "error", "error_type": "FileEditError", "error_message": "old_string not found in file"}),
        call(4, "run_skill_script", {"skill_name": "repo_aid", "file_path": "scripts/edit_retry.py", "args": {"filepath": "pkg/mod.py"}}),
        result(4, {"status": "ok", "stdout": "OK: edit applied to pkg/mod.py (matched by: similar)."}),
    ]
    r = am.aid_row(msgs, {"pkg/mod.py"})
    assert r["find_called"] and r["find_in_first3"] and r["gold_listed"] and r["seen"]
    assert (r["nf_edits"], r["retry_follow"], r["retry_calls"], r["retry_ok"]) == (1, 1, 1, 1)


def test_report_applies_the_rules():
    row = {"find_called": True, "find_in_first3": True, "gold_listed": True, "seen": True, "nf_edits": 2, "retry_follow": 1, "retry_calls": 1, "retry_ok": 0,
           "resolved": False, "multi": False, "short": True}
    rep = am.report([row] * 10)
    assert rep["rules"]["compliance >= 80% and in first 3 calls"] and rep["rules"]["short seen >= 40%"]
    assert rep["rules"]["retry follow >= 50%"] and not rep["rules"]["retry ok >= 40%"] and not rep["rules"]["resolved >= 10 of 66"]
