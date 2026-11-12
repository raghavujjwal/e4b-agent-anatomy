"""mine/ver_metrics.py on made-up conversations: reproduction script before the edit, rerun after it."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mine"))
import ver_metrics as vm  # noqa: E402


def call(i, name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def result(i, payload):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(payload)}


def test_routine_followed():
    msgs = [call(1, "run_command", {"command": "cat > /tmp/repro.py <<'X'\nprint(1)\nX\npython /tmp/repro.py"}), result(1, {"status": "ok"}),
            call(2, "edit_file", {"filepath": "a.py", "old_string": "x", "new_string": "y"}), result(2, {"status": "ok"}),
            call(3, "run_command", {"command": "python /tmp/repro.py"}), result(3, {"status": "ok"})]
    r = vm.run_row(msgs)
    assert r["repro_before_edit"] and r["rerun_after_edit"] and r["applied_edit"] and r["first_edit_call"] == 1


def test_routine_skipped():
    msgs = [call(1, "run_command", {"command": "grep -rn foo ."}), result(1, {"status": "ok"}),
            call(2, "edit_file", {"filepath": "a.py", "old_string": "x", "new_string": "y"}), result(2, {"status": "error", "error_message": "old_string not found in file"})]
    r = vm.run_row(msgs)
    assert not r["repro_before_edit"] and not r["rerun_after_edit"] and not r["applied_edit"]
