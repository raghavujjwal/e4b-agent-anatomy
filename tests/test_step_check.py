"""train/step_check.py: classification and the gate, without a server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import step_check as sc  # noqa: E402


def rows(pairs):
    return [{"recorded": r, "generated": g, "tokens": 10} for r, g in pairs]


def test_classify():
    assert sc.classify('<|channel>thought\nx<channel|><|tool_call>call:read_file{filepath:<|"|>a<|"|>}<tool_call|>') == "read_file"
    assert sc.classify("I think we are done.") == "no_call"


def test_gate_catches_an_adapter_that_always_edits():
    base = sc.summarize(rows([("read_file", "read_file")] * 40 + [("run_command", "run_command")] * 30 + [("submit_patch", "submit_patch")] * 20
                             + [("edit_file", "edit_file")] * 20))
    collapsed = sc.summarize(rows([("read_file", "edit_file")] * 40 + [("run_command", "edit_file")] * 30 + [("submit_patch", "edit_file")] * 20
                                  + [("edit_file", "edit_file")] * 20))
    fails = sc.gate(base, collapsed)
    assert any(f.startswith("read_file") for f in fails) and any(f.startswith("run_command") for f in fails)


def test_gate_is_invalid_when_requests_failed():
    # the 2026-11-02 run: connection refused for every request, base and adapters alike; the old gate passed it
    dead = sc.summarize(rows([("read_file", "request_error")] * 40 + [("edit_file", "request_error")] * 20))
    assert dead["request_error_%"] == 100.0
    assert sc.gate(dead, dead)[0].startswith("INVALID")
    ok = sc.summarize(rows([("read_file", "read_file")] * 40 + [("edit_file", "edit_file")] * 20))
    assert sc.gate(ok, dead)[0].startswith("INVALID")


def test_gate_passes_the_base_model_against_itself_and_ignores_edit_file():
    base = sc.summarize(rows([("read_file", "read_file")] * 20 + [("edit_file", "edit_file")] * 20))
    same = sc.summarize(rows([("read_file", "read_file")] * 18 + [("read_file", "run_command")] * 2 + [("edit_file", "read_file")] * 20))
    assert sc.gate(base, same) == []
