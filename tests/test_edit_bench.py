"""train/edit_bench.py grading, without a server: the gold call must pass, a corrupted anchor must not."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
import edit_bench  # noqa: E402

DATA = Path("D:/data/edit_sft/validation.jsonl")


def fake(args, name="edit_file", finish="tool_calls"):
    return {"choices": [{"finish_reason": finish, "message": {"content": None, "reasoning_content": "x",
            "tool_calls": [{"function": {"name": name, "arguments": json.dumps(args)}}]}}], "usage": {"completion_tokens": 10}}


@pytest.fixture(scope="module")
def recs():
    if not DATA.exists():
        pytest.skip("validation data not built")
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    return [r for r in rows if r["format"] == "edit_file"][:40]


def test_gold_call_reproduces(recs):
    for r in recs:
        res = edit_bench.grade(r, fake(edit_bench.gold_call(r["messages"])))
        assert res["status"] == "applied" and res["tier"] == "exact" and res["reproduces"], res


def test_wrong_text_is_not_found(recs):
    r = recs[0]
    g = edit_bench.gold_call(r["messages"])
    res = edit_bench.grade(r, fake({**g, "old_string": "zz_not_in_file_zz = 1"}))
    assert res["status"] == "not_found"


def test_extra_key_and_wrong_tool(recs):
    r = recs[0]
    g = edit_bench.gold_call(r["messages"])
    assert edit_bench.grade(r, fake({**g, "filename": "x"}))["status"] == "bad_keys"
    assert edit_bench.grade(r, fake(g, name="write_file"))["status"] == "wrong_tool"
