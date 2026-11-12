"""train/s2_audit.py: each detector must flag the defect it exists for and pass a clean conversation. No tokenizer needed (a fake control-token map)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
pytest.importorskip("transformers")
import s2_audit as A  # noqa: E402

CID = {name: 100 + k for k, name in enumerate(A.CONTROL)}
BOS, TURN, TURN_END, CALL, CALL_END, RESP, RESP_END, THINK = (CID[n] for n in A.CONTROL)
X = 5                                                                    # any ordinary token


def episode(n_calls=2, think=False):
    """ids and loss mask of: bos, system turn, user turn, a model turn of n_calls x (call, stop token, response), then the closing text."""
    ids = [BOS] + ([THINK] if think else []) + [TURN, X, TURN_END, TURN, X, X, TURN_END, TURN, X]
    mask = [0] * len(ids)
    for _ in range(n_calls):
        ids += [CALL, X, X, CALL_END, RESP, X, X, RESP_END]           # the model writes the first five; the response body is context
        mask += [1] * 5 + [0] * 3
    ids += [X, X, TURN_END]
    mask += [1, 1, 1]
    return ids, mask


def test_a_clean_episode_passes_the_token_and_mask_checks():
    ids, mask = episode(3, think=True)
    assert A.control_problems(ids, True, CID, 3) == []
    assert A.mask_problems(ids, mask, CID, 3) == []


def test_an_injected_control_token_is_flagged():
    """A repository file that contains the text `<tool_response|>` becomes a real token when the sample is tokenized."""
    ids, _ = episode(2)
    ids.insert(10, RESP_END)
    bad = A.control_problems(ids, False, CID, 2)
    assert any("<tool_response|>" in p for p in bad)
    assert A.control_problems(ids, True, CID, 2) != []                   # thinking claimed but no <|think|> in the ids


def test_a_sample_that_does_not_start_with_bos_is_flagged():
    ids, _ = episode(1)
    assert any("bos" in p for p in A.control_problems([X] + ids[1:], False, CID, 1))


def test_a_loss_run_that_starts_mid_call_is_flagged():
    ids, mask = episode(2)
    first = mask.index(1)
    mask[first] = 0                                                      # the run no longer starts with <|tool_call>
    assert any("not exactly one tool call" in p for p in A.mask_problems(ids, mask, CID, 2))
    ids2, mask2 = episode(2)
    mask2[-1] = 0                                                        # closing text without <turn|>
    assert any("<turn|>" in p for p in A.mask_problems(ids2, mask2, CID, 2))
    assert any("loss runs" in p for p in A.mask_problems(*episode(2), CID, 3))


DECLARED = [
    {"type": "function", "function": {"name": "read_file", "parameters": {"properties": {"filepath": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "edit_file", "parameters": {"properties": {"filepath": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["filepath", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "run_command", "parameters": {"properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "submit_patch", "parameters": {"properties": {}, "required": []}}},
]


def call(i, name, **args):
    return {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "id": f"c{i}", "function": {"name": name, "arguments": json.dumps(args)}}]}


def result(i, **d):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps({"status": "ok", **d})}


def good_messages():
    return [{"role": "user", "content": "task"},
            call(0, "read_file", filepath="a.py", start_line=1, end_line=40), result(0, filepath="a.py", content="x\n" * 39, start_line=1, end_line=40, total_lines=90, is_truncated=False),
            call(1, "edit_file", filepath="a.py", old_string="x", new_string="y"), result(1, filepath="a.py", occurrences=1, strategy="exact", diff="...", is_truncated=False),
            call(2, "run_command", command="python -c 1"), result(2, stdout="ok", stderr="", exit_code=0),
            call(3, "submit_patch"), result(3, patch_size=10, files_changed=1),
            {"role": "assistant", "content": "I changed a.py and submitted the patch."}]


def test_a_clean_conversation_passes_the_message_checks():
    msgs = good_messages()
    assert A.argument_problems(msgs, DECLARED) == []
    assert A.result_problems(msgs) == ([], [])


@pytest.mark.parametrize("edit,expect", [({"strategy": "fuzzy"}, "strategy 'fuzzy'"), ({"occurrences": 2}, "occurrences 2")])
def test_an_edit_that_is_not_an_exact_single_replacement_is_flagged(edit, expect):
    msgs = good_messages()
    d = json.loads(msgs[4]["content"])
    d.update(edit)
    msgs[4]["content"] = json.dumps(d)
    bad, _ = A.result_problems(msgs)
    assert any(expect in p for p in bad) and any("not an exact single replacement" in p for p in bad)


def test_the_patch_must_touch_one_to_three_files():
    msgs = good_messages()
    msgs[8]["content"] = json.dumps({"status": "ok", "patch_size": 10, "files_changed": 4})
    assert any("4 files" in p for p in A.result_problems(msgs)[0])


@pytest.mark.parametrize("what", ["window", "result", "stdout"])
def test_results_over_the_caps_are_flagged(what):
    msgs = good_messages()
    if what == "window":
        msgs[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"filepath": "a.py", "start_line": 1, "end_line": 200})
    elif what == "result":
        msgs[2]["content"] = json.dumps({"status": "ok", "content": "x" * 10001, "start_line": 1, "end_line": 2})
    else:
        msgs[6]["content"] = json.dumps({"status": "ok", "stdout": "x" * 5001, "stderr": "", "exit_code": 0})
    assert A.result_problems(msgs)[1] != []


def test_a_call_with_an_undeclared_missing_or_mistyped_argument_is_flagged():
    msgs = good_messages()
    msgs[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"filepath": "a.py", "start": 3})
    assert any("undeclared" in p for p in A.argument_problems(msgs, DECLARED))
    msgs = good_messages()
    msgs[3]["tool_calls"][0]["function"]["arguments"] = json.dumps({"filepath": "a.py", "old_string": "x"})
    assert any("missing ['new_string']" in p for p in A.argument_problems(msgs, DECLARED))
    msgs = good_messages()
    msgs[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"filepath": "a.py", "start_line": "1"})
    assert any("expected integer" in p for p in A.argument_problems(msgs, DECLARED))
    msgs = good_messages()
    msgs[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"filepath": "a.py", "start_line": True})
    assert any("expected integer" in p for p in A.argument_problems(msgs, DECLARED))


def verify_messages(path, cmd):
    return [call(0, "edit_file", filepath=path, old_string="a", new_string="b"), result(0, strategy="exact", occurrences=1),
            call(1, "run_command", command=cmd), result(1, stdout="", stderr="", exit_code=0), call(2, "submit_patch")]


@pytest.mark.parametrize("path,cmd,ok", [
    ("src/pkg/core.py", "python -m pytest tests/test_core.py -v", True),
    ("src/pkg/core.py", 'python -c "from pkg.core import run; run()"', True),           # names the package
    ("webauthn/helpers/structs.py", "python3 -c \"mv = memoryview(b'x'); print(type(mv))\"", False),
    ("pkg/mod.py", 'python -c "print(round(500, -3))"', False),
    ("pkg/mod.py", "cd /workspace && python -m unittest discover", True),
])
def test_a_verification_must_run_tests_or_name_the_edited_code(path, cmd, ok):
    assert (A.unrelated_verification(verify_messages(path, cmd)) == "") is ok


def test_no_command_after_the_last_edit_is_not_an_unrelated_verification():
    msgs = verify_messages("pkg/mod.py", "ls")[:2] + [call(2, "submit_patch")]
    assert A.unrelated_verification(msgs) == ""


Q = '<|"|>'


def test_key_order_must_match_the_served_template():
    ok = (f"<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},new_string:{Q}y{Q},old_string:{Q}x{Q}}}<tool_call|><|tool_response>"
          f"<|tool_call>call:read_file{{end_line:9,filepath:{Q}a.py{Q},start_line:1}}<tool_call|><|tool_response>"
          f"<|tool_call>call:read_file{{filepath:{Q}b.py{Q}}}<tool_call|><|tool_response>")
    assert A.order_problems(ok) == []
    swapped = f"<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},old_string:{Q}x{Q},new_string:{Q}y{Q}}}<tool_call|>"
    assert A.order_problems(swapped) == ["edit_file keys out of order"]
    bad_read = f"<|tool_call>call:read_file{{filepath:{Q}a.py{Q},end_line:9,start_line:1}}<tool_call|>"
    assert A.order_problems(bad_read) == ["read_file keys out of order"]


def test_leaks_flag_teacher_paths_and_evaluation_repositories_but_not_a_plain_import():
    assert A.leak_problems("File /opt/conda/envs/testbed/lib/python3.9/x.py") != []
    assert A.leak_problems("cd /testbed && pytest") != []
    assert A.leak_problems("see https://github.com/psf/requests/issues/12") != []
    assert A.leak_problems("path /workspace/scrapy/settings.py") != []
    assert A.leak_problems("import urllib3\nimport requests\n") == []
    assert A.leak_problems("/workspace/owner__repo__1234") != []


CAPTURED = json.loads(A.CAPTURE.read_text(encoding="utf-8"))["requests"][0]


def test_the_first_message_is_split_into_task_fields_and_fixed_text():
    u = CAPTURED["messages"][1]["content"]
    repo, statement, middle, tree = A.split_first_message(u)
    assert (repo, statement) == ("demo/repo", "foo() should return 2, not 1.")
    assert tree == "./README.md\n./pkg\n./pkg/__init__.py\n./pkg/mod.py"
    assert middle.startswith("\n\n## Task Budget") and middle.endswith("(up to 3 levels):\n```\n") and "Do NOT attempt to run pip install" in middle
    assert A.first_message_problems(u, "demo/repo", middle) == []
    # a different task in the same template is fine: only the three task fields change
    other = u.replace("demo/repo", "a/b.c").replace("foo() should return 2, not 1.", "Line one\n\nLine two\n## Task Budget is mentioned here").replace("./pkg/mod.py", "./x/y.py")
    assert A.first_message_problems(other, "a/b.c", middle) == []


@pytest.mark.parametrize("edit,expect", [
    (lambda u: u.replace("Tool calls allowance: 24 calls", "Tool calls allowance: 25 calls"), "fixed text"),
    (lambda u: u.replace("File view limit: 150", "File view limit: 200"), "fixed text"),
    (lambda u: u.replace("demo/repo", "other/repo", 1), "repository field"),
    (lambda u: u.replace("Problem Statement:\nfoo() should return 2, not 1.", "Problem Statement:\n"), "empty"),
    (lambda u: u[len("You are "):], "shape"),
    (lambda u: u + "extra", "shape"),
])
def test_a_first_message_that_differs_from_the_capture_is_flagged(edit, expect):
    u = CAPTURED["messages"][1]["content"]
    middle = A.split_first_message(u)[2]
    assert any(expect in p for p in A.first_message_problems(edit(u), "demo/repo", middle))


def test_the_system_turn_must_be_shared_up_to_the_user_turn():
    T = 9                                                                # the id of <|turn>: the second one starts the user turn
    a, b = [T, 1, 2, T, 4], [T, 1, 2, T, 7, 7]
    assert A.shared_system_turn({False: [a, b]}, T) == {False: (4, 3, True)}
    c = [T, 1, 8, T, 4]
    lcp, start, ok = A.shared_system_turn({False: [a, c], True: [[5] + a, [5] + b]}, T)[False]
    assert (lcp, start, ok) == (2, 3, False)
    assert A.shared_system_turn({False: [a, c], True: [[5] + a, [5] + b]}, T)[True] == (5, 4, True)       # one extra token (<|think|>) moves the user turn by one
