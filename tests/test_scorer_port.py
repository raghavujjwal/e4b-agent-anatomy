"""The local harness must show the model what the scorer shows it (gate G0 in docs/PLAN_E4B_SFT.md).

Fixtures in tests/fixtures were recorded on 2026-11-04 by running the scorer's own code (swegemma 0.2.11,
adk-submission 0.2.13, google-adk 1.36.1) against a mock model server and a fake sandbox. These tests replay the
same scripted conversations through the local runner and tools and require identical request bodies and tool
results. mine/diff_scorer_port.py does the same against the real modules with random inputs; the last test here
runs it when the scorer venv exists.
"""
import copy
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "agent" / "vendor"
FIX = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(VENDOR))

pytest.importorskip("yaml")
pytest.importorskip("numpy")
import eval as EV  # noqa: E402  (agent/vendor/eval.py)
import runner as R  # noqa: E402
import scorer_port as P  # noqa: E402
import tools as T  # noqa: E402

REQS = json.loads((FIX / "scorer_requests_v2rec.json").read_text(encoding="utf-8"))
ENVS = json.loads((FIX / "scorer_envelopes.json").read_text(encoding="utf-8"))
SUB = ROOT / "agent" / "submission_e4b_v2rec"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"


class FakeContainer:
    """The five container methods over an in-memory workspace, answering the commands the scripts use the way
    the capture's fake sandbox did."""

    def __init__(self, ws_files, tree, diff):
        self.files = {"/workspace/" + k: v.encode("utf-8") for k, v in ws_files.items()}
        self.tree, self.diff, self.cmds = tree, diff, []

    def sh(self, cmd, timeout=600):
        self.cmds.append(cmd)
        if "find . -mindepth 1" in cmd:
            return 0, self.tree, ""
        if "git diff" in cmd:
            return 0, self.diff, ""
        if cmd.startswith("cd /workspace && git add") or cmd.startswith("mkdir -p"):
            return 0, "", ""
        if cmd.startswith("grep"):
            return 0, "pkg/mod.py:1:def foo():\n", ""
        if cmd.startswith("false"):
            return 1, "", "boom\n"
        return 0, "ok\n", ""

    def get_bytes(self, path):
        return self.files.get(path)

    def put_bytes(self, path, data):
        self.files[path] = data


class ScriptedServer:
    """OpenAI-compatible mock that replays a list of scripted replies and records every request body."""

    def __init__(self, script):
        self.script, self.bodies = script, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a, **k):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                idx = len(outer.bodies)
                outer.bodies.append(body)
                item = outer.script[idx] if idx < len(outer.script) else {"content": "done"}
                if item.get("http_error"):
                    code, text = item["http_error"]
                    data = json.dumps({"error": {"message": text, "code": code}}).encode()
                    self.send_response(code)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                calls = item.get("calls") or []
                msg = {"role": "assistant", "content": item.get("content")}
                if calls:
                    msg["tool_calls"] = [{"id": f"chatcmpl-tool-{idx}-{i}", "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c.get("args", {}))}}
                                         for i, c in enumerate(calls)]
                if item.get("reasoning"):
                    msg["reasoning"] = msg["reasoning_content"] = item["reasoning"]
                resp = {"id": f"chatcmpl-{idx}", "object": "chat.completion", "created": 0, "model": body.get("model"),
                        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                        "usage": {"prompt_tokens": 100 + idx, "completion_tokens": 20, "total_tokens": 120 + idx}}
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()


def local_tree():
    cfg, cfg_path = R.load_submission(str(SUB))
    return R.resolve_tree(cfg, cfg_path)


# ----------------------------------------------------------------------------------------- request bodies
def test_requests_match_scorer_end_to_end():
    """Same scripted model, same files: the local runner must send the scorer's exact bodies, all six of them."""
    tree = local_tree()
    server = ScriptedServer(REQS["script"])
    try:
        c = FakeContainer(REQS["ws_files"], REQS["tree"], REQS["diff"])
        cfg = REQS["config"]
        budget = T.Budget(tool_calls=cfg["tool_calls"], time_minutes=cfg["time_minutes"], turns=cfg["turns"],
                          command_timeout=cfg["cmd_timeout"])
        tools = T.Tools(c, budget)
        task = {"repo": "demo/repo", "problem_statement": "foo() should return 2, not 1.", "hints_text": ""}
        state = {"problem_description": task["problem_statement"],
                 "_contents": [{"role": "user", "content": EV.first_message(task, budget, tools, c, tree)}]}
        budget.start()
        backend = R.OpenAIBackend(server.url, MODEL)
        stop = R.drive(tree, tools, state, backend, str(SUB), [])
    finally:
        server.close()
    assert stop == "submitted"
    assert len(server.bodies) == len(REQS["requests"]) == 6
    for i, (got, want) in enumerate(zip(server.bodies, REQS["requests"])):
        got = {k: v for k, v in got.items() if k != "return_token_ids"}     # local-only: exact token ids for SFT data
        assert got.keys() == want.keys(), f"request {i}: keys differ: {sorted(set(got) ^ set(want))}"
        for key in want:
            assert got[key] == want[key], f"request {i}: `{key}` differs"


def test_system_message_and_tool_block_are_the_scorers():
    tree = local_tree()
    instr, _ = R.interpolate(tree["instruction"], {"problem_description": "x"})
    system = P.adk_system_text(instr, tree["name"], tree.get("description") or "")
    assert system == REQS["requests"][0]["messages"][0]["content"]
    names = [t for t in tree["tools"] if isinstance(t, str)]
    assert R.schemas_for(names) == REQS["requests"][0]["tools"]


# ----------------------------------------------------------------------------------------- tool results
def mask_clock(s):
    d = json.loads(s)
    for k in ("time_seconds_remaining", "agent_elapsed_seconds"):
        if k in d:
            assert isinstance(d[k], float)
            d[k] = "<float>"
    return d


def test_tool_results_match_scorer_envelopes():
    c = FakeContainer(ENVS["ws_files"], "", "")
    tools = T.Tools(c, T.Budget(tool_calls=24, time_minutes=5.5, turns=40, command_timeout=90))
    tools.b.start()
    calls = [x["calls"][0] for x in ENVS["script"] if x.get("calls")]
    assert len(calls) == len(ENVS["tool_results"])
    for i, (call, want) in enumerate(zip(calls, ENVS["tool_results"])):
        got = tools.call(call["name"], call["args"])
        if i == 1:                                  # missing file: docker's text differs from the fake sandbox's
            assert json.loads(got)["error_type"] == "FileReadError"
        elif i == 7:                                # get_status: clock fields differ; keys, order and types must not
            assert list(json.loads(got)) == list(json.loads(want))
            assert mask_clock(got) == mask_clock(want)
        else:
            assert got == want, f"call {i} ({call['name']})"


def test_docker_not_found_message_shape():
    msg = P.docker_not_found_message("/workspace/nope.py", cid="ab" * 32)
    assert msg.startswith("404 Client Error for http+docker://localhost/v")
    assert 'Not Found ("Could not find the file /workspace/nope.py in container ' in msg


# ----------------------------------------------------------------------------------------- budget semantics
def make_tools(calls=24, turns=40, minutes=5.5, ws=None):
    c = FakeContainer(ws or {"a.py": "x = 1\n"}, "", "")
    t = T.Tools(c, T.Budget(tool_calls=calls, time_minutes=minutes, turns=turns, command_timeout=90))
    t.b.start()
    return t


def test_budget_exceeded_and_free_tools():
    t = make_tools(calls=3)
    for _ in range(3):
        assert json.loads(t.call("read_file", {"filepath": "a.py"}))["status"] == "ok"
    refused = json.loads(t.call("read_file", {"filepath": "a.py"}))
    assert refused == {"status": "error", "error_type": "BudgetExceeded", "error_message": "Tool call budget exhausted (3 calls)"}
    assert t.b.used == 3                                         # a refused call is not counted
    assert json.loads(t.call("get_status", {}))["tool_calls_remaining"] == 0
    assert json.loads(t.call("submit_patch", {}))["status"] == "ok"      # free tools still work


def test_turn_budget_message():
    t = make_tools(turns=2)
    t.b.turns_used = 2
    assert json.loads(t.call("read_file", {"filepath": "a.py"}))["error_message"] == "Turn budget exhausted (2 turns)"


def test_budget_warning_only_on_counted_tools_and_from_the_14th_call():
    t = make_tools(calls=24)
    seen = []
    for i in range(1, 25):
        r = json.loads(t.call("read_file", {"filepath": "a.py"}))
        seen.append(r.get("budget_warning"))
        if i == 14:
            assert r["budget_warning"] == "Only 10 tool call(s) remaining (14/24 used). Finalize your edits and call submit_patch soon."
    assert seen[:13] == [None] * 13 and all(seen[13:])
    assert "budget_warning" not in json.loads(t.call("get_status", {}))
    assert "budget_warning" not in json.loads(t.call("submit_patch", {}))
    small = make_tools(calls=19)                                 # below 20 calls the warning is never attached
    for _ in range(19):
        assert "budget_warning" not in json.loads(small.call("read_file", {"filepath": "a.py"}))


def test_unknown_tool_and_missing_argument_use_adk_text():
    t = make_tools()
    t.available = ["read_file", "edit_file"]
    r = json.loads(t.call("bash", {"command": "ls"}))
    assert r["error_type"] == "ValueError"
    assert r["error"].startswith("Tool 'bash' not found.\nAvailable tools: read_file, edit_file\n\nPossible causes:")
    missing = json.loads(t.call("edit_file", {"filepath": "a.py", "old_string": "x"}))
    assert missing == {"error": "Invoking `edit_file()` failed as the following mandatory input parameters are not present:\n"
                                "new_string\nYou could retry calling this tool, but it is IMPORTANT for you to provide all "
                                "the mandatory parameters."}
    assert t.b.used == 0                                         # neither is counted
    extra = json.loads(t.call("read_file", {"filepath": "a.py", "bogus": 1}))    # unknown keys are dropped
    assert extra["status"] == "ok"


# ----------------------------------------------------------------------------------------- file semantics
def test_edit_keeps_crlf_and_read_folds_it():
    ws = {"w.py": "a = 1\r\nb = 2\r\n"}
    t = make_tools(ws=ws)
    r = json.loads(t.call("edit_file", {"filepath": "w.py", "old_string": "a = 1", "new_string": "a = 5"}))
    assert r["status"] == "ok" and r["strategy"] == "exact"
    assert t.c.files["/workspace/w.py"] == b"a = 5\r\nb = 2\r\n"
    assert json.loads(t.call("read_file", {"filepath": "w.py"}))["content"] == "a = 5\nb = 2"


def test_read_file_truncation_does_not_recompute_end_line():
    long_line = "x" * 6000
    t = make_tools(ws={"big.txt": "\n".join([long_line, long_line, "tail"]) + "\n"})
    r = json.loads(t.call("read_file", {"filepath": "big.txt"}))
    assert r["is_truncated"] is True and len(r["content"]) == 10000
    assert r["end_line"] == 3 and r["total_lines"] == 3         # the scorer reports the requested end, not the cut one


def test_json_envelope_keeps_non_ascii():
    t = make_tools(ws={"u.py": "N = 'caf\u00e9 \u2192'\n"})
    assert "caf\u00e9 \u2192" in t.call("read_file", {"filepath": "u.py"})


def test_protected_files_are_stripped_from_the_submitted_patch():
    diff = ("diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n")
    t = make_tools()
    t.c.diff = diff
    r = json.loads(t.call("submit_patch", {}))
    assert r["files_changed"] == 1 and "tests/test_a.py" not in t.submitted_patch


# ----------------------------------------------------------------------------------------- thinking bridge
@pytest.mark.parametrize("cfg,extra,max_out", [
    ({"thinking_config": {"include_thoughts": True, "thinking_budget": 4096}, "max_output_tokens": 8192},
     {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}, 8192),
    ({"thinking_config": {"thinking_budget": 0}, "max_output_tokens": 8192},
     {"chat_template_kwargs": {"enable_thinking": False}}, 8192),
    ({}, {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}, 16384),
])
def test_thinking_bridge_matches_captures(cfg, extra, max_out):
    merged = P.merge_generation_config(cfg)
    assert merged["max_output_tokens"] == max_out
    assert P.thinking_request_fields(merged)["extra_body"] == extra


def test_thinking_off_request_has_no_think_marker_and_no_budget():
    server = ScriptedServer([{"content": "ok"}])
    try:
        b = R.OpenAIBackend(server.url, MODEL, gen={"temperature": 1.0, "thinking_config": {"thinking_budget": 0}})
        b.step("sys", [{"role": "user", "content": "hi"}], [])
    finally:
        server.close()
    body = server.bodies[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and "thinking_token_budget" not in body


def test_summary_call_carries_thinking_and_seed_but_no_sampling():
    """ADK's summarizer builds its request without a config, but adk-submission attached thinking and seed to the
    model object, so they go out anyway."""
    server = ScriptedServer([{"content": "summary"}])
    gen = {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "max_output_tokens": 8192, "seed": 7,
           "thinking_config": {"thinking_budget": 2048, "include_thoughts": True}}
    try:
        text, _reasoning, _tokens = R.OpenAIBackend(server.url, MODEL, gen=gen).summarize("conversation text")
    finally:
        server.close()
    body = server.bodies[0]
    assert text == "summary"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["thinking_token_budget"] == 2048 and body["seed"] == 7
    for key in ("temperature", "top_p", "top_k", "max_completion_tokens", "tools"):
        assert key not in body, key


def test_missing_reasoning_config_gives_an_actionable_error():
    server = ScriptedServer([{"http_error": (400, "thinking_token_budget is set but reasoning_config is not configured. "
                                                  "Please set --reasoning-config to use thinking_token_budget.")}])
    try:
        b = R.OpenAIBackend(server.url, MODEL, gen={"thinking_config": {"thinking_budget": 4096}})
        with pytest.raises(RuntimeError, match=r"--reasoning-config .*reasoning_start_str"):
            b.step("sys", [{"role": "user", "content": "hi"}], [])
    finally:
        server.close()
    assert len(server.bodies) == 1                           # a 400 is not retried


def test_include_thoughts_false_strips_history_reasoning():
    server = ScriptedServer([{"content": "ok"}])
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_status", "arguments": "{}"}}],
             "reasoning_content": "thought"},
            {"role": "tool", "tool_call_id": "1", "content": "{}"}]
    try:
        b = R.OpenAIBackend(server.url, MODEL, gen={"thinking_config": {"include_thoughts": False, "thinking_budget": 4096}})
        b.step("sys", copy.deepcopy(msgs), [])
        keep = R.OpenAIBackend(server.url, MODEL, gen={"thinking_config": {"include_thoughts": True, "thinking_budget": 4096}})
        keep.step("sys", copy.deepcopy(msgs), [])
    finally:
        server.close()
    stripped, kept = server.bodies[0]["messages"][2], server.bodies[1]["messages"][2]
    assert "reasoning" not in stripped and "reasoning_content" not in stripped
    assert kept["reasoning"] == kept["reasoning_content"] == "thought" and "content" not in kept


# ----------------------------------------------------------------------------------------- the matcher
def test_matcher_tiers():
    exact = P.apply_replacement_core("a\nb\nc\n", "b", "B")
    assert (exact.strategy, exact.new_content) == ("exact", "a\nB\nc\n")
    flex = P.apply_replacement_core("  a = 1\n  b = 2\n", "a = 1", "a = 9")       # exact fails? no: substring -> exact
    assert flex.strategy == "exact"
    flex = P.apply_replacement_core("    x = 1  \n    y = 2\n", "x = 1\ny = 2", "x = 5\ny = 6")
    assert flex.strategy == "flexible" and flex.new_content == "    x = 5\n    y = 6\n"
    rx = P.apply_replacement_core("def f(a,b):\n    return a\n", "def f( a , b ):", "def f(a, b):")
    assert rx.strategy == "regex" and rx.new_content.startswith("def f(a, b):")
    un = P.apply_replacement_core('s = "caf\u00e9"\n', 's = \\"caf\\u00e9\\"', 's = "x"')
    assert un.strategy == "unescaped"
    assert P.apply_replacement_core("a\n", "zzz", "y").error_message.startswith("Failed to replace: old_string not found.")
    assert P.apply_replacement_core("a\n", "a", "a").error_message == "No changes to apply. old_string and new_string are identical."
    with pytest.raises(ValueError, match=r"Target string occurs 2 times in f.py\. Please provide more surrounding context\."):
        P.apply_replacement("a\na\n", "a", "b", filepath="f.py")


def test_compat_apply_replacement_for_train_scripts():
    assert T.apply_replacement("a\nb\n", "b", "c", False) == ("a\nc\n", 1, "exact")
    assert T.apply_replacement("a\nb\n", "q", "c", False)[0] is None


# ----------------------------------------------------------------------------------------- the real modules
SCORER_PY = os.environ.get("SCORER_PYTHON", r"D:\envs\scorer\Scripts\python.exe")


@pytest.mark.skipif(not Path(SCORER_PY).exists(), reason="scorer venv not installed here (see docs/MEASUREMENTS.md)")
def test_differential_against_the_scorers_own_modules():
    r = subprocess.run([SCORER_PY, "-I", "-X", "utf8", str(ROOT / "mine" / "diff_scorer_port.py"), "--cases", "400"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1500:]
