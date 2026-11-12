"""train/s4_gates.py: the offline gates G3, G4 and G5.

The parser and the gate arithmetic are plain functions and run anywhere. The tests that need the competition model's tokenizer files (as in test_served.py) or
the harness are skipped where those are absent. The end-to-end test talks to a fake /v1/completions server on localhost; no GPU and no model are involved.
"""
import copy
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import s4_gates as G  # noqa: E402

Q = G.QUOTE


def call(name, **kw):
    """A call as the chat template writes it: keys in alphabetical order, strings between <|"|> marks."""
    def val(v):
        return f"{Q}{v}{Q}" if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else str(v)
    return f"<|tool_call>call:{name}{{" + ",".join(f"{k}:{val(v)}" for k, v in sorted(kw.items())) + "}<tool_call|>"


# ---- the parser -------------------------------------------------------------------------------------------------------------
def test_parse_args_reads_strings_numbers_booleans_and_null():
    body = f"allow_multiple:true,end_line:12,filepath:{Q}a/b.py{Q},new_string:{Q}x, y: {{z}}\nline2{Q},old_string:{Q}{Q},ratio:1.5,start_line:3,x:null"
    assert G.parse_args(body) == {"allow_multiple": True, "end_line": 12, "filepath": "a/b.py", "new_string": "x, y: {z}\nline2", "old_string": "",
                                  "ratio": 1.5, "start_line": 3, "x": None}


def test_parse_args_of_a_tool_without_parameters_is_empty():
    assert G.parse_args("") == {}


@pytest.mark.parametrize("body", [
    "filepath",                                   # no colon
    f"filepath:{Q}a.py",                          # string not closed
    "start_line:abc",                             # not a number, boolean or null
    f"a:{Q}x{Q}b:1",                              # comma missing
    f"a:{Q}x{Q},",                                # trailing comma
    "1a:2",                                       # not an identifier
    "a:1,,b:2",                                   # empty argument
    "start_line: 100",                            # a space the template never writes (the base model does it in 4 of 2,529 generations)
    "new_string:license = 1,old_string:x",        # a string without its marks
])
def test_parse_args_refuses_what_the_template_would_not_write(body):
    assert G.parse_args(body) is None


def test_first_call_finds_the_call_after_a_thought_and_ignores_what_follows():
    text = "<|channel>thought\nI should look at the file.<channel|>" + call("read_file", filepath="a.py") + "<|tool_response>"
    assert G.first_call(text) == ("read_file", {"filepath": "a.py"})
    assert G.first_call("just text") is None
    assert G.first_call(f"<|tool_call>call:read_file{{filepath:{Q}a.py{Q}") is None             # never closed
    assert G.first_call(call("read_file", filepath="a.py") + call("run_command", command="ls")) == ("read_file", {"filepath": "a.py"})


def test_a_brace_inside_a_string_does_not_end_the_call():
    c = call("edit_file", filepath="a.py", old_string="def f():\n    return {}\n", new_string="x = {1: 2}\n")
    assert G.first_call(c) == ("edit_file", {"filepath": "a.py", "old_string": "def f():\n    return {}\n", "new_string": "x = {1: 2}\n"})


@pytest.mark.parametrize("text", [
    call("read_file", filepath="a.py"),
    call("read_file", end_line=50, filepath="a.py", start_line=1),
    call("edit_file", filepath="a.py", new_string="b", old_string="a"),
    call("edit_file", allow_multiple=True, filepath="a.py", new_string="b", old_string="a"),
    call("write_file", content="x\n", filepath="a.py"),
    call("run_command", command="python -m pytest tests/test_a.py -x"),
    call("get_status"),
    call("submit_patch"),
    "<|channel>thought\nlet me think<channel|>" + call("run_command", command="ls") + "<|tool_response>",
])
def test_well_formed_accepts_one_declared_call(text):
    assert G.well_formed(text) == (True, "")


@pytest.mark.parametrize("text,why", [
    ("I give up.", "0 tool calls"),
    (call("read_file", filepath="a.py") + call("read_file", filepath="b.py"), "2 tool calls"),
    (f"<|tool_call>call:read_file{{filepath:{Q}a.py{Q}", "call not closed"),
    (call("delete_file", filepath="a.py"), "undeclared tool delete_file"),
    (f"<|tool_call>call:read_file{{filepath}}<tool_call|>", "arguments do not parse"),
    (call("edit_file", filepath="a.py", old_string="b"), "missing new_string"),
    (call("read_file", filepath="a.py", mode="r"), "undeclared parameter mode"),
    (call("read_file", filepath="a.py", start_line="3"), "start_line is not int"),
    (call("read_file", filepath="a.py", start_line=True), "start_line is not int"),
    (call("read_file", filepath=3), "filepath is not str"),
    (call("edit_file", allow_multiple="yes", filepath="a", new_string="b", old_string="c"), "allow_multiple is not bool"),
])
def test_well_formed_names_what_is_wrong(text, why):
    assert G.well_formed(text) == (False, why)


def test_the_tool_table_matches_the_declarations_the_harness_sends():
    try:
        import s2_render as M
        _, tools = M.system_text(ROOT / "agent" / "submission_e4b_v2rec")
    except ImportError as e:            # the vendored harness needs packages this interpreter lacks
        pytest.skip(str(e))
    declared = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    assert set(declared) == set(G.TOOLS)
    py = {"string": str, "integer": int, "boolean": bool}
    for name, (req, opt) in G.TOOLS.items():
        props = declared[name].get("properties") or {}
        assert set(props) == set(req) | set(opt), name
        assert set(declared[name].get("required") or []) == set(req), name
        for key, typ in {**req, **opt}.items():
            assert py[props[key]["type"]] is typ, (name, key)


# ---- the parser against the real chat template -------------------------------------------------------------------------------
try:
    import served as SV
    TOK_DIR = Path(os.environ.get("E4B_TOKENIZER_DIR") or SV.DEFAULT_TOKENIZER_DIR)
except Exception:                       # transformers missing
    TOK_DIR = None
need_tok = pytest.mark.skipif(TOK_DIR is None or not (TOK_DIR / "chat_template.jinja").exists(), reason="E4B tokenizer files not present")
REQS = json.loads((ROOT / "tests" / "fixtures" / "scorer_requests_v2rec.json").read_text(encoding="utf-8"))["requests"]


@need_tok
def test_the_parser_reads_every_call_the_chat_template_renders():
    pytest.importorskip("transformers")
    import s2_render as M
    tok = SV.load_tokenizer()
    _, tools = M.system_text(ROOT / "agent" / "submission_e4b_v2rec")
    msgs = copy.deepcopy(REQS[-1]["messages"])
    for m in msgs:
        m.pop("reasoning", None)
        m.pop("reasoning_content", None)
    sent = [(tc["function"]["name"], json.loads(tc["function"]["arguments"])) for m in msgs for tc in m.get("tool_calls") or []]
    assert len(sent) >= 4
    text = SV.render(tok, msgs, tools, enable_thinking=False, add_generation_prompt=False)
    spans = [s + "<tool_call|>" for s in text.split("<tool_call|>")[:-1]]
    spans = [s[s.index("<|tool_call>"):] for s in spans if "<|tool_call>" in s]
    assert len(spans) == len(sent)
    for span, (name, args) in zip(spans, sent):
        assert G.well_formed(span) == (True, ""), span[:200]
        assert G.first_call(span) == (name, args)


# ---- repeat / re-read ---------------------------------------------------------------------------------------------------------
FAILED = {"name": "edit_file", "args": {"filepath": "a.py", "old_string": "foo", "new_string": "bar"}}


@pytest.mark.parametrize("text,kind", [
    (call("edit_file", filepath="a.py", old_string="foo", new_string="bar"), "repeat"),
    (call("edit_file", filepath="a.py", old_string="foo", new_string="baz"), "repeat"),         # an anchor that was not found is not found with another replacement
    (call("edit_file", filepath="a.py", old_string="foo2", new_string="bar"), "edit"),
    (call("edit_file", filepath="b.py", old_string="foo", new_string="bar"), "edit"),
    (call("write_file", filepath="a.py", content="x"), "edit"),
    (call("read_file", filepath="a.py"), "reread"),
    (call("read_file", filepath="a.py", start_line=1, end_line=40), "reread"),
    (call("run_command", command="python -m pytest -x"), "other"),
    (call("submit_patch"), "other"),
    ("I could not apply the edit.", "no_call"),
    (f"<|tool_call>call:edit_file{{filepath:{Q}a.py", "malformed"),
    (call("nonsense", x="1"), "malformed"),
    ("<|tool_call>call:edit_file{filepath}<tool_call|>", "malformed"),
    ("", "no_call"),
])
def test_next_kind_after_a_failed_edit_file(text, kind):
    assert G.next_kind(text, FAILED) == kind


def test_a_repeated_write_file_is_a_repeat_only_with_the_same_arguments():
    failed = {"name": "write_file", "args": {"filepath": "a.py", "content": "x"}}
    assert G.next_kind(call("write_file", filepath="a.py", content="x"), failed) == "repeat"
    assert G.next_kind(call("write_file", filepath="a.py", content="y"), failed) == "edit"
    assert G.next_kind(call("edit_file", filepath="a.py", old_string="x", new_string="y"), failed) == "edit"


def rows4(kinds, tasks=10):
    return [{"instance_id": f"t{i % tasks}", "kind": k} for i, k in enumerate(kinds)]


def test_g4_stats_are_shares_of_the_generations_that_came_back():
    s = G.g4_stats(rows4(["repeat"] * 3 + ["reread"] * 5 + ["edit"] * 1 + ["request_error"]), prefixes=4)
    assert s["n"] == 9 and s["repeat_%"] == 33.3 and s["reread_%"] == 55.6 and s["request_error_%"] == 10.0 and s["prefixes"] == 4


def test_clustered_se_counts_a_task_as_one_unit():
    # 20 generations, 4 tasks of 5: one task repeats every time, the others never. The share is 25%; if the 20 were independent the SE would be about 9.7 points,
    # with the task as the unit it is 25.
    rows = [{"instance_id": f"t{i // 5}", "kind": "repeat" if i < 5 else "edit"} for i in range(20)]
    assert G.clustered_se(rows, "repeat") == 25.0
    assert G.clustered_se(rows[:5], "repeat") == 0.0                      # one task: nothing to estimate it from


def g4(rep, rer, n=200, prefixes=60, errors=0):
    k = ["repeat"] * rep + ["reread"] * rer + ["edit"] * (n - rep - rer)
    return G.g4_stats(rows4(k, tasks=40) + rows4(["request_error"] * errors), prefixes)


def test_g4_gate_is_relative_to_the_base_on_the_same_prefixes():
    base = g4(60, 20)                                        # 30% repeat, 10% re-read: what the old traces show for the base
    assert G.g4_gate(base, g4(60, 20)) == []
    assert G.g4_gate(base, g4(75, 20)) == []                 # 37.5%: inside 8 points
    assert G.g4_gate(base, g4(80, 20)) == [f"repeat 40.0% against 30.0% for the base (tolerance {G.G4_TOLERANCE} points)"]
    assert G.g4_gate(base, g4(60, 10)) == []                 # 5%: inside 8 points
    assert G.g4_gate(base, g4(60, 2)) == [f"re-read 1.0% against 10.0% for the base (tolerance {G.G4_TOLERANCE} points)"]
    assert len(G.g4_gate(base, g4(180, 0))) == 2             # the edit_v1 collapse: 90% repeat, never re-reads
    assert G.g4_gate(g4(10, 100), g4(0, 70)) == [f"re-read 35.0% against 50.0% for the base (tolerance {G.G4_TOLERANCE} points)"]


def test_g4_gate_does_not_pass_on_too_little_or_broken_data():
    base = g4(60, 20)
    assert G.verdict(G.g4_gate(g4(60, 20, prefixes=30), g4(60, 20, prefixes=30))) == "INSUFFICIENT"
    assert G.verdict(G.g4_gate(base, g4(60, 20, errors=20))) == "INVALID"           # 9% of the requests failed
    assert G.verdict(G.g4_gate(g4(60, 20, errors=20), g4(60, 20))) == "INVALID"     # the reference failed: nothing was compared


# ---- G4b ----------------------------------------------------------------------------------------------------------------------
def rows4b(sub, recover, other=0, applied=0, errors=0, prefixes=60):
    """Generations on failed-edit prefixes that show no applied edit: `sub` submit_patch, `recover` a re-read or another edit (half each), `other` another tool. Plus `applied`
    submit_patch generations on prefixes that DO show an applied edit (G4b must ignore them) and `errors` request errors."""
    def row(i, kind, first, visible):
        return {"instance_id": f"t{i % 40}", "turn": i, "prefix": f"p{i % prefixes}", "kind": kind, "first_tool": first, "applied_visible": visible}
    out = [row(i, "other", "submit_patch", False) for i in range(sub)]
    out += [row(sub + i, "reread" if i % 2 else "edit", "read_file" if i % 2 else "edit_file", False) for i in range(recover)]
    out += [row(sub + recover + i, "other", "run_command", False) for i in range(other)]
    out += [row(i, "other", "submit_patch", True) for i in range(applied)]
    out += [row(i, "request_error", None, False) for i in range(errors)]
    return out


def test_g4b_counts_submit_and_recover_only_where_no_applied_edit_is_visible():
    s = G.g4b_stats(rows4b(sub=20, recover=60, other=20, applied=50))
    assert s["available"] and s["n"] == 100 and s["submit_%"] == 20.0 and s["recover_%"] == 60.0 and s["prefixes"] == 60 and s["request_error_%"] == 0.0
    s = G.g4b_stats(rows4b(sub=10, recover=30, other=0, errors=10))                          # a request error counts toward the error share only
    assert s["n"] == 40 and s["submit_%"] == 25.0 and s["recover_%"] == 75.0 and s["request_error_%"] == 20.0


def test_g4b_gate_is_relative_to_the_base_on_the_same_prefixes():
    base = G.g4b_stats(rows4b(sub=6, recover=56, other=38))                                  # the base on the 2026-11-06 S4 run: submit 6%, recover 56%
    assert G.g4b_gate(base, base) == []
    assert G.g4b_gate(base, G.g4b_stats(rows4b(sub=11, recover=46, other=43))) == []         # +5 and -10 points: on the line, inside
    assert G.g4b_gate(base, G.g4b_stats(rows4b(sub=12, recover=56, other=32))) == [f"submit 12.0% against 6.0% for the base (tolerance {G.G4B_SUBMIT_TOLERANCE} points)"]
    assert G.g4b_gate(base, G.g4b_stats(rows4b(sub=6, recover=45, other=49))) == [
        f"recover (re-read or another edit) 45.0% against 56.0% for the base (tolerance {G.G4B_RECOVER_TOLERANCE} points)"]
    assert len(G.g4b_gate(base, G.g4b_stats(rows4b(sub=22, recover=36, other=42)))) == 2    # e4b_sft_v2 on that run: fails both


def test_g4b_gate_does_not_pass_on_too_little_or_broken_data():
    base = G.g4b_stats(rows4b(6, 56, 38))
    few = G.g4b_stats(rows4b(6, 56, 38, prefixes=30))
    assert G.verdict(G.g4b_gate(few, few)) == "INSUFFICIENT"                                  # 30 prefixes without an applied edit
    old = G.g4b_stats([{"instance_id": "t", "turn": 1, "kind": "reread", "applied_visible": None}] * 50)      # a check file built before the flag existed
    assert not old["available"] and G.verdict(G.g4b_gate(old, old)) == "INSUFFICIENT" and G.verdict(G.g4b_gate(base, old)) == "INSUFFICIENT"
    broken = G.g4b_stats(rows4b(6, 56, 38, errors=20))                                         # 20 of 120 requests failed
    assert G.verdict(G.g4b_gate(base, broken)) == "INVALID" and G.verdict(G.g4b_gate(broken, base)) == "INVALID"


# ---- G5 -----------------------------------------------------------------------------------------------------------------------
def rows5(n, bad=0, length=0, errors=0, thought=0, tokens=100):
    out = []
    for i in range(n):
        out.append({"finish": "stop", "well_formed": True, "why": "", "tokens": tokens, "text": call("get_status")})
    for r in out[:bad]:
        r["well_formed"], r["why"] = False, "2 tool calls"
    for r in out[bad:bad + length]:
        r["finish"] = "length"
    for r in out[:thought]:
        r["text"] = "<|channel>thought\nhm<channel|>" + r["text"]
    return out + [{"finish": "error", "well_formed": False, "why": "request error", "tokens": 0, "text": ""} for _ in range(errors)]


def test_g5_stats_and_gate():
    s = G.g5_stats(rows5(400, bad=4, length=4, thought=100, tokens=250))
    assert (s["well_formed_%"], s["length_%"], s["thought_%"], s["mean_tokens"], s["request_error_%"]) == (99.0, 1.0, 25.0, 250, 0.0)
    assert s["reasons"] == {"2": 4}
    assert G.g5_gate(s) == []
    assert G.g5_gate(G.g5_stats(rows5(400, bad=12))) == ["well-formed 97.0% below 98.0%"]
    assert G.g5_gate(G.g5_stats(rows5(400, length=12))) == ["length-limit hits 3.0% above 2.0%"]
    assert G.verdict(G.g5_gate(G.g5_stats(rows5(100)))) == "INSUFFICIENT"
    assert G.verdict(G.g5_gate(G.g5_stats(rows5(380, errors=20)))) == "INVALID"


def test_verdict_puts_invalid_before_fail_before_insufficient():
    assert G.verdict([]) == "PASS"
    assert G.verdict(["x"]) == "FAIL"
    assert G.verdict(["INSUFFICIENT: y"]) == "INSUFFICIENT"
    assert G.verdict(["x", "INSUFFICIENT: y"]) == "FAIL"                     # a comparison with enough data came out against the adapter: too little data elsewhere does not hide it
    assert G.verdict(["x", "INSUFFICIENT: y", "INVALID: z"]) == "INVALID"


def summary3(n_per_tool, agree=1.0, tools=("read_file", "run_command", "submit_patch", "edit_file")):
    import step_check as SC
    rows = []
    for t in tools:
        for i in range(n_per_tool):
            rows.append({"recorded": t, "generated": t if i < agree * n_per_tool else "run_command" if t != "run_command" else "read_file", "tokens": 10})
    return SC.summarize(rows)


def test_g3_gate_needs_enough_prefixes_to_compare_anything():
    assert G.g3_gate(summary3(30), summary3(30)) == []
    small = G.g3_gate(summary3(9), summary3(9))                                # every tool under 10 prefixes: step_check skips all of them and passes
    assert G.verdict(small) == "INSUFFICIENT" and "0 gated tools" in small[0]
    two = G.g3_gate(summary3(30, tools=("read_file", "edit_file")), summary3(30, tools=("read_file", "edit_file")))
    assert G.verdict(two) == "INSUFFICIENT" and "1 gated tools" in two[0]    # edit_file is never gated
    few = G.g3_gate(summary3(10, tools=("read_file", "run_command", "submit_patch")), summary3(10, tools=("read_file", "run_command", "submit_patch")))
    assert G.verdict(few) == "INSUFFICIENT" and "30 generations" in few[0]
    fail = G.g3_gate(summary3(30), summary3(30, agree=0.5))                    # enough data and a real drop: FAIL, not INSUFFICIENT
    assert G.verdict(fail) == "FAIL" and any(f.startswith("read_file") for f in fail)


# ---- the whole program against a fake server ------------------------------------------------------------------------------------
TOOLS3 = ["read_file", "run_command", "edit_file", "submit_patch"]
ARGS3 = {"read_file": {"filepath": "a.py"}, "run_command": {"command": "ls"}, "edit_file": {"filepath": "a.py", "old_string": "q", "new_string": "r"}, "submit_patch": {}}


def reply(model, ids):
    """What the fake model says to a prompt. A G3 prompt is [0, index of the recorded tool]; a G4 prompt is [1, n, 1 if an applied edit is visible else 0]."""
    kind, k = ids[:2]
    visible = bool(ids[2]) if len(ids) > 2 else True
    if model == "broken":
        return None
    if kind == 0:                                           # G3: the same tool as the recorded run, or (the loopy adapter) always an edit
        tool = "edit_file" if model == "loopy" else TOOLS3[k]
        return call(tool, **ARGS3[tool])
    if model == "loopy":                                    # G4: always the failed call again
        return call("edit_file", filepath="a.py", old_string="foo", new_string="bar")
    if model == "quitter" and not visible and k % 4 == 3:   # G4: gives up where the base tries another edit; the shares of repeat and re-read stay the base's
        return call("submit_patch")
    if k % 4 == 0:
        return call("edit_file", filepath="a.py", old_string="foo", new_string="bar")          # repeat: 25%
    if k % 4 in (1, 2):
        return call("read_file", filepath="a.py")                                                  # re-read: 50%
    return call("edit_file", filepath="a.py", old_string="foo2", new_string="bar")                 # another edit: 25%


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["top_k"] == 64 and body["skip_special_tokens"] is False
        text = reply(body["model"], body["prompt"])
        if text is None:
            self.send_response(500)
            self.end_headers()
            return
        data = json.dumps({"choices": [{"text": text, "finish_reason": "stop"}], "usage": {"completion_tokens": 9}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


@pytest.fixture()
def checks(tmp_path):
    g3 = [{"instance_id": f"t{i % 20}", "turn": i, "tool": TOOLS3[i % 4], "prompt_ids": [0, i % 4]} for i in range(100)]
    g4_ = [{"instance_id": f"t{i % 20}", "turn": i, "tool": "read_file", "prompt_ids": [1, i, int(i % 5 == 0)], "failed_call": FAILED, "applied_edit_visible": i % 5 == 0}
           for i in range(50)]                                                                 # 10 prefixes show an applied edit, 40 do not (the INSUFFICIENT line is 40)
    for name, rows in (("check.jsonl", g3), ("failed.jsonl", g4_)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return tmp_path


def run(server, checks, *models, capsys, extra=()):
    argv = ["--check", str(checks / "check.jsonl"), "--failed-edit", str(checks / "failed.jsonl"), "--base-url", server, "--out", str(checks / "out"),
            "--samples", "2", "--g4-samples", "2", "--workers", "4", *extra]
    for m in models:
        argv += ["--model", m]
    code = G.main(argv)
    return code, capsys.readouterr().out


def test_an_adapter_that_behaves_like_the_base_passes_everything(server, checks, capsys):
    code, out = run(server, checks, "base", "good", capsys=capsys)
    assert code == 0, out
    s = json.load(open(checks / "out" / "summary.json"))
    assert s["base"]["G4"]["repeat_%"] == 26.0 and s["base"]["G4"]["reread_%"] == 50.0          # of 50 prefixes: 13 repeat, 25 re-read, 12 edit elsewhere
    assert s["good"]["gates"] == {"G3": [], "G4": [], "G4b": [], "G5": []}
    assert s["base"]["G4b"]["prefixes"] == 40 and s["base"]["G4b"]["submit_%"] == 0.0 and s["base"]["G4b"]["recover_%"] == 75.0
    assert s["good"]["G5"]["n"] == 300 and s["good"]["G5"]["well_formed_%"] == 100.0
    assert "PASS" in out and "FAIL" not in out
    assert (checks / "out" / "good_g3.jsonl").exists() and (checks / "out" / "good_g4.jsonl").exists()


def test_a_collapsed_adapter_fails_g3_and_g4(server, checks, capsys):
    code, out = run(server, checks, "base", "loopy", capsys=capsys)
    assert code == 1, out
    s = json.load(open(checks / "out" / "summary.json"))
    assert s["loopy"]["gates"]["G5"] == []                    # its calls are well formed: the format gate does not see a loop
    assert any(f.startswith("read_file") for f in s["loopy"]["gates"]["G3"])
    assert len(s["loopy"]["gates"]["G4"]) == 2 and s["loopy"]["G4"]["repeat_%"] == 100.0
    assert "G3 step types  FAIL" in out and "G4 after a failed edit  FAIL" in out


def test_an_adapter_that_gives_up_after_a_failed_edit_passes_g4_and_fails_g4b(server, checks, capsys):
    code, out = run(server, checks, "base", "quitter", capsys=capsys)
    assert code == 1, out
    s = json.load(open(checks / "out" / "summary.json"))
    g = s["quitter"]["gates"]
    assert g["G3"] == [] and g["G4"] == [] and g["G5"] == []                            # repeat and re-read are the base's: the old gates see nothing
    assert len(g["G4b"]) == 2 and s["quitter"]["G4b"]["submit_%"] == 25.0 and s["quitter"]["G4b"]["recover_%"] == 50.0
    assert "G4 after a failed edit  PASS" in out and "G4b no applied edit     FAIL" in out


def test_a_model_the_server_cannot_answer_for_is_invalid_not_passing(server, checks, capsys):
    code, out = run(server, checks, "base", "broken", capsys=capsys)
    assert code == 2, out
    assert out.count("INVALID") >= 3
    code, out = run(server, checks, "broken", "good", capsys=capsys)                      # the reference is the one that failed
    assert code == 2, out


def test_a_run_with_few_generations_is_insufficient_not_passing(server, checks, capsys):
    code, out = run(server, checks, "base", "good", capsys=capsys, extra=("--n", "20", "--n-failed", "10"))
    assert code == 1 and "INSUFFICIENT" in out
