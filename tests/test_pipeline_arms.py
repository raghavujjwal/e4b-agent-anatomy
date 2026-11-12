"""The three pipeline rungs (agent/submission_e4b_pipe_s1 .. _s3, written by mine/make_pipeline_arms.py).

A rung is a candidate for the scorer only if it beats the control in the lab (docs/EVAL_PROTOCOL.md rule 16), and the lab only means something if each
folder is exactly what its README says: the control's wording where it keeps it, the control's sampling and budget byte for byte, and one thing added per
rung. These tests pin that, the shape of the tree the scorer's loader accepts, and how the vendored runner moves the note from stage 1 to stage 2."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
sys.path.insert(0, str(ROOT / "mine"))
sys.path.insert(0, str(ROOT / "agent" / "vendor"))
sys.path.insert(0, str(Path(__file__).parent))
pytest.importorskip("yaml")
pytest.importorskip("numpy")
import eval as EV  # noqa: E402  (agent/vendor/eval.py)
import lab_report as LR  # noqa: E402
import make_pipeline_arms as G  # noqa: E402
import runner as R  # noqa: E402
from test_scorer_port import MODEL, FakeContainer, ScriptedServer  # noqa: E402
import tools as T  # noqa: E402

CONTROL = AGENT / "submission_e4b_v2rec"
PIPE = json.loads((ROOT / "tests" / "fixtures" / "scorer_requests_pipe_s2.json").read_text(encoding="utf-8"))
RUNGS = sorted(G.RUNGS)
ARM = lambda rung: AGENT / f"submission_e4b_pipe_{rung}"                                  # noqa: E731
SIX = ["read_file", "edit_file", "write_file", "run_command", "get_status", "submit_patch"]
# the fields adk_submission 0.2.13 schema.py accepts (extra="forbid"), so a typo is caught without the scorer's venv
LLM_KEYS = {"agent_class", "name", "description", "before_agent_callbacks", "after_agent_callbacks", "model", "adapter", "instruction", "output_key",
            "include_contents", "disallow_transfer_to_parent", "disallow_transfer_to_peers", "tools", "skills", "sub_agents", "generate_content_config",
            "before_model_callbacks", "after_model_callbacks", "before_tool_callbacks", "after_tool_callbacks"}
SEQ_KEYS = {"agent_class", "name", "description", "before_agent_callbacks", "after_agent_callbacks", "sub_agents"}


def tree(rung):
    cfg, path = R.load_submission(str(ARM(rung)))
    return R.resolve_tree(cfg, path)


def prompt(rung, stage):
    return (ARM(rung) / "prompts" / f"{stage}.md").read_text(encoding="utf-8")


def control_sections():
    return G.sections((CONTROL / "prompts" / "flat_search.md").read_text(encoding="utf-8"))[1]


# ------------------------------------------------------------------------------------------------ what is on disk
def test_the_folders_are_what_the_generator_writes():
    """No hand edits: the arm folders are artifacts of mine/make_pipeline_arms.py."""
    for arm, files in G.render_all().items():
        folder = AGENT / arm
        on_disk = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        assert on_disk == set(files), arm
        for name, data in files.items():
            assert (folder / name).read_bytes() == data, f"{arm}/{name} differs from the generator's output; run mine/make_pipeline_arms.py"


def test_the_sampling_and_the_budget_are_the_controls_bytes():
    for rung in RUNGS:
        for name in ("configs/sampling.yaml", "eval_config.yaml"):
            assert (ARM(rung) / name).read_bytes() == (CONTROL / name).read_bytes(), f"{rung}/{name}"


def test_unix_line_endings_and_no_weights():
    for rung in RUNGS:
        for p in ARM(rung).rglob("*"):
            if p.is_file():
                assert b"\r\n" not in p.read_bytes(), f"{p} has Windows line endings"
                assert p.suffix not in (".safetensors", ".bin", ".pt"), p


# ------------------------------------------------------------------------------------------------ the tree
@pytest.mark.parametrize("rung", RUNGS)
def test_the_tree_is_two_llm_agents_in_a_sequence(rung):
    root = tree(rung)
    control, _ = R.load_submission(str(CONTROL))
    assert root["agent_class"] == "SequentialAgent" and root["name"] == "swe_pipeline" and set(root) - {"_sub", "_agent_tools"} <= SEQ_KEYS
    investigate, fix = root["_sub"]
    assert [investigate["name"], fix["name"]] == ["investigate", "fix"]
    assert investigate["tools"] == ["read_file", "run_command", "get_status"] and fix["tools"] == SIX == control["tools"]
    for forbidden in ("edit_file", "write_file", "submit_patch"):
        assert forbidden not in investigate["tools"]
    assert investigate["output_key"] == "findings" and "output_key" not in fix
    assert fix["include_contents"] == "none" and "include_contents" not in investigate
    for stage in (investigate, fix):
        assert stage["model"] == control["model"] == MODEL
        assert stage["generate_content_config"] == control["generate_content_config"]
        assert set(stage) - {"_sub", "_agent_tools"} <= LLM_KEYS
        assert "adapter" not in stage and not stage["_sub"] and not stage["_agent_tools"]


# ------------------------------------------------------------------------------------------------ the prompts
@pytest.mark.parametrize("rung", RUNGS)
def test_the_only_braces_are_the_two_variables_of_the_fix_prompt(rung):
    """google-adk 1.36.1 renders every {identifier} from the session state and raises on a missing one; the pattern has no escape, so there must be no other."""
    pattern = re.compile(r"{+[^{}]*}+")
    assert pattern.findall(prompt(rung, "investigate")) == []
    assert pattern.findall(prompt(rung, "fix")) == ["{problem_description}", "{findings?}"]
    text, missing = R.interpolate(prompt(rung, "fix"), {"problem_description": "THE ISSUE", "findings": "THE NOTE"})
    assert missing == [] and "THE ISSUE" in text and "THE NOTE" in text and "{" not in text
    text, missing = R.interpolate(prompt(rung, "fix"), {"problem_description": "THE ISSUE"})        # stage 1 left no note: the stage still renders
    assert missing == ["findings"] and "THE ISSUE" in text
    with pytest.raises(R.TemplateKeyError):
        R.interpolate(prompt(rung, "fix"), {"findings": "x"})


@pytest.mark.parametrize("rung", RUNGS)
def test_what_a_stage_keeps_of_the_control_prompt_is_verbatim(rung):
    sec = control_sections()
    inv, fix = prompt(rung, "investigate"), prompt(rung, "fix")
    for title in (G.FIND, G.CORRECT):
        assert sec[title] in inv, title
    for title in (G.CORRECT, G.EDIT, G.RULES, G.FINISH):
        assert sec[title] in fix, title
    assert (sec[G.CHECK] in fix) is (rung == "s1")                  # S2 on: the check is the reproduction


def sections_of(text):
    intro, sec = G.sections(text)
    return intro, sec


def delta(a, b):
    """(sections only in b, sections only in a, sections in both that differ) between two prompts, and whether the opening paragraph changed."""
    (ia, sa), (ib, sb) = sections_of(a), sections_of(b)
    return (sorted(set(sb) - set(sa)), sorted(set(sa) - set(sb)), sorted(k for k in set(sa) & set(sb) if sa[k] != sb[k]), ia != ib)


def test_each_rung_adds_one_thing_to_the_one_below():
    """Prompts compared heading by heading, everything else file by file."""
    inv = lambda r: prompt(r, "investigate")                                              # noqa: E731
    fix = lambda r: prompt(r, "fix")                                                      # noqa: E731
    assert delta(inv("s1"), inv("s2")) == (["Reproducing the problem"], [], ["Your budget", "Your final message: the hand-over note"], False)
    assert delta(fix("s1"), fix("s2")) == ([], [], [G.CHECK], False)
    assert delta(inv("s2"), inv("s3")) == (["Example of a finished investigation"], [], [], False)
    assert delta(fix("s2"), fix("s3")) == (["Example of an edit"], [], [], False)

    def yaml_lines(path):
        return [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]
    for lower, upper in (("s1", "s2"), ("s2", "s3")):
        for name in ("agent.yaml", "sub_agents/investigate.yaml", "sub_agents/fix.yaml", "configs/sampling.yaml", "eval_config.yaml"):
            assert yaml_lines(ARM(lower) / name) == yaml_lines(ARM(upper) / name), f"{lower} -> {upper}: {name}"


@pytest.mark.parametrize("rung", RUNGS)
def test_both_stages_know_what_a_restart_looks_like(rung):
    """The scorer's loop sends a nudge whenever a pass ends without submit_patch (32 of the control's 70 tune episodes had one) and a SequentialAgent root then
    starts over at stage 1. So each stage says what to do when the other's work is already in the session: stage 1 marks the note, stage 2 does not edit the
    same lines again, and the marker one writes is the one the other looks for."""
    inv, fix = prompt(rung, "investigate"), prompt(rung, "fix")
    sec = G.sections(inv)[1]
    assert "For context: [fix]" in sec["If this step starts again"] and "only your teammate can submit" in sec["If this step starts again"]
    titles = list(sec)
    assert titles.index("If this step starts again") < titles.index("Your final message: the hand-over note")
    assert re.search(r'begin the CHANGE line with\s+"ALREADY MADE:"', inv) and re.search(r'CHANGE line\s+starts with "ALREADY MADE"', fix)
    assert "do not edit those lines again" in fix and "call submit_patch" in fix and "without any memory of the earlier ones" in fix


def test_the_note_format_names_the_reproduction_only_from_s2_on():
    assert "REPRO" not in prompt("s1", "investigate") and "REPRO" not in prompt("s1", "fix")
    for rung in ("s2", "s3"):
        assert "REPRO: the command" in prompt(rung, "investigate") and "REPRO line of the note" in prompt(rung, "fix")
    for rung in RUNGS:
        for field in ("FILE:", "LINES:", "CAUSE:", "CHANGE:", "MATCH:"):
            assert field in prompt(rung, "investigate")


def test_the_examples_are_invented_and_use_the_tool_texts_of_the_harness():
    demo = prompt("s3", "fix")
    assert "Target string occurs 2 times in src/units/sizes.py. Please provide more surrounding context." in demo
    assert "Failed to replace: old_string not found" in demo
    for rung in ("s1", "s2"):
        assert "units/sizes" not in prompt(rung, "investigate") + prompt(rung, "fix")
    # the invented project must not be a project of any pool
    for word in ("click", "falcon", "werkzeug", "tornado", "packaging", "loguru", "black", "pygments", "fastapi", "requests", "httpx", "rich"):
        assert word not in (G.INV_DEMO + G.FIX_DEMO).lower(), word


def test_the_edit_in_the_example_behaves_as_the_example_says():
    """The refusal text and the fix in the worked example are what the harness's edit tool really answers for that file."""
    src = ("def parse_size(text: str) -> int:\n    number, unit = _split(text)\n    value = int(number)\n    return value * SIZE_UNITS[unit]\n\n\n"
           "def parse_rate(text: str) -> int:\n    number, unit = _split(text)\n    value = int(number)\n    return value * RATE_UNITS[unit]\n")
    c = FakeContainer({"src/units/sizes.py": src}, "", "")
    tools = T.Tools(c, T.Budget(tool_calls=24, time_minutes=5.5, turns=40, command_timeout=90))
    tools.b.start()
    first = json.loads(tools.call("edit_file", {"filepath": "src/units/sizes.py", "old_string": "    value = int(number)", "new_string": "    value = float(number)"}))
    assert first["status"] == "error" and first["error_message"] == "Target string occurs 2 times in src/units/sizes.py. Please provide more surrounding context."
    ok = json.loads(tools.call("edit_file", {"filepath": "src/units/sizes.py", "old_string": "    value = int(number)\n    return value * SIZE_UNITS[unit]",
                                             "new_string": "    value = float(number)\n    return int(value * SIZE_UNITS[unit])"}))
    assert ok["status"] == "ok" and ok["occurrences"] == 1
    assert "return int(value * SIZE_UNITS[unit])" in c.files["/workspace/src/units/sizes.py"].decode() and c.files["/workspace/src/units/sizes.py"].decode().count("int(number)") == 1
    gone = json.loads(tools.call("edit_file", {"filepath": "src/units/sizes.py", "old_string": "  value = int(numbr)", "new_string": "x"}))
    assert gone["error_message"].startswith("Failed to replace: old_string not found")


# ------------------------------------------------------------------------------------------------ the vendored runner against the scorer's own code
def run_pipeline(rung, script, problem="foo() should return 2, not 1.", calls=24):
    """The tree of a rung through the vendored runner, a scripted server and the in-memory workspace of the fixture."""
    root = tree(rung)
    server = ScriptedServer(script)
    try:
        c = FakeContainer(PIPE["ws_files"], PIPE["tree"], PIPE["diff"])
        budget = T.Budget(tool_calls=calls, time_minutes=5.5, turns=40, command_timeout=90)
        tools = T.Tools(c, budget)
        task = {"repo": "demo/repo", "problem_statement": problem, "hints_text": ""}
        state = {"problem_description": problem, "_contents": [{"role": "user", "content": EV.first_message(task, budget, tools, c, root)}]}
        budget.start()
        trace = []
        stop = R.drive(root, tools, state, R.OpenAIBackend(server.url, MODEL), str(ARM(rung)), trace)
    finally:
        server.close()
    return {"stop": stop, "bodies": server.bodies, "tools": tools, "container": c, "trace": trace, "state": state}


def test_the_pipeline_requests_match_the_scorers_end_to_end():
    """Same scripted model, same files: the vendored runner must send the bodies that swegemma 0.2.11 + adk-submission 0.2.13 + google-adk 1.36.1 sent for
    the S2 tree (tests/fixtures/scorer_requests_pipe_s2.json), stage 2's 'For context' message included."""
    run = run_pipeline("s2", PIPE["script"])
    assert run["stop"] == "submitted" and len(run["bodies"]) == len(PIPE["requests"]) == 9
    for i, (got, want) in enumerate(zip(run["bodies"], PIPE["requests"])):
        got = {k: v for k, v in got.items() if k != "return_token_ids"}                    # local-only: exact token ids for SFT data
        assert got.keys() == want.keys(), f"request {i}: keys differ: {sorted(set(got) ^ set(want))}"
        for key in want:
            assert got[key] == want[key], f"request {i}: `{key}` differs"


def test_the_other_rungs_load_and_run_through_the_same_path():
    for rung in ("s1", "s3"):
        run = run_pipeline(rung, PIPE["script"])
        assert run["stop"] == "submitted" and len(run["bodies"]) == 9
        assert [t["function"]["name"] for t in run["bodies"][0]["tools"]] == ["read_file", "run_command", "get_status"]
        assert len(run["bodies"][4]["tools"]) == 6 and run["bodies"][4]["messages"][1]["content"][0] == {"type": "text", "text": "For context:"}


def test_stage_two_starts_from_the_note_and_not_from_the_exploration():
    run = run_pipeline("s2", PIPE["script"])
    first_fix = run["bodies"][4]
    roles = [m["role"] for m in first_fix["messages"]]
    assert roles == ["system", "user"]                                          # no assistant turn, no tool result, no first task message of stage 1
    note = run["state"]["findings"]
    assert note.startswith("FILE: pkg/mod.py") and "Now the hand-over note." not in note        # the thoughts of the last reply are not part of the note
    assert note in first_fix["messages"][0]["content"] and "[investigate] said: " + note == first_fix["messages"][1]["content"][1]["text"]
    assert "foo() should return 2, not 1." in first_fix["messages"][0]["content"]
    assert "You are evaluating a software engineering task" not in json.dumps(first_fix["messages"])      # the scorer's first message is stage 1's alone


def test_the_budget_is_one_budget_and_stage_one_cannot_reach_the_edit_tools():
    run = run_pipeline("s2", PIPE["script"])
    assert run["tools"].b.used == 6                                              # grep, read, repro | read, edit, rerun; submit_patch and get_status are free
    stage1 = [[t["function"]["name"] for t in b["tools"]] for b in run["bodies"][:4]]
    assert all(names == ["read_file", "run_command", "get_status"] for names in stage1)
    # a model that calls a tool it was not given gets the harness's own refusal and the workspace stays as it was
    script = [{"calls": [{"name": "edit_file", "args": {"filepath": "pkg/mod.py", "old_string": "return 1", "new_string": "return 2"}}]},
              {"content": "FILE: pkg/mod.py"}, {"calls": [{"name": "submit_patch", "args": {}}]}, {"content": "done"}]
    run = run_pipeline("s1", script)
    refused = run["bodies"][1]["messages"][-1]
    assert refused["role"] == "tool" and "Tool 'edit_file' not found" in refused["content"] and "read_file, run_command, get_status" in refused["content"]
    assert run["container"].files["/workspace/pkg/mod.py"] == PIPE["ws_files"]["pkg/mod.py"].encode()


EMPTY = json.loads((ROOT / "tests" / "fixtures" / "scorer_requests_pipe_s2_empty.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("which", ["script", "script_thoughts_only"])
def test_an_empty_note_reaches_stage_two_as_the_scorer_delivers_it(which):
    """Stage 1 ends without text (nothing at all, or thoughts only, e.g. cut by the token limit inside the thinking). The scorer leaves no event for it,
    so {findings?} renders empty, the key is never set, and stage 2 starts from the scorer's first message (tests/fixtures/scorer_requests_pipe_s2_empty.json);
    the vendored runner must send the same bodies."""
    run = run_pipeline("s2", EMPTY[which])
    assert run["stop"] == "submitted" and len(run["bodies"]) == len(EMPTY["requests"]) == 5
    for i, (got, want) in enumerate(zip(run["bodies"], EMPTY["requests"])):
        got = {k: v for k, v in got.items() if k != "return_token_ids"}
        assert got == want, f"request {i} differs from the scorer's"
    first_fix = run["bodies"][1]
    assert [m["role"] for m in first_fix["messages"]] == ["system", "user"]
    assert first_fix["messages"][1]["content"].startswith("You are evaluating a software engineering task")
    assert "## Hand-over note from your teammate\n\n\n\nThe note says" in first_fix["messages"][0]["content"]
    assert "findings" not in run["state"]                                       # a reply without text never sets the output key


def test_the_reporter_reads_the_trace_of_a_pipeline_run():
    run = run_pipeline("s2", PIPE["script"])
    split = LR.stage_split({"t": run["trace"]})
    assert split["calls"] == {"investigate": 3, "fix": 3} and (split["empty_note"], split["fix_without_edit"]) == (0, 0)
    empty = run_pipeline("s2", [{"content": ""}, {"calls": [{"name": "submit_patch", "args": {}}]}, {"content": "done"}])
    assert LR.stage_split({"t": empty["trace"]}) == {"calls": {"investigate": 0, "fix": 0}, "empty_note": 1, "fix_without_edit": 1,
                                                     "investigate_over_limit": 0, "nudged": 0, "episodes": 1}


NUDGE = json.loads((ROOT / "tests" / "fixtures" / "scorer_requests_pipe_s2_nudge.json").read_text(encoding="utf-8"))
CASES = json.loads((ROOT / "tests" / "fixtures" / "scorer_requests_pipe_s2_cases.json").read_text(encoding="utf-8"))["cases"]


def same_bodies(run, want):
    assert len(run["bodies"]) == len(want), f"{len(run['bodies'])} requests, the scorer sent {len(want)}"
    for i, (got, real) in enumerate(zip(run["bodies"], want)):
        got = {k: v for k, v in got.items() if k != "return_token_ids"}
        assert got == real, f"request {i} differs from the scorer's"


def test_the_second_pass_after_a_nudge_matches_the_scorers():
    """Pass 1 ends after stage 2's edit with text and no submit_patch; the scorer sends its nudge as a new user message of the SAME session and a SequentialAgent
    root starts over at stage 1. There `investigate` sees the whole session (its own history and note, `fix`'s events as 'For context' user messages, the nudge)
    and `fix` starts again from the new note alone (tests/fixtures/scorer_requests_pipe_s2_nudge.json)."""
    run = run_pipeline("s2", NUDGE["script"])
    assert run["stop"] == "submitted"
    same_bodies(run, NUDGE["requests"])
    assert run["tools"].b.used == 6                                              # one budget over both passes: 2 + 2, then 1 + 1 (submit_patch is free)
    assert [e["n"] for e in run["trace"] if e.get("event") == "nudge"] == [0]
    assert LR.stage_split({"t": run["trace"]})["nudged"] == 1


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_edge_cases_match_the_scorers(case):
    run = run_pipeline("s2", CASES[case]["script"])
    assert run["stop"] == "submitted", CASES[case]["why"]
    same_bodies(run, CASES[case]["requests"])


def test_the_view_is_the_log_for_a_single_agent_and_a_story_for_the_others():
    """R.view: the user's and the agent's own events as they are, another agent's as 'For context' user messages, include_contents none from the latest user or
    other-agent event on."""
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"filepath": "a.py"})}}
    log = [{"role": "user", "content": "task"},
           {"role": "assistant", "content": None, "tool_calls": [call], "reasoning_content": "mine", "_author": "a"},
           {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"status": "ok", "n": 1, "flag": True}), "_author": "a"},
           {"role": "assistant", "content": "note", "reasoning_content": "mine too", "_author": "a"},
           {"role": "assistant", "content": "   ", "_author": "b"},                                          # blank text and no call: nothing to show
           {"role": "user", "content": "nudge"}]
    assert R.view(log, "a") == [m for m in log if m["role"] != "assistant" or m["content"] != "   "]           # own events untouched; b's blank one has nothing to show `a`
    seen = R.view(log, "b")
    assert [m["role"] for m in seen] == ["user", "user", "user", "user", "assistant", "user"]
    assert seen[0] == log[0] and seen[4] == log[4] and seen[5] == log[5]                                       # b's own blank reply stays (ADK strips only what it re-presents)
    assert seen[1]["_parts"] == ["For context:", "[a] called tool `read_file` with parameters: {'filepath': 'a.py'}"]
    assert seen[2]["_parts"] == ["For context:", "[a] `read_file` tool returned result: {'status': 'ok', 'n': 1, 'flag': True}"]
    assert seen[3]["_parts"] == ["For context:", "[a] said: note"] and "mine" not in json.dumps(seen)         # thoughts of another agent never show
    assert R.view(log, "b", "none") == [seen[5]]                                                               # the nudge is the latest user event
    assert R.view(log[:4], "b", "none") == [seen[3]]                                                           # no user event after `a`'s note: it starts there
    assert R.view(log[:3], "b", "none") == [seen[2]]                                                           # ... or at a tool result of `a`, whichever is the latest
    assert R.view(log[:3], "a", "none") == log[:1] + log[1:3]                                                  # `a` itself starts at the user's message
