"""The lab's one-factor arms differ from their parent in exactly the line their README names. A stray edit would make a screening run mean something else,
and a pack that lacks a folder costs a GPU session."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
sys.path.insert(0, str(ROOT / "mine"))
import make_lab_pack as L  # noqa: E402


def files(folder):
    return {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def changed(parent, arm):
    """{file: (lines only in the parent, lines only in the arm)} over every file but README.md. Comments and blank lines of a yaml file do not count."""
    a, b = files(AGENT / parent), files(AGENT / arm)
    out = {}
    for name in sorted(set(a) | set(b)):
        if name == "README.md":
            continue
        ta = a[name].read_text(encoding="utf-8").splitlines() if name in a else []
        tb = b[name].read_text(encoding="utf-8").splitlines() if name in b else []
        if name.endswith(".yaml"):
            ta, tb = [x for x in ta if x.strip() and not x.lstrip().startswith("#")], [x for x in tb if x.strip() and not x.lstrip().startswith("#")]
        if ta != tb:
            out[name] = (sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta)))
    return out


S = "configs/sampling.yaml"
ONE_FACTOR = [
    ("submission_e4b_v2rec", "submission_e4b_v2rec_th0", {S: (["  include_thoughts: true", "  thinking_budget: 4096"], ["  include_thoughts: false", "  thinking_budget: 0"])}),
    ("submission_e4b_v2rec", "submission_e4b_v2rec_th512", {S: (["  thinking_budget: 4096"], ["  thinking_budget: 512"])}),
    ("submission_e4b_v2rec", "submission_e4b_v2rec_t07", {S: (["temperature: 1.0"], ["temperature: 0.7"])}),
    ("submission_e4b_v2rec", "submission_e4b_v2rec_noth", {S: (["  include_thoughts: true"], ["  include_thoughts: false"])}),
    ("submission_e4b_v2rec", "submission_e4b_v2rec_sft2", {"agent.yaml": ([], ["adapter: e4b_sft_v2"])}),
    ("submission_e4b_v2rec", "submission_e4b_v2", {S: (["temperature: 1.0", "top_k: 64"], ["temperature: 0.2"])}),       # the prompt of v2rec at the baseline's sampling
    ("submission_e4b", "submission_e4b_rec", {S: (["temperature: 0.2"], ["temperature: 1.0", "top_k: 64"])}),            # the baseline's prompt at the model's own sampling
]


@pytest.mark.parametrize("parent,arm,expected", ONE_FACTOR, ids=[a for _, a, _ in ONE_FACTOR])
def test_a_one_factor_arm_changes_only_its_line(parent, arm, expected):
    assert changed(parent, arm) == expected


def test_the_four_cells_of_prompt_by_sampling_are_what_their_names_say():
    """baseline = (prompt v1, temperature 0.2), rec = (v1, 1.0), v2 = (v2, 0.2), v2rec = (v2, 1.0): the prompts and the sampling files pair up byte for byte."""
    prompt = lambda a: (AGENT / a / "prompts" / "flat_search.md").read_bytes()                      # noqa: E731
    assert prompt("submission_e4b") == prompt("submission_e4b_rec") != prompt("submission_e4b_v2") == prompt("submission_e4b_v2rec")
    assert changed("submission_e4b_v2", "submission_e4b_v2rec") == {S: (["temperature: 0.2"], ["temperature: 1.0", "top_k: 64"])}
    assert S not in changed("submission_e4b_v2rec", "submission_e4b_rec"), "v2rec must sample exactly like rec"
    assert S not in changed("submission_e4b", "submission_e4b_v2"), "v2 must sample exactly like the baseline"


#                              enable_thinking, thinking_token_budget, earlier thoughts sent back, temperature, top_k
ON_THE_WIRE = {
    "submission_e4b":             (True, 4096, True, 0.2, None),
    "submission_e4b_rec":         (True, 4096, True, 1.0, 64.0),
    "submission_e4b_v2":          (True, 4096, True, 0.2, None),
    "submission_e4b_v2rec":       (True, 4096, True, 1.0, 64.0),
    "submission_e4b_v2rec_t07":   (True, 4096, True, 0.7, 64.0),
    "submission_e4b_v2rec_th512": (True, 512, True, 1.0, 64.0),
    "submission_e4b_v2rec_noth":  (True, 4096, False, 1.0, 64.0),     # the switch stays on, the history loses the thoughts
    "submission_e4b_v2rec_th0":   (False, None, False, 1.0, 64.0),    # the switch is off AND the history has no thoughts
    "submission_e4b_v2rec_sft2":  (True, 4096, True, 1.0, 64.0),
}


@pytest.mark.parametrize("arm", sorted(ON_THE_WIRE))
def test_what_each_arm_sends_to_the_server(arm):
    """The arm folder goes through the vendored runner (adk-submission 0.2.13's thinking bridge) and the request body is read back from a mock server."""
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(ROOT / "agent" / "vendor"))
    import runner as R                                                                 # noqa: E402
    from test_scorer_port import MODEL, ScriptedServer                                 # noqa: E402
    enable, budget, thoughts_back, temperature, top_k = ON_THE_WIRE[arm]
    cfg, _ = R.load_submission(str(AGENT / arm))
    history = [{"role": "user", "content": "fix it"},
               {"role": "assistant", "content": None, "reasoning_content": "I should read the file first.",
                "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "c0", "content": "{}"}]
    server = ScriptedServer([{"content": "ok"}])
    try:
        R.OpenAIBackend(server.url, MODEL, gen=cfg["generate_content_config"]).step("sys", history, [])
    finally:
        server.close()
    body = server.bodies[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": enable}
    assert body.get("thinking_token_budget") == budget
    assert body["temperature"] == temperature and body.get("top_k") == top_k and body["top_p"] == 0.95 and body["max_completion_tokens"] == 8192
    sent = next(m for m in body["messages"] if m["role"] == "assistant")
    assert ("reasoning" in sent or "reasoning_content" in sent) is thoughts_back
    if thoughts_back:
        assert sent["reasoning"] == sent["reasoning_content"] == "I should read the file first."
    assert (cfg.get("adapter") == "e4b_sft_v2") is (arm == "submission_e4b_v2rec_sft2") and (cfg.get("adapter") is None or arm.endswith("sft2"))


def test_every_arm_the_pack_carries_exists_and_has_an_agent_yaml():
    for arm in L.DEFAULT_ARMS:
        assert (AGENT / arm / "agent.yaml").exists(), arm
        assert (AGENT / arm / "eval_config.yaml").exists(), arm
    assert len(set(L.DEFAULT_ARMS)) == len(L.DEFAULT_ARMS)


def test_no_arm_folder_ships_adapter_weights():
    """The adapter files live on Drive; a weights file in an arm folder would go into the pack and into git."""
    for arm in L.DEFAULT_ARMS:
        assert not [n for n in files(AGENT / arm) if n.endswith((".safetensors", ".bin", ".pt"))], arm


def test_the_unchanged_files_keep_the_unix_line_endings_of_their_parent():
    for parent, arm, _ in ONE_FACTOR:
        for name, p in files(AGENT / arm).items():
            assert b"\r\n" not in p.read_bytes(), f"{arm}/{name} has Windows line endings"
