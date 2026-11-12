"""train/served.py: the renderer must reproduce what the scorer's vLLM feeds the model.

Needs `transformers` and the competition model's tokenizer files (E4B_TOKENIZER_DIR or the default Kaggle copy);
skipped where they are absent.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
pytest.importorskip("transformers")
import served as S  # noqa: E402

TOK_DIR = Path(os.environ.get("E4B_TOKENIZER_DIR") or S.DEFAULT_TOKENIZER_DIR)
CANONICAL_DIR = Path(r"D:\data\gemma4_tok")
pytestmark = pytest.mark.skipif(not (TOK_DIR / "chat_template.jinja").exists(), reason="E4B tokenizer files not present")
REQS = json.loads((ROOT / "tests" / "fixtures" / "scorer_requests_v2rec.json").read_text(encoding="utf-8"))["requests"]


@pytest.fixture(scope="module")
def tok():
    return S.load_tokenizer(TOK_DIR)


def test_template_is_the_competition_original():
    assert S.template_sha256(TOK_DIR) == S.ORIGINAL_TEMPLATE_SHA256


@pytest.mark.skipif(not (CANONICAL_DIR / "chat_template.jinja").exists(), reason="canonical template copy not present")
def test_canonical_template_is_refused():
    with pytest.raises(S.WrongTemplate, match="July canonical"):
        S.check_template(CANONICAL_DIR)


def test_successive_requests_render_as_prefixes(tok):
    prev = None
    for i, body in enumerate(REQS):
        text = S.render_request(tok, body)
        if prev is not None:
            assert text.startswith(prev), f"request {i} does not extend request {i - 1}"
        prev = text


def test_thinking_marker_only_when_enabled(tok):
    on = S.render_request(tok, REQS[0])
    off = S.render_request(tok, {**REQS[0], "chat_template_kwargs": {"enable_thinking": False}})
    assert on.count("<|think|>") == 1 and "<|think|>" not in off
    assert on.replace("<|think|>\n", "") == off


def test_thoughts_are_replayed_only_for_tool_call_turns_after_the_last_user_message(tok):
    text = S.render_request(tok, REQS[-1])
    assert "<|channel>thought\nI should locate foo first.\n<channel|>" in text
    assert "<|channel>thought\nOpen the file around the match.\n<channel|>" in text
    assert text.count("<|channel>thought") == 2                    # the two turns that carried reasoning, no others


def test_generation_prompt_is_a_bare_model_turn(tok):
    text = S.render_request(tok, REQS[0])
    assert text.endswith("<|turn>model\n")
    assert "<|channel>" not in text[text.rindex("<|turn>model"):]   # nothing forces a thought channel open


def test_tool_arguments_and_results_use_the_gemma_call_format(tok):
    text = S.render_request(tok, REQS[2])
    assert "<|tool_call>call:read_file{" in text and "end_line:6" in text and "filepath:<|\"|>pkg/mod.py<|\"|>" in text
    assert "<|tool_response>response:read_file{value:<|\"|>" in text
