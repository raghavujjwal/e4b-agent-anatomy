"""Render a conversation the way the scorer's vLLM serves it to Gemma-4 E4B.

Every training sample, every prompt-length estimate and every "does this render match the scorer" check goes
through here, so there is exactly one definition of "the prompt the model sees".

What the server does to a request before the chat template runs (vllm 0.19.1, entrypoints/chat_utils.py):
  * a string `content` becomes [{"type": "text", "text": ...}]; the template then puts a space after system text
  * tool-call `arguments` arrive as JSON strings and are parsed into dicts
  * assistant `reasoning` is kept; adk-submission's reasoning patch mirrors `reasoning_content` into it
The template is the competition model's ORIGINAL chat_template.jinja (sha256 below), not the July canonical one:
  * enable_thinking only adds `<|think|>` to the first system turn
  * the generation prompt is `<|turn>model\\n` and nothing else (no forced `<|channel>thought\\n`)
  * an assistant turn's thought is rendered only when it has tool_calls AND comes after the last user message
Rendering with the canonical template instead would silently train on a prompt the scorer never produces.

Usage:
    tok = load_tokenizer()                       # E4B_TOKENIZER_DIR, else the Kaggle model copy
    text = render(tok, messages, tools, enable_thinking=True)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

ORIGINAL_TEMPLATE_SHA256 = "33204f1acb5bd0002713e16a593847f24ceeafe711ed88bda2a352dc996a3373"
CANONICAL_TEMPLATE_SHA256 = "0a2c8073"        # prefix only: the July 9 file, kept so the error can name it
DEFAULT_TOKENIZER_DIR = r"D:\data\kagglehub\models\google\gemma-4\transformers\gemma-4-e4b-it-qat-q4_0-unquantized\2"


class WrongTemplate(RuntimeError):
    pass


def template_sha256(model_dir: str | os.PathLike) -> str | None:
    p = Path(model_dir) / "chat_template.jinja"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def check_template(model_dir: str | os.PathLike) -> None:
    sha = template_sha256(model_dir)
    if sha is None:
        raise WrongTemplate(f"{model_dir}/chat_template.jinja not found; cannot verify the template")
    if sha != ORIGINAL_TEMPLATE_SHA256:
        which = "the July canonical template" if sha.startswith(CANONICAL_TEMPLATE_SHA256) else "an unknown template"
        raise WrongTemplate(f"{model_dir} holds {which} (sha256 {sha}); the scorer serves {ORIGINAL_TEMPLATE_SHA256}")


def load_tokenizer(model_dir: str | os.PathLike | None = None, verify: bool = True):
    from transformers import AutoTokenizer
    d = str(model_dir or os.environ.get("E4B_TOKENIZER_DIR") or DEFAULT_TOKENIZER_DIR)
    if verify:
        check_template(d)
    return AutoTokenizer.from_pretrained(d)


def served_messages(messages: list[dict]) -> list[dict]:
    """The messages as the template receives them after vLLM's pre-processing."""
    out = []
    for m in copy.deepcopy(messages):
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                tc["function"]["arguments"] = json.loads(a)
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = [{"type": "text", "text": c}]
        out.append(m)
    return out


def render(tok, messages: list[dict], tools: list[dict] | None = None, enable_thinking: bool = True,
           add_generation_prompt: bool = True, tokenize: bool = False, **template_kwargs):
    """Render a conversation. `messages` use the OpenAI wire format (as in a request body or a trace)."""
    return tok.apply_chat_template(served_messages(messages), tools=tools or None,
                                   add_generation_prompt=add_generation_prompt, tokenize=tokenize,
                                   enable_thinking=enable_thinking, **template_kwargs)


def render_request(tok, body: dict, add_generation_prompt: bool = True, tokenize: bool = False):
    """Render a recorded request body, taking enable_thinking from its chat_template_kwargs."""
    kw = dict(body.get("chat_template_kwargs") or {})
    return tok.apply_chat_template(served_messages(body["messages"]), tools=body.get("tools") or None,
                                   add_generation_prompt=add_generation_prompt, tokenize=tokenize, **kw)
