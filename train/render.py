"""Turn a sample of train/make_edit_data.py (OpenAI-style messages) into prompt and completion token ids, the way the scorer renders them.

  python train/render.py --data D:/data/edit_sft --tokenizer <checkpoint dir> [--format edit_file] [--thought empty]

The prompt is `apply_chat_template(messages[:-1], tools, add_generation_prompt=True, enable_thinking=True)`; the submissions run with a thinking
budget, and the template then puts `<|think|>` in the system turn. The completion is what follows the prompt in the template's own render of the
whole conversation, which ends with the `<|tool_response>` stop token. The template drops past reasoning from history, so the prompt is exactly
what the server renders at that step; the thought channel of the target itself is written here:
  empty  `<|channel>thought\n<channel|>` then the call (the model learns to call without thinking at this step)
  copy   the thought holds the exact lines to be replaced, then the call (copy first, then write the call)
Samples whose render is not prefix-stable are dropped and counted.
"""
from __future__ import annotations
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

TOOLS_FILE = {"edit_file": "tools_edit_file.json", "lines": "tools_lines.json"}


def thought_text(rec: dict, mode: str) -> str:
    if mode == "empty":
        return ""
    call = json.loads(rec["messages"][-1]["tool_calls"][0]["function"]["arguments"])
    old = call.get("old_string")
    if old is None:
        old = (call.get("args") or {}).get("expect", "")
    return "Lines to change, copied exactly from the file:\n" + str(old) + "\n"


def render(tok, rec: dict, tools: list, thought: str = "empty"):
    """Returns ({'prompt_ids', 'gen_ids'}, None) or (None, reason)."""
    msgs = copy.deepcopy(rec["messages"])
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"]["arguments"], str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    if msgs[-1]["role"] != "assistant" or not msgs[-1].get("tool_calls"):
        return None, "target is not a tool call"
    kw = dict(tools=tools, tokenize=False, enable_thinking=True)
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, **kw)
    full = tok.apply_chat_template(msgs, add_generation_prompt=False, **kw)
    if not full.startswith(prompt):
        return None, "render is not prefix-stable"
    tail = full[len(prompt):]
    if not tail.startswith("<|tool_call>") or not tail.endswith("<|tool_response>"):
        return None, "unexpected completion shape"
    gen = "<|channel>thought\n" + thought_text(rec, thought) + "<channel|>" + tail
    return {"prompt_ids": tok(prompt, add_special_tokens=False)["input_ids"],
            "gen_ids": tok(gen, add_special_tokens=False)["input_ids"]}, None


def load_split(data_dir: str | Path, split: str, formats: list[str]):
    data_dir = Path(data_dir)
    tools = {f: json.load(open(data_dir / TOOLS_FILE[f], encoding="utf-8")) for f in formats}
    for line in open(data_dir / f"{split}.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        if rec["format"] in formats:
            yield rec, tools[rec["format"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--format", action="append", default=[])
    ap.add_argument("--thought", choices=["empty", "copy"], default="empty")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--show", type=int, default=1, help="print this many rendered samples")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    formats = a.format or ["edit_file"]
    drop, n, plen, glen, shown = Counter(), 0, [], [], 0
    for rec, tools in load_split(a.data, a.split, formats):
        out, why = render(tok, rec, tools, a.thought)
        if out is None:
            drop[why] += 1
            continue
        n += 1
        plen.append(len(out["prompt_ids"]))
        glen.append(len(out["gen_ids"]))
        if shown < a.show:
            shown += 1
            print("PROMPT TAIL:", repr(tok.decode(out["prompt_ids"][-60:])))
            print("COMPLETION :", repr(tok.decode(out["gen_ids"])))
    plen.sort(), glen.sort()
    pct = lambda v, q: v[min(len(v) - 1, int(q * len(v)))] if v else 0
    print(json.dumps({"samples": n, "dropped": dict(drop), "prompt_tokens_median": pct(plen, .5), "prompt_tokens_p95": pct(plen, .95),
                      "prompt_tokens_max": plen[-1] if plen else 0, "gen_tokens_median": pct(glen, .5), "gen_tokens_p95": pct(glen, .95),
                      "gen_tokens_max": glen[-1] if glen else 0}))


if __name__ == "__main__":
    main()
