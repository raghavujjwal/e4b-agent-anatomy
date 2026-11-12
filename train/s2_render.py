"""S2.4 / S2.7: render replayed trajectories as one-pass training sequences in format T0 (no thought channel) and audit them.

  python -I -X utf8 train/s2_render.py --replayed D:/data/s2/pilot/replayed.jsonl --out D:/data/s2/pilot/sft.jsonl \
      [--submission agent/submission_e4b_v2rec] [--think-share 0.5] [--show 3]

A sample is the whole conversation rendered once with the scorer's chat template (train/served.py). The loss covers what the model generates and
nothing else: each `<|tool_call>...<tool_call|>` plus the `<|tool_response>` stop token that follows it, and the closing text with `<turn|>`.
Half of the samples carry `<|think|>` in the system turn (the scorer's thinking-on requests) and half do not (thinking-off requests); the model's
own turns have no thought channel in either (T0). Checks (plan S2.7, pass conditions in docs/PLAN_E4B_SFT.md): prefix stability at every assistant
turn, tool results parse as the harness envelope, read windows within the caps, no evaluation-pool repository or teacher path in the text,
token lengths. Samples that fail a check are dropped and counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agent" / "vendor"))
import served  # noqa: E402
import scorer_port as P  # noqa: E402

TOOL_ORDER = ["read_file", "edit_file", "write_file", "run_command", "get_status", "submit_patch"]
# owner/name of the evaluation pools, as in s2_condense.EVAL_REPOS (tests/test_s2.py checks that the two agree). A bare `import urllib3` inside a sample is not
# a leak; the repository itself, a link to it or a path inside it is.
EVAL_REPOS = ["fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx", "litestar-org/litestar", "pydantic/pydantic-settings",
              "pycqa/isort", "urllib3/urllib3", "marshmallow-code/marshmallow", "scrapy/scrapy"]
EVAL_SHORT = {r.split("/")[1] for r in EVAL_REPOS}
CALL = re.compile(r"<\|tool_call>.*?<tool_call\|><\|tool_response>", re.S)
LEAK = re.compile(r"/opt/conda|/testbed|/workspace/[A-Za-z0-9_.\-]+__[A-Za-z0-9_.\-]+__")


RUNNER = re.compile(r"pytest|unittest|\btox\b|nosetests|py\.test|manage\.py test|make test")
GENERIC = {"src", "lib", "tests", "test", "init", "__init__", "python", "utils", "util", "core", "main", "base", "common", "py"}


def unrelated_verification(messages: list[dict]) -> str:
    """'' unless the command run after the last edit runs a test runner or names a part of an edited file's path. A snippet that checks how Python or a library
    behaves (`python -c "print(round(500, -3))"`) says nothing about the patch, and training on it teaches the model to spend a call on nothing."""
    edited, last = set(), -1
    for k, m in enumerate(messages):
        for tc in m.get("tool_calls") or []:
            if tc["function"]["name"] == "edit_file":
                path = json.loads(tc["function"]["arguments"])["filepath"]
                edited |= {p.lower() for p in re.split(r"[/\\.]", path) if len(p) >= 3 and p.lower() not in GENERIC}
                last = k
    for m in messages[last + 1:]:
        for tc in m.get("tool_calls") or []:
            if tc["function"]["name"] == "run_command":
                cmd = json.loads(tc["function"]["arguments"])["command"].lower()
                return "" if RUNNER.search(cmd) or any(t in cmd for t in edited) else "verification does not touch the edit"
    return ""


def system_text(sub_dir: Path) -> tuple[str, list[dict]]:
    import runner
    cfg, _ = runner.load_submission(str(sub_dir))
    instr, missing = runner.interpolate(cfg.get("instruction") or "", {})
    assert not missing, missing
    text = P.adk_system_text(instr, cfg.get("name", "swe_agent"), cfg.get("description") or "")
    declared = runner.schemas_for([t for t in cfg.get("tools") or [] if isinstance(t, str)])
    return text, declared


def thinking_for(instance_id: str, share: float) -> bool:
    h = int(hashlib.sha256(("think/" + instance_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < share


def mask_spans(text: str) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in CALL.finditer(text)]
    last = text.rfind("<tool_response|>")
    end = text.find("<turn|>", last)
    if last >= 0 and end > last:
        spans.append((last + len("<tool_response|>"), end + len("<turn|>")))
    return spans


def audit_messages(messages: list[dict]) -> str:
    """'' if every tool result is a harness envelope within the caps and nothing in the text identifies an evaluation pool or the teacher's paths."""
    for m in messages:
        c = m.get("content")
        if m["role"] == "tool":
            try:
                d = json.loads(c)
            except Exception:
                return "tool result is not JSON"
            if "status" not in d:
                return "tool result without status"
            body = d.get("content")
            if isinstance(body, str) and (body.count("\n") + 1 > 150 or len(body) > 10000):
                return "read window over the caps"
        text = c if isinstance(c, str) else json.dumps(m.get("tool_calls") or "")
        low = text.lower()
        if any(n in low for n in EVAL_REPOS) or any(f"/workspace/{s}" in low for s in EVAL_SHORT):
            return "evaluation-pool repository in " + m["role"]
        if LEAK.search(text):
            return "teacher path left in " + m["role"]
    return ""


MAX_TOKENS = 12000          # plan S2.7: a sample longer than this is left out (the trainer's --max-len is 12288, the card reports the median)


UNLICENSED = (None, "", "None")        # what the task metadata holds for a repository GitHub could not match to a licence


def render_one(tok, rec: dict, sys_text: str, tools: list[dict], think: bool, max_tokens: int = MAX_TOKENS, drop_unlicensed: bool = False):
    repo = (rec.get("repo") or "").lower()
    if repo in EVAL_REPOS or repo.split("/")[-1] in EVAL_SHORT:
        return None, "evaluation-pool repository"
    if drop_unlicensed and rec.get("license") in UNLICENSED:
        return None, "repository has no declared licence"
    messages = [{"role": "system", "content": sys_text}] + rec["messages"]
    bad = audit_messages(messages[1:]) or unrelated_verification(rec["messages"])
    if bad:
        return None, bad
    text = served.render(tok, messages, tools, enable_thinking=think, add_generation_prompt=False)
    for k, m in enumerate(messages):
        if m["role"] == "assistant":
            prompt = served.render(tok, messages[:k], tools, enable_thinking=think, add_generation_prompt=True)
            if not text.startswith(prompt):
                return None, f"render not prefix-stable at message {k}"
    spans = mask_spans(text)
    n_calls = sum(1 for m in messages if m.get("tool_calls"))
    if len(spans) != n_calls + 1:
        return None, f"expected {n_calls + 1} generated spans, found {len(spans)}"
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_tokens:
        return None, f"over {max_tokens} tokens"
    mask = [1 if any(a <= s and e <= b for a, b in spans) else 0 for s, e in offs]
    if sum(mask) == 0:
        return None, "empty loss mask"
    return {"instance_id": rec["instance_id"], "repo": rec["repo"], "license": rec.get("license"), "teacher": rec.get("teacher"),
            "enable_thinking": think, "input_ids": ids, "loss_mask": mask, "n_tokens": len(ids), "n_loss_tokens": sum(mask),
            "n_calls": n_calls, "synthesized": rec.get("synthesized", []), "text": text}, ""


_W: dict = {}


def _init_worker(submission: str, max_tokens: int, drop_unlicensed: bool) -> None:
    _W["tok"] = served.load_tokenizer()
    _W["sys"], _W["tools"] = system_text(Path(submission))
    _W["max"], _W["unlicensed"] = max_tokens, drop_unlicensed


def _work(item):
    rec, think = item
    return render_one(_W["tok"], rec, _W["sys"], _W["tools"], think, _W["max"], _W["unlicensed"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="render in this many processes (a sample takes about 4 s)")
    ap.add_argument("--replayed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--submission", default=str(HERE.parent / "agent" / "submission_e4b_v2rec"))
    ap.add_argument("--think-share", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="leave out samples longer than this many tokens (plan S2.7: at most 12,000)")
    ap.add_argument("--drop-unlicensed", action="store_true", help="leave out samples from repositories with no declared licence (the task metadata says None)")
    ap.add_argument("--show", type=int, default=0, help="print this many rendered samples")
    ap.add_argument("--no-text", action="store_true", help="leave the rendered text out of the output file")
    a = ap.parse_args(argv)
    sys_text, tools = system_text(Path(a.submission))
    names = [t["function"]["name"] for t in tools]
    assert names == TOOL_ORDER, names
    kept, why, dropped_earlier, todo = [], Counter(), Counter(), []
    for line in open(a.replayed, encoding="utf-8"):
        rec = json.loads(line)
        if "messages" not in rec:
            dropped_earlier[(rec.get("dropped") or "?")[:40]] += 1
            continue
        todo.append((rec, thinking_for(rec["instance_id"], a.think_share)))
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker, initargs=(a.submission, a.max_tokens, a.drop_unlicensed)) as ex:
            results = list(ex.map(_work, todo, chunksize=4))
    else:
        tok = served.load_tokenizer()
        results = [render_one(tok, rec, sys_text, tools, think, a.max_tokens, a.drop_unlicensed) for rec, think in todo]
    for out, bad in results:
        if out is None:
            why[bad] += 1
        else:
            kept.append(out)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for o in kept:
            if a.no_text:
                o = {k: v for k, v in o.items() if k != "text"}
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    toks = sorted(o["n_tokens"] for o in kept)
    summary = {"rendered": len(kept), "dropped_in_replay": dict(dropped_earlier), "dropped_in_render": dict(why),
               "tokens": {"median": toks[len(toks) // 2] if toks else 0, "max": toks[-1] if toks else 0},
               "loss_share": round(sum(o["n_loss_tokens"] for o in kept) / max(1, sum(o["n_tokens"] for o in kept)), 4),
               "think_on": sum(o["enable_thinking"] for o in kept), "calls": dict(Counter(o["n_calls"] for o in kept)),
               "repos": len({o["repo"] for o in kept})}
    print(json.dumps(summary, indent=1))
    for o in kept[: a.show]:
        print("=" * 100, o["instance_id"], "think" if o["enable_thinking"] else "no-think", o["n_tokens"], "tokens")
        print(o["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
