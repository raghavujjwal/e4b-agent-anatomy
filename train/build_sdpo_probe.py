"""Probe set for the self-teacher diagnostic (train/sdpo_probe.py): edit_file turns of the base model's own runs, with the tool's answer as feedback.

  python train/build_sdpo_probe.py --runs D:/data/e4b_runs/base1 ... --tokenizer <checkpoint dir> --out D:/data/sdpo_probe

Why (docs/MEASUREMENTS.md, `ad2`): after a failed edit_file the adapted model resent the same call 89% of the time. SDPO (Hubotter et al. 2026, arXiv 2601.20802) trains a
model by asking the SAME model, with the feedback of an attempt in its context, how likely it now finds each token of that attempt. This file builds the data to ask the
cheap question first: does E4B, shown "old_string not found", see which tokens of its failed call were wrong? Every sample keeps the exact token ids of the turn
(prompt_ids, gen_ids; nothing re-rendered) and the tool answer that followed it (feedback, read from the NEXT turn's prompt).
Kinds:
  notfound     edit failed with "not found" and old_string is a near miss of text the model had read (similarity >= 0.9): the characters of old_string that differ from the best
               matching window of the file are the known mistakes; their tokens go to wrong_tok, the other old_string tokens to right_tok
  repeat_fail  edit_file call identical to the previous edit call of the run, which had failed, and this one failed again (the loop of the adapted models)
  first_fail   failed edit that is not such a repeat
  ok_edit      edit applied
Each failed sample also carries fb_mismatch, the feedback of another failed sample, to separate "the teacher read this error" from "any error text moves the probabilities".
"""
from __future__ import annotations
import argparse
import difflib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mine"))
from build_replay_data import turns_of, tool_of  # noqa: E402
from trace_analysis import parse_calls, file_texts  # noqa: E402

FB_RE = re.compile(r'<\|tool_response>response:edit_file\{value:<\|"\|>(.*?)<\|"\|>\}<tool_response\|>', re.S)
CALL_RE = re.compile(r"<\|tool_call>call:edit_file\{.*?\}<tool_call\|>", re.S)
MIN_SIM = 0.9    # kind notfound needs old_string to be a close copy of the file text, so that the differing characters are the model's mistakes (at 0.7 whole regions were marked)
OLD_RE = re.compile(r'old_string:<\|"\|>(.*?)<\|"\|>', re.S)
PATH_RE = re.compile(r'filepath:<\|"\|>(.*?)<\|"\|>', re.S)


def char_ends(tok, ids):
    """End offset in the decoded text of each token (decoding prefixes, so multi-byte pieces are counted where they complete)."""
    ends = []
    for i in range(1, len(ids) + 1):
        ends.append(len(tok.decode(ids[:i], skip_special_tokens=False)))
    return ends


def tokens_overlapping(ends, a, b):
    """Indices of tokens whose text overlaps the character range [a, b)."""
    out = []
    for i, e in enumerate(ends):
        s = ends[i - 1] if i else 0
        if s < b and e > a:
            out.append(i)
    return out


def best_window(old, texts):
    """(similarity, window text) of the lines of the file that look most like old, same line count as old, over all texts the model read."""
    best = (0.0, "")
    k = max(len(old.split("\n")), 1)
    for text in texts:
        tl = text.split("\n")
        for i in range(0, max(len(tl) - k + 1, 1)):
            win = "\n".join(tl[i:i + k])
            sm = difflib.SequenceMatcher(None, old, win, autojunk=False)
            if sm.quick_ratio() > max(best[0], 0.6):
                r = sm.ratio()
                if r > best[0]:
                    best = (r, win)
    return best


def wrong_chars(old, window):
    """Positions in old that the best matching file text does not contain there (replaced or deleted characters)."""
    bad = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, old, window, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            bad.update(range(i1, i2))
    return bad


def classify_tokens(ends, base_char, old, bad):
    """(wrong, right) token indices inside old_string: old starts at character base_char of the decoded generation."""
    wrong, right = set(), set()
    for ci in range(len(old)):
        for t in tokens_overlapping(ends, base_char + ci, base_char + ci + 1):
            (wrong if ci in bad else right).add(t)
    right -= wrong
    return sorted(wrong), sorted(right)


def feedback_of(tok, turns, k):
    if k + 1 >= len(turns):
        return None
    m = None
    for m in FB_RE.finditer(tok.decode(turns[k + 1][4], skip_special_tokens=False)):
        pass
    return m.group(1) if m else None


def run_samples(run, tok, max_len):
    res = {json.loads(l)["instance_id"] for l in open(f"{run}.jsonl", encoding="utf-8")}
    out = []
    for line in open(f"{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        if tr["instance_id"] not in res:
            continue
        turns, msgs = turns_of(tr)
        if not turns:
            continue
        calls = [c for m in msgs for c in (parse_calls(m) if m.get("role") == "assistant" else [])]
        shown = file_texts(msgs, {c["id"]: c for c in calls})
        prev_call, prev_status = None, None
        for k, (prompt, gen, finish, _none, _suf) in enumerate(turns):
            if finish == "length":
                continue
            text = tok.decode(gen, skip_special_tokens=False)
            if tool_of(text) != "edit_file":
                continue
            fb = feedback_of(tok, turns, k)
            cm = CALL_RE.search(text)
            if fb is None or not cm:
                continue
            status = "ok" if '"status": "ok"' in fb[:40] else "error"
            call_text = cm.group(0)
            row = {"id": f"{Path(run).name}/{tr['instance_id']}/{k}", "prompt_ids": prompt, "gen_ids": gen, "fb": fb[:1500]}
            is_repeat = status == "error" and prev_status == "error" and call_text == prev_call
            prev_call, prev_status = call_text, status
            if len(prompt) + len(gen) > max_len:
                continue
            ends = char_ends(tok, gen)
            row["call_tok"] = [tokens_overlapping(ends, cm.start(), cm.end())[0], tokens_overlapping(ends, cm.start(), cm.end())[-1] + 1]
            if status == "ok":
                row["kind"] = "ok_edit"
            elif is_repeat:
                row["kind"] = "repeat_fail"
            else:
                row["kind"] = "first_fail"
                om, pm = OLD_RE.search(text, cm.start()), PATH_RE.search(text, cm.start())
                if "not found" in fb.lower() and om and pm and shown.get(pm.group(1)):
                    old = om.group(1)
                    sim, win = best_window(old, shown[pm.group(1)])
                    if sim >= MIN_SIM and old not in win:
                        bad = wrong_chars(old, win)
                        wrong, right = classify_tokens(ends, om.start(1), old, bad)
                        if wrong and len(right) >= 3:
                            row.update(kind="notfound", wrong_tok=wrong, right_tok=right, similarity=round(sim, 3))
            out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run paths without suffix (<run>.jsonl and <run>_trace.jsonl); base-model runs only, no adapters")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=12000)
    ap.add_argument("--cap", nargs=4, type=int, default=[150, 150, 100, 100], metavar=("NOTFOUND", "REPEAT_FAIL", "FIRST_FAIL", "OK_EDIT"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [r for run in a.runs for r in run_samples(run, tok, a.max_len)]
    found = Counter(r["kind"] for r in rows)
    rng = random.Random(a.seed)
    keep = []
    for kind, cap in zip(("notfound", "repeat_fail", "first_fail", "ok_edit"), a.cap):
        pool = [r for r in rows if r["kind"] == kind]
        rng.shuffle(pool)
        keep += pool[:cap]
    failed = [r for r in keep if r["kind"] != "ok_edit"]
    for r in failed:
        other = [o for o in failed if o["fb"] != r["fb"]]
        r["fb_mismatch"] = rng.choice(other)["fb"] if other else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe.jsonl", "w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    stats = {"found": dict(found), "kept": dict(Counter(r["kind"] for r in keep)), "runs": [Path(r).name for r in a.runs],
             "tokens": sum(len(r["prompt_ids"]) + len(r["gen_ids"]) for r in keep)}
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    sys.exit(main())
