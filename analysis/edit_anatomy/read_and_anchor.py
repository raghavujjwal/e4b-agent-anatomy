"""Does the model see the code it edits? Control runs only (ctl_v2rec1/2). Analysis helper.
usage: python -I -X utf8 read_and_anchor.py <folder with ctl_v2rec{1,2}_trace.jsonl>
For every edit_file call whose status is known: is the longest line of its old_string visible in the prompt the model had (raw or JSON-escaped)?
For every read_file call: did it pass start_line / end_line?
"""
import collections
import json
import statistics as st
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import s4_gates as G  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
SRC = sys.argv[1]
RUNS = ("ctl_v2rec1", "ctl_v2rec2")


def anchor_visible(old_string, prompt_text):
    lines = [l.strip() for l in old_string.split("\n") if l.strip()]
    if not lines:
        return False
    probe = max(lines, key=len)[:80]
    esc = json.dumps(probe)[1:-1]
    return probe in prompt_text or esc in prompt_text or esc.replace("\\\\", "\\") in prompt_text


reads = collections.Counter()
read_tokens = collections.defaultdict(list)
rows = []                      # (status, kind, visible, post_compaction_prompt)
first_read_ranged = collections.Counter()
for run in RUNS:
    for line in open(f"{SRC}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        seen_read = False
        for turns in B.chains_of(tr):
            for k, (prompt, gen, finish, _n, suffix) in enumerate(turns):
                text = tok.decode(gen, skip_special_tokens=False)
                tool = B.tool_of(text)
                c = G.first_call(text)
                if tool == "read_file" and c and c[1] is not None:
                    ranged = ("start_line" in c[1]) or ("end_line" in c[1])
                    reads["ranged" if ranged else "unranged"] += 1
                    if k + 1 < len(turns):
                        read_tokens["ranged" if ranged else "unranged"].append(len(turns[k + 1][4]))
                    if not seen_read:
                        first_read_ranged["ranged" if ranged else "unranged"] += 1
                        seen_read = True
                if tool == "edit_file" and c and c[1] and isinstance(c[1].get("old_string"), str):
                    s = B.edit_status(tok, turns, k, tool)
                    if s not in ("ok", "error"):
                        continue
                    ptxt = tok.decode(prompt, skip_special_tokens=False)
                    first_user, first_model = ptxt.find("<|turn>user"), ptxt.find("<|turn>model")
                    post = first_model != -1 and (first_user == -1 or first_model < first_user)
                    nxt = tok.decode(turns[k + 1][4], skip_special_tokens=False) if k + 1 < len(turns) else ""
                    blocks = list(B.RESP_BLOCK.finditer(nxt))
                    body = blocks[-1].group(2)[:400] if blocks else ""
                    kind = "ok" if s == "ok" else ("occurs N times" if "occurs" in body else ("not found" if "not found" in body else "other error"))
                    rows.append((s, kind, anchor_visible(c[1]["old_string"], ptxt), post))

n_reads = sum(reads.values())
print(f"== read_file calls: {n_reads}; with start_line/end_line {reads['ranged']} ({100 * reads['ranged'] / n_reads:.0f}%), whole-file default {reads['unranged']} ({100 * reads['unranged'] / n_reads:.0f}%)")
for k_, v in read_tokens.items():
    print(f"   {k_:<9} tokens appended per read: mean {st.mean(v):.0f}, median {st.median(v):.0f}, n={len(v)}")
print("   the first read of an episode:", dict(first_read_ranged))
print(f"\n== edit_file calls with a known status and a parsed old_string: {len(rows)}")
by = collections.defaultdict(list)
for s, kind, vis, post in rows:
    by[kind].append(vis)
for kind, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print(f"   {kind:<15} n={len(v):>4}  longest line of old_string visible in the prompt: {100 * sum(v) / len(v):>4.0f}%")
print("\n== same, split by whether the prompt is after a compaction")
for post in (False, True):
    for kind in ("ok", "not found", "occurs N times"):
        v = [vis for s, k_, vis, p in rows if p == post and k_ == kind]
        if v:
            print(f"   after compaction={post!s:<5} {kind:<15} n={len(v):>4} visible {100 * sum(v) / len(v):>4.0f}%")
