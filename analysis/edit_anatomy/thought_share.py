"""How much of the context is the model's own earlier thinking? Control runs only (ctl_v2rec1/2). Analysis helper.
usage: python -I -X utf8 thought_share.py <folder with ctl_v2rec{1,2}_trace.jsonl>
Looks at the longest prompt of each episode (the one just before a compaction, for compacted episodes).
"""
import collections
import json
import re
import statistics as st
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
SRC = sys.argv[1]
RUNS = ("ctl_v2rec1", "ctl_v2rec2")
THOUGHT = re.compile(r"<\|channel>thought.*?<channel\|>", re.S)
CALL = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.S)
RESP = re.compile(r"<\|tool_response>.*?<tool_response\|>", re.S)

shown = False
rows = []
for run in RUNS:
    for line in open(f"{SRC}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        best = None
        for turns in B.chains_of(tr):
            for t in turns:
                if best is None or len(t[0]) > len(best):
                    best = t[0]
        if best is None:
            continue
        text = tok.decode(best, skip_special_tokens=False)
        th = THOUGHT.findall(text)
        if not shown and not th and "<|channel>" in text:
            i = text.find("<|channel>")
            print("DEBUG sample around the first <|channel>:", repr(text[i - 60:i + 300]))
            shown = True
        n = len(best)
        n_th = sum(len(tok.encode(s, add_special_tokens=False)) for s in th)
        n_call = sum(len(tok.encode(s, add_special_tokens=False)) for s in CALL.findall(text))
        n_resp = sum(len(tok.encode(s, add_special_tokens=False)) for s in RESP.findall(text))
        rows.append({"tokens": n, "thought": n_th, "calls": n_call, "responses": n_resp, "n_thoughts": len(th)})

print(f"episodes: {len(rows)}")
big = [r for r in rows if r["tokens"] >= 13000]
for name, sub in (("all episodes", rows), ("episodes whose longest prompt is >= 13,000 tokens (about to compact)", big)):
    if not sub:
        continue
    print(f"\n== {name}: n={len(sub)}")
    for key in ("tokens", "thought", "calls", "responses"):
        v = [r[key] for r in sub]
        print(f"   {key:<10} mean {st.mean(v):>7.0f}  median {st.median(v):>7.0f}")
    tot = sum(r["tokens"] for r in sub)
    print(f"   share of the prompt: thoughts {100 * sum(r['thought'] for r in sub) / tot:.0f}%, the model's tool calls {100 * sum(r['calls'] for r in sub) / tot:.0f}%, tool responses {100 * sum(r['responses'] for r in sub) / tot:.0f}%")
    print(f"   thought blocks still in the prompt: mean {st.mean(r['n_thoughts'] for r in sub):.1f}")
