"""Does the context compaction coincide with worse edits, and what fills the context? Control runs only (ctl_v2rec1/2). Analysis helper.
usage: python -I -X utf8 compaction_effect.py <folder with ctl_v2rec{1,2}[_trace].jsonl>
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
STATUS = re.compile(r'^\{"status": "(\w+)"')

rows = {}
for run in RUNS:
    for l in open(f"{SRC}/{run}.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            rows[(run, r["instance_id"])] = r

pre, post = collections.Counter(), collections.Counter()
comp_sample = None
first_comp_at_call = []
first_comp_prompt_len = []
growth = collections.defaultdict(list)          # tool -> new prompt tokens that its response added
growth_total = collections.Counter()
ep_rows = []
order_ok = 0
for run in RUNS:
    for line in open(f"{SRC}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        ev = tr["trace"]
        r = rows[(run, tr["instance_id"])]
        seen = False
        calls = 0
        edits_pre = edits_post = 0
        for e in ev:
            if e.get("event") == "compaction":
                if comp_sample is None:
                    comp_sample = e
                if not seen:
                    first_comp_at_call.append(calls)
                seen = True
                continue
            if e.get("tool"):
                calls += 1
                if e["tool"] in ("edit_file", "write_file"):
                    m = STATUS.match(e.get("result_head", ""))
                    st_ = m.group(1) if m else "unknown"
                    (post if seen else pre)[st_] += 1
        # what fills the context: the tokens each response adds to the next prompt
        for turns in B.chains_of(tr):
            for k in range(len(turns) - 1):
                text = tok.decode(turns[k][1], skip_special_tokens=False)
                tool = B.tool_of(text)
                if tool is None:
                    continue
                new = len(turns[k + 1][4])          # suffix of the next prompt = what was appended
                growth[tool].append(new)
                growth_total[tool] += new
        # longest prompt of the episode
        mx = 0
        for turns in B.chains_of(tr):
            for t in turns:
                mx = max(mx, len(t[0]))
        ep_rows.append({"run": run, "id": tr["instance_id"], "compacted": seen, "resolved": bool(r["resolved"]), "calls": r["tool_calls"], "max_prompt": mx})

print("compaction event sample keys:", None if comp_sample is None else {k: (str(v)[:80]) for k, v in comp_sample.items()})
print("\n== edit/write calls before vs after the first compaction in the same episode")
for name, c in (("before (or no compaction)", pre), ("after a compaction", post)):
    n = sum(c.values())
    print(f"  {name:<28} n={n:>4}  " + ", ".join(f"{k} {v} ({100 * v / n:.1f}%)" for k, v in c.most_common()))
print("\n== first compaction happens after this many tool calls: median %s, p10 %s, p90 %s (n=%d)" % (
    st.median(first_comp_at_call), sorted(first_comp_at_call)[int(.1 * len(first_comp_at_call))], sorted(first_comp_at_call)[int(.9 * len(first_comp_at_call))], len(first_comp_at_call)))
print("\n== tokens a tool response adds to the next prompt (suffix length), by the tool that was called")
tot = sum(growth_total.values())
for tool, v in sorted(growth.items(), key=lambda kv: -growth_total[kv[0]]):
    v2 = sorted(v)
    print(f"  {tool:<14} calls {len(v):>5}  mean {st.mean(v):>6.0f}  median {st.median(v):>5.0f}  p90 {v2[int(.9 * len(v2))]:>6}  share of all appended tokens {100 * growth_total[tool] / tot:>4.0f}%")
print("\n== longest prompt per episode: with a compaction vs without")
for flag in (True, False):
    sub = [e["max_prompt"] for e in ep_rows if e["compacted"] == flag]
    sub.sort()
    print(f"  compacted={flag!s:<5} n={len(sub):>3}  median {sub[len(sub) // 2]}  p10 {sub[int(.1 * len(sub))]}  p90 {sub[int(.9 * len(sub))]}  max {sub[-1]}")
print("\n== resolved by compaction and by number of tool calls (to see how much is just 'long episodes')")
for lo, hi in ((0, 6), (7, 10), (11, 16), (17, 24)):
    for flag in (False, True):
        sub = [e for e in ep_rows if e["compacted"] == flag and lo <= e["calls"] <= hi]
        if sub:
            print(f"  calls {lo:>2}-{hi:<2} compacted={flag!s:<5} n={len(sub):>3} resolved {sum(e['resolved'] for e in sub):>2} ({100 * sum(e['resolved'] for e in sub) / len(sub):>4.1f}%)")
