import os
import json, re
from collections import Counter
D = os.environ.get("CTL_DIR", "data/ctl_files")
def trace_of(name):
    out = {}
    for l in open(f"{D}/{name}_trace.jsonl", encoding="utf-8"):
        if l.strip():
            t = json.loads(l); out[t["instance_id"]] = t["trace"]
    return out
ta, tb = trace_of("ctl_v2rec1"), trace_of("ctl_v2rec2")
ids = sorted(set(ta) & set(tb))
def calls(trace):
    return [e for e in trace if e.get("tool")]
# 1. which tool is the first call, in each run
print("first tool, run1:", Counter(calls(ta[i])[0]["tool"] for i in ids if calls(ta[i])))
print("first tool, run2:", Counter(calls(tb[i])[0]["tool"] for i in ids if calls(tb[i])))
# 2. the strictest artifact-free key: tool name + the file path that appears in the result head (if any)
def key(e):
    head = e.get("result_head") or ""
    m = re.search(r'"filepath": "([^"]+)"', head)
    return (e["tool"], m.group(1) if m else None)
def first_diff(ca, cb, keyf):
    n = min(len(ca), len(cb))
    for j in range(n):
        if keyf(ca[j]) != keyf(cb[j]):
            return j
    return -1 if len(ca) == len(cb) else n
for label, keyf in (("tool name only", lambda e: e["tool"]), ("tool name + file path", key)):
    c = Counter(first_diff(calls(ta[i]), calls(tb[i]), keyf) for i in ids)
    n0 = c.get(0, 0); n01 = n0 + c.get(1, 0)
    print(f"{label:24s}: differ at call 0 in {n0}/{len(ids)}, within the first two calls {n01}/{len(ids)}, identical sequence {c.get(-1, 0)}  | histogram {dict(sorted(c.items()))}")
# 3. what a differing first call looks like (five tasks)
shown = 0
for i in ids:
    ca, cb = calls(ta[i]), calls(tb[i])
    if ca and cb and key(ca[0]) != key(cb[0]) and shown < 6:
        print(f"  {i}: run1 first {key(ca[0])}  |  run2 first {key(cb[0])}")
        shown += 1
# 4. do the heads of identical first calls differ in volatile text?
vol = 0; same = 0
for i in ids:
    ca, cb = calls(ta[i]), calls(tb[i])
    if ca and cb and key(ca[0]) == key(cb[0]):
        same += 1
        vol += (ca[0].get("result_head") or "") != (cb[0].get("result_head") or "")
print("first calls with the same key:", same, "| of those, result heads differ in raw text:", vol)
