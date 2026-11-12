import os
import json, statistics as st
from collections import Counter
D = os.environ.get("CTL_DIR", "data/ctl_files")
def rows_of(name):
    out = {}
    for l in open(f"{D}/{name}.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l); out[r["instance_id"]] = r
    return out
def trace_of(name):
    out = {}
    for l in open(f"{D}/{name}_trace.jsonl", encoding="utf-8"):
        if l.strip():
            t = json.loads(l); out[t["instance_id"]] = t["trace"]
    return out
a, b = rows_of("ctl_v2rec1"), rows_of("ctl_v2rec2")
ta, tb = trace_of("ctl_v2rec1"), trace_of("ctl_v2rec2")
print("keys of a row:", sorted(next(iter(a.values())).keys()))
ids = sorted(set(a) & set(b))
print("tasks in both:", len(ids), "| resolved run1", sum(bool(a[i]["resolved"]) for i in ids), "run2", sum(bool(b[i]["resolved"]) for i in ids))
both = sum(bool(a[i]["resolved"]) and bool(b[i]["resolved"]) for i in ids)
only1 = sum(bool(a[i]["resolved"]) and not b[i]["resolved"] for i in ids)
only2 = sum(not a[i]["resolved"] and bool(b[i]["resolved"]) for i in ids)
print(f"resolved in both {both}, only run1 {only1}, only run2 {only2}, neither {len(ids)-both-only1-only2}")
p = (sum(bool(a[i]['resolved']) for i in ids) + sum(bool(b[i]['resolved']) for i in ids)) / (2*len(ids))
print(f"if every task had the same chance p={p:.3f} on each run: expected only-one-run solves {2*len(ids)*p*(1-p):.1f}, both {len(ids)*p*p:.1f}; observed only-one {only1+only2}, both {both}")
# how different are the runs on the same task
same_class = sum(a[i]["outcome_class"] == b[i]["outcome_class"] for i in ids)
print("same outcome class:", same_class, "of", len(ids))
rel = [abs(a[i]["tool_calls"] - b[i]["tool_calls"]) for i in ids]
print("tool_calls: same count in", sum(x == 0 for x in rel), "of", len(ids), "| median |difference|", st.median(rel))
ps = [(a[i].get("patch_bytes"), b[i].get("patch_bytes")) for i in ids]
print("patch_bytes identical in", sum(x == y for x, y in ps), "of", len(ids), "(both empty:", sum(x == y == 0 for x, y in ps), ")")
sec = [abs(a[i]["agent_seconds"] - b[i]["agent_seconds"]) / max(1.0, (a[i]["agent_seconds"] + b[i]["agent_seconds"]) / 2) for i in ids]
print("agent_seconds: median relative difference between the two runs of the same task", round(st.median(sec), 2))
tok = [abs(a[i]["completion_tokens"] - b[i]["completion_tokens"]) / max(1.0, (a[i]["completion_tokens"] + b[i]["completion_tokens"]) / 2) for i in ids]
print("completion_tokens: median relative difference", round(st.median(tok), 2))
# where do the trajectories first part ways
def calls(trace):
    return [(e.get("tool"), tuple(e.get("args_keys") or ()), (e.get("result_head") or "")[:90]) for e in trace if e.get("tool")]
first = []
for i in ids:
    ca, cb = calls(ta.get(i, [])), calls(tb.get(i, []))
    n = min(len(ca), len(cb))
    k = next((j for j in range(n) if ca[j] != cb[j]), None)
    first.append(k if k is not None else (n if len(ca) != len(cb) else -1))
c = Counter(first)
print("call index at which the two runs of the same task first differ (0 = the very first call; -1 = identical sequences):")
print("  ", dict(sorted(c.items())))
print("  runs differing at the first call:", c.get(0, 0), "| within the first 2 calls:", c.get(0, 0) + c.get(1, 0), "| identical in full:", c.get(-1, 0), "of", len(ids))
# among tasks resolved by exactly one run: what did the other run do
for i in ids:
    if bool(a[i]["resolved"]) != bool(b[i]["resolved"]):
        r1, r2 = a[i], b[i]
        print(f"  flip {i}: run1 {r1['outcome_class']} ({r1['tool_calls']} calls, {r1.get('patch_bytes')} B) vs run2 {r2['outcome_class']} ({r2['tool_calls']} calls, {r2.get('patch_bytes')} B)")
