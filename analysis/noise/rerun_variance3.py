import os
import json
from collections import Counter
D = os.environ.get("CTL_DIR", "data/ctl_files")
def rows_of(name):
    out = {}
    for l in open(f"{D}/{name}.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l); out[r["instance_id"]] = r
    return out
a, b = rows_of("ctl_v2rec1"), rows_of("ctl_v2rec2")
for n, r in (("run1", a), ("run2", b)):
    print(n, "clock:", Counter(x.get("clock") for x in r.values()), "| par_k:", Counter(x.get("par_k") for x in r.values()),
          "| stop_reason:", dict(Counter(x["stop_reason"] for x in r.values())),
          "| max agent_seconds:", round(max(x["agent_seconds"] for x in r.values()), 0),
          "| over 330 s:", sum(x["agent_seconds"] > 330 for x in r.values()))
ids = sorted(set(a) & set(b))
same_size = [i for i in ids if a[i]["patch_bytes"] == b[i]["patch_bytes"] and a[i]["patch_bytes"] > 0]
print("non-empty patches with the same size in both runs:", len(same_size), [(i, bool(a[i]["resolved"]), bool(b[i]["resolved"])) for i in same_size])
# does the time spent decide anything: resolved vs not, by seconds, across the two runs
over = [(i, n) for n, r in (("run1", a), ("run2", b)) for i in r if r[i]["agent_seconds"] > 300]
print("episodes above 300 s:", len(over), "| resolved among them:", sum(bool((a if n == 'run1' else b)[i]['resolved']) for i, n in over))
