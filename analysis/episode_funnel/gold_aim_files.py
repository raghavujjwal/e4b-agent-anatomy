"""File-level view: how many refused edits were aimed at a file the gold patch changes? Outcome classes of the 132 control episodes."""
import os
import collections, json

SP = os.environ.get("WORK_DIR", "work") + "/"
tasks = {}
for line in open(os.environ.get("TASKS_JSONL", "data/tasks.jsonl"), encoding="utf-8"):
    t = json.loads(line)
    tasks[t["instance_id"]] = t


def gold_hunks(patch):
    """{path: [(start, length), ...]} in base-file coordinates, from the unified diff of the source patch."""
    out, cur = {}, None
    for l in patch.split(chr(10)):
        if l.startswith("--- a/"):
            cur = l[6:].strip()
            out.setdefault(cur, [])
        elif l.startswith("--- /dev/null"):
            cur = None
        elif l.startswith("@@") and cur is not None:
            minus = l.split()[1]
            a, _, b = minus[1:].partition(",")
            out[cur].append((int(a), int(b) if b else 1))
    return out


gold = {iid: gold_hunks(t["patch"]) for iid, t in tasks.items()}
n_new_only = sum(1 for iid, g in gold.items() if not g)
print("tasks:", len(tasks), "| gold patches that only add new files (no base-file hunk):", n_new_only)

# outcome classes of the 132 episodes
res = {}
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(os.environ.get("CTL_DIR", "data/ctl_files") + "/" + run + ".jsonl", encoding="utf-8"):
        r = json.loads(line)
        res[(run, r["instance_id"])] = r
print()
print("outcome classes of the 132 control episodes:", dict(collections.Counter(r["outcome_class"] for r in res.values())))
print("stop reasons:", dict(collections.Counter(r["stop_reason"] for r in res.values())))

rows = json.load(open(SP + "edits_all.json", encoding="utf-8"))


def clean(p):
    p = p or ""
    for pre in ("/workspace/", "./", "/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


c = collections.Counter()
per_kind = collections.defaultdict(collections.Counter)
for r in rows:
    if "old" not in r:
        continue
    g = gold.get(r["iid"], {})
    fp = clean(r["filepath"])
    where = "gold file" if fp in g else ("test file" if fp.startswith("tests/") or "/tests/" in fp else "other file")
    per_kind[r["kind"]][where] += 1
print()
print("edit calls by outcome and whether the file is in the gold patch:")
for k in ("ok", "multiple", "not_found", "unchanged", "other"):
    if per_kind[k]:
        tot = sum(per_kind[k].values())
        print("  %-10s n=%3d  %s" % (k, tot, {w: "%d (%.0f%%)" % (v, 100 * v / tot) for w, v in per_kind[k].most_common()}))

# episode view: resolved vs. had a refused edit in a gold file
ep = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    if "old" not in r:
        continue
    g = gold.get(r["iid"], {})
    fp = clean(r["filepath"])
    key = (r["run"], r["iid"])
    if fp in g:
        ep[key]["gold_" + r["kind"]] += 1
    ep[key]["any_" + r["kind"]] += 1
tot = collections.Counter()
for key, r in res.items():
    e = ep.get(key, collections.Counter())
    refused_gold = e["gold_multiple"] + e["gold_not_found"]
    ok_gold = e["gold_ok"]
    cls = "resolved" if str(r["resolved"]) == "True" else "unresolved"
    tot[(cls, "refused in gold file" if refused_gold else "no refusal in gold file", "applied in gold file" if ok_gold else "nothing applied in gold file")] += 1
print()
print("episodes: (resolved?, refused edit in a gold file?, any applied edit in a gold file?)")
for k, v in sorted(tot.items()):
    print("  %3d  %s" % (v, k))
