"""What do the episodes that never applied an edit in a gold file do instead?"""
import os
import collections, json, posixpath

SP = os.environ.get("WORK_DIR", "work") + "/"
eps = json.load(open(SP + "episodes.json", encoding="utf-8"))
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.environ.get("TASKS_JSONL", "data/tasks.jsonl"), encoding="utf-8")}


def gold_files(patch):
    return {l[6:].strip() for l in patch.split(chr(10)) if l.startswith("--- a/")}


def clean(p):
    p = p or ""
    for pre in ("/workspace/", "./", "/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


cls = collections.Counter()
ex = []
for e in eps:
    g = gold_files(tasks[e["iid"]]["patch"])
    applied = []
    for c in e["calls"]:
        a = c.get("args") or {}
        r = c.get("resp") or {}
        if c["tool"] in ("edit_file", "write_file") and r.get("status") == "ok":
            applied.append(clean(a.get("filepath")))
    if any(p in g for p in applied):
        continue
    gd = {posixpath.dirname(p) for p in g}
    if not applied:
        kind = "no applied edit at all (%s)" % e["result"]["outcome_class"]
    elif all(p.startswith("tests/") or "/tests/" in p or posixpath.basename(p).startswith("test_") for p in applied):
        kind = "edited only test files"
    elif any(posixpath.dirname(p) in gd for p in applied):
        kind = "edited another source file in the same directory as a gold file"
    elif any(p.endswith(".md") or p.endswith(".rst") or p.startswith("docs/") for p in applied):
        kind = "edited docs / markdown (and no gold file)"
    else:
        kind = "edited a source file elsewhere"
    cls[kind] += 1
    if len(ex) < 14 and applied:
        ex.append((e["iid"], sorted(g)[:2], applied[:3]))
print("episodes without an applied edit in a gold file:", sum(cls.values()))
for k, v in cls.most_common():
    print("  %3d  %s" % (v, k))
print()
for iid, g, a in ex:
    print("  %-14s gold %s | edited %s" % (iid, g, a))

# those with no applied edit at all: why? last tool calls
print()
print("episodes with no applied edit at all: tool-call mix (read/edit/run/submit) and whether an edit was even attempted")
for e in eps:
    pass
no_edit = []
for e in eps:
    ok = [c for c in e["calls"] if c["tool"] in ("edit_file", "write_file") and (c.get("resp") or {}).get("status") == "ok"]
    if ok:
        continue
    attempted = sum(1 for c in e["calls"] if c["tool"] == "edit_file")
    no_edit.append((e["iid"], len(e["calls"]), attempted, e["result"]["stop_reason"], e["result"]["outcome_class"]))
print("  n =", len(no_edit), "| attempted at least one edit:", sum(1 for x in no_edit if x[2] > 0), "| stop reasons:", dict(collections.Counter(x[3] for x in no_edit)))
