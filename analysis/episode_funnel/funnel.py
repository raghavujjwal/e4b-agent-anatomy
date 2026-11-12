"""Funnel of the 132 control episodes: read a gold file -> applied an edit in a gold file -> resolved. Plus test-running habits.
Uses episodes.json (extract_episodes.py) and the gold patches of the benchmark tasks.jsonl (TASKS_JSONL)."""
import os
import collections, json, statistics as S

SP = os.environ.get("WORK_DIR", "work") + "/"
eps = json.load(open(SP + "episodes.json", encoding="utf-8"))
tasks = {}
for line in open(os.environ.get("TASKS_JSONL", "data/tasks.jsonl"), encoding="utf-8"):
    t = json.loads(line)
    tasks[t["instance_id"]] = t


def gold_files(patch):
    return {l[6:].strip() for l in patch.split(chr(10)) if l.startswith("--- a/")}


def clean(p):
    p = p or ""
    for pre in ("/workspace/", "./", "/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


rows = []
for e in eps:
    g = gold_files(tasks[e["iid"]]["patch"])
    r = e["result"]
    read_g = edit_g = False
    edits_ok = edits_err = 0
    ran_tests = ran_repro = 0
    grep_cmds = 0
    n_read = 0
    first_gold_read = None
    for i, c in enumerate(e["calls"]):
        a = c.get("args") or {}
        resp = c.get("resp") or {}
        if c["tool"] == "read_file":
            n_read += 1
            if clean(a.get("filepath")) in g:
                read_g = True
                if first_gold_read is None:
                    first_gold_read = i
        if c["tool"] == "edit_file":
            if resp.get("status") == "ok":
                edits_ok += 1
                if clean(a.get("filepath")) in g:
                    edit_g = True
            else:
                edits_err += 1
        if c["tool"] == "run_command":
            cmd = a.get("command", "")
            if "pytest" in cmd or "unittest" in cmd:
                ran_tests += 1
            if "grep" in cmd or "rg " in cmd or "find " in cmd or "ls " in cmd:
                grep_cmds += 1
            if "python" in cmd and "pytest" not in cmd:
                ran_repro += 1
    rows.append(dict(run=e["run"], iid=e["iid"], res=str(r["resolved"]) == "True", cls=r["outcome_class"], read_g=read_g, edit_g=edit_g,
                     edits_ok=edits_ok, edits_err=edits_err, ran_tests=ran_tests, ran_repro=ran_repro, grep_cmds=grep_cmds, n_read=n_read,
                     n_gold=len(g), first_gold_read=first_gold_read, calls=len(e["calls"])))

N = len(rows)
print("episodes", N)
print()
print("FUNNEL")
a = [r for r in rows if r["read_g"]]
b = [r for r in rows if r["edit_g"]]
c = [r for r in rows if r["res"]]
print("  read a gold file          %3d / %d (%.0f%%)" % (len(a), N, 100 * len(a) / N))
print("  applied an edit in a gold file %3d / %d (%.0f%%)" % (len(b), N, 100 * len(b) / N))
print("  resolved                  %3d / %d (%.0f%%)" % (len(c), N, 100 * len(c) / N))
print("  resolved | read gold: %d/%d = %.0f%%; resolved | edited gold: %d/%d = %.0f%%; resolved | did not edit gold: %d/%d" % (
    sum(r["res"] for r in a), len(a), 100 * sum(r["res"] for r in a) / max(len(a), 1),
    sum(r["res"] for r in b), len(b), 100 * sum(r["res"] for r in b) / max(len(b), 1),
    sum(r["res"] for r in rows if not r["edit_g"]), sum(1 for r in rows if not r["edit_g"])))
print("  read gold but never applied an edit in it: %d (of those with a refused edit: %d)" % (
    sum(1 for r in a if not r["edit_g"]), sum(1 for r in a if not r["edit_g"] and r["edits_err"] > 0)))
print("  never read any gold file: %d (empty patch among them: %d)" % (
    sum(1 for r in rows if not r["read_g"]), sum(1 for r in rows if not r["read_g"] and r["cls"].startswith("empty"))))

print()
print("by number of gold files in the real patch (resolved / edited-gold / n):")
by = collections.defaultdict(list)
for r in rows:
    by[min(r["n_gold"], 3)].append(r)
for k in sorted(by):
    xs = by[k]
    print("  %d%s gold file(s): n=%3d  edited gold %3d  resolved %3d" % (k, "+" if k == 3 else "", len(xs), sum(r["edit_g"] for r in xs), sum(r["res"] for r in xs)))

print()
print("test running (pytest/unittest command inside the episode):")
for name, f in (("ran tests", lambda r: r["ran_tests"] > 0), ("never ran tests", lambda r: r["ran_tests"] == 0)):
    xs = [r for r in rows if f(r)]
    print("  %-16s n=%3d  resolved %2d (%.0f%%)  edited gold %3d" % (name, len(xs), sum(r["res"] for r in xs), 100 * sum(r["res"] for r in xs) / max(len(xs), 1), sum(r["edit_g"] for r in xs)))
print("  ran a python script (repro) and no pytest:", sum(1 for r in rows if r["ran_repro"] > 0 and r["ran_tests"] == 0))

print()
print("edits per episode: applied median %.0f, refused median %.0f; episodes with >=1 refused edit: %d (%.0f%%)" % (
    S.median([r["edits_ok"] for r in rows]), S.median([r["edits_err"] for r in rows]), sum(1 for r in rows if r["edits_err"]), 100 * sum(1 for r in rows if r["edits_err"]) / N))
print("resolved rate by refusals in the episode: 0 refused: %d/%d; 1-2: %d/%d; 3+: %d/%d" % (
    sum(r["res"] for r in rows if r["edits_err"] == 0), sum(1 for r in rows if r["edits_err"] == 0),
    sum(r["res"] for r in rows if 1 <= r["edits_err"] <= 2), sum(1 for r in rows if 1 <= r["edits_err"] <= 2),
    sum(r["res"] for r in rows if r["edits_err"] >= 3), sum(1 for r in rows if r["edits_err"] >= 3)))
json.dump(rows, open(SP + "funnel_rows.json", "w"))
