"""Does the task statement even point at the gold file? Relates 'statement names the gold file / a gold identifier' to 'the model edited a gold file' and to 'resolved'."""
import os
import collections, json, posixpath, re

SP = os.environ.get("WORK_DIR", "work") + "/"
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.environ.get("TASKS_JSONL", "data/tasks.jsonl"), encoding="utf-8")}
rows = json.load(open(SP + "funnel_rows.json"))
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{4,}")


def distinctive(tok):
    return "_" in tok.strip("_") or any(a.islower() and b.isupper() for a, b in zip(tok, tok[1:]))


def gold_files(patch):
    return [l[6:].strip() for l in patch.split(chr(10)) if l.startswith("--- a/")]


def gold_idents(patch):
    out = set()
    for l in patch.split(chr(10)):
        if l.startswith("@@"):
            tail = l.split("@@")[-1]
            out.update(t for t in IDENT.findall(tail) if t not in ("def", "class"))
        elif (l.startswith("+") or l.startswith("-")) and not l.startswith("+++") and not l.startswith("---"):
            out.update(t for t in IDENT.findall(l) if distinctive(t))
    return out


per_task = {}
for iid, t in tasks.items():
    st = (t.get("problem_statement") or "").lower()
    gf = [g for g in gold_files(t["patch"]) if not g.startswith("tests/")]
    mention_file = any(g.lower() in st or posixpath.basename(g).lower() in st or posixpath.splitext(posixpath.basename(g))[0].lower() in st for g in gf)
    ids = gold_idents(t["patch"])
    mention_id = [i for i in ids if i.lower() in st]
    per_task[iid] = dict(mention_file=mention_file, n_ident_mentioned=len(mention_id), n_gold_files=len(gf), stmt_len=len(st), kinds={posixpath.splitext(g)[1] for g in gf},
                         docs_src=any(g.startswith("docs_src/") for g in gf))

tab = collections.Counter()
for r in rows:
    p = per_task[r["iid"]]
    tab[(p["mention_file"] or p["n_ident_mentioned"] > 0, r["edit_g"], r["res"])] += 1
print("episodes (132): statement names the gold file or a distinctive gold identifier?  x  edited a gold file?  x  resolved?")
for k, v in sorted(tab.items(), key=lambda kv: (not kv[0][0], not kv[0][1], not kv[0][2])):
    print("   points-at-gold=%-5s edited-gold=%-5s resolved=%-5s  %3d" % (k + (v,)))
print()
for name, f in (("statement names the gold file", lambda p: p["mention_file"]), ("statement names a distinctive gold identifier (not the file)", lambda p: (not p["mention_file"]) and p["n_ident_mentioned"] > 0),
                ("statement points at neither", lambda p: (not p["mention_file"]) and p["n_ident_mentioned"] == 0)):
    xs = [r for r in rows if f(per_task[r["iid"]])]
    print("%-62s n=%3d  edited gold %3d (%.0f%%)  resolved %2d (%.0f%%)" % (name, len(xs), sum(r["edit_g"] for r in xs), 100 * sum(r["edit_g"] for r in xs) / max(len(xs), 1), sum(r["res"] for r in xs), 100 * sum(r["res"] for r in xs) / max(len(xs), 1)))
print()
xs = [r for r in rows if per_task[r["iid"]]["docs_src"]]
print("tasks whose gold patch touches docs_src/ (tutorial example code): %d episodes, edited gold %d, resolved %d" % (len(xs), sum(r["edit_g"] for r in xs), sum(r["res"] for r in xs)))
xs = [r for r in rows if not per_task[r["iid"]]["docs_src"]]
print("the rest: %d episodes, edited gold %d, resolved %d" % (len(xs), sum(r["edit_g"] for r in xs), sum(r["res"] for r in xs)))
print()
print("statement length (chars) median:", sorted(p["stmt_len"] for p in per_task.values())[len(per_task) // 2])
