"""How does the model search before its first edit, in episodes that edited a gold file vs those that never did?"""
import os
import collections, json, posixpath, re

SP = os.environ.get("WORK_DIR", "work") + "/"
eps = json.load(open(SP + "episodes.json", encoding="utf-8"))
rows = {(r["run"], r["iid"]): r for r in json.load(open(SP + "funnel_rows.json"))}
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(os.environ.get("TASKS_JSONL", "data/tasks.jsonl"), encoding="utf-8")}
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


def clean(p):
    p = p or ""
    for pre in ("/workspace/", "./", "/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def gold_files(patch):
    return [l[6:].strip() for l in patch.split(chr(10)) if l.startswith("--- a/")]


def brief(c):
    a = c.get("args") or {}
    t = c["tool"]
    st = (c.get("resp") or {}).get("status")
    if t == "run_command":
        return "RUN  " + a.get("command", "").replace(chr(10), " ; ")[:110]
    if t == "read_file":
        rng = "" if a.get("start_line") is None else " [%s-%s]" % (a.get("start_line"), a.get("end_line"))
        return "READ " + str(a.get("filepath")) + rng
    if t == "edit_file":
        return "EDIT %s (%s)" % (a.get("filepath"), st)
    return t.upper()


# per group: number of calls before the first edit, number of search commands, grep usage
SEARCH = ("grep", "rg ", "find ", "ls ", "tree", "cat ", "head ", "sed -n", "git ")
stat = collections.defaultdict(list)
for e in eps:
    key = (e["run"], e["iid"])
    r = rows[key]
    grp = "edited gold" if r["edit_g"] else "never edited gold"
    calls = e["calls"]
    first_edit = next((i for i, c in enumerate(calls) if c["tool"] == "edit_file"), len(calls))
    pre = calls[:first_edit]
    n_search = sum(1 for c in pre if c["tool"] == "run_command" and any(s in (c.get("args") or {}).get("command", "") for s in SEARCH))
    n_grep = sum(1 for c in pre if c["tool"] == "run_command" and ("grep" in (c.get("args") or {}).get("command", "") or "rg " in (c.get("args") or {}).get("command", "")))
    n_read = sum(1 for c in pre if c["tool"] == "read_file")
    stat[grp].append((first_edit, n_search, n_grep, n_read, len(calls)))
print("calls before the first edit (median) / search-type commands / grep commands / reads / total calls")
for g, xs in stat.items():
    med = lambda i: sorted(x[i] for x in xs)[len(xs) // 2]
    print("  %-18s n=%3d  before-first-edit %2d | search cmds %d | grep cmds %d | reads %d | total %d | no grep at all before first edit: %d%%" % (
        g, len(xs), med(0), med(1), med(2), med(3), med(4), round(100 * sum(1 for x in xs if x[2] == 0) / len(xs))))

# the 32 episodes where the statement names the target but the model never edited a gold file: show first calls
shown = 0
print()
print("EXAMPLES: statement names a gold file or distinctive identifier, model never edited a gold file")
for e in eps:
    key = (e["run"], e["iid"])
    r = rows[key]
    if r["edit_g"]:
        continue
    t = tasks[e["iid"]]
    st = (t.get("problem_statement") or "")
    g = [x for x in gold_files(t["patch"]) if not x.startswith("tests/")]
    if not any(posixpath.basename(x).lower().replace(".py", "") in st.lower() for x in g):
        continue
    shown += 1
    if shown > 5:
        break
    print("--- %s | gold %s | statement: %s" % (e["iid"], g[:2], st[:200].replace(chr(10), " ")))
    for c in e["calls"][:9]:
        print("     ", brief(c))
