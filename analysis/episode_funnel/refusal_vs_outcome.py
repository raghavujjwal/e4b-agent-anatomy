"""Is a refused edit associated with failure within the same kind of task? Stratify by number of gold files and by the first attempt in the gold file.
Association only: harder tasks may cause both refusals and failures."""
import os
import collections, json

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


rows = []
for e in eps:
    g = gold_files(tasks[e["iid"]]["patch"])
    edits = [c for c in e["calls"] if c["tool"] == "edit_file" and (c.get("resp") or {}).get("status") in ("ok", "error")]
    gold_edits = [c for c in edits if clean((c.get("args") or {}).get("filepath")) in g]
    refused_any = sum(1 for c in edits if c["resp"]["status"] == "error")
    refused_gold = sum(1 for c in gold_edits if c["resp"]["status"] == "error")
    applied_gold = sum(1 for c in gold_edits if c["resp"]["status"] == "ok")
    first = gold_edits[0]["resp"]["status"] if gold_edits else None
    rows.append(dict(iid=e["iid"], n_gold=len(g), res=str(e["result"]["resolved"]) == "True", refused_any=refused_any, refused_gold=refused_gold,
                     applied_gold=applied_gold, first=first, tried_gold=bool(gold_edits)))


def rate(xs):
    return "%d/%d (%.0f%%)" % (sum(r["res"] for r in xs), len(xs), 100 * sum(r["res"] for r in xs) / max(len(xs), 1))


single = [r for r in rows if r["n_gold"] == 1]
print("single-gold-file tasks:", len(single), "| resolved", rate(single))
print("  by refused edits anywhere in the episode: 0:", rate([r for r in single if r["refused_any"] == 0]), "| 1+:", rate([r for r in single if r["refused_any"] >= 1]))
print("  tried an edit in the gold file:", rate([r for r in single if r["tried_gold"]]), "| never tried:", rate([r for r in single if not r["tried_gold"]]))
tg = [r for r in single if r["tried_gold"]]
print("  of those that tried: first gold edit applied:", rate([r for r in tg if r["first"] == "ok"]), "| first gold edit refused:", rate([r for r in tg if r["first"] == "error"]))
print("  of those that tried: no refusal in the gold file:", rate([r for r in tg if r["refused_gold"] == 0]), "| 1+ refusal in the gold file:", rate([r for r in tg if r["refused_gold"] >= 1]))
print("  of those that tried and ended with an applied gold edit: no refusal:", rate([r for r in tg if r["applied_gold"] and r["refused_gold"] == 0]),
      "| refused then applied:", rate([r for r in tg if r["applied_gold"] and r["refused_gold"] >= 1]), "| refused and never applied:", rate([r for r in tg if not r["applied_gold"]]))
print()
print("all tasks, tried an edit in a gold file: n=%d" % sum(r["tried_gold"] for r in rows))
tg = [r for r in rows if r["tried_gold"]]
print("  first gold edit applied:", rate([r for r in tg if r["first"] == "ok"]), "| refused:", rate([r for r in tg if r["first"] == "error"]))
print("  refused-then-applied:", rate([r for r in tg if r["applied_gold"] and r["refused_gold"] >= 1]), "| refused, never applied:", rate([r for r in tg if not r["applied_gold"]]))
print()
print("single-file tasks that tried the gold file but never applied an edit there: %d episodes" % sum(1 for r in single if r["tried_gold"] and not r["applied_gold"]))
print("  the same across all tasks: %d episodes; with 3+ refused gold edits: %d" % (
    sum(1 for r in rows if r["tried_gold"] and not r["applied_gold"]), sum(1 for r in rows if r["tried_gold"] and not r["applied_gold"] and r["refused_gold"] >= 3)))
