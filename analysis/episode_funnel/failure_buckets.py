"""Why do the 118 unresolved control episodes fail? Reads the verifier fields of the two control runs."""
import os
import collections, json

res = []
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(os.environ.get("CTL_DIR", "data/ctl_files") + "/" + run + ".jsonl", encoding="utf-8"):
        r = json.loads(line)
        r["run"] = run
        res.append(r)
print("episodes", len(res), "| keys:", sorted(res[0].keys()))
print()
print("required_source (what the hidden tests need from the source) by outcome:")
tab = collections.defaultdict(collections.Counter)
for r in res:
    tab[str(r.get("required_source"))][r["outcome_class"]] += 1
for k, c in sorted(tab.items(), key=lambda kv: -sum(kv[1].values())):
    print("  %-22s n=%3d  %s" % (k, sum(c.values()), dict(c)))


def bucket(r):
    if r["outcome_class"] == "resolved":
        return "resolved"
    if r["outcome_class"].startswith("empty"):
        return "empty patch"
    t = ((r.get("verify_error") or "") + " " + (r.get("pytest_tail") or "")).lower()
    for key, name in (("importerror", "ImportError / cannot import name"), ("cannot import", "ImportError / cannot import name"),
                      ("modulenotfound", "ImportError / cannot import name"), ("attributeerror", "AttributeError"),
                      ("nameerror", "NameError"), ("typeerror", "TypeError"), ("syntaxerror", "SyntaxError / IndentationError"),
                      ("indentationerror", "SyntaxError / IndentationError"), ("assert", "AssertionError (wrong behaviour)"),
                      ("validationerror", "ValidationError"), ("timeout", "timeout"), ("collect", "collection error")):
        if key in t:
            return name
    return "other: " + (r.get("verify_error") or "")[:50]


cnt = collections.Counter(bucket(r) for r in res)
print()
print("failure buckets (verifier text):")
for k, v in cnt.most_common():
    print("  %3d  %s" % (v, k))

print()
print("verify_error heads of the first 25 unresolved non-empty episodes:")
n = 0
for r in res:
    if r["outcome_class"] == "unresolved_nonempty_patch" and n < 25:
        n += 1
        print("  %-16s %s | %s" % (r["instance_id"], (r.get("verify_error") or "")[:90], (r.get("pytest_tail") or "")[-70:].replace(chr(10), " ")))

print()
pb = [int(r["patch_bytes"] or 0) for r in res if r["outcome_class"] == "unresolved_nonempty_patch"]
pr = [int(r["patch_bytes"] or 0) for r in res if r["outcome_class"] == "resolved"]
import statistics as S
print("patch bytes: unresolved-nonempty median %.0f (n=%d); resolved median %.0f (n=%d)" % (S.median(pb), len(pb), S.median(pr), len(pr)))
print("tool calls: resolved median %.0f; unresolved median %.0f; empty median %.0f" % (
    S.median([int(r["tool_calls"]) for r in res if r["outcome_class"] == "resolved"]),
    S.median([int(r["tool_calls"]) for r in res if r["outcome_class"] == "unresolved_nonempty_patch"]),
    S.median([int(r["tool_calls"]) for r in res if r["outcome_class"].startswith("empty")])))
