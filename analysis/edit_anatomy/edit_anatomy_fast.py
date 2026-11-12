"""Second pass over edits_all.json (extract_edits.py): ambiguity, anchor size, recovery after an ambiguity error, and what the content mismatches look like."""
import os
import collections, difflib, json, statistics as S

BS, NL, Q = chr(92), chr(10), chr(34)
rows = json.load(open(os.environ.get("WORK_DIR", "work") + "/edits_all.json", encoding="utf-8"))
print("edit calls:", len(rows), dict(collections.Counter(r["kind"] for r in rows)))
parsed = [r for r in rows if "old" in r]


def nlines(s):
    return len([l for l in s.split(NL) if l.strip()])


print()
print("anchor size by outcome (lines with text in old_string; chars)")
for kind in ("ok", "multiple", "not_found"):
    xs = [r for r in parsed if r["kind"] == kind]
    if not xs:
        continue
    L = [nlines(r["old"]) for r in xs]
    C = [len(r["old"]) for r in xs]
    one = sum(1 for l in L if l == 1) / len(L)
    ge3 = sum(1 for l in L if l >= 3) / len(L)
    print("  %-10s n=%3d  lines median %.0f mean %.1f | one-line %2.0f%% | 3+ lines %2.0f%% | chars median %.0f" % (kind, len(xs), S.median(L), S.mean(L), 100 * one, 100 * ge3, S.median(C)))

print()
print("apply rate by anchor size (all parsed ok/error edits)")
for lo, hi, name in ((1, 1, "1 line"), (2, 2, "2 lines"), (3, 4, "3-4 lines"), (5, 9, "5-9 lines"), (10, 999, "10+ lines")):
    xs = [r for r in parsed if r["status"] in ("ok", "error") and lo <= nlines(r["old"]) <= hi]
    if xs:
        ok = sum(r["kind"] == "ok" for r in xs); amb = sum(r["kind"] == "multiple" for r in xs); nf = sum(r["kind"] == "not_found" for r in xs)
        print("  %-10s n=%3d  applied %2.0f%%  ambiguous %2.0f%%  not found %2.0f%%" % (name, len(xs), 100 * ok / len(xs), 100 * amb / len(xs), 100 * nf / len(xs)))

amb = [r for r in parsed if r["kind"] == "multiple"]
vis = [r["corp"].count(r["old"]) for r in amb if r.get("corp")]
print()
print("ambiguity errors: %d. The anchor occurs in the VISIBLE text 0 / 1 / 2+ times: %d / %d / %d (so the model could have seen the repetition only in the 2+ cases)" %
      (len(amb), sum(v == 0 for v in vis), sum(v == 1 for v in vis), sum(v >= 2 for v in vis)))
files = collections.Counter(r["filepath"] for r in amb)
print("   top files:", files.most_common(5), "| share in the top 3 files: %.0f%%" % (100 * sum(v for _, v in files.most_common(3)) / max(len(amb), 1)))

# recovery after an ambiguity error: the next edit on the same file in the same chain
by_chain = collections.defaultdict(list)
for r in rows:
    by_chain[(r["run"], r["iid"], r["chain"])].append(r)
rec = collections.Counter()
for key, seq in by_chain.items():
    seq.sort(key=lambda r: r["turn"])
    for i, r in enumerate(seq):
        if r["kind"] != "multiple":
            continue
        nxt = next((s for s in seq[i + 1:] if s.get("filepath") == r.get("filepath")), None)
        if nxt is None:
            rec["no later edit on that file"] += 1
        elif "old" not in nxt:
            rec["next edit unreadable"] += 1
        else:
            rel = "same text" if nxt["old"] == r["old"] else ("longer, contains the old anchor" if r["old"] in nxt["old"] else "different text")
            rec["%s -> %s" % (rel, nxt["kind"])] += 1
print()
print("what the model does after an ambiguity error (next edit on that file in the same run of the loop):")
for k, v in rec.most_common():
    print("  %3d  %s" % (v, k))

