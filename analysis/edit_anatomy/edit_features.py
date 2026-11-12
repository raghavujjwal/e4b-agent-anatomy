"""Which old_strings apply? Control runs only (ctl_v2rec1/2), edit_file calls with a known status. Analysis helper.
usage: python -I -X utf8 edit_features.py <folder with ctl_v2rec{1,2}_trace.jsonl>
"""
import collections
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import s4_gates as G  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
SRC = sys.argv[1]
RUNS = ("ctl_v2rec1", "ctl_v2rec2")

rows = []
for run in RUNS:
    for line in open(f"{SRC}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        for turns in B.chains_of(tr):
            for k, (prompt, gen, finish, _n, suffix) in enumerate(turns):
                text = tok.decode(gen, skip_special_tokens=False)
                if B.tool_of(text) != "edit_file":
                    continue
                c = G.first_call(text)
                if not c or not c[1] or not isinstance(c[1].get("old_string"), str):
                    continue
                s = B.edit_status(tok, turns, k, "edit_file")
                if s not in ("ok", "error"):
                    continue
                old = c[1]["old_string"]
                new = c[1].get("new_string") if isinstance(c[1].get("new_string"), str) else ""
                rows.append({"ok": s == "ok", "old": old, "new": new})

print(f"edit_file calls with a known status and a string old_string: {len(rows)}; applied {sum(r['ok'] for r in rows)} ({100 * sum(r['ok'] for r in rows) / len(rows):.1f}%)")


def rate(name, pred):
    sub = [r for r in rows if pred(r["old"])]
    if sub:
        print(f"  {name:<44} n={len(sub):>4} ({100 * len(sub) / len(rows):>3.0f}% of edits)  applied {100 * sum(r['ok'] for r in sub) / len(sub):>5.1f}%")
    return sub


print("\n== apply rate by what old_string contains")
rate("contains a backslash", lambda o: "\\" in o)
rate("no backslash", lambda o: "\\" not in o)
rate("contains a double quote", lambda o: '"' in o)
rate("no double quote", lambda o: '"' not in o)
rate("no backslash and no double quote", lambda o: "\\" not in o and '"' not in o)
rate("backslash or double quote", lambda o: "\\" in o or '"' in o)
print()
rate("single line", lambda o: "\n" not in o.strip("\n"))
rate("2-3 lines", lambda o: 1 <= o.strip("\n").count("\n") <= 2)
rate("4+ lines", lambda o: o.strip("\n").count("\n") >= 3)
print()
rate("shorter than 40 chars", lambda o: len(o) < 40)
rate("40-100 chars", lambda o: 40 <= len(o) <= 100)
rate("over 100 chars", lambda o: len(o) > 100)
print()
rate("starts with whitespace", lambda o: o[:1] in (" ", "\t"))
rate("starts with a non-space", lambda o: o[:1] not in (" ", "\t", ""))
rate("ends with newline", lambda o: o.endswith("\n"))
rate("contains a tab", lambda o: "\t" in o)
print()
clean = rate("CLEAN: single line, no backslash, no double quote", lambda o: "\n" not in o.strip("\n") and "\\" not in o and '"' not in o)
dirty = [r for r in rows if r not in clean]
print(f"  {'everything else':<44} n={len(dirty):>4} ({100 * len(dirty) / len(rows):>3.0f}% of edits)  applied {100 * sum(r['ok'] for r in dirty) / len(dirty):>5.1f}%")
rate("no backslash, no dq, any number of lines", lambda o: "\\" not in o and '"' not in o)
print("\n== of the edits that FAILED: what they contain")
fail = [r for r in rows if not r["ok"]]
print(f"  failed {len(fail)}: backslash {sum('\\' in r['old'] for r in fail)} | double quote {sum('\"' in r['old'] for r in fail)} | either {sum(('\\' in r['old'] or '\"' in r['old']) for r in fail)} | neither {sum(('\\' not in r['old'] and '\"' not in r['old']) for r in fail)}")
print(f"  of the failed edits, with neither a backslash nor a double quote: single line {sum(('\\' not in r['old'] and '\"' not in r['old'] and chr(10) not in r['old'].strip(chr(10))) for r in fail)}, multi-line {sum(('\\' not in r['old'] and '\"' not in r['old'] and chr(10) in r['old'].strip(chr(10))) for r in fail)}")
print("\n== of the APPLIED edits: how many contain a double quote or backslash in old_string?")
okr = [r for r in rows if r["ok"]]
print(f"  applied {len(okr)}: double quote {sum('\"' in r['old'] for r in okr)} ({100 * sum('\"' in r['old'] for r in okr) / len(okr):.0f}%), backslash {sum('\\' in r['old'] for r in okr)} ({100 * sum('\\' in r['old'] for r in okr) / len(okr):.0f}%)")
