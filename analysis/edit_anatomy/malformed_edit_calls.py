"""The raw generations of the edit_file calls that ADK rejected for a missing mandatory parameter. Analysis helper.
usage: python -I -X utf8 malformed_edit_calls.py <folder with ctl_v2rec{1,2}_trace.jsonl>
"""
import collections
import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import s4_gates as G  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
SRC = sys.argv[1]
MISSING = re.compile(r"mandatory input parameters are not present:\\n((?:\w+\\n)+)")
rows = []
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(f"{SRC}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        for ci, chain in enumerate(B.chains_of(tr)):
            for k, (prompt, gen, finish, _n, suffix) in enumerate(chain):
                text = tok.decode(gen, skip_special_tokens=False)
                tool = B.tool_of(text)
                if tool != "edit_file" or k + 1 >= len(chain):
                    continue
                nsuf = tok.decode(chain[k + 1][4], skip_special_tokens=False)
                if "mandatory input parameters are not present" not in nsuf:
                    continue
                m = MISSING.search(nsuf)
                missing = tuple(m.group(1).split("\\n")[:-1]) if m else None
                ok, why = G.well_formed(text)
                call_start = text.find("<|tool_call>")
                rows.append({"run": run, "iid": tr["instance_id"], "k": k, "len_gen": len(gen), "finish": finish, "missing": missing, "wf": ok, "why": why,
                             "thought_chars": call_start, "text": text, "ngen": len(gen)})

print("malformed edit_file calls:", len(rows))
print("missing parameters :", dict(collections.Counter(r["missing"] for r in rows)))
print("our well_formed says:", dict(collections.Counter((r["wf"], r["why"]) for r in rows)))
print("finish             :", dict(collections.Counter(r["finish"] for r in rows)))
print("generation tokens  : median", sorted(r["len_gen"] for r in rows)[len(rows) // 2], "max", max(r["len_gen"] for r in rows))
print("tasks              :", len({r["iid"] for r in rows}), dict(collections.Counter(r["iid"] for r in rows).most_common(8)))
print()
for r in rows[:5] + rows[-2:]:
    t = r["text"]
    i = t.find("<|tool_call>")
    print(f"--- {r['run']} {r['iid']} turn {r['k']} | {r['len_gen']} tokens | finish {r['finish']} | missing {r['missing']} | well_formed={r['wf']} {r['why']!r}")
    print("call:", repr(t[i:i + 900]))
    print("tail:", repr(t[-160:]))
    print()
