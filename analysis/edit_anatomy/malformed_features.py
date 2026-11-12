"""Which parse failures does the REAL framework reject, and which does it still run? Control runs. Analysis helper.
usage: python -I -X utf8 malformed_features.py <folder with ctl_v2rec{1,2}_trace.jsonl>
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
REQ = ("filepath", "old_string", "new_string")
Q = '<|"|>'


def offline_fail(text):
    if text.count("<|tool_call>") != 1:
        return "multi/none"
    c = G.first_call(text)
    if c is None:
        return "not closed"
    if c[1] is None:
        return "args do not parse"
    if [k for k in REQ if k not in c[1]]:
        return "missing"
    return None


def features(body):
    f = []
    if re.search(r"\b(old_string|new_string|filepath|content)=", body):
        f.append("kwarg=")
    if '"""' in body:
        f.append('triple"')
    if re.search(r'\b(old_string|new_string|filepath|content):"', body):
        f.append('key:"')
    if re.search(r'\b(old_string|new_string|filepath|content): +["\']', body):
        f.append('key: "')
    if re.search(r"\b(old_string|new_string|content):\s*\n", body):
        f.append("key:<newline>")
    if re.search(r"\b(old_string|new_string|content):(None|null|true|false|\d)", body):
        f.append("key:bare-literal")
    if re.search(r"\b(old_string|new_string|content):'", body):
        f.append("key:'")
    n_open = body.count(Q)
    if n_open % 2:
        f.append("odd-<|\"|>")
    return f or ["(none of the above)"]


rows = []
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(f"{sys.argv[1]}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        for turns in B.chains_of(tr):
            for k, (prompt, gen, finish, _n, suffix) in enumerate(turns):
                text = tok.decode(gen, skip_special_tokens=False)
                if B.tool_of(text) != "edit_file":
                    continue
                st = B.edit_status(tok, turns, k, "edit_file")
                if st not in ("ok", "error", "rejected"):
                    continue
                why = offline_fail(text)
                if not why:
                    continue
                i = text.find("<|tool_call>call:edit_file{")
                body = text[i + len("<|tool_call>call:edit_file{"):] if i >= 0 else text
                rows.append((st, why, features(body), body[:150].replace("\n", "\n")))
by = collections.defaultdict(collections.Counter)
for st, why, f, head in rows:
    by[tuple(f)][st] += 1
print(f"{len(rows)} generations the offline rule calls malformed; real status per feature set:")
for f, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
    print(f"  {' + '.join(f):<46} rejected {c['rejected']:>3} | ran: ok {c['ok']:>2} error {c['error']:>2}")
print("\nthe ones the real framework RAN:")
for st, why, f, head in rows:
    if st != "rejected":
        print(f"  real={st:<6} {why:<18} {'+'.join(f):<30} {head[:140]}")
