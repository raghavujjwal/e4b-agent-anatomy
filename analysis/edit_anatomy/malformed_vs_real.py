"""Does the offline `malformed` rule (s4_gates.first_call / parse_args + mandatory keys) agree with what the real framework did? Control runs (ctl_v2rec1/2).
Analysis helper.   usage: python -I -X utf8 malformed_vs_real.py <folder with ctl_v2rec{1,2}_trace.jsonl>
For every edit_file generation with a known real status (ok / error / rejected): would the offline rule call it malformed?
"""
import collections
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import s4_gates as G  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
REQ = ("filepath", "old_string", "new_string")


def offline_malformed(text):
    if text.count("<|tool_call>") != 1:
        return "multi/none"
    c = G.first_call(text)
    if c is None:
        return "not closed"
    if c[1] is None:
        return "args do not parse"
    miss = [k for k in REQ if k not in c[1]]
    if miss:
        return "missing " + ",".join(miss)
    if not all(isinstance(c[1][k], str) for k in REQ):
        return "not a string"
    return None


tab = collections.Counter()
shown = collections.Counter()
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
                why = offline_malformed(text)
                tab[(st, "malformed" if why else "parsed")] += 1
                if (st == "rejected") != bool(why) and shown[(st, bool(why))] < 3:
                    shown[(st, bool(why))] += 1
                    i = text.find("<|tool_call>")
                    print(f"DISAGREE real={st} offline={why}: {text[i:i+420]!r}")
print("\nreal status x offline rule:")
for st in ("ok", "error", "rejected"):
    print(f"  real {st:<9} offline malformed {tab[(st, 'malformed')]:>4} | parsed {tab[(st, 'parsed')]:>4}")
r = tab[("rejected", "malformed")] + tab[("rejected", "parsed")]
print(f"recall of real 'rejected' by the offline rule: {tab[('rejected', 'malformed')]}/{r}")
fp = tab[("ok", "malformed")] + tab[("error", "malformed")]
print(f"offline 'malformed' that the real framework actually ran (ok or error): {fp}")
