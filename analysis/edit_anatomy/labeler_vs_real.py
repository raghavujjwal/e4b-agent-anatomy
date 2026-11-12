"""edit_pairs.parse_generation against what the real framework did, on the corrected control runs. Analysis helper.
usage: python -I -X utf8 labeler_vs_real.py <folder with ctl_v2rec{1,2}_trace.jsonl>
"""
import collections
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import edit_pairs as E  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
tab = collections.Counter()
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
                i = text.find("<|tool_call>")
                tab[(E.parse_generation(text[i:])["status"], st)] += 1
print(f"{'parse_generation':<18} {'real ok':>8} {'real error':>11} {'real rejected':>14}")
for p in ("call", "malformed", "unparsed", "multi_call", "wrong_tool", "no_call"):
    if any(tab[(p, s)] for s in ("ok", "error", "rejected")):
        print(f"{p:<18} {tab[(p, 'ok')]:>8} {tab[(p, 'error')]:>11} {tab[(p, 'rejected')]:>14}")
m_all = sum(tab[("malformed", s)] for s in ("ok", "error", "rejected"))
print(f"\nprecision of `malformed` as 'the framework rejects it': {tab[('malformed', 'rejected')]}/{m_all}")
print(f"real rejections that are `malformed`: {tab[('malformed', 'rejected')]}/{sum(tab[(p, 'rejected')] for p in ('call','malformed','unparsed','multi_call','wrong_tool','no_call'))}")
