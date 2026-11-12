"""Do rejected (malformed) edit calls come after earlier failed edits? Control runs only (ctl_v2rec1/2). Analysis helper.
usage: python -I -X utf8 rejected_vs_prior_failures.py <folder with ctl_v2rec{1,2}_trace.jsonl>
For every edit_file generation: its status (ok / error / rejected) and how many edit_file responses with status error are visible in its prompt.
"""
import collections
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
SRC = sys.argv[1]
rows = []
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(f"{SRC}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        for turns in B.chains_of(tr):
            for k, (prompt, gen, finish, _n, suffix) in enumerate(turns):
                if B.tool_of(tok.decode(gen, skip_special_tokens=False)) != "edit_file":
                    continue
                st = B.edit_status(tok, turns, k, "edit_file")
                if st not in ("ok", "error", "rejected"):
                    continue
                n_err = n_rej = 0
                for m in B.RESP_BLOCK.finditer(tok.decode(prompt, skip_special_tokens=False)):
                    if m.group(1) != "edit_file":
                        continue
                    s = B.STATUS_RE.match(m.group(2))
                    if s and s.group(1) == "error":
                        n_err += 1
                    elif m.group(2).startswith(B.REJECTED_PREFIX):
                        n_rej += 1
                rows.append({"status": st, "n_err": n_err, "n_rej": n_rej, "iid": tr["instance_id"]})
print(f"edit_file generations with a known status: {len(rows)} ({collections.Counter(r['status'] for r in rows)})")
print("\n== by the number of earlier FAILED (error) edit responses visible in the prompt")
print(f"{'earlier failed':>14} {'n':>5} {'applied':>9} {'error':>9} {'rejected':>9}")
for lo, hi, name in ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 99, "3+")):
    sub = [r for r in rows if lo <= r["n_err"] <= hi]
    if sub:
        c = collections.Counter(r["status"] for r in sub)
        print(f"{name:>14} {len(sub):>5} {100*c['ok']/len(sub):>8.1f}% {100*c['error']/len(sub):>8.1f}% {100*c['rejected']/len(sub):>8.1f}%")
print("\n== by earlier REJECTED edit responses visible")
for name, pred in (("none", lambda r: r["n_rej"] == 0), ("1+", lambda r: r["n_rej"] >= 1)):
    sub = [r for r in rows if pred(r)]
    c = collections.Counter(r["status"] for r in sub)
    print(f"{name:>14} {len(sub):>5} applied {100*c['ok']/len(sub):.1f}%  error {100*c['error']/len(sub):.1f}%  rejected {100*c['rejected']/len(sub):.1f}%")
rej = [r for r in rows if r["status"] == "rejected"]
print(f"\nof the {len(rej)} rejected calls: preceded by >=1 failed edit {sum(r['n_err'] >= 1 for r in rej)} ({100*sum(r['n_err'] >= 1 for r in rej)/len(rej):.0f}%), by >=1 rejected {sum(r['n_rej'] >= 1 for r in rej)}, by neither {sum(r['n_err'] == 0 and r['n_rej'] == 0 for r in rej)}; tasks {len({r['iid'] for r in rej})}")
