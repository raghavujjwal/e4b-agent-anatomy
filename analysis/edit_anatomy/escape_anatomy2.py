"""Anatomy of refused edit_file calls in the corrected-harness control runs: how many are plain escape-copy errors?
For every edit the matcher refused (status error), the file text the model had seen is rebuilt from the read_file responses in that turn's prompt
(JSON-decoded), then the model's old_string is tested against it. First match wins.
No backslash is typed in this file (BS = chr(92)): an escape layer between the author and the disk halved doubled backslashes once already."""
import os
import collections, json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B
import s4_gates as G
import served

tok = served.load_tokenizer()
BS, NL, TAB, CR, Q = chr(92), chr(10), chr(9), chr(13), chr(34)
UN = {"n": NL, "t": TAB, "r": CR, Q: Q, BS: BS, "/": "/"}
VALUE_OPEN, VALUE_CLOSE = 'value:<|' + Q + '|>', '<|' + Q + '|>'


def unescape(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == BS and i + 1 < len(s) and s[i + 1] in UN:
            out.append(UN[s[i + 1]])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def norm(s):
    return " ".join(s.split())


def clean(p):
    p = p or ""
    for pre in ("/workspace/", "./", "/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def body_json(body):
    if body.startswith(VALUE_OPEN):
        body = body[len(VALUE_OPEN):]
    if body.endswith(VALUE_CLOSE):
        body = body[:-len(VALUE_CLOSE)]
    return json.loads(body)


def corpus_of(prompt_text, filepath):
    parts = []
    for m in B.RESP_BLOCK.finditer(prompt_text):
        if m.group(1) != "read_file":
            continue
        try:
            d = body_json(m.group(2))
        except Exception:
            continue
        if d.get("status") == "ok" and clean(d.get("filepath")) == clean(filepath):
            parts.append(d.get("content", ""))
    return NL.join(parts)


def err_kind(turns, k):
    blocks = list(B.RESP_BLOCK.finditer(tok.decode(turns[k + 1][4], skip_special_tokens=False)))
    if not blocks:
        return "?"
    try:
        msg = body_json(blocks[-1].group(2)).get("error_message", "")
    except Exception:
        return "?"
    return "not_found" if "not found" in msg else ("multiple" if ("times" in msg or "multiple" in msg or "several" in msg) else "other")


rows, skipped = [], collections.Counter()
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(os.environ.get("CTL_DIR", "data/ctl_files") + "/" + run + "_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        for turns in B.chains_of(tr):
            for k in range(len(turns) - 1):
                gen = tok.decode(turns[k][1], skip_special_tokens=False)
                if B.tool_of(gen) != "edit_file" or gen.count("<|tool_call>") != 1:
                    continue
                if B.edit_status(tok, turns, k, "edit_file") != "error":
                    continue
                fc = G.first_call(gen)
                if not fc or not fc[1] or "old_string" not in fc[1]:
                    skipped["unparsed"] += 1
                    continue
                old, fp = fc[1]["old_string"], fc[1].get("filepath", "")
                kind = err_kind(turns, k)
                corp = corpus_of(tok.decode(turns[k][0], skip_special_tokens=False), fp)
                rows.append((old, corp, kind, run, tr.get("instance_id")))
print("refused edits with a parsed old_string:", len(rows), "| skipped:", dict(skipped), flush=True)

cls = collections.Counter()
ex = collections.defaultdict(list)
esc = collections.Counter()
has = collections.Counter()
for old, corp, kind, run, iid in rows:
    has_nl, has_q, has_bs = (BS + "n") in old, (BS + Q) in old, (BS + BS) in old
    uo = unescape(old)
    if kind == "multiple":
        c = "ambiguous (matches several places)"
    elif not corp:
        c = "file text not in prompt"
    elif old in corp:
        c = "exact text present (other cause)"
    elif uo in corp:
        c = "A. plain un-escape makes it exact"
        esc["newline"] += has_nl
        esc["quote"] += has_q
        esc["backslash"] += has_bs
    elif norm(uo) in norm(corp):
        c = "B. un-escape plus whitespace or indent fix"
    elif norm(old) in norm(corp):
        c = "C. whitespace or indent only"
    else:
        ol = [l.strip() for l in uo.split(NL) if l.strip()]
        cl = {l.strip() for l in corp.split(NL)}
        frac = sum(l in cl for l in ol) / max(len(ol), 1)
        if frac >= 0.8:
            c = "D. most lines exist (>=80%), text differs"
        elif frac >= 0.3:
            c = "E. some lines exist (30-80%)"
        else:
            c = "F. little or nothing exists (<30%)"
    cls[c] += 1
    ex[c].append(old)
    has["backslash-n"] += has_nl
    has["backslash-quote"] += has_q
    has["any backslash"] += (BS in old)
n = len(rows)
print()
for c, v in sorted(cls.items()):
    print("%4d  %5.1f%%  %s" % (v, 100 * v / n, c))
print()
print("escape kinds inside class A:", dict(esc))
print("share of ALL refused edits whose old_string contains a literal backslash-n: %.0f%%, a backslash-quote: %.0f%%, any backslash: %.0f%%" %
      (100 * has["backslash-n"] / n, 100 * has["backslash-quote"] / n, 100 * has["any backslash"] / n))
for c in ("A. plain un-escape makes it exact", "B. un-escape plus whitespace or indent fix", "C. whitespace or indent only"):
    print()
    print("examples of", c)
    for o in ex[c][:3]:
        print("   ", repr(o[:160]))
