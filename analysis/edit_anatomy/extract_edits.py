"""Dump every edit_file call of the corrected-harness control runs (ok / error / rejected) with the file text the model had seen, to one JSON file,
so later analyses do not pay for the tokenizer again. No backslash is typed here (BS = chr(92))."""
import os
import collections, json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B
import s4_gates as G
import served

tok = served.load_tokenizer()
BS, NL, Q = chr(92), chr(10), chr(34)
VALUE_OPEN, VALUE_CLOSE = 'value:<|' + Q + '|>', '<|' + Q + '|>'


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
    if "not found" in msg:
        return "not_found"
    if "occurs" in msg and "times" in msg:
        return "multiple"
    if "identical" in msg:
        return "unchanged"
    return "other"


out = []
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(os.environ.get("CTL_DIR", "data/ctl_files") + "/" + run + "_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        for ci, turns in enumerate(B.chains_of(tr)):
            n_prev_err = 0
            for k in range(len(turns) - 1):
                gen = tok.decode(turns[k][1], skip_special_tokens=False)
                if B.tool_of(gen) != "edit_file" or gen.count("<|tool_call>") != 1:
                    continue
                st = B.edit_status(tok, turns, k, "edit_file")
                if st not in ("ok", "error", "rejected"):
                    continue
                fc = G.first_call(gen)
                rec = {"run": run, "iid": tr.get("instance_id"), "chain": ci, "turn": k, "status": st, "gen": gen[:5000]}
                if fc and fc[1] and "old_string" in fc[1]:
                    a = fc[1]
                    rec.update(old=a["old_string"], new=a.get("new_string"), filepath=a.get("filepath", ""), allow_multiple=a.get("allow_multiple"))
                    if st in ("ok", "error"):
                        rec["corp"] = corpus_of(tok.decode(turns[k][0], skip_special_tokens=False), rec["filepath"])[:80000]
                    rec["kind"] = err_kind(turns, k) if st == "error" else st
                else:
                    rec["kind"] = "unparsed"
                rec["prev_errors_in_chain"] = n_prev_err
                if st != "ok":
                    n_prev_err += 1
                out.append(rec)
json.dump(out, open(os.environ.get("WORK_DIR", "work") + "/edits_all.json", "w", encoding="utf-8"))
print("wrote", len(out), "edit calls;", dict(collections.Counter(r["kind"] for r in out)), flush=True)
