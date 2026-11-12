"""How many failed 'old_string not found' edits are near-miss copy slips, and what kind?  (no model, offline)

Reuses the trace parsers of mine/trace_analysis.py. For every such failure whose file the model had read, find the
window of the same number of lines (in any text of that file the model was shown) that is closest to old_string and
describe the difference with difflib opcodes.

  python -I -X utf8 mine/copy_slip_census.py --runs D:/data/e4b_runs/base1 D:/data/e4b_runs/base2 ... --out D:/data/analysis/copy_slip_census.json
"""
import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_analysis import parse_calls, result_info, edit_outcome, file_texts  # noqa: E402


def windows(text, k):
    tl = text.split("\n")
    return ["\n".join(tl[i:i + k]) for i in range(0, max(len(tl) - k + 1, 1))]


def closest(old, texts):
    k = max(len(old.split("\n")), 1)
    best_r, best_w = 0.0, None
    for t in texts:
        for w in windows(t, k):
            sm = difflib.SequenceMatcher(None, old, w, autojunk=False)
            if sm.quick_ratio() <= max(best_r, 0.5):
                continue
            r = sm.ratio()
            if r > best_r:
                best_r, best_w = r, w
    return best_r, best_w


def describe(old, win):
    """Classify the difference between the model's old_string and the closest real window."""
    ops = [o for o in difflib.SequenceMatcher(None, old, win, autojunk=False).get_opcodes() if o[0] != "equal"]
    if not ops:
        return "identical", []
    kinds = {o[0] for o in ops}
    chunks = []
    for tag, i1, i2, j1, j2 in ops:
        chunks.append((tag, old[i1:i2], win[j1:j2], old[max(i1 - 1, 0):i1], old[i2:i2 + 1] or win[j2:j2 + 1]))
    if kinds == {"insert"}:       # the file has characters the model left out
        word_final = all(c[3].isalpha() and not (c[4].isalnum() or c[4] == "_") and len(c[2]) <= 2 for c in chunks)
        only_s = all(c[2] == "s" for c in chunks)
        return ("dropped_final_s" if word_final and only_s else "dropped_final_letter" if word_final else "dropped_chars"), chunks
    if kinds == {"delete"}:       # the model added characters the file does not have
        return "extra_chars", chunks
    if kinds == {"replace"}:
        ws = all(not c[1].strip() and not c[2].strip() for c in chunks)
        return ("whitespace_only" if ws else "substitution"), chunks
    return "mixed", chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        name = Path(run).name
        for line in open(f"{run}_trace.jsonl", encoding="utf-8"):
            tr = json.loads(line)
            convs = [e for e in tr["trace"] if e.get("event") == "conversation"]
            if not convs:
                continue
            msgs = convs[-1]["messages"]
            results = {m["tool_call_id"]: m.get("content", "") for m in msgs if m.get("role") == "tool" and m.get("tool_call_id")}
            calls = [c for m in msgs if m.get("role") == "assistant" for c in parse_calls(m)]
            shown = file_texts(msgs, {c["id"]: c for c in calls})
            for c in calls:
                if c["name"] != "edit_file" or not isinstance(c["args"], dict):
                    continue
                if edit_outcome(*result_info(results.get(c["id"], ""))) != "old_string not found":
                    continue
                old, path = str(c["args"].get("old_string", "")), c["args"].get("filepath")
                texts = shown.get(path, [])
                row = {"run": name, "task": tr["instance_id"], "file": path, "read": bool(texts), "old_lines": len(old.split("\n"))}
                if texts and old.strip():
                    r, w = closest(old, texts)
                    row["ratio"] = round(r, 3)
                    row["kind"], chunks = describe(old, w) if w is not None else ("none", [])
                    row["chunks"] = [[c0, c1[:40], c2[:40]] for c0, c1, c2, _p, _n in chunks[:4]]
                rows.append(row)
    n = len(rows)
    read = [r for r in rows if "ratio" in r]
    summary = {"not_found_edits": n, "file_read": len(read)}
    for lo, hi in ((0.95, 1.01), (0.9, 0.95), (0.8, 0.9), (0.0, 0.8)):
        sel = [r for r in read if lo <= r["ratio"] < hi]
        summary[f"ratio {lo}-{hi}"] = {"n": len(sel), "kinds": dict(Counter(r["kind"] for r in sel))}
    near = [r for r in read if r["ratio"] >= 0.9]
    summary["near_miss_total_ge_0.9"] = len(near)
    summary["dropped_final_s_ge_0.9"] = sum(r["kind"] == "dropped_final_s" for r in near)
    summary["dropped_final_letter_incl_s_ge_0.9"] = sum(r["kind"] in ("dropped_final_s", "dropped_final_letter") for r in near)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
