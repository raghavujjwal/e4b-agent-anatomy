"""After a failed edit with nothing applied yet: what does each model do next? Analysis helper.
usage: python -I -X utf8 analyze_s4_g4.py <s4 result folder> <check folder>
"""
import collections
import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import s4_gates as G  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
RES, CHK = sys.argv[1], sys.argv[2]
BASE, AD = "gemma-4-e4b-it-qat-w4a16-ct", "e4b_sft_v2"
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
fe = jl(f"{CHK}/check_failed_edit.jsonl")
g4 = {m: jl(f"{RES}/{m}_g4.jsonl") for m in (BASE, AD)}
APPLIED = re.compile(r'response:(?:edit_file|write_file)\{value:<\|"\|>\{"status": "ok"')
texts = [tok.decode(p["prompt_ids"], skip_special_tokens=False) for p in fe]
applied = [bool(APPLIED.search(t)) for t in texts]
print(f"G4 prefixes: {len(fe)}; with an edit already applied earlier: {sum(applied)}; nothing applied yet: {len(fe) - sum(applied)}")
# what the RECORDED run did next at those prefixes
rec = collections.Counter()
for p, a in zip(fe, applied):
    rec[(("applied" if a else "nothing applied"), p["recorded_next"]["name"] if p["recorded_next"] else "no call")] += 1
print("recorded run, next call by state:", dict(sorted(rec.items())))


def tool_of_text(text):
    c = G.first_call(text)
    return c[0] if c else ("no_call" if "<|tool_call>" not in text else "malformed")


for state in (False, True):
    label = "an edit already applied" if state else "NOTHING applied yet"
    print(f"\n== {label}: {sum(1 for a in applied if a == state)} prefixes x 3 samples")
    for m in (BASE, AD):
        rows = [r for i, r in enumerate(g4[m]) if applied[i // 3] == state]
        n = len(rows)
        tools = collections.Counter(tool_of_text(r["text"]) for r in rows)
        kinds = collections.Counter(r["kind"] for r in rows)
        tests = 0
        for r in rows:
            c = G.first_call(r["text"])
            if c and c[0] == "run_command" and c[1] and re.search(r"\b(pytest|unittest|tox)\b", c[1].get("command", "")):
                tests += 1
        print(f"  {m[:12]:<12} n={n:>3} | " + ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in tools.most_common()) + f" | of which run tests {100 * tests / n:.0f}%"
              + f" | repeat {100 * kinds['repeat'] / n:.0f}% re-read {100 * kinds['reread'] / n:.0f}% another edit {100 * kinds['edit'] / n:.0f}%")
