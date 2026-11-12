"""What do the generations the framework rejected look like? Analysis helper.
usage: python -I -X utf8 rejected_styles.py <check_rejected.jsonl> [n_show]
"""
import collections
import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import served  # noqa: E402

tok = served.load_tokenizer()
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
show = int(sys.argv[2]) if len(sys.argv) > 2 else 10
Q = '<|"|>'
styles = collections.Counter()
lens = []
for i, r in enumerate(rows):
    text = tok.decode(r["gen_ids"], skip_special_tokens=False)
    lens.append(len(r["gen_ids"]))
    m = re.search(r"<\|tool_call>call:(\w+)\{(.*)", text, re.S)
    body = m.group(2) if m else ""
    tags = []
    if re.search(r'\b(old_string|new_string|content|filepath)\s*=\s*', body):
        tags.append("python-kwarg(=)")
    if '"""' in body or "'''" in body:
        tags.append("triple-quote")
    if re.search(r'\b(old_string|new_string|content|filepath):\s*"', body):
        tags.append("json-quote(key:\"...\")")
    if re.search(r"\b(old_string|new_string|content|filepath):\s*'", body):
        tags.append("single-quote")
    if "\n" in body and Q not in body:
        tags.append("escaped-newlines-no-Q")
    if Q not in body:
        tags.append("no <|\"|> delimiters at all")
    styles[tuple(tags) or ("other",)] += 1
    if i < show:
        print(f"\n--- {r['instance_id']} turn {r['turn']} ({r['tool']}), {len(r['gen_ids'])} tokens, tags={tags}")
        print(text[:700].replace("\n", "\n"))
print("\n== styles (combinations of tags):")
for k, v in styles.most_common():
    print(f"  {v:>3}  {' + '.join(k)}")
print(f"generation length: median {sorted(lens)[len(lens)//2]}, max {max(lens)} tokens; {len(rows)} rejected generations from {len({r['instance_id'] for r in rows})} tasks")
