"""What did the adapter do where the base reads, and after a failed edit? Reads the S4 per-generation files and the check files they were made from.
Analysis helper.  usage: python -I -X utf8 analyze_s4.py <s4 result folder> <check folder (ctl_check)>
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

check = jl(f"{CHK}/check.jsonl")
fe = jl(f"{CHK}/check_failed_edit.jsonl")
g3 = {m: jl(f"{RES}/{m}_g3.jsonl") for m in (BASE, AD)}
g4 = {m: jl(f"{RES}/{m}_g4.jsonl") for m in (BASE, AD)}

# ---- alignment: rows come in prefix order, N samples per prefix
def aligned(rows, prefixes, per, key):
    bad = sum(1 for i, r in enumerate(rows) if key(prefixes[i // per]) != (r["instance_id"], r["turn"], r["recorded"]))
    return bad
print("alignment (rows whose prefix by position does not match their instance/turn/tool):",
      {m: aligned(g3[m], check, 2, lambda p: (p["instance_id"], p["turn"], p["tool"])) for m in g3},
      {m: aligned(g4[m], fe, 3, lambda p: (p["instance_id"], p["turn"], p["tool"])) for m in g4})

# ---- prefix features
APPLIED = re.compile(r'response:(?:edit_file|write_file)\{value:<\|"\|>\{"status": "ok"')


def features(ids):
    t = tok.decode(ids, skip_special_tokens=False)
    first_user, first_model = t.find("<|turn>user"), t.find("<|turn>model")
    post_compaction = first_model != -1 and (first_user == -1 or first_model < first_user)
    users = t.count("<|turn>user")
    return {"text": t, "post_compaction": post_compaction, "nudge": users >= (1 if post_compaction else 2),
            "thoughts": "<|channel>thought" in t, "applied_edit": bool(APPLIED.search(t)), "tokens": len(ids)}


F3 = [features(p["prompt_ids"]) for p in check]
F4 = [features(p["prompt_ids"]) for p in fe]


def tool_of_text(text):
    c = G.first_call(text)
    return c[0] if c else ("no_call" if "<|tool_call>" not in text else "malformed")


def cmd_kind(cmd):
    c = cmd.strip()
    if re.search(r"\b(pytest|unittest|tox|nosetests)\b", c):
        return "run tests"
    if re.match(r"(cat|head|tail|sed|less|grep|rg|find|ls|git (diff|show|log|status)|awk|wc|tree|file)\b", c):
        return "read-like (cat/grep/sed/ls/git diff)"
    if re.match(r"python3?\b", c):
        return "python script / -c"
    return "other"


# ---- 1. read_file prefixes: where does the adapter's non-read go, by kind of prefix
print("\n== read_file prefixes (50 prefixes, 2 samples each): agreement with a read, by kind of prefix")
for name, pred in (("all", lambda f: True), ("after a compaction", lambda f: f["post_compaction"]), ("not after a compaction", lambda f: not f["post_compaction"]),
                   ("after a nudge", lambda f: f["nudge"]), ("not after a nudge", lambda f: not f["nudge"]),
                   ("thoughts in the history", lambda f: f["thoughts"]), ("no thoughts in the history", lambda f: not f["thoughts"]),
                   ("an edit already applied in the history", lambda f: f["applied_edit"]), ("no edit applied yet", lambda f: not f["applied_edit"])):
    out = []
    for m in (BASE, AD):
        rows = [(r, F3[i // 2]) for i, r in enumerate(g3[m]) if r["recorded"] == "read_file" and pred(F3[i // 2])]
        n = len(rows)
        out.append((n, sum(1 for r, f in rows if r["generated"] == "read_file"), sum(1 for r, f in rows if r["generated"] == "edit_file")))
    (nb, rb, eb), (na, ra, ea) = out
    if nb:
        print(f"  {name:<40} prefixes {nb // 2:>2} | base: read {100 * rb / nb:>4.0f}% edit {100 * eb / nb:>3.0f}% | adapter: read {100 * ra / na:>4.0f}% edit {100 * ea / na:>3.0f}%")

# ---- 2. the adapter's edit_file at read prefixes: is the anchor visible in the context?
def anchor_visible(call_text, prefix_text):
    c = G.first_call(call_text)
    if not c or not c[1] or "old_string" not in c[1]:
        return None
    lines = [l.strip() for l in c[1]["old_string"].split("\n") if l.strip()]
    if not lines:
        return False
    probe = max(lines, key=len)[:80]
    esc = json.dumps(probe)[1:-1]
    return probe in prefix_text or esc in prefix_text or esc.replace("\\\\", "\\") in prefix_text


print("\n== is the old_string of a generated edit_file visible in the prefix (its longest line, raw or JSON-escaped)?")
for m in (BASE, AD):
    for rec in ("read_file", "edit_file", "run_command"):
        rows = [(r, F3[i // 2]) for i, r in enumerate(g3[m]) if r["recorded"] == rec and r["generated"] == "edit_file"]
        vis = [anchor_visible(r["text"], f["text"]) for r, f in rows]
        vis = [v for v in vis if v is not None]
        if vis:
            print(f"  {m[:12]:<12} edits at {rec:<12} prefixes: {len(vis):>3} edits, anchor visible in {100 * sum(vis) / len(vis):.0f}%")

# ---- 3. submit_patch: is anything applied yet?
print("\n== submit_patch generated, and whether an edit has been applied in the history (an empty patch is the worst outcome)")
for m in (BASE, AD):
    rows = [(r, F3[i // 2]) for i, r in enumerate(g3[m]) if r["generated"] == "submit_patch"]
    by_rec = collections.Counter((r["recorded"], f["applied_edit"]) for r, f in rows)
    tot = len(rows)
    noedit = sum(1 for r, f in rows if not f["applied_edit"])
    print(f"  {m[:12]:<12} submits {tot:>3} of 402; with NO applied edit in the history: {noedit}; by (recorded tool, applied edit): {dict(sorted(by_rec.items()))}")

# ---- 4. G4: what is 'other'?
print("\n== G4, after a failed edit (150 prefixes x 3 samples): what the first call is")
for m in (BASE, AD):
    rows = g4[m]
    tools = collections.Counter(tool_of_text(r["text"]) for r in rows)
    print(f"  {m[:12]:<12} {dict(tools.most_common())}")
    other = [r for r in rows if r["kind"] == "other"]
    sub = collections.Counter(tool_of_text(r["text"]) for r in other)
    cmds = collections.Counter()
    for r in other:
        c = G.first_call(r["text"])
        if c and c[0] == "run_command" and c[1]:
            cmds[cmd_kind(c[1].get("command", ""))] += 1
    print(f"     'other' = {len(other)}: by tool {dict(sub.most_common())}; run_command by kind {dict(cmds.most_common())}")
    idx = [i for i, r in enumerate(rows) if r["kind"] == "other" and tool_of_text(r["text"]) == "submit_patch"]
    if idx:
        print(f"     submit_patch right after a failed edit: {len(idx)} generations; with an applied edit earlier in the history: {sum(F4[i // 3]['applied_edit'] for i in idx)}")
    # by failure kind
    fk = collections.defaultdict(collections.Counter)
    for i, r in enumerate(rows):
        kind = "occurs N times" if "occurs" in F4[i // 3]["text"][-500:] else "not found"
        fk[kind][r["kind"]] += 1
    print("     by failure kind:", {k: dict(v) for k, v in fk.items()})

# ---- 5. what the adapter's run_command is at read prefixes
print("\n== run_command generated at read_file prefixes, by kind")
for m in (BASE, AD):
    cmds = collections.Counter()
    for i, r in enumerate(g3[m]):
        if r["recorded"] == "read_file" and r["generated"] == "run_command":
            c = G.first_call(r["text"])
            cmds[cmd_kind(c[1].get("command", "")) if c and c[1] else "unparsed"] += 1
    print(f"  {m[:12]:<12} {dict(cmds)}")

# ---- 6. overall tool shares
print("\n== share of first calls by tool over all 402 G3 generations")
for m in (BASE, AD):
    c = collections.Counter(r["generated"] for r in g3[m])
    print(f"  {m[:12]:<12} " + ", ".join(f"{k} {100 * v / 402:.0f}%" for k, v in c.most_common()))
print("  the control's own calls (all 1,653 tool-call generations): read_file 37%, edit_file 31%, run_command 25%, submit_patch 8%")
