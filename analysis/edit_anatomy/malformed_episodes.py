"""How many episodes does the rejected-call class touch, what became of them, and what did the recorded run do after a failed edit? Analysis helper.
usage: python -I -X utf8 malformed_episodes.py <folder with ctl_v2rec{1,2}[_trace].jsonl> <check folder>
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
SRC, CHK = sys.argv[1], sys.argv[2]
BLOCK = re.compile(r"<\|tool_response>response:(\w+)\{(.*?)\}<tool_response\|>", re.S)
STATUS = re.compile(r'^value:<\|"\|>\{"status": "(\w+)"')


def status_of(body: str):
    m = STATUS.match(body)
    if m:
        return m.group(1)
    return "rejected" if body.startswith('value:<|"|>{"error": "Invoking `') else "other"


def style(text: str) -> str:
    i = text.find("<|tool_call>")
    c = text[i:]
    if re.search(r'\w+:"""', c):
        return 'python triple quotes   key:"""..."""'
    if re.search(r'\w+="', c) or re.search(r"\w+='", c):
        return 'equals sign            key="..."'
    if re.search(r'\w+: "', c) or re.search(r'"\w+":', c):
        return 'json-style quotes      key: "..." / "key":'
    if re.search(r"\w+:'", c):
        return "single quotes          key:'...'"
    return "other"


outcome = {}
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for l in open(f"{SRC}/{run}.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            outcome[(run, r["instance_id"])] = r

per_ep = collections.defaultdict(collections.Counter)
styles = collections.Counter()
status_all = collections.Counter()
per_tool = collections.Counter()
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(f"{SRC}/{run}_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        for chain in B.chains_of(tr):
            for k, (prompt, gen, finish, _n, suffix) in enumerate(chain):
                text = tok.decode(gen, skip_special_tokens=False)
                tool = B.tool_of(text)
                if tool is None:
                    continue
                per_tool[tool] += 1
                if tool not in B.EDIT_TOOLS:
                    continue
                if k + 1 >= len(chain):
                    status_all["no response (end of a run)"] += 1
                    continue
                nsuf = tok.decode(chain[k + 1][4], skip_special_tokens=False)
                same = [b for b in BLOCK.finditer(nsuf) if b.group(1) == tool]
                st = status_of(same[-1].group(2)) if same else "no response block"
                status_all[st] += 1
                per_ep[(run, tr["instance_id"])][st] += 1
                if st == "rejected":
                    styles[style(text)] += 1

print("edit/write calls by status (last same-tool response in the next suffix):", dict(status_all))
tot = sum(v for k, v in status_all.items() if k in ("ok", "error", "rejected", "other"))
print("   of %d answered: ok %.1f%%, FileEditError %.1f%%, rejected for a missing parameter %.1f%%" % (
    tot, 100 * status_all["ok"] / tot, 100 * status_all["error"] / tot, 100 * status_all["rejected"] / tot))
print("tool calls by tool (all generations with a call):", dict(per_tool.most_common()))
print("quoting style of the rejected calls:", dict(styles.most_common()))

n_ep = len(outcome)
aff = [k for k, c in per_ep.items() if c["rejected"]]
print(f"\nepisodes with >=1 rejected call: {len(aff)} of {n_ep} ({100 * len(aff) / n_ep:.1f}%) on {len({k[1] for k in aff})} tasks")
oc = collections.Counter(outcome[k]["outcome_class"] for k in aff)
print("   their outcomes:", dict(oc))
rest = [k for k in outcome if k not in set(aff)]
oc2 = collections.Counter(outcome[k]["outcome_class"] for k in rest)
print("   the other %d episodes:" % len(rest), dict(oc2), "-> resolved %.1f%%, empty patch %.1f%%" % (100 * oc2["resolved"] / len(rest), 100 * oc2["empty_patch_without_error"] / len(rest)))
heavy = [k for k in aff if per_ep[k]["rejected"] >= 3]
print(f"   episodes with >=3 rejected calls: {len(heavy)}; their outcomes", dict(collections.Counter(outcome[k]['outcome_class'] for k in heavy)))
print("   rejected calls in the 5 worst episodes:", sorted(((per_ep[k]["rejected"], k[0][-1], k[1]) for k in aff), reverse=True)[:5])
# is the same task also in the other seed?
by_task = collections.defaultdict(list)
for k in aff:
    by_task[k[1]].append(per_ep[k]["rejected"])
print("   tasks that are hit in both seeds:", sum(1 for v in by_task.values() if len(v) == 2), "of", len(by_task))

# what the recorded run did after a failed edit (the 150 G4 prefixes), by proper comparison, not by rebuilding text
fe = [json.loads(l) for l in open(f"{CHK}/check_failed_edit.jsonl", encoding="utf-8")]
kinds = collections.Counter()
for r in fe:
    nx = r["recorded_next"]
    failed = (r["failed_call"]["name"], r["failed_call"]["args"])
    if not nx:
        kinds["no_call"] += 1
    elif nx["args"] is None:
        kinds["malformed"] += 1
    elif G.same_call(failed, (nx["name"], nx["args"])):
        kinds["repeat"] += 1
    elif nx["name"] == "read_file":
        kinds["reread"] += 1
    elif nx["name"] in ("edit_file", "write_file"):
        kinds["edit"] += 1
    else:
        kinds["other"] += 1
n = len(fe)
print(f"\nrecorded next action after a failed edit, {n} prefixes:", dict(kinds), "| repeat %.1f%%, re-read %.1f%%, another edit %.1f%%" % (
    100 * kinds["repeat"] / n, 100 * kinds["reread"] / n, 100 * kinds["edit"] / n))
fails = collections.Counter("not found" if "old_string not found" in tok.decode(r["prompt_ids"][-400:], skip_special_tokens=False) else "occurs N times" if "occurs" in tok.decode(r["prompt_ids"][-400:], skip_special_tokens=False) else "other" for r in fe)
print("failure kinds among the G4 prefixes:", dict(fails))
