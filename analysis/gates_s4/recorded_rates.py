import os
"""The base model's recorded behaviour in a check folder built by train/build_replay_data.py: G4 shares (what the recorded run did after a failed edit) and G5 well-formedness of the
kept generations. usage: python -I -X utf8 recorded_rates.py <folder> [<folder> ...]"""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import s4_gates as G
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER_DIR", "gemma-4-e4b-it-qat-q4_0-unquantized"))
Q = G.QUOTE


def serialize(name, args):
    def val(v):
        return f"{Q}{v}{Q}" if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else str(v)
    return f"<|tool_call>call:{name}{{" + ",".join(f"{k}:{val(v)}" for k, v in sorted(args.items())) + "}<tool_call|>"


def kind_of(r):
    nx = r["recorded_next"]
    text = serialize(nx["name"], nx["args"]) if nx and nx["args"] is not None else "I cannot continue."
    return G.next_kind(text, r["failed_call"])


def show(label, rows):
    s = G.g4_stats(rows, len(rows))
    print(f"  {label:<14} prefixes {s['prefixes']:>4}  tasks {s['tasks']:>3}  repeat {s['repeat_%']:>5}% (se {s['repeat_se']})  re-read {s['reread_%']:>5}% (se {s['reread_se']})  kinds {s['kinds']}")


for folder in sys.argv[1:]:
    print(folder)
    rows = []
    for line in open(f"{folder}/check_failed_edit.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows.append({"instance_id": r["instance_id"], "run": r["run"], "kind": kind_of(r)})
    show("all", rows)
    show("base arm", [r for r in rows if r["run"].startswith("base")])
    show("v2rec arm", [r for r in rows if r["run"].startswith("v2rec")])
    ok = bad = 0
    why = {}
    for name in ("turns_train.jsonl", "turns_holdout.jsonl"):
        for line in open(f"{folder}/{name}", encoding="utf-8"):
            r = json.loads(line)
            good, reason = G.well_formed(tok.decode(r["gen_ids"], skip_special_tokens=False))
            ok += good
            bad += not good
            if not good:
                why[reason] = why.get(reason, 0) + 1
    print(f"  G5 well formed (kept generations): {ok} of {ok + bad} = {100 * ok / (ok + bad):.1f}%   misses: {why}")
