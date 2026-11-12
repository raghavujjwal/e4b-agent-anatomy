"""On-policy training data from the agent's own traces: every turn the base model took, with the exact token ids the server saw and produced.

  python train/build_replay_data.py --runs D:/data/e4b_runs/base1 D:/data/e4b_runs/base2 ... --tokenizer <checkpoint dir> --out D:/data/replay_v1

Why: an adapter trained only on "next action = edit_file" learned to emit edit_file at every step (ad1 run, docs/MEASUREMENTS.md). This builds the
opposite kind of data: the model's own turns at every step of the loop, so training can only move behaviour where the filter says so.
  keep   every turn whose action is read_file, run_command, submit_patch, get_status (the model's own choice, thoughts included)
  keep   edit_file / write_file turns only when the tool applied them (status ok); failed edits are dropped, so the only change relative to
         the model's own distribution is "fewer of the edits that fail"
  drop   turns that hit the length limit, turns longer than --max-len tokens
Token ids come from the trace's `turn_tokens` (prefix_len, suffix, gen, finish), which is the delta encoding used by teammates' training/build_sft.py:
prompt = previous prompt[:prefix_len] + suffix. Nothing is re-rendered. An episode has one `conversation` event per run of the agent's loop (the first run and one more
after every nudge), each with its own delta chain; all of them are read (`chains_of`), and `turn` counts on across them. The result is one jsonl line per turn: run,
instance_id, turn, tool, prompt_ids, gen_ids; and stats.json. `--exclude-task` removes tasks (for example to keep one pool unseen).
Tasks are split by a fixed hash (`sha256("replay-split-v1/" + id)`, first byte % 5 == 0, about 20%): their turns go to turns_holdout.jsonl, the rest to
turns_train.jsonl. check.jsonl is a stratified sample of the holdout (up to --check-per-tool turns of each tool): the prompts for train/step_check.py,
which asks whether an adapter still chooses the same tool as the base model at real prefixes. Train only on turns_train.jsonl.
check_failed_edit.jsonl (S4 gate G4, train/s4_gates.py): the prefixes that END in the result of a failed edit_file/write_file call, with that call parsed. Scope by
--g4-tasks: `all` (the default: the first adapter has seen no task of these runs, and the holdout alone holds too few failed edits) or `holdout` (use this once an adapter has
been trained on own-run turns of these tasks). --g3-tasks does the same for check.jsonl (default `holdout`, as before). Every row also carries `applied_edit_visible`
(an earlier edit that applied is in the prompt), which gate G4b needs; check files built before 2026-11-07 lack it.
check_rejected.jsonl (S4 format probe, report only): the prefixes BEFORE an edit_file/write_file call that the framework refused without running it, because a mandatory
parameter was missing from the arguments it could read (the model wrote a string the way the template's parser cannot read, e.g. `new_string="..."`), with that generation.
All tasks, up to --check-rejected. docs/PLAN_E4B_SFT.md, note of 2026-11-07.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s4_gates import first_call  # noqa: E402

CALL_RE = re.compile(r"<\|tool_call>call:([A-Za-z_]\w*)")
RESP_BLOCK = re.compile(r"<\|tool_response>response:(\w+)\{(.*?)\}<tool_response\|>", re.S)
STATUS_RE = re.compile(r'^value:<\|"\|>\{"status": "(\w+)"')
REJECTED_PREFIX = 'value:<|"|>{"error": "Invoking `'          # ADK's answer to a call with a mandatory parameter missing: the tool never ran
EDIT_TOOLS = {"edit_file", "write_file"}


def tool_of(text: str):
    m = CALL_RE.search(text)
    return m.group(1) if m else None


def turns_of(trace_line):
    """The LAST run of the agent's loop only: [(prompt_ids, gen_ids, finish, None, suffix_ids)], prompts rebuilt from the delta encoding, or None.
    Kept for train/build_sdpo_probe.py, which lines these turns up with the last `conversation` event's messages. Everything else uses chains_of."""
    convs = [e for e in trace_line.get("trace", []) if e.get("event") == "conversation"]
    if not convs or not convs[-1].get("turn_tokens"):
        return None, None
    out, prev = [], []
    for t in convs[-1]["turn_tokens"]:
        prompt = prev[:t["prefix_len"]] + t["suffix"]
        out.append((prompt, t["gen"], t.get("finish"), None, t["suffix"]))
        prev = prompt
    return out, convs[-1].get("messages") or []


def chains_of(trace_line):
    """One list of turns per run of the root agent's loop, in order; a turn is (prompt_ids, gen_ids, finish, None, suffix_ids), the prompt rebuilt from that run's own delta encoding.
    The agent writes one `conversation` event per run of its loop: the first run, and one more after every nudge (the scorer's outer loop nudges a model that ends on text
    before submit_patch). Each run starts a new delta chain (its first turn has prefix_len 0 and holds the whole history). This module read only the last run until 2026-11-06;
    in the six pre-fidelity traces 145 of 396 episodes have several runs and the last alone held 3,446 of the 4,952 turns."""
    convs = [e for e in trace_line.get("trace", []) if e.get("event") == "conversation" and e.get("turn_tokens")]
    if not convs:
        return []
    root = convs[-1].get("agent")             # a sub-agent's conversation is not the root's
    chains = []
    for e in convs:
        if e.get("agent") != root:
            continue
        out, prev = [], []
        for t in e["turn_tokens"]:
            prompt = prev[:t["prefix_len"]] + t["suffix"]
            out.append((prompt, t["gen"], t.get("finish"), None, t["suffix"]))
            prev = prompt
        chains.append(out)
    return chains


def applied_edit_visible(tok, prompt_ids) -> bool:
    """Does the prompt show the result of an edit_file/write_file call that applied (status ok)? 'Visible', not 'happened': a compaction can hide an earlier edit.
    Gate G4b (train/s4_gates.py) judges a model only on the failed-edit prefixes where this is False, because only there is a submit or a give-up likely to leave an empty patch."""
    for m in RESP_BLOCK.finditer(tok.decode(prompt_ids, skip_special_tokens=False)):
        s = STATUS_RE.match(m.group(2))
        if m.group(1) in EDIT_TOOLS and s and s.group(1) == "ok":
            return True
    return False


def edit_status(tok, turns, k, tool):
    """'ok' / 'error' / 'rejected' / None: what happened to the edit made at turn k, read from the tool response that the NEXT turn's suffix adds.
    'error': the tool ran and refused (FileEditError). 'rejected': the framework refused the call before the tool ran, a mandatory parameter was missing from the arguments
    it could read. None: no response (the run ended on this call) or one that is neither.
    A generation with n calls gets n responses, in order, and they are the LAST n blocks of the suffix: when something earlier in the history was re-rendered, the suffix
    also holds older responses in front of them, and the first match is the wrong one (6 of 504 edit turns of the 2026-11-06 control runs: 3 failures read as successes,
    3 successes as failures).
    (Index alignment with the message list is unreliable: compaction makes the two lists differ in length in about half of the traces.)"""
    if k + 1 >= len(turns):
        return None
    n_calls = max(1, tok.decode(turns[k][1], skip_special_tokens=False).count("<|tool_call>"))
    blocks = list(RESP_BLOCK.finditer(tok.decode(turns[k + 1][4], skip_special_tokens=False)))
    if len(blocks) < n_calls:
        return None
    name, body = blocks[-n_calls].groups()
    if name != tool:
        return None
    m = STATUS_RE.match(body)
    if m:
        return m.group(1)
    return "rejected" if body.startswith(REJECTED_PREFIX) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run prefixes: <prefix>_trace.jsonl is read")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=12000, help="drop turns with more prompt+completion tokens than this")
    ap.add_argument("--exclude-task", action="append", default=[])
    ap.add_argument("--check-per-tool", type=int, default=50)
    ap.add_argument("--g3-tasks", choices=["holdout", "all"], default="holdout", help="tasks that check.jsonl (gate G3) is drawn from")
    ap.add_argument("--g4-tasks", choices=["holdout", "all"], default="all", help="tasks that check_failed_edit.jsonl (gate G4) is drawn from")
    ap.add_argument("--check-failed-edit", type=int, default=150, help="at most this many failed-edit prefixes in check_failed_edit.jsonl (0 = all)")
    ap.add_argument("--check-rejected", type=int, default=150, help="at most this many prefixes before a rejected call in check_rejected.jsonl (0 = all)")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats, kept_by_tool, tok_prompt, tok_gen = Counter(), Counter(), 0, 0
    lens = []
    holdout_rows, failed_rows, rejected_rows = [], [], []
    is_holdout = lambda iid: hashlib.sha256(("replay-split-v1/" + iid).encode()).digest()[0] % 5 == 0
    with open(out / "turns_train.jsonl", "w", encoding="utf-8") as fh, open(out / "turns_holdout.jsonl", "w", encoding="utf-8") as fh_h:
        for run in a.runs:
            name = Path(run).name
            for line in open(f"{run}_trace.jsonl", encoding="utf-8"):
                tr = json.loads(line)
                iid = tr["instance_id"]
                if iid in a.exclude_task:
                    stats["excluded_task"] += 1
                    continue
                chains = chains_of(tr)
                if not chains:
                    stats["no_token_record"] += 1
                    continue
                stats["extra_runs"] += len(chains) - 1
                steps, base = [], 0                       # turn numbers run on across the runs of one episode, so (run, instance_id, turn) stays unique
                for run_turns in chains:
                    steps += [(run_turns, j, base + j) for j in range(len(run_turns))]
                    base += len(run_turns)
                for turns, k, n in steps:                 # k: the turn inside its own run (the next turn's prompt holds the result of this one), n: the number written out
                    prompt, gen, finish, _none, _suffix = turns[k]
                    stats["turns"] += 1
                    if finish == "length":
                        stats["dropped_length"] += 1
                        continue
                    if len(prompt) + len(gen) > a.max_len:
                        stats["dropped_too_long"] += 1
                        continue
                    tool = tool_of(tok.decode(gen, skip_special_tokens=False))
                    if tool is None:
                        stats["no_tool_call"] += 1       # a final message or a nudged turn: kept out, not a tool decision
                        continue
                    if tool in EDIT_TOOLS:
                        st = edit_status(tok, turns, k, tool)
                        if st != "ok":
                            stats[f"dropped_{tool}_{st or 'unknown'}"] += 1
                            if st == "error" and k + 1 < len(turns) and len(turns[k + 1][0]) <= a.max_len:      # the state after the failure, for gate G4
                                call = first_call(tok.decode(gen, skip_special_tokens=False))
                                if call and call[1] is not None:
                                    nxt = first_call(tok.decode(turns[k + 1][1], skip_special_tokens=False))      # what the recorded run did next
                                    failed_rows.append({"run": name, "instance_id": iid, "turn": n + 1, "tool": nxt[0] if nxt else None,
                                                        "prompt_ids": turns[k + 1][0], "failed_call": {"name": call[0], "args": call[1]},
                                                        "recorded_next": {"name": nxt[0], "args": nxt[1]} if nxt else None, "applied_edit_visible": applied_edit_visible(tok, turns[k + 1][0]), "holdout": is_holdout(iid)})
                            elif st == "rejected":                                                                                   # the prefix before the call, for the format probe
                                rejected_rows.append({"run": name, "instance_id": iid, "turn": n, "tool": tool, "prompt_ids": prompt, "gen_ids": gen})
                            continue
                    row = {"run": name, "instance_id": iid, "turn": n, "tool": tool, "prompt_ids": prompt, "gen_ids": gen}
                    if is_holdout(iid):
                        fh_h.write(json.dumps(row) + "\n")
                        holdout_rows.append((row["run"], row["instance_id"], row["turn"], row["tool"]))
                        kept_by_tool["holdout_" + tool] += 1
                        continue
                    fh.write(json.dumps(row) + "\n")
                    if a.g3_tasks == "all":
                        holdout_rows.append((row["run"], row["instance_id"], row["turn"], row["tool"]))
                    kept_by_tool[tool] += 1
                    tok_prompt, tok_gen = tok_prompt + len(prompt), tok_gen + len(gen)
                    lens.append(len(prompt) + len(gen))
    lens.sort()
    # stratified check set from the held-out tasks
    pick, by_tool = set(), defaultdict(list)
    for r in holdout_rows:
        by_tool[r[3]].append(r)
    for tool, rows in by_tool.items():
        random.Random(0).shuffle(rows)
        pick |= set(rows[:a.check_per_tool])
    n_check = 0
    with open(out / "check.jsonl", "w", encoding="utf-8") as fc:
        for src in ["turns_holdout.jsonl"] + (["turns_train.jsonl"] if a.g3_tasks == "all" else []):
            for line in open(out / src, encoding="utf-8"):
                row = json.loads(line)
                if (row["run"], row["instance_id"], row["turn"], row["tool"]) in pick:
                    fc.write(line)
                    n_check += 1
    # gate G4: the prefixes that end in a failed edit
    pool = [r for r in failed_rows if a.g4_tasks == "all" or r["holdout"]]
    random.Random(0).shuffle(pool)
    pool = pool[:a.check_failed_edit] if a.check_failed_edit else pool
    with open(out / "check_failed_edit.jsonl", "w", encoding="utf-8") as fe:
        for r in pool:
            fe.write(json.dumps({k: v for k, v in r.items() if k != "holdout"}) + "\n")
    # format probe (report only): the prefixes before a call the framework rejected, whatever the task
    random.Random(0).shuffle(rejected_rows)
    probe = rejected_rows[:a.check_rejected] if a.check_rejected else rejected_rows
    with open(out / "check_rejected.jsonl", "w", encoding="utf-8") as fr:
        for r in probe:
            fr.write(json.dumps(r) + "\n")
    rep = {**stats, "check_turns": n_check, "failed_edit_prefixes": len(failed_rows), "failed_edit_checked": len(pool),
           "failed_edit_tasks": len({r["instance_id"] for r in pool}), "g3_tasks": a.g3_tasks, "g4_tasks": a.g4_tasks,
           "rejected_prefixes": len(rejected_rows), "rejected_checked": len(probe), "rejected_tasks": len({r["instance_id"] for r in probe}),
           "holdout_tasks": len({r[1] for r in holdout_rows}), "kept": sum(kept_by_tool.values()), "kept_by_tool": dict(kept_by_tool),
           "tokens_prompt": tok_prompt, "tokens_completion": tok_gen,
           "len_median": lens[len(lens) // 2] if lens else 0, "len_p95": lens[int(.95 * len(lens))] if lens else 0}
    json.dump(rep, open(out / "stats.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    sys.exit(main())
