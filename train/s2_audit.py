"""S2.7: audit a built training set against the pass conditions of the plan, separately from the render step.

  python -I -X utf8 train/s2_audit.py --sft D:/data/s2/full3/sft.jsonl --replayed D:/data/s2/full3/replayed.jsonl [--render-log D:/data/s2/full3/render.log] [--json audit.json]

Needs the file written WITHOUT --no-text: it checks that the stored ids are the tokenization of the stored text. The render step drops samples that fail
its own checks; this script looks at what is left, re-derives each condition from the stored data, and exits 0 only if every one holds:

  lengths        median <= 7,000 tokens, max <= 12,000
  control tokens each sample has exactly the control tokens its conversation implies (a repository file that contains the text `<tool_response|>` would
                 otherwise become a real control token and teach the model a broken format)
  loss mask      one run per tool call, each exactly `<|tool_call>` ... `<tool_call|><|tool_response>`, then the closing text through `<turn|>`
  text and ids   the stored ids are the tokenization of the stored text
  arguments      every tool call parses, uses declared parameters only, has the required ones, with the declared types
  edits          every edit_file result is ok, strategy exact, one occurrence; the patch touches one to three files
  envelope       read windows <= 150 lines and <= 10,000 characters, command output <= 5,000 characters, every result ok
  verification   the command run after the last edit runs a test runner or names a part of an edited file (not a snippet that tests Python itself)
  format         the keys of each call are in the order the served template writes them (alphabetical)
  leaks          no teacher path and no evaluation-pool repository anywhere in the text
  teacher names  no teacher or benchmark name in what the model is trained to say (the loss-masked spans)
  first message  (plan D0) the fixed text of every sample's first user message equals the scorer capture byte for byte; repository field = the sample's repository
  system prompt  the system turn is identical in every sample with the same thinking setting
  scorer capture (plan D0) the system message and the tool block equal the captured request's, as text and as the tokens the server would send
  samples        instance ids are unique; thinking is on for 40-60% of the samples
  prefix         the render log shows no sample dropped for an unstable prefix
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import served  # noqa: E402
from s2_render import EVAL_REPOS, EVAL_SHORT, LEAK, system_text, unrelated_verification  # noqa: E402

CONTROL = ["<bos>", "<|turn>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|think|>"]
IDENTITY = re.compile(r"openhands|qwen|nebius|rebench|swe-?bench|str_replace_editor|execute_bash|<function=|<parameter=", re.I)
MAX_MEDIAN, MAX_TOKENS = 7000, 12000
READ_LINES, READ_CHARS, STDOUT_CHARS = 150, 10000, 5000
JSON_TYPES = {"string": str, "integer": int, "boolean": bool, "number": (int, float), "object": dict, "array": list}


def control_ids(tok) -> dict[str, int]:
    ids = {}
    for name in CONTROL:
        i = tok.convert_tokens_to_ids(name)
        if i is None or i == tok.unk_token_id:
            raise SystemExit(f"{name} is not a token of this tokenizer")
        ids[name] = i
    return ids


def control_problems(ids: list[int], think: bool, cid: dict[str, int], n_calls: int) -> list[str]:
    """The control tokens a whole episode implies: one bos at the start, system + user + model turns, one tool_call / tool_response pair per call."""
    c = Counter(ids)
    want = {"<bos>": 1, "<|turn>": 3, "<turn|>": 3, "<|tool_call>": n_calls, "<tool_call|>": n_calls, "<|tool_response>": n_calls,
            "<tool_response|>": n_calls, "<|think|>": 1 if think else 0}
    bad = [f"{k} x{c[cid[k]]}, expected {w}" for k, w in want.items() if c[cid[k]] != w]
    if not ids or ids[0] != cid["<bos>"]:
        bad.append("does not start with <bos>")
    return bad


def mask_runs(mask: list[int]) -> list[tuple[int, int]]:
    runs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def mask_problems(ids: list[int], mask: list[int], cid: dict[str, int], n_calls: int) -> list[str]:
    runs = mask_runs(mask)
    bad = []
    if len(mask) != len(ids):
        return [f"mask has {len(mask)} entries for {len(ids)} ids"]
    if len(runs) != n_calls + 1:
        bad.append(f"{len(runs)} loss runs for {n_calls} calls")
    for k, (a, b) in enumerate(runs[:-1]):
        if not (ids[a] == cid["<|tool_call>"] and ids[b - 1] == cid["<|tool_response>"] and ids[b - 2] == cid["<tool_call|>"]):
            bad.append(f"loss run {k} is not exactly one tool call plus its stop token")
        elif ids[a:b].count(cid["<|tool_call>"]) != 1:
            bad.append(f"loss run {k} holds more than one tool call")
    if runs:
        a, b = runs[-1]
        if ids[b - 1] != cid["<turn|>"]:
            bad.append("the last loss run does not end with <turn|>")
        if a == 0 or ids[a - 1] != cid["<tool_response|>"]:
            bad.append("the closing text does not follow a tool response")
        if cid["<|tool_call>"] in ids[a:b]:
            bad.append("the closing text contains a tool call")
    return bad


def argument_problems(messages: list[dict], declared: list[dict]) -> list[str]:
    schema = {t["function"]["name"]: t["function"].get("parameters") or {} for t in declared}
    bad = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except ValueError:
                bad.append(f"{name}: arguments are not JSON")
                continue
            if name not in schema:
                bad.append(f"{name}: not a declared tool")
                continue
            props = schema[name].get("properties") or {}
            extra = set(args) - set(props)
            missing = set(schema[name].get("required") or []) - set(args)
            if extra:
                bad.append(f"{name}: undeclared parameters {sorted(extra)}")
            if missing:
                bad.append(f"{name}: missing {sorted(missing)}")
            for k, v in args.items():
                want = JSON_TYPES.get((props.get(k) or {}).get("type"))
                if want and (not isinstance(v, want) or (want is int and isinstance(v, bool))):
                    bad.append(f"{name}.{k}: expected {props[k]['type']}, got {type(v).__name__}")
    return bad


def result_problems(messages: list[dict]) -> tuple[list[str], list[str]]:
    """(edit problems, envelope problems) from the tool calls and their results."""
    names, edit_bad, env_bad, n_edit, n_edit_ok, files = {}, [], [], 0, 0, None
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names[tc["id"]] = tc["function"]["name"]
            if tc["function"]["name"] == "read_file":
                a = json.loads(tc["function"]["arguments"])
                if "start_line" in a and "end_line" in a and a["end_line"] - a["start_line"] + 1 > READ_LINES:
                    env_bad.append(f"read window of {a['end_line'] - a['start_line'] + 1} lines requested")
            if tc["function"]["name"] == "edit_file":
                n_edit += 1
        if m["role"] != "tool":
            continue
        name, d = names.get(m["tool_call_id"]), json.loads(m["content"])
        if d.get("status") != "ok":
            env_bad.append(f"{name}: status {d.get('status')!r}")
        if name == "edit_file":
            if d.get("strategy") == "exact" and d.get("occurrences") == 1 and d.get("status") == "ok":
                n_edit_ok += 1
            else:
                edit_bad.append(f"edit_file strategy {d.get('strategy')!r}, occurrences {d.get('occurrences')!r}")
        elif name == "read_file":
            body = d.get("content") or ""
            if body.count("\n") + 1 > READ_LINES or len(body) > READ_CHARS:
                env_bad.append(f"read result of {body.count(chr(10)) + 1} lines / {len(body)} characters")
        elif name == "run_command":
            if len(d.get("stdout") or "") > STDOUT_CHARS or len(d.get("stderr") or "") > STDOUT_CHARS:
                env_bad.append("command output over 5,000 characters")
            if not isinstance(d.get("exit_code"), int):
                env_bad.append("command result without an exit code")
        elif name == "submit_patch":
            files = d.get("files_changed")
    if n_edit == 0:
        edit_bad.append("no edit_file call")
    if n_edit_ok != n_edit:
        edit_bad.append(f"{n_edit - n_edit_ok} of {n_edit} edits were not an exact single replacement")
    if not isinstance(files, int) or not 1 <= files <= 3:
        edit_bad.append(f"patch touches {files!r} files")
    return edit_bad, env_bad


def order_problems(text: str) -> list[str]:
    """The served template writes the keys of a call alphabetically; training text in any other order would teach one the server never produces."""
    calls = re.findall(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.S)
    bad = []
    for name, body in calls:
        if name == "edit_file" and not re.match(r'filepath:<\|"\|>.*?<\|"\|>,new_string:<\|"\|>.*?<\|"\|>,old_string:<\|"\|>', body, re.S):
            bad.append("edit_file keys out of order")
        if name == "read_file" and not re.match(r'(end_line:\d+,)?filepath:<\|"\|>[^<]*<\|"\|>(,start_line:\d+)?$', body):
            bad.append("read_file keys out of order")
    return bad


def leak_problems(text: str) -> list[str]:
    low = text.lower()
    bad = []
    if LEAK.search(text):
        bad.append("teacher path: " + LEAK.search(text).group(0))
    hit = [n for n in EVAL_REPOS if n in low] + [f"/workspace/{s}" for s in EVAL_SHORT if f"/workspace/{s}" in low]
    if hit:
        bad.append("evaluation pool: " + hit[0])
    return bad


def shared_system_turn(groups: dict[bool, list[list[int]]], turn_id: int) -> dict[bool, tuple[int, int, bool]]:
    """Per thinking setting: (tokens every sample shares, where the user turn starts, whether the shared prefix reaches it, i.e. the system turn is identical)."""
    out = {}
    for think, rows in groups.items():
        ref, lcp = rows[0], len(rows[0])
        for ids in rows[1:]:
            k, m = 0, min(lcp, len(ids))
            while k < m and ids[k] == ref[k]:
                k += 1
            lcp = k
        start = [i for i, t in enumerate(ref) if t == turn_id][1]
        out[think] = (lcp, start, lcp >= start)
    return out


FIRST_HEAD = "You are evaluating a software engineering task for repository "
STATEMENT_HEAD = ".\n\nProblem Statement:\n"
BUDGET_HEAD = "\n\n## Task Budget (Session terminates"
TREE_HEAD = "(up to 3 levels):\n```\n"
TREE_TAIL = "\n```\n"
CAPTURE = HERE.parent / "tests" / "fixtures" / "scorer_requests_v2rec.json"


def split_first_message(u: str):
    """(repo, statement, fixed middle, tree) of the scorer's first user message, or None if it does not have that shape. Everything between the statement and the
    tree is fixed text (budget, rules, instructions, layout heading); only the repository, the statement and the tree are task fields."""
    if not u.startswith(FIRST_HEAD) or STATEMENT_HEAD not in u or BUDGET_HEAD not in u or TREE_HEAD not in u or not u.endswith(TREE_TAIL):
        return None
    i = u.index(STATEMENT_HEAD, len(FIRST_HEAD))
    j = u.index(BUDGET_HEAD, i)
    k = u.index(TREE_HEAD, j) + len(TREE_HEAD)
    return u[len(FIRST_HEAD):i], u[i + len(STATEMENT_HEAD):j], u[j:k], u[k:-len(TREE_TAIL)]


def first_message_problems(u: str, repo: str, captured_middle: str) -> list[str]:
    """Plan D0: the fixed parts of the first user message equal the scorer capture byte for byte; the task fields are the sample's own."""
    parts = split_first_message(u)
    if parts is None:
        return ["first user message does not have the scorer's shape"]
    bad = []
    if parts[0] != repo:
        bad.append(f"repository field {parts[0]!r} is not {repo!r}")
    if parts[2] != captured_middle:
        bad.append("the fixed text (budget, rules, instructions) differs from the scorer capture")
    if not parts[1].strip() or not parts[3].strip():
        bad.append("empty problem statement or tree")
    return bad


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--replayed", required=True)
    ap.add_argument("--render-log")
    ap.add_argument("--submission", default=str(HERE.parent / "agent" / "submission_e4b_v2rec"))
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args(argv)
    sft = read(a.sft)
    if sft and "text" not in sft[0]:
        raise SystemExit("the file has no `text`: audit the file written without --no-text")
    rep = {r["instance_id"]: r for r in read(a.replayed) if "messages" in r}
    tok = served.load_tokenizer()
    cid = control_ids(tok)
    sys_msg, declared = system_text(Path(a.submission))
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))["requests"][0]
    captured_middle = split_first_message(capture["messages"][1]["content"])[2]
    found: dict[str, list[tuple[str, str]]] = {k: [] for k in ("control tokens", "loss mask", "text and ids", "arguments", "edits", "envelope", "verification", "format", "leaks", "teacher names",
                                                               "first message")}
    for s in sft:
        iid, ids, rec = s["instance_id"], s["input_ids"], rep.get(s["instance_id"])
        if rec is None:
            found["text and ids"].append((iid, "no replayed record"))
            continue
        for p in control_problems(ids, s["enable_thinking"], cid, s["n_calls"]):
            found["control tokens"].append((iid, p))
        for p in mask_problems(ids, s["loss_mask"], cid, s["n_calls"]):
            found["loss mask"].append((iid, p))
        if tok(s["text"], add_special_tokens=False)["input_ids"] != ids:
            found["text and ids"].append((iid, "ids are not the tokenization of the text"))
        for p in argument_problems(rec["messages"], declared):
            found["arguments"].append((iid, p))
        edit_bad, env_bad = result_problems(rec["messages"])
        found["edits"] += [(iid, p) for p in edit_bad]
        found["envelope"] += [(iid, p) for p in env_bad]
        found["verification"] += [(iid, p) for p in [unrelated_verification(rec["messages"])] if p]
        found["first message"] += [(iid, p) for p in first_message_problems(rec["messages"][0]["content"], s["repo"], captured_middle)]
        found["format"] += [(iid, p) for p in order_problems(s["text"])]
        found["leaks"] += [(iid, p) for p in leak_problems(s["text"])]
        said = " ".join(tok.decode(ids[x:y]) for x, y in mask_runs(s["loss_mask"]))
        if IDENTITY.search(said):
            found["teacher names"].append((iid, "the model is trained to write: " + IDENTITY.search(said).group(0)))
    toks = [s["n_tokens"] for s in sft]
    groups = {t: [s["input_ids"] for s in sft if s["enable_thinking"] == t] for t in (False, True)}
    groups = {t: rows for t, rows in groups.items() if rows}
    sysprompt = shared_system_turn(groups, cid["<|turn>"])
    # plan D0: the system turn and the tool block of the samples are the scorer capture's, token for token (render the captured request the way the server does)
    cap_ok, cap_detail = sys_msg == capture["messages"][0]["content"] and json.dumps(declared, sort_keys=True) == json.dumps(capture["tools"], sort_keys=True), []
    for think, (_, start, _) in sysprompt.items():
        ref = tok(served.render(tok, capture["messages"], capture["tools"], enable_thinking=think, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        same = ref[:start] == groups[think][0][:start] and ref[start] == cid["<|turn>"]
        cap_ok = cap_ok and same
        cap_detail.append(f"thinking {'on' if think else 'off'}: {start} tokens {'identical' if same else 'DIFFER'}")
    think_share = sum(s["enable_thinking"] for s in sft) / max(1, len(sft))
    dup = [k for k, v in Counter(s["instance_id"] for s in sft).items() if v > 1]
    render_drops = {}
    if a.render_log:
        txt = Path(a.render_log).read_text(encoding="utf-8")
        render_drops = json.loads(txt[txt.index("{"):]).get("dropped_in_render", {})
    rows = [
        ("lengths", statistics.median(toks) <= MAX_MEDIAN and max(toks) <= MAX_TOKENS, f"median {statistics.median(toks):.0f} (<= {MAX_MEDIAN}), max {max(toks)} (<= {MAX_TOKENS})"),
        *[(k, not v, f"{len(v)} problems in {len({i for i, _ in v})} samples") for k, v in found.items()],
        ("system prompt", all(ok for _, _, ok in sysprompt.values()),
         "; ".join(f"thinking {'on' if t else 'off'}: {n} shared tokens, user turn starts at {start}" for t, (n, start, _) in sysprompt.items())),
        ("scorer capture", cap_ok, "system message and tool block equal the captured request's" + (" (" + "; ".join(cap_detail) + ")")),
        ("samples", not dup and 0.4 <= think_share <= 0.6, f"{len(dup)} duplicated ids, thinking on for {think_share:.0%}"),
        ("prefix", not any("prefix" in k for k in render_drops) if a.render_log else True,
         ("render drops: " + (json.dumps(render_drops) if render_drops else "none")) if a.render_log else "not checked (no --render-log)"),
    ]
    width = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    for k, v in found.items():
        for iid, p in v[:5]:
            print(f"      {k}: {iid}: {p}")
    all_ok = all(ok for _, ok, _ in rows)
    print(f"\n{len(sft)} samples from {len({s['repo'] for s in sft})} repositories: " + ("every pass condition holds" if all_ok else "NOT ALL CONDITIONS HOLD"))
    if a.json:
        Path(a.json).write_text(json.dumps({"samples": len(sft), "pass": all_ok, "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in rows]}, indent=1) + "\n", encoding="utf-8")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
