"""On-policy edit pairs: the model's own `edit_file` calls at an edit step, labelled by the harness's real matcher, as chosen / rejected pairs for preference training.
docs/PLAN_EDIT_PAIRS.md fixes every rule below before any sample exists.

  python train/edit_pairs.py export-spec                                    # train/prompt_spec_v2rec.json from the captured scorer request (once)
  python train/edit_pairs.py pack-data --data D:/data/edit_sft --out edit_states.zip      # the draw `sample` and `eval` make, 205 MB -> a few MB, for the notebook
  python train/edit_pairs.py sample --data D:/data/edit_sft --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it-qat-w4a16-ct --out /content/pairs_r1
  python train/edit_pairs.py pairs  --samples /content/pairs_r1/samples.jsonl --out /content/pairs_r1
  python train/edit_pairs.py eval   --data D:/data/edit_sft --base-url ... --model gemma-4-e4b-it-qat-w4a16-ct --model arm_dpo --out /content/pairs_eval

What a state is. A prefix of train/make_edit_data.py (issue, one read_file call and its result, the gold edit withheld), re-rendered the way the scorer renders the shipped
agent: the v2rec system text, the six real tool declarations and the real user wrapper (budget, rules, workspace layout), from tests/fixtures/scorer_requests_v2rec.json. (The
samples of D:/data/edit_sft carry an older system prompt and shorter tool descriptions: the prompt of the edit_v1 run, not of the shipped agent.) The prompt is token ids, as
train/s4_gates.py sends them.
How a state is sampled. One full generation (thought and call) at the shipped sampling; the thought it opened with is kept; k-1 more calls are drawn after that same thought
(one request with n = k-1, so the server reuses the prefix). All k calls therefore share the thought, and a preference between two of them is a preference between calls.
A second round (`--round2`) takes one failed call of a state, adds it to the history with the real error text (the harness's own message for the matcher's verdict on the
shown window) and samples the same way at that state: the recovery step.
Labels, from the window the model was shown and the harness's matcher (agent/vendor/scorer_port.py: exact, flexible, regex, unescaped):
  applied      the edit applies (exactly one match, or allow_multiple); `tier` is the matcher's; `changed` the window differs; `localized` the changed lines are within MARGIN
               lines of the gold edit's; `reproduces` the edited window equals the gold-edited window
  not_found | ambiguous | unchanged | empty_old     the tool's own refusals
  malformed    the framework refuses it without running the tool: Python-style keyword arguments (`old_string="..."`), a mandatory argument missing, or a call never closed
  unparsed     the template's reader cannot read the arguments in some other way; the server's parser sometimes can (13 of 38 such calls in the control runs were executed)
  wrong_tool | wrong_file | multi_call | no_call | unparsed     not an edit of the shown file, or not certain: no preference is drawn from them
The window is not the whole file, so uniqueness is checked inside it (optimistic, as in train/edit_bench.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import threading
import time
import urllib.request
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "agent" / "vendor"))
import scorer_port as P  # noqa: E402
import s4_gates as G  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "scorer_requests_v2rec.json"
SPEC_FILE = HERE / "prompt_spec_v2rec.json"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}        # the shipped arm (v2rec)
MARGIN = 3                          # lines: an edit is `localized` when its changed lines come within this many lines of the gold edit's
REQUIRED = ("filepath", "old_string", "new_string")
REJECTED = ("not_found", "ambiguous", "malformed", "unchanged", "empty_old")
FAILED_BY_TOOL = ("not_found", "ambiguous")                   # the refusals whose answer is the matcher's own message: round 2 starts from one of these
THOUGHT_OPEN, THOUGHT_CLOSE, CALL_OPEN, CALL_CLOSE = "<|channel>thought\n", "<channel|>", "<|tool_call>", "<tool_call|>"
PLACEHOLDERS = re.compile(r"(\{repo\}|\{statement\}|\{tree\})")
CALL_NAME = re.compile(r"<\|tool_call>call:([A-Za-z_]\w*)")
KWARG_STYLE = re.compile(r"[{,]\s*(?:filepath|old_string|new_string|allow_multiple)\s*=")


# ---- the prompt spec: what the shipped agent sends -------------------------------------------------------------------------------
def export_spec(fixture: Path = FIXTURE, out: Path = SPEC_FILE) -> dict:
    """System text, user wrapper (with {repo}, {statement}, {tree} where the captured task had its own) and tool declarations of the first captured request."""
    req = json.load(open(fixture, encoding="utf-8"))["requests"][0]
    sysm, user = req["messages"][0]["content"], req["messages"][1]["content"]
    repo = re.search(r"for repository (\S+?)\.\n", user).group(1)
    statement = re.search(r"Problem Statement:\n(.*?)\n\n## Task Budget", user, re.S).group(1)
    tree = re.search(r"\(up to 3 levels\):\n```\n(.*?)\n```", user, re.S).group(1)
    tpl = user.replace(repo, "{repo}", 1).replace(statement, "{statement}", 1).replace(tree, "{tree}", 1)
    assert all(tpl.count(p) == 1 for p in ("{repo}", "{statement}", "{tree}")), "the captured user message does not have the expected shape"
    spec = {"source": f"{fixture.name}, request 0", "enable_thinking": req["chat_template_kwargs"]["enable_thinking"], "system": sysm, "user_template": tpl, "tools": req["tools"]}
    out.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    return spec


def load_spec(path: Path | str | None = None) -> dict:
    p = Path(path) if path else SPEC_FILE
    if not p.exists():
        raise SystemExit(f"{p} not found: run `python train/edit_pairs.py export-spec` (needs {FIXTURE.name})")
    return json.load(open(p, encoding="utf-8"))


def fill(template: str, **values: str) -> str:
    """Substitute {repo}, {statement} and {tree} without str.format (a problem statement is full of braces) and without re-scanning what was inserted."""
    return "".join(values[p[1:-1]] if p in ("{repo}", "{statement}", "{tree}") else p for p in PLACEHOLDERS.split(template))


def tree_of(path: str) -> str:
    """A stand-in for the workspace tree of the user wrapper: the directories down to the edited file (3 levels, as the real listing) plus the README."""
    parts = path.split("/")[:3]
    lines = ["./README.md"]
    for i in range(1, len(parts) + 1):
        lines.append("./" + "/".join(parts[:i]))
    return "\n".join(dict.fromkeys(lines))


def statement_of(rec: dict) -> str:
    return rec["messages"][1]["content"].split("Problem Statement:\n", 1)[1]


def window_of(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "tool":
            return json.loads(m["content"]).get("content", "")
    return ""


def gold_of(rec: dict) -> dict:
    return json.loads(rec["messages"][-1]["tool_calls"][0]["function"]["arguments"])


def prefix_messages(rec: dict, spec: dict) -> list[dict]:
    """The messages up to (not including) the gold edit, with the shipped agent's system text and user wrapper."""
    msgs = copy.deepcopy(rec["messages"][:-1])
    msgs[0] = {"role": "system", "content": spec["system"]}
    msgs[1] = {"role": "user", "content": fill(spec["user_template"], repo=rec["meta"]["repo"], statement=statement_of(rec), tree=tree_of(rec["meta"]["file"]))}
    return msgs


def render_ids(tok, messages: list[dict], spec: dict) -> list[int]:
    import served
    text = served.render(tok, messages, spec["tools"], enable_thinking=spec["enable_thinking"], add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


# ---- diffs and labels -------------------------------------------------------------------------------------------------------------
def changed_span(before: str, after: str):
    """(first, last) changed line of `before` (0-based, inclusive); a pure insertion gives first == last == the line it goes in front of. None when nothing changed."""
    a, b = before.split("\n"), after.split("\n")
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    if i == len(a) == len(b):
        return None
    j = 0
    while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    return i, max(i, len(a) - j - 1)


def near(span, gold_span, margin: int = MARGIN) -> bool:
    return bool(span and gold_span) and span[0] <= gold_span[1] + margin and gold_span[0] <= span[1] + margin


def parse_generation(text: str) -> dict:
    """What the generation is, before the matcher sees it: no_call | multi_call | wrong_tool | malformed | call (with the parsed arguments)."""
    text = text or ""
    opens = text.count(CALL_OPEN)
    if opens == 0:
        return {"status": "no_call"}
    m = CALL_NAME.search(text)
    name = m.group(1) if m else None
    if opens > 1:
        return {"status": "multi_call", "tool": name}
    if name != "edit_file":
        return {"status": "wrong_tool", "tool": name}
    first = G.first_call(text)
    if first is None:
        return {"status": "malformed", "why": "call not closed"}
    args = first[1]
    if args is None:
        # The template's own reader is stricter than the server's tool parser. On the 501 edit calls of the corrected control runs it refuses all 55 calls the framework rejected, and
        # also 14 that the framework ran (3 applied, 11 refused by the matcher). Python keyword arguments (`old_string="..."`) were rejected 32 times in 32; the other styles (JSON
        # quotes after a space, triple quotes, bare values, odd delimiters) 23 times in 37. Only the first is `malformed`; the rest is `unparsed`, which draws no preference.
        if KWARG_STYLE.search(text[text.index(CALL_OPEN):]):
            return {"status": "malformed", "why": "python-style keyword arguments"}
        return {"status": "unparsed", "why": "arguments do not parse"}
    missing = [k for k in REQUIRED if k not in args]
    if missing:
        return {"status": "malformed", "why": "missing " + ",".join(missing)}
    if not all(isinstance(args[k], str) for k in REQUIRED):
        return {"status": "unparsed", "why": "an argument is not a string"}
    return {"status": "call", "args": args}


def label_sample(text: str, window: str, gold: dict, expected: str, gold_span) -> dict:
    """Status and details of one generated call against the window the model was shown."""
    g = parse_generation(text)
    if g["status"] != "call":
        return g
    args = g["args"]
    out = {"status": None, "args": {k: args[k] for k in REQUIRED}}
    if args["filepath"] != gold["filepath"]:
        return {**out, "status": "wrong_file"}
    old, new = args["old_string"], args["new_string"]
    if not old:
        return {**out, "status": "empty_old"}
    if old == new:
        return {**out, "status": "unchanged"}
    res = P.apply_replacement_core(window, old, new, args.get("allow_multiple") is True)
    if res.error_message:
        return {**out, "status": "ambiguous" if res.occurrences > 1 else "not_found", "tier": res.strategy}
    span = changed_span(window, res.new_content)
    return {**out, "status": "applied", "tier": res.strategy, "changed": span is not None, "span": list(span) if span else None,
            "localized": near(span, gold_span), "reproduces": res.new_content == expected}


def is_chosen(s: dict) -> bool:
    return s["status"] == "applied" and s.get("tier") == "exact" and bool(s.get("changed")) and bool(s.get("localized"))


def is_rejected(s: dict) -> bool:
    return s["status"] in REJECTED


def is_mixed(samples: list[dict]) -> bool:
    """A state that can give a pair: a chosen call next to a refused one (or an `unparsed` one, which `pairs --unparsed-as-rejected` can use). Only these keep their prompt ids."""
    return any(is_chosen(s) for s in samples) and any(is_rejected(s) or s["status"] == "unparsed" for s in samples)


def tool_error_message(window: str, args: dict) -> str:
    """The harness's own answer to an edit the matcher refused (agent/vendor/scorer_port.py), for the window as the file."""
    try:
        P.apply_replacement(window, args["old_string"], args["new_string"], filepath=args["filepath"], allow_multiple=args.get("allow_multiple") is True)
    except ValueError as e:
        return str(e)
    raise ValueError("the edit applies: there is no error to report")


def call_text(text: str) -> str:
    """The call of a generation: from the call opener to the closer (inclusive); to the end, minus a trailing stop marker, when it never closes."""
    i = text.find(CALL_OPEN)
    if i < 0:
        return ""
    j = text.find(CALL_CLOSE, i)
    if j >= 0:
        return text[i:j + len(CALL_CLOSE)]
    return text[i:].replace("<|tool_response>", "").rstrip()


def thought_of(head: str) -> str:
    """The reasoning text inside a generation head, as the message field the chat template writes back between the thought markers."""
    if not head.startswith(THOUGHT_OPEN):
        return ""
    body = head[len(THOUGHT_OPEN):]
    return body[:body.index(THOUGHT_CLOSE)] if THOUGHT_CLOSE in body else body


# ---- states -----------------------------------------------------------------------------------------------------------------------
def make_state(rec: dict, spec: dict, tok) -> dict | None:
    msgs = prefix_messages(rec, spec)
    window, gold = window_of(msgs), gold_of(rec)
    res = P.apply_replacement_core(window, gold["old_string"], gold["new_string"], False)
    if res.error_message:
        return None                                              # the gold edit does not apply to its own window: not a usable state
    return {"id": rec["id"], "round": 1, "repo": rec["meta"]["repo"], "file": rec["meta"]["file"], "source": rec["meta"]["source"], "messages": msgs,
            "prompt_ids": render_ids(tok, msgs, spec), "window": window, "gold": gold, "expected": res.new_content, "gold_span": changed_span(window, res.new_content)}


def failure_state(state: dict, head: str, sample: dict, spec: dict, tok) -> dict:
    """The state after `sample` (a refused edit) was made and the harness answered: the history gets the call with its thought and the error."""
    args, tid = sample["args"], f"chatcmpl-tool-{state['id']}-r1"
    thought = thought_of(head)
    msgs = copy.deepcopy(state["messages"])
    msgs.append({"role": "assistant", "tool_calls": [{"type": "function", "id": tid, "function": {"name": "edit_file", "arguments": json.dumps(args)}}],
                 "reasoning_content": thought, "reasoning": thought})
    msgs.append({"role": "tool", "tool_call_id": tid, "content": P.error_response("FileEditError", tool_error_message(state["window"], args))})
    return {**state, "id": state["id"] + "/r2", "round": 2, "messages": msgs, "prompt_ids": render_ids(tok, msgs, spec)}


# ---- the server -------------------------------------------------------------------------------------------------------------------
def complete(base_url: str, model: str, prompt_ids: list[int], n: int, max_tokens: int, seed: int | None = None, timeout: int = 900, tries: int = 3) -> list[tuple[str, str]]:
    body = {"model": model, "prompt": prompt_ids, "n": n, "max_tokens": max_tokens, "skip_special_tokens": False, **SAMPLING}
    if seed is not None:
        body["seed"] = seed
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(f"{base_url}/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            return [(c["text"], c.get("finish_reason") or "") for c in sorted(out["choices"], key=lambda c: c["index"])]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"request failed {tries} times: {last}")


def seed_of(state_id: str, extra: int = 0) -> int:
    return zlib.crc32(state_id.encode()) % (2 ** 31) + extra


def sample_state(a, tok, state: dict) -> dict:
    """k labelled calls that share the thought of the first full generation. Returns the row for samples.jsonl (and `_head`, the thought, for round 2)."""
    row = {"id": state["id"], "round": state["round"], "repo": state["repo"], "file": state["file"], "source": state["source"]}
    first = None
    for attempt in range(3):
        text, finish = complete(a.base_url, a.model, state["prompt_ids"], 1, a.max_tokens_full, seed_of(state["id"], attempt))[0]
        if finish == "length":
            continue
        if CALL_OPEN in text:
            first = text
            break
    if first is None:
        return {**row, "dropped": "no finished generation with a call in 3 tries", "samples": []}
    i = first.index(CALL_OPEN)
    head, calls = first[:i], [first[i:]]
    prompt = state["prompt_ids"] + (tok(head, add_special_tokens=False)["input_ids"] if head else [])
    if a.k > 1:
        for text, finish in complete(a.base_url, a.model, prompt, a.k - 1, a.max_tokens_call, seed_of(state["id"], 99)):
            calls.append(text if finish != "length" else text + "<runaway>")
    samples = []
    for text in calls:
        lab = label_sample(text, state["window"], state["gold"], state["expected"], state["gold_span"])
        if text.endswith("<runaway>"):
            lab = {"status": "malformed", "why": "ran into the length limit", "args": lab.get("args")}
        samples.append({**lab, "call": call_text(text.replace("<runaway>", ""))})
    mixed = is_mixed(samples)
    return {**row, "head": head, "samples": samples, "mixed": mixed, "prompt_ids": prompt if mixed else None, "n_prompt": len(prompt)}


def draw_records(data_dir, split: str, n: int, seed: int) -> list[dict]:
    """The `edit_file` records of one split of train/make_edit_data.py's output, shuffled with `seed`, the first `n`. `sample`, `eval` and `pack-data` all draw this way, so a
    pack holds exactly the states a run on the full files would use."""
    recs = [json.loads(l) for l in open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["format"] == "edit_file"]
    random.Random(seed).shuffle(recs)
    return recs[:n]


def cmd_pack_data(a) -> int:
    import zipfile
    counts = {}
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for split, n in (("train", a.n_train), ("validation", a.n_validation)):
            recs = draw_records(a.data, split, n, a.seed)
            counts[split] = len(recs)
            z.writestr(f"{split}.jsonl", "".join(json.dumps(r) + "\n" for r in recs))
        z.writestr("pack_manifest.json", json.dumps({"seed": a.seed, "records": counts}, indent=1) + "\n")
    print(f"wrote {a.out}: {counts} edit_file records (seed {a.seed}), {Path(a.out).stat().st_size // 1024} KiB")
    return 0


def cmd_sample(a) -> int:
    import served
    tok = served.load_tokenizer(a.tokenizer, verify=not a.no_verify)
    spec = load_spec(a.spec)
    recs = draw_records(a.data, a.split, a.n, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "samples.jsonl"
    prior = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                prior[r["id"]] = r
    lock, stats = threading.Lock(), Counter()
    t0 = time.time()

    def work(rec):
        sid = rec["id"]
        have1, have2 = sid in prior, sid + "/r2" in prior
        if have1 and (have2 or not a.round2):
            return None                                              # a resumed run does not sample a state twice; a crash between the two rounds still gets its round 2
        state = make_state(rec, spec, tok)
        if state is None:
            return None if have1 else [{"id": sid, "round": 1, "dropped": "gold edit does not apply to its window", "samples": []}]
        rows = []
        row1 = prior[sid] if have1 else None
        if row1 is None:
            row1 = sample_state(a, tok, state)
            rows.append(row1)
        if a.round2 and row1.get("samples") and not have2:
            bad = next((s for s in row1["samples"] if s["status"] in FAILED_BY_TOOL), None)
            if bad:
                rows.append(sample_state(a, tok, failure_state(state, row1["head"], bad, spec, tok)))
        return rows

    with ThreadPoolExecutor(a.workers) as ex, open(path, "a", encoding="utf-8") as fh:
        futs = [ex.submit(work, r) for r in recs]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                rows = f.result()
            except Exception as e:  # noqa: BLE001
                stats["error"] += 1
                print("error:", str(e)[:200], flush=True)
                continue
            for row in rows or []:
                with lock:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                stats[f"round{row['round']}"] += 1
                stats["mixed"] += bool(row.get("mixed"))
                stats["dropped"] += "dropped" in row
            if n % 50 == 0:
                print(f"{n}/{len(futs)} prefixes  {dict(stats)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done: {dict(stats)} in {time.time() - t0:.0f}s -> {path}", flush=True)
    return 0


# ---- pairs ------------------------------------------------------------------------------------------------------------------------
def build_pairs(rows: list[dict], tok, max_pairs: int = 2, max_call_tokens: int = 1024, seed: int = 0, unparsed_rejected: bool = False) -> tuple[list[dict], dict]:
    """Pairs under the rules of docs/PLAN_EDIT_PAIRS.md: chosen = applied at the exact tier, changed, localized; rejected = refused by the tool or the framework (`unparsed` calls
    too only with `unparsed_rejected`). Samples of any other status draw no preference. At most `max_pairs` per state, rejected calls taken round-robin over their statuses, no
    call longer than `max_call_tokens` tokens."""
    rejected_ok = lambda s: is_rejected(s) or (unparsed_rejected and s["status"] == "unparsed")      # noqa: E731
    pairs, stats = [], Counter()
    by_round = defaultdict(Counter)
    for row in rows:
        for s in row.get("samples", []):
            by_round[row["round"]][s["status"]] += 1
        if not row.get("mixed"):
            continue
        rng = random.Random(f"{seed}/{row['id']}")
        ok = lambda s: len(tok(s["call"], add_special_tokens=False)["input_ids"]) <= max_call_tokens           # noqa: E731
        chosen, seen = [], set()
        for s in row["samples"]:
            if is_chosen(s) and s["call"] not in seen and ok(s):
                seen.add(s["call"])
                chosen.append(s)
        groups = defaultdict(list)
        for s in row["samples"]:
            if rejected_ok(s) and s["call"] not in seen and ok(s):
                seen.add(s["call"])
                groups[s["status"]].append(s)
        rng.shuffle(chosen)
        rejected = []
        for lst in groups.values():
            rng.shuffle(lst)
        while any(groups.values()):
            for st in sorted(groups):
                if groups[st]:
                    rejected.append(groups[st].pop())
        stats["states_mixed"] += 1
        for j, (c, r) in enumerate(zip(chosen[:max_pairs], rejected[:max_pairs])):
            pairs.append({"id": f"{row['id']}#{j}", "state": row["id"], "round": row["round"], "repo": row["repo"], "file": row["file"], "source": row["source"],
                          "prompt_ids": row["prompt_ids"], "chosen": c["call"], "rejected": r["call"], "chosen_tier": c["tier"], "rejected_status": r["status"],
                          "chosen_reproduces": bool(c.get("reproduces"))})
    lens = {"chosen": [len(tok(p["chosen"], add_special_tokens=False)["input_ids"]) for p in pairs], "rejected": [len(tok(p["rejected"], add_special_tokens=False)["input_ids"]) for p in pairs]}
    mean = lambda v: round(sum(v) / len(v), 1) if v else 0.0                 # noqa: E731
    stats.update({"states": len(rows), "pairs": len(pairs)})
    out = {**dict(stats), "status_by_round": {str(k): dict(v) for k, v in by_round.items()}, "pairs_by_round": dict(Counter(p["round"] for p in pairs)),
           "rejected_status": dict(Counter(p["rejected_status"] for p in pairs)), "chosen_reproduces": sum(p["chosen_reproduces"] for p in pairs),
           "mean_call_tokens": {k: mean(v) for k, v in lens.items()}, "pairs_by_repo": dict(Counter(p["repo"] for p in pairs))}
    return pairs, out


def cmd_pairs(a) -> int:
    import served
    tok = served.load_tokenizer(a.tokenizer, verify=not a.no_verify)
    rows = [json.loads(l) for l in open(a.samples, encoding="utf-8") if l.strip()]
    pairs, stats = build_pairs(rows, tok, a.max_pairs, a.max_call_tokens, a.seed, a.unparsed_as_rejected)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    json.dump(stats, open(out / "pairs_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    return 0


# ---- evaluation on held-out prefixes -----------------------------------------------------------------------------------------------
def summarize_eval(rows: list[dict]) -> dict:
    n = len(rows)
    pct = lambda k: round(100 * k / n, 1) if n else 0.0                      # noqa: E731
    c = Counter(r["status"] for r in rows)
    applied = [r for r in rows if r["status"] == "applied"]
    return {"n": n, "edit_call_%": pct(sum(r["status"] not in ("no_call", "wrong_tool", "multi_call") for r in rows)),
            "good_%": pct(sum(is_chosen(r) for r in rows)), "applied_any_tier_%": pct(len(applied)), "applied_exact_%": pct(sum(r.get("tier") == "exact" for r in applied)),
            "reproduces_%": pct(sum(bool(r.get("reproduces")) for r in applied)), "not_found_%": pct(c["not_found"]), "ambiguous_%": pct(c["ambiguous"]),
            "malformed_%": pct(c["malformed"]), "wrong_tool_%": pct(c["wrong_tool"]), "no_call_%": pct(c["no_call"]), "wrong_file_%": pct(c["wrong_file"]),
            "localized_given_applied_%": round(100 * sum(bool(r.get("localized")) for r in applied) / len(applied), 1) if applied else 0.0, "status": dict(c)}


def paired_bootstrap(base: dict[str, list[int]], other: dict[str, list[int]], iters: int = 2000, seed: int = 0) -> dict:
    """Mean difference (other - base) of a 0/1 outcome, resampling prefixes (the unit: the samples of one prefix are not independent). 95% interval."""
    ids = sorted(base.keys() & other.keys())
    d = [sum(other[i]) / len(other[i]) - sum(base[i]) / len(base[i]) for i in ids]
    if not d:
        return {"n_prefixes": 0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(iters))
    return {"n_prefixes": len(d), "diff_points": round(100 * sum(d) / len(d), 1), "ci95_points": [round(100 * means[int(.025 * iters)], 1), round(100 * means[int(.975 * iters) - 1], 1)]}


def cmd_eval(a) -> int:
    import served
    tok = served.load_tokenizer(a.tokenizer, verify=not a.no_verify)
    spec = load_spec(a.spec)
    states = [s for s in (make_state(r, spec, tok) for r in draw_records(a.data, a.split, a.n, a.seed)) if s]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result, per_prefix = {}, {}
    for model in a.model:
        def one(st):
            try:
                gens = complete(a.base_url, model, st["prompt_ids"], a.samples, a.max_tokens_full, seed_of(st["id"], 7))
            except Exception as e:  # noqa: BLE001
                return st["id"], [{"status": "request_error", "error": str(e)[:200]}]
            return st["id"], [label_sample(t[t.index(CALL_OPEN):] if CALL_OPEN in t else t, st["window"], st["gold"], st["expected"], st["gold_span"]) for t, _ in gens]
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, states))
        rows = [dict(r, prefix=i) for i, rs in res for r in rs]
        with open(out / f"{model}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "args"}) + "\n")
        result[model] = summarize_eval([r for r in rows if r["status"] != "request_error"])
        result[model]["request_errors"] = sum(r["status"] == "request_error" for r in rows)
        per_prefix[model] = {i: [int(is_chosen(r)) for r in rs if r["status"] != "request_error"] or [0] for i, rs in res}
        print(model, json.dumps({k: v for k, v in result[model].items() if k != "status"}), flush=True)
    ref = a.model[0]
    for m in a.model[1:]:
        result[m]["good_vs_" + ref] = paired_bootstrap(per_prefix[ref], per_prefix[m])
        print(f"{m} good-edit rate against {ref}:", result[m]["good_vs_" + ref], flush=True)
    json.dump(result, open(out / "eval_summary.json", "w"), indent=1)
    return 0


PRIMARY_MIN_GAIN = 10.0              # points of good_%, with the 95% interval above 0
GUARD_REPRODUCES_DROP = 2.0          # points
GUARD_EDIT_CALL_BAND = 10.0          # points, either direction
GATES = ("G3", "G4", "G4b", "G5")


def decide(evals: dict, gates: dict | None, ref: str) -> dict:
    """The pre-registered reading (docs/PLAN_EDIT_PAIRS.md, section 3) of `eval_summary.json` and train/s4_gates.py's `summary.json`: for every model but `ref`, the primary outcome, the
    three guards, the four gate verdicts, and which row of the plan's outcome table applies. Nothing here is tuned after seeing a result."""
    base, out = evals[ref], {}
    for m, s in evals.items():
        if m == ref:
            continue
        cmp_ = s.get(f"good_vs_{ref}") or {}
        diff, ci = cmp_.get("diff_points"), cmp_.get("ci95_points")
        primary = diff is not None and diff >= PRIMARY_MIN_GAIN and ci[0] > 0
        guards = {"reproduces": s["reproduces_%"] >= base["reproduces_%"] - GUARD_REPRODUCES_DROP,
                  "edit_call": abs(s["edit_call_%"] - base["edit_call_%"]) <= GUARD_EDIT_CALL_BAND,
                  "malformed": s["malformed_%"] <= base["malformed_%"]}
        verdicts = {g: G.verdict(f) for g, f in ((gates or {}).get(m) or {}).get("gates", {}).items()}
        failed = [f"guard {k}" for k, ok in guards.items() if not ok] + [f"gate {g} {v}" for g, v in verdicts.items() if v != "PASS"] + [f"gate {g} missing" for g in GATES if g not in verdicts]
        if not primary:
            outcome = "primary not met"
        elif failed:
            outcome = "primary met, but " + ", ".join(failed)
        else:
            outcome = "primary met, guards and gates pass"
        out[m] = {"primary": primary, "diff_points": diff, "ci95_points": ci, "guards": guards, "gates": verdicts, "outcome": outcome, "agent_run_allowed": primary and not failed}
    return out


def cmd_decide(a) -> int:
    evals = json.load(open(a.eval, encoding="utf-8"))
    gates = json.load(open(a.gates, encoding="utf-8")) if a.gates else None
    res = decide(evals, gates, a.ref)
    for m, r in res.items():
        print(f"{m}: good-edit rate {r['diff_points']} points against {a.ref} (95% interval {r['ci95_points']}) | guards {r['guards']} | gates {r['gates']}\n    -> {r['outcome']}"
              + ("  (an agent-level run is allowed)" if r["agent_run_allowed"] else ""))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export-spec", help="write train/prompt_spec_v2rec.json from the captured scorer request")
    d = sub.add_parser("decide", help="apply the pre-registered outcome rules to eval_summary.json (and s4_gates' summary.json)")
    d.add_argument("--eval", required=True)
    d.add_argument("--gates", help="summary.json of train/s4_gates.py")
    d.add_argument("--ref", required=True, help="the base model's served name (the first --model of both runs)")
    d.add_argument("--out")
    k = sub.add_parser("pack-data", help="zip the records `sample` and `eval` would draw (a few MB instead of the 205 MB of the full files)")
    k.add_argument("--data", required=True, help="train/make_edit_data.py output (train.jsonl, validation.jsonl)")
    k.add_argument("--out", required=True)
    k.add_argument("--n-train", type=int, default=2400)
    k.add_argument("--n-validation", type=int, default=300)
    k.add_argument("--seed", type=int, default=0)

    def common(p):
        p.add_argument("--tokenizer", help="directory of the checkpoint (default: train/served.py's)")
        p.add_argument("--no-verify", action="store_true", help="do not check the chat template's hash (tests)")
        p.add_argument("--spec", help="prompt spec (default: train/prompt_spec_v2rec.json)")

    def server(p):
        p.add_argument("--data", required=True, help="train/make_edit_data.py output (train.jsonl, validation.jsonl)")
        p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=24)
        p.add_argument("--max-tokens-full", type=int, default=6144, help="a thought that does not end in a call within this is dropped (the scorer's thinking budget is 4,096)")
        p.add_argument("--out", required=True)
    s = sub.add_parser("sample", help="sample k calls per state and label them")
    common(s)
    server(s)
    s.add_argument("--model", required=True)
    s.add_argument("--split", default="train")
    s.add_argument("--n", type=int, default=2400, help="prefixes")
    s.add_argument("--k", type=int, default=8, help="calls per state")
    s.add_argument("--max-tokens-call", type=int, default=2048)
    s.add_argument("--round2", action="store_true", help="also sample the state after one refused edit")
    p = sub.add_parser("pairs", help="build chosen/rejected pairs from samples.jsonl")
    common(p)
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-pairs", type=int, default=2)
    p.add_argument("--max-call-tokens", type=int, default=1024)
    p.add_argument("--unparsed-as-rejected", action="store_true", help="also use calls the template's reader cannot read but that are not Python-style (the server ran 13 of 38 in the control runs)")
    p.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval", help="good-edit rate of one or more served models on held-out prefixes (the first --model is the reference)")
    common(e)
    server(e)
    e.add_argument("--model", action="append", required=True)
    e.add_argument("--split", default="validation")
    e.add_argument("--n", type=int, default=300)
    e.add_argument("--samples", type=int, default=2, help="independent generations per prefix")
    a = ap.parse_args(argv)
    if a.cmd == "export-spec":
        spec = export_spec()
        print(f"wrote {SPEC_FILE} ({len(spec['system'])} chars of system text, {len(spec['tools'])} tools)")
        return 0
    return {"sample": cmd_sample, "pairs": cmd_pairs, "eval": cmd_eval, "pack-data": cmd_pack_data, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
