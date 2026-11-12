"""S4: the offline gates G3, G4 and G5 of docs/PLAN_E4B_SFT.md for one or more adapters, against the base model, on prefixes of the base model's own runs.

  python train/s4_gates.py --check D:/data/s4/check.jsonl --failed-edit D:/data/s4/check_failed_edit.jsonl --base-url http://127.0.0.1:8000/v1 \\
      --model gemma-4-e4b-it-qat-w4a16-ct --model e4b_sft_v2 --out /content/s4

The first --model is the reference (the base model); every other one is judged against it. Both check files come from train/build_replay_data.py (runs of the
corrected harness). Each prompt is sent to /v1/completions as token ids, so nothing is re-rendered, with the sampling of the shipped arm (v2rec: temperature 1.0,
top_p 0.95, top_k 64, 8,192 tokens).

  G3  step types: for every tool the base model chose in a recorded run (edit_file excepted) the adapter's agreement is within 10 points of the base's, its share
      of "no call" is at most 5 points above the base's, request errors at most 2% (train/step_check.py, unchanged).
  G4  after a failed edit: the prefixes end in the result of a failed edit_file/write_file. repeat = the next call is the failed call again (same tool and arguments;
      for edit_file the same file and the same old_string, because an anchor that was not found is not found again); re-read = the next call is read_file.
      Relative to the base model's rates on the same prefixes in the same run: repeat at most 8 points above the base's, re-read at most 8 points below it. (The plan's
      first draft said "repeat at most 12%"; that figure is the share of ALL edit calls that repeat the call before, 135 of 1,077. After a failed edit the base model repeats
      about 30% of the time on the old traces, so an absolute 12% would fail the base itself.)
  G4b the second part of G4 (docs/PLAN_E4B_SFT.md, note of 2026-11-07), on the G4 prefixes whose prompt shows no applied edit (`applied_edit_visible` False, written by
      train/build_replay_data.py): the share of generations whose first call is submit_patch at most the base's + 5 points, the share whose first call is a re-read or
      another (not identical) edit at least the base's - 10 points. G4 counts only repeat and re-read, so it was blind to giving up (e4b_sft_v2 passed it).
  G5  format: the share of generations with exactly one well-formed tool call, the share that ran into the length limit, the mean tokens per generation. Reported, not
      gated: the share of generations that open with a thought channel.

The definitions and thresholds are fixed in docs/PLAN_E4B_SFT.md, amendment 2026-11-06 (S4), written before any adapter had produced an output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step_check as SC  # noqa: E402

# tool -> (required, optional) parameters with their python types. tests/test_s4_gates.py checks the table against the declarations the harness sends.
TOOLS = {
    "read_file": ({"filepath": str}, {"start_line": int, "end_line": int}),
    "edit_file": ({"filepath": str, "old_string": str, "new_string": str}, {"allow_multiple": bool}),
    "write_file": ({"filepath": str, "content": str}, {}),
    "run_command": ({"command": str}, {}),
    "get_status": ({}, {}),
    "submit_patch": ({}, {}),
}
QUOTE = '<|"|>'
CALL_BLOCK = re.compile(r"<\|tool_call>call:([A-Za-z_]\w*)\{(.*?)\}<tool_call\|>", re.S)
KEY = re.compile(r"([A-Za-z_]\w*):")
INT = re.compile(r"-?\d+$")
FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?$")

THOUGHT = "<|channel>thought"
MIN_G3_GENERATIONS = 100      # step_check.gate skips a tool with fewer than 10 prefixes; with too few of them it would pass without having compared anything
MIN_G3_TOOLS = 3              # tools other than edit_file with at least 10 prefixes
MIN_G4_PREFIXES = 40          # fewer failed-edit prefixes than this say nothing about a rate
MIN_G5_GENERATIONS = 300      # the plan: "parse 300 generations"
G4_TOLERANCE = 8.0            # points. On the old traces the task-clustered standard error of one rate is 2-4 points (324 prefixes, 54 tasks), so two samplings of the
                              # same model differ by about 4; 8 points is two standard errors of the difference. It catches a collapse (edit_v1 repeated 89%), not a drift.
WELL_FORMED_MIN, LENGTH_HITS_MAX, ERROR_MAX = 98.0, 2.0, 2.0
G4B_SUBMIT_TOLERANCE, G4B_RECOVER_TOLERANCE = 5.0, 10.0    # points; fixed in the plan's note of 2026-11-07 (G4b), same INSUFFICIENT rule as G4 (MIN_G4_PREFIXES)


def parse_args(body: str):
    """The arguments of a call in the template's syntax (`key:<|"|>text<|"|>,key:12,key:true`) as a dict, or None if the text does not parse."""
    args, i, n = {}, 0, len(body)
    while i < n:
        m = KEY.match(body, i)
        if not m:
            return None
        key, i = m.group(1), m.end()
        if body.startswith(QUOTE, i):
            j = body.find(QUOTE, i + len(QUOTE))
            if j < 0:
                return None
            args[key], i = body[i + len(QUOTE):j], j + len(QUOTE)
        else:
            j = body.find(",", i)
            j = n if j < 0 else j
            raw, i = body[i:j], j
            if raw in ("true", "false"):
                args[key] = raw == "true"
            elif raw == "null":
                args[key] = None
            elif INT.match(raw):
                args[key] = int(raw)
            elif FLOAT.match(raw):
                args[key] = float(raw)
            else:
                return None
        if i < n:
            if body[i] != ",":
                return None
            i += 1
            if i >= n:                                   # a trailing comma
                return None
    return args


def first_call(text: str):
    """(name, arguments or None) of the first complete call in a generation, or None if there is none."""
    m = CALL_BLOCK.search(text or "")
    return (m.group(1), parse_args(m.group(2))) if m else None


def well_formed(text: str) -> tuple[bool, str]:
    """(ok, why not): exactly one complete call, a declared tool, arguments that parse, the required ones present, no undeclared ones, the declared types."""
    text = text or ""
    opens = text.count("<|tool_call>")
    if opens != 1:
        return False, f"{opens} tool calls"
    blocks = CALL_BLOCK.findall(text)
    if len(blocks) != 1:
        return False, "call not closed"
    name, body = blocks[0]
    if name not in TOOLS:
        return False, f"undeclared tool {name}"
    args = parse_args(body)
    if args is None:
        return False, "arguments do not parse"
    req, opt = TOOLS[name]
    if set(req) - set(args):
        return False, "missing " + ",".join(sorted(set(req) - set(args)))
    if set(args) - set(req) - set(opt):
        return False, "undeclared parameter " + ",".join(sorted(set(args) - set(req) - set(opt)))
    for k, v in args.items():
        typ = {**req, **opt}[k]
        if not isinstance(v, typ) or (typ is int and isinstance(v, bool)):
            return False, f"{k} is not {typ.__name__}"
    return True, ""


def same_call(a, b) -> bool:
    """Is call b a repeat of the failed call a? edit_file: the same file and the same old_string (whatever the replacement says, an anchor that was not found is not
    found again). Every other tool: the same arguments."""
    if not a or not b or a[0] != b[0] or a[1] is None or b[1] is None:
        return False
    if a[0] == "edit_file":
        return a[1].get("filepath") == b[1].get("filepath") and a[1].get("old_string") == b[1].get("old_string")
    return a[1] == b[1]


def next_kind(text: str, failed: dict) -> str:
    """What the first call generated after a failed edit is: repeat | reread | edit (another edit) | other (another tool) | no_call | malformed."""
    c = first_call(text)
    if c is None:
        return "malformed" if "<|tool_call>" in (text or "") else "no_call"
    name, args = c
    if args is None or name not in TOOLS:
        return "malformed"
    if same_call((failed["name"], failed["args"]), (name, args)):
        return "repeat"
    if name == "read_file":
        return "reread"
    if name in ("edit_file", "write_file"):
        return "edit"
    return "other"


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def g3_gate(base: dict, other: dict) -> list[str]:
    """train/step_check.gate plus a floor on the data: it only compares a tool that has at least 10 check prefixes, so a small check file would pass untested."""
    fails = SC.gate(base, other)
    if any(f.startswith("INVALID") for f in fails):
        return fails
    gated = [t for t, b in base["by_recorded_tool"].items() if t != "edit_file" and b["n"] >= 10 and t in other["by_recorded_tool"]]
    if base["n"] < MIN_G3_GENERATIONS or len(gated) < MIN_G3_TOOLS:
        fails.append(f"INSUFFICIENT: {base['n']} generations and {len(gated)} gated tools with at least 10 prefixes (need {MIN_G3_GENERATIONS} and {MIN_G3_TOOLS})")
    return fails


def clustered_se(rows: list[dict], kind: str) -> float:
    """Standard error, in percentage points, of the share of rows of this kind. The task is the unit of resampling: the prefixes and the samples of one task are not
    independent. 0.0 with fewer than two tasks (nothing to estimate it from)."""
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        c = by[r["instance_id"]]
        c[0] += r["kind"] == kind
        c[1] += 1
    m, tot = len(by), len(rows)
    if m < 2:
        return 0.0
    p = sum(c[0] for c in by.values()) / tot
    var = sum((c[0] - p * c[1]) ** 2 for c in by.values()) / tot ** 2 * m / (m - 1)
    return round(100 * var ** 0.5, 1)


def g4_stats(rows: list[dict], prefixes: int) -> dict:
    """Shares are of the generations that came back; a request error counts toward request_error_% only."""
    valid = [r for r in rows if r["kind"] != "request_error"]
    n = len(valid)
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("repeat", "reread", "edit", "other", "no_call", "malformed", "request_error")}
    return {"prefixes": prefixes, "n": n, "tasks": len({r["instance_id"] for r in valid}), "repeat_%": pct(kinds["repeat"], n), "reread_%": pct(kinds["reread"], n),
            "repeat_se": clustered_se(valid, "repeat"), "reread_se": clustered_se(valid, "reread"),
            "request_error_%": pct(kinds["request_error"], len(rows)), "kinds": kinds}


def g4_gate(base: dict, other: dict) -> list[str]:
    """Failures of `other` against `base`, the base model's rates on the same prefixes: repeat more than G4_TOLERANCE points above the base's, or re-read more than
    G4_TOLERANCE points below it."""
    if base["request_error_%"] > ERROR_MAX:
        return [f"INVALID: the base model had {base['request_error_%']}% request errors, so nothing was compared"]
    if other["request_error_%"] > ERROR_MAX:
        return [f"INVALID: {other['request_error_%']}% request errors"]
    if base["prefixes"] < MIN_G4_PREFIXES:
        return [f"INSUFFICIENT: {base['prefixes']} failed-edit prefixes, fewer than {MIN_G4_PREFIXES}"]
    fails = []
    if other["repeat_%"] > base["repeat_%"] + G4_TOLERANCE:
        fails.append(f"repeat {other['repeat_%']}% against {base['repeat_%']}% for the base (tolerance {G4_TOLERANCE} points)")
    if other["reread_%"] < base["reread_%"] - G4_TOLERANCE:
        fails.append(f"re-read {other['reread_%']}% against {base['reread_%']}% for the base (tolerance {G4_TOLERANCE} points)")
    return fails


def g4b_stats(rows: list[dict]) -> dict:
    """G4b: the generations on the failed-edit prefixes that show no applied edit (`applied_visible` False). submit = the first call is submit_patch (very likely an empty
    patch); recover = the first call is a re-read or another edit that is not identical to the failed one. `available` is False when no row carries the flag (a check
    file built before train/build_replay_data.py wrote it)."""
    sub = [r for r in rows if r.get("applied_visible") is False]
    valid = [r for r in sub if r["kind"] != "request_error"]
    n = len(valid)
    marked = [{"instance_id": r["instance_id"], "kind": "submit" if r.get("first_tool") == "submit_patch" else "recover" if r["kind"] in ("reread", "edit") else "other"} for r in valid]
    return {"available": any(r.get("applied_visible") is not None for r in rows), "prefixes": len({r.get("prefix") or (r["instance_id"], r.get("turn")) for r in sub}), "n": n,
            "tasks": len({r["instance_id"] for r in valid}), "submit_%": pct(sum(m["kind"] == "submit" for m in marked), n), "recover_%": pct(sum(m["kind"] == "recover" for m in marked), n),
            "submit_se": clustered_se(marked, "submit"), "recover_se": clustered_se(marked, "recover"), "request_error_%": pct(len(sub) - n, len(sub))}


def g4b_gate(base: dict, other: dict) -> list[str]:
    """Failures of `other` against `base` on the same prefixes: submit more than G4B_SUBMIT_TOLERANCE points above the base's, or recover more than G4B_RECOVER_TOLERANCE
    points below it."""
    if not base["available"] or not other["available"]:
        return ["INSUFFICIENT: the failed-edit prefixes lack applied_edit_visible (rebuild the check files with train/build_replay_data.py of 2026-11-07 or later)"]
    if base["request_error_%"] > ERROR_MAX:
        return [f"INVALID: the base model had {base['request_error_%']}% request errors, so nothing was compared"]
    if other["request_error_%"] > ERROR_MAX:
        return [f"INVALID: {other['request_error_%']}% request errors"]
    if base["prefixes"] < MIN_G4_PREFIXES:
        return [f"INSUFFICIENT: {base['prefixes']} failed-edit prefixes without an applied edit, fewer than {MIN_G4_PREFIXES}"]
    fails = []
    if other["submit_%"] > base["submit_%"] + G4B_SUBMIT_TOLERANCE:
        fails.append(f"submit {other['submit_%']}% against {base['submit_%']}% for the base (tolerance {G4B_SUBMIT_TOLERANCE} points)")
    if other["recover_%"] < base["recover_%"] - G4B_RECOVER_TOLERANCE:
        fails.append(f"recover (re-read or another edit) {other['recover_%']}% against {base['recover_%']}% for the base (tolerance {G4B_RECOVER_TOLERANCE} points)")
    return fails


def g5_stats(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["finish"] != "error"]
    return {"n": n, "well_formed_%": pct(sum(r["well_formed"] for r in ok), len(ok)), "length_%": pct(sum(r["finish"] == "length" for r in ok), len(ok)),
            "thought_%": pct(sum(THOUGHT in r.get("text", "") for r in ok), len(ok)),
            "mean_tokens": round(sum(r["tokens"] for r in ok) / len(ok)) if ok else 0, "request_error_%": pct(n - len(ok), n),
            "reasons": {k: sum(1 for r in ok if not r["well_formed"] and r.get("why", "").startswith(k)) for k in
                        sorted({r.get("why", "").split(" ")[0] for r in ok if not r["well_formed"]})}}


def g5_gate(s: dict) -> list[str]:
    if s["request_error_%"] > ERROR_MAX:
        return [f"INVALID: {s['request_error_%']}% request errors"]
    if s["n"] < MIN_G5_GENERATIONS:
        return [f"INSUFFICIENT: {s['n']} generations, fewer than {MIN_G5_GENERATIONS}"]
    fails = []
    if s["well_formed_%"] < WELL_FORMED_MIN:
        fails.append(f"well-formed {s['well_formed_%']}% below {WELL_FORMED_MIN}%")
    if s["length_%"] > LENGTH_HITS_MAX:
        fails.append(f"length-limit hits {s['length_%']}% above {LENGTH_HITS_MAX}%")
    return fails


def evaluate(models: list[str], g3_rows: dict, g4_rows: dict, g4_prefixes: int) -> dict:
    """Summaries of every model and, for every model but the first, the failures of each gate (an empty list = pass)."""
    out = {m: {"G3": SC.summarize(g3_rows[m]) if g3_rows[m] else None, "G4": g4_stats(g4_rows[m], g4_prefixes), "G4b": g4b_stats(g4_rows[m]), "G5": g5_stats(g3_rows[m] + g4_rows[m])} for m in models}
    ref = models[0]
    for m in models[1:]:
        out[m]["gates"] = {"G3": g3_gate(out[ref]["G3"], out[m]["G3"]) if out[ref]["G3"] and out[m]["G3"] else ["INSUFFICIENT: no G3 prefixes"],
                           "G4": g4_gate(out[ref]["G4"], out[m]["G4"]), "G4b": g4b_gate(out[ref]["G4b"], out[m]["G4b"]), "G5": g5_gate(out[m]["G5"])}
    return out


def verdict(fails: list[str]) -> str:
    """INVALID (requests failed, nothing was compared) > FAIL (a comparison that had enough data came out against the adapter) > INSUFFICIENT (too little data) > PASS."""
    if any(f.startswith("INVALID") for f in fails):
        return "INVALID"
    if any(not f.startswith("INSUFFICIENT") for f in fails):
        return "FAIL"
    return "INSUFFICIENT" if fails else "PASS"


def report(summary: dict, models: list[str]) -> str:
    def g4(s):
        return (f"repeat {s['repeat_%']}% (+-{s['repeat_se']}) re-read {s['reread_%']}% (+-{s['reread_se']}) on {s['prefixes']} prefixes from {s['tasks']} tasks, "
                f"{s['n']} generations")
    lines = []
    ref = summary[models[0]]
    lines.append(f"reference {models[0]}: G3 n={ref['G3']['n'] if ref['G3'] else 0} | G4 {g4(ref['G4'])} | "
                 f"G5 well-formed {ref['G5']['well_formed_%']}% length hits {ref['G5']['length_%']}% thought {ref['G5']['thought_%']}% mean tokens {ref['G5']['mean_tokens']}")
    for m in models[1:]:
        s, g = summary[m], summary[m]["gates"]
        lines.append(f"{m}:")
        lines.append(f"  G3 step types  {verdict(g['G3']):12s} " + ("; ".join(g["G3"]) or "within the base's range on every gated tool"))
        lines.append(f"  G4 after a failed edit  {verdict(g['G4']):12s} {g4(s['G4'])}" + ("  -> " + "; ".join(g["G4"]) if g["G4"] else ""))
        b, rb = s["G4b"], ref["G4b"]
        lines.append(f"  G4b no applied edit     {verdict(g['G4b']):12s} " + (f"submit {b['submit_%']}% (+-{b['submit_se']}) recover {b['recover_%']}% (+-{b['recover_se']}) on {b['prefixes']} prefixes, {b['n']} generations "
                                                                             f"(base submit {rb['submit_%']}%, recover {rb['recover_%']}%)" if b["available"] else "no applied_edit_visible flag in the check file")
                     + ("  -> " + "; ".join(g["G4b"]) if g["G4b"] else ""))
        lines.append(f"  G5 format      {verdict(g['G5']):12s} well-formed {s['G5']['well_formed_%']}%, length hits {s['G5']['length_%']}%, mean tokens {s['G5']['mean_tokens']} "
                     f"(base {ref['G5']['mean_tokens']}), thought channel in {s['G5']['thought_%']}% (base {ref['G5']['thought_%']}%)"
                     + ("  -> " + "; ".join(g["G5"]) if g["G5"] else ""))
    return "\n".join(lines)


def generate(base_url: str, model: str, prompt_ids: list[int], a) -> dict:
    try:
        text, finish, ntok = SC.complete(base_url, model, prompt_ids, a.temperature, a.top_p, a.max_tokens, top_k=a.top_k)
        ok, why = well_formed(text)
        return {"text": text, "finish": finish, "tokens": ntok, "well_formed": ok, "why": why}
    except Exception as e:  # noqa: BLE001
        return {"text": "", "finish": "error", "tokens": 0, "well_formed": False, "why": "request error", "error": str(e)[:200]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", required=True, help="check.jsonl of train/build_replay_data.py (G3 prefixes)")
    ap.add_argument("--failed-edit", required=True, help="check_failed_edit.jsonl of train/build_replay_data.py (G4 prefixes)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", action="append", required=True, help="first one is the reference (the base model)")
    ap.add_argument("--n", type=int, default=0, help="use at most this many G3 prefixes (0 = all)")
    ap.add_argument("--n-failed", type=int, default=0, help="use at most this many G4 prefixes (0 = all)")
    ap.add_argument("--samples", type=int, default=2, help="generations per G3 prefix")
    ap.add_argument("--g4-samples", type=int, default=3, help="generations per G4 prefix")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    g3 = [json.loads(l) for l in open(a.check, encoding="utf-8") if l.strip()]
    g4 = [json.loads(l) for l in open(a.failed_edit, encoding="utf-8") if l.strip()]
    g3, g4 = (g3[:a.n] if a.n else g3), (g4[:a.n_failed] if a.n_failed else g4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g3_rows, g4_rows = {}, {}
    for model in a.model:
        def one3(t):
            g = generate(a.base_url, model, t["prompt_ids"], a)
            return {"instance_id": t["instance_id"], "turn": t["turn"], "recorded": t["tool"], "generated": "request_error" if g["finish"] == "error" else SC.classify(g["text"]),
                    "finish": g["finish"], "tokens": g["tokens"], "well_formed": g["well_formed"], "why": g["why"], "text": g["text"][:20000]}

        def one4(t):
            g = generate(a.base_url, model, t["prompt_ids"], a)
            kind = "request_error" if g["finish"] == "error" else next_kind(g["text"], t["failed_call"])
            first = first_call(g["text"])
            return {"instance_id": t["instance_id"], "turn": t["turn"], "prefix": f'{t.get("run", "")}|{t["instance_id"]}|{t["turn"]}', "recorded": t["tool"],
                    "failed": t["failed_call"]["name"], "kind": kind, "first_tool": first[0] if first else None, "applied_visible": t.get("applied_edit_visible"),
                    "finish": g["finish"], "tokens": g["tokens"], "well_formed": g["well_formed"], "why": g["why"], "text": g["text"][:20000]}
        with ThreadPoolExecutor(a.workers) as ex:
            g3_rows[model] = list(ex.map(one3, [t for t in g3 for _ in range(a.samples)]))
            g4_rows[model] = list(ex.map(one4, [t for t in g4 for _ in range(a.g4_samples)]))
        for name, rows in (("g3", g3_rows[model]), ("g4", g4_rows[model])):
            with open(out / f"{model}_{name}.jsonl", "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        print(f"{model}: {len(g3_rows[model])} G3 and {len(g4_rows[model])} G4 generations", flush=True)
    summary = evaluate(a.model, g3_rows, g4_rows, len(g4))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(report(summary, a.model))
    verdicts = [verdict(f) for m in a.model[1:] for f in summary[m]["gates"].values()]
    return 2 if "INVALID" in verdicts else (1 if any(v != "PASS" for v in verdicts) else 0)


if __name__ == "__main__":
    sys.exit(main())
