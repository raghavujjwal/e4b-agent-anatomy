"""CPU-only census of nebius/SWE-rebench-openhands-trajectories (OpenHands v0.54, Qwen3-Coder-480B).

  python -I census_nebius.py --parquet D:/data/ext_traj/nebius_openhands/trajectories.parquet --out D:/data/ext_traj/census [--limit N]
  python -I census_nebius.py --jsonl D:/data/ext_traj/sample/first600.jsonl --out D:/data/ext_traj/census_test

Writes into --out:
  census.parquet   one row per trajectory: tool-call counts by type, edit failures, grounding of edits, git-history use, patch shape,
                   and the CRITICAL-PATH estimate (cond_*): how many calls remain if only the calls that found, showed and changed the
                   edited files (plus one verification) are kept
  examples.jsonl   a few full trajectories that pass the critical-path filter, for inspecting formats and developing the converter
  progress.log     one line per row group

Nothing here trains or converts; it only measures how much usable data there is under each filter.
"""
from __future__ import annotations

import argparse
import collections
import json
import posixpath
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CAT_N = re.compile(r"^\s*\d+\t", re.M)
GIT_HIST = re.compile(r"\bgit\s+(?:--no-pager\s+)?(?:log|show|reflog|blame|rev-list|shortlog|describe)\b")
GIT_RISKY = re.compile(
    r"\bgit\s+(?:--no-pager\s+)?(?:fetch|pull|cherry-pick|stash|reset|ls-remote|remote\b|branch\s+-a|checkout\s+(?!--)[^\s-]\S*|log\b[^|;&\n]*--all)")
TEST_RUN = re.compile(r"\b(pytest|py\.test|unittest|tox|nosetests|runtests|manage\.py\s+test)\b")
VERIF_CMD = re.compile(r"\b(pytest|py\.test|python3?\s+-m\s+pytest|python3?\s+-c|unittest)\b")
PIP_INSTALL = re.compile(r"\bpip\d?\s+install\b")
EDIT_FAIL = ("No replacement was performed", "did not appear verbatim", "Multiple occurrences of old_str")
DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
TEST_PATH = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$")
ISSUE = re.compile(r"<issue_description>\s*(.*?)\s*</issue_description>", re.S)
REL = re.compile(r"^/workspace/[^/]+/(.*)$")
SCRATCH_NAME = re.compile(r"(reproduc|repro|test_|_test|check|debug|verif|demo|example|script|fix|simple|temp|tmp|comprehensive|final|quick)", re.I)
DOC_FILE = re.compile(r"\.(md|rst|txt)$", re.I)
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]{3,}")
STOP_TOKENS = {"grep", "include", "find", "name", "head", "exec", "print", "type", "workspace", "xargs", "sort", "wc"}
COLS = ["trajectory_id", "instance_id", "repo", "trajectory", "model_patch", "exit_status", "resolved",
        "gen_tests_correct", "pred_passes_gen_tests"]


def patch_stats(p: str) -> dict:
    files = [b for _, b in DIFF_FILE.findall(p or "")]
    lines = (p or "").splitlines()
    return {
        "p_files": len(files),
        "p_added": sum(1 for l in lines if l.startswith("+") and not l.startswith("+++")),
        "p_removed": sum(1 for l in lines if l.startswith("-") and not l.startswith("---")),
        "p_test_files": sum(1 for f in files if TEST_PATH.search(f)),
        "p_nonpy_files": sum(1 for f in files if not f.endswith(".py")),
        "p_paths": files,
    }


def rel_path(p: str) -> str:
    m = REL.match(p or "")
    return m.group(1) if m else (p or "").lstrip("/")


def build_events(tr: list[dict]) -> tuple[list[dict], collections.Counter, int, int]:
    """One event per tool call, in order, with its result text filled in from the matching tool message."""
    events: list[dict] = []
    by_id: dict[str, dict] = {}
    c: collections.Counter = collections.Counter()
    chars_total = chars_tool = 0
    for m in tr:
        role = m.get("role")
        content = m.get("content") or ""
        chars_total += len(content)
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if content.strip():
                c["asst_text_msgs"] += 1
                c["asst_text_chars"] += len(content)
            if len(tcs) > 1:
                c["multi_call_msgs"] += 1
            for tc in tcs:
                f = tc.get("function") or {}
                raw = f.get("arguments") or ""
                chars_total += len(raw)
                try:
                    args = json.loads(raw)
                except Exception:  # noqa: BLE001
                    args = None
                    c["bad_args"] += 1
                ev = {"i": len(events), "name": f.get("name"), "args": args if isinstance(args, dict) else {}, "result": "", "id": tc.get("id")}
                ev["cmd"] = ev["args"].get("command") if ev["name"] == "str_replace_editor" else None
                events.append(ev)
                by_id[tc.get("id")] = ev
                key = ev["name"] if ev["cmd"] is None else f"{ev['name']}:{ev['cmd']}"
                c["c_" + str(key)] += 1
        elif role == "tool":
            chars_tool += len(content)
            ev = by_id.get(m.get("tool_call_id"))
            if ev is not None:
                ev["result"] = content
    return events, c, chars_total, chars_tool


def critical_path(events: list[dict], patch_paths: list[str], issue: str, created: list[str]) -> dict:
    """Estimate how many calls are left if only the calls on the critical path are kept.

    Kept: the successful str_replace edits on the existing source files of the final patch; for each edit, the latest earlier `view` of that
    file whose text contains old_str (the call that grounds the edit); the earliest call that revealed the file's path (the discovery call),
    unless the issue already names the file; and the first self-contained passing verification command after the last kept edit.
    Other files in the patch (tests, scratch scripts the agent created, docs) are ignored; any other kind of file rejects the trajectory.
    """
    out = {"cond_ok": 0, "cond_calls": 0, "cond_has_verif": 0, "cond_edits": 0, "cond_views": 0, "cond_disc": 0, "cond_disc_kind": "",
           "cond_disc_grep_n": 0, "cond_disc_pat_in_issue": 0, "cond_file_in_issue": 0, "cond_ungrounded": 0, "cond_files": 0,
           "cond_chars": 0, "cond_why": "", "cond_view_last_line": 0, "cond_small_file": 0}
    created_rel = {rel_path(p) for p in created}
    scratch = {p for p in created_rel if "/" not in p or SCRATCH_NAME.search(posixpath.basename(p))}
    edited_ok: dict[str, list[dict]] = {}
    for ev in events:
        if ev["name"] == "str_replace_editor" and ev["cmd"] in ("insert", "undo_edit"):
            out["cond_why"] = "insert_or_undo"
            return out
        if ev["name"] == "str_replace_editor" and ev["cmd"] == "str_replace" and not any(s in ev["result"] for s in EDIT_FAIL):
            edited_ok.setdefault(rel_path(ev["args"].get("path", "")), []).append(ev)
    src, other = [], []
    for p in patch_paths:
        if TEST_PATH.search(p) or p in scratch or DOC_FILE.search(p):
            continue
        if p.endswith(".py") and p in edited_ok:
            src.append(p)
        elif p in created_rel and p.endswith(".py"):
            out["cond_why"] = "new_module"
            return out
        else:
            other.append(p)
    if other:
        out["cond_why"] = "other_files"
        return out
    if not src or len(src) > 3:
        out["cond_why"] = "no_source_edit" if not src else "too_many_files"
        return out
    srcset = set(src)
    kept = [ev for p in srcset for ev in edited_ok[p]]
    kept.sort(key=lambda e: e["i"])
    if len(kept) > 4:
        out["cond_why"] = "edit_count"
        return out
    grounded: dict[int, dict] = {}
    for ed in kept:
        old = ed["args"].get("old_str") or ""
        rp = rel_path(ed["args"].get("path", ""))
        hit = None
        for ev in events[: ed["i"]]:
            if ev["name"] == "str_replace_editor" and ev["cmd"] == "view" and rel_path(ev["args"].get("path", "")) == rp:
                if old.expandtabs() in CAT_N.sub("", ev["result"]).expandtabs():
                    hit = ev
        if hit is None:
            out["cond_ungrounded"] += 1
        else:
            grounded[hit["i"]] = hit
    if out["cond_ungrounded"]:
        out["cond_why"] = "ungrounded_edit"
        return out
    first_view = min(grounded)
    last_lines = [int(x) for v in grounded.values() for x in re.findall(r"^\s*(\d+)\t", v["result"], re.M)[-1:]]
    out["cond_view_last_line"] = max(last_lines) if last_lines else 0
    out["cond_small_file"] = int(bool(last_lines) and max(last_lines) <= 150 and not any(v["args"].get("view_range") for v in grounded.values()))
    out["cond_files"] = len(srcset)
    out["cond_file_in_issue"] = int(all((p in issue or posixpath.basename(p) in issue) for p in srcset))
    disc = None
    if out["cond_file_in_issue"]:
        out["cond_disc_kind"] = "issue"
    else:
        for ev in events[:first_view]:
            if ev["name"] == "execute_bash" and any(p in ev["result"] for p in srcset):
                disc, out["cond_disc_kind"] = ev, "bash"
                break
        if disc is None:
            for ev in events[:first_view]:
                if ev["name"] == "str_replace_editor" and any(posixpath.basename(p) in ev["result"] for p in srcset):
                    disc, out["cond_disc_kind"] = ev, "dirview"
                    break
        if disc is None:
            out["cond_why"] = "no_discovery"
            return out
        if disc["name"] == "execute_bash":
            cmd = str(disc["args"].get("command", ""))
            out["cond_disc_grep_n"] = int(bool(re.search(r"\bgrep\b[^|;&]*\s-[a-zA-Z]*n", cmd)) or bool(re.search(r"\brg\b", cmd)))
            toks = [t for t in TOKEN.findall(cmd) if t.lower() not in STOP_TOKENS]
            out["cond_disc_pat_in_issue"] = int(any(t in issue for t in toks))
    last_edit = max(e["i"] for e in kept)
    verif = None
    for ev in events[last_edit + 1:]:
        if ev["name"] != "execute_bash":
            continue
        cmd = str(ev["args"].get("command", ""))
        if VERIF_CMD.search(cmd) and "exit code 0" in ev["result"] and not any(cr and cr in cmd for cr in created_rel):
            verif = ev
            break
    out["cond_ok"] = 1
    out["cond_edits"] = len(kept)
    out["cond_views"] = len(grounded)
    out["cond_disc"] = int(disc is not None)
    out["cond_has_verif"] = int(verif is not None)
    out["cond_calls"] = len(kept) + len(grounded) + out["cond_disc"] + out["cond_has_verif"]
    ch = len(issue) + 3000
    if disc:
        ch += min(len(disc["result"]), 5000) + len(str(disc["args"]))
    ch += sum(min(len(v["result"]), 9000) for v in grounded.values())
    ch += sum(len(json.dumps(e["args"])) + min(len(e["result"]), 1500) for e in kept)
    if verif:
        ch += min(len(verif["result"]), 3000) + len(str(verif["args"]))
    out["cond_chars"] = ch
    return out


COND_KEYS = ["cond_ok", "cond_calls", "cond_has_verif", "cond_edits", "cond_views", "cond_disc", "cond_disc_kind", "cond_disc_grep_n",
             "cond_disc_pat_in_issue", "cond_file_in_issue", "cond_ungrounded", "cond_files", "cond_chars", "cond_why",
             "cond_view_last_line", "cond_small_file"]


def analyze(row: dict) -> tuple[dict, list[dict]]:
    tr = row["trajectory"] or []
    events, c, chars_total, chars_tool = build_events(tr)
    seen: list[str] = []
    created: list[str] = []
    for ev in events:
        name, args, res = ev["name"], ev["args"], ev["result"]
        if name == "str_replace_editor":
            if ev["cmd"] == "create":
                created.append(str(args.get("path", "")))
            elif ev["cmd"] == "str_replace":
                if any(s in res for s in EDIT_FAIL):
                    c["edit_fail"] += 1
                else:
                    c["edit_ok"] += 1
                    old = args.get("old_str")
                    if isinstance(old, str) and old and any(old.expandtabs() in s.expandtabs() for s in seen):
                        c["edit_grounded"] += 1
                    else:
                        c["edit_ungrounded"] += 1
            elif ev["cmd"] == "view":
                if "files and directories up to 2 levels deep" in res:
                    c["view_dir"] += 1
                else:
                    c["view_file"] += 1
                    if args.get("view_range"):
                        c["view_ranged"] += 1
                seen.append(CAT_N.sub("", res))
        elif name == "execute_bash":
            cmd = str(args.get("command", ""))
            if str(args.get("is_input")).lower() == "true":
                c["bash_is_input"] += 1
            if GIT_HIST.search(cmd):
                c["git_hist"] += 1
            if GIT_RISKY.search(cmd):
                c["git_risky"] += 1
            if TEST_RUN.search(cmd):
                c["test_cmds"] += 1
            if PIP_INSTALL.search(cmd):
                c["pip_install"] += 1
            seen.append(res)
    ps = patch_stats(row.get("model_patch") or "")
    paths = ps.pop("p_paths")
    pset = set(paths)
    scratch = sum(1 for p in created if rel_path(p) not in pset)
    mapped = c["c_execute_bash"] + c["c_str_replace_editor:view"] + c["c_str_replace_editor:str_replace"] + c["c_str_replace_editor:create"]
    unmappable = c["c_str_replace_editor:insert"] + c["c_str_replace_editor:undo_edit"] + c["c_task_tracker"] + c["bash_is_input"]
    issue = ""
    if len(tr) > 1:
        m = ISSUE.search(tr[1].get("content") or "")
        issue = m.group(1) if m else ""
    cp = critical_path(events, paths, issue, created) if int(row["resolved"] or 0) == 1 else {"cond_ok": 0, "cond_why": "unresolved"}
    out = {
        "trajectory_id": row["trajectory_id"], "instance_id": row["instance_id"], "repo": row["repo"],
        "resolved": int(row["resolved"] or 0), "exit_status": row["exit_status"],
        "gen_tests_correct": row.get("gen_tests_correct"), "pred_passes_gen_tests": row.get("pred_passes_gen_tests"),
        "n_msgs": len(tr), "mapped_calls": mapped, "unmappable_calls": unmappable, "issue_chars": len(issue),
        "chars_total": chars_total, "chars_tool": chars_tool, "scratch_created": scratch,
    }
    for k, srck in [("c_bash", "c_execute_bash"), ("c_view", "c_str_replace_editor:view"), ("c_str_replace", "c_str_replace_editor:str_replace"),
                    ("c_create", "c_str_replace_editor:create"), ("c_insert", "c_str_replace_editor:insert"),
                    ("c_undo", "c_str_replace_editor:undo_edit"), ("c_think", "c_think"), ("c_finish", "c_finish"),
                    ("c_task_tracker", "c_task_tracker")]:
        out[k] = c[srck]
    for k in ["bash_is_input", "edit_ok", "edit_fail", "edit_grounded", "edit_ungrounded", "view_dir", "view_file", "view_ranged",
              "git_hist", "git_risky", "test_cmds", "pip_install", "asst_text_msgs", "asst_text_chars", "multi_call_msgs", "bad_args"]:
        out[k] = c[k]
    out.update(ps)
    out.update({k: cp.get(k, "" if k in ("cond_why", "cond_disc_kind") else 0) for k in COND_KEYS})
    return out, tr


def rows_from_parquet(path: str):
    pf = pq.ParquetFile(path)
    for g in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(g, columns=COLS)
        for start in range(0, tbl.num_rows, 128):
            yield from tbl.slice(start, 128).to_pylist()
        yield {"__row_group_done__": g}
        del tbl


def rows_from_jsonl(path: str):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        r.setdefault("gen_tests_correct", None)
        r.setdefault("pred_passes_gen_tests", None)
        yield r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet")
    src.add_argument("--jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many rows (0 = all)")
    ap.add_argument("--examples", type=int, default=60)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ex_fh = open(out / "examples.jsonl", "w", encoding="utf-8")
    log = open(out / "progress.log", "a", encoding="utf-8")
    n_ex = 0
    t0 = time.time()
    it = rows_from_parquet(a.parquet) if a.parquet else rows_from_jsonl(a.jsonl)
    for r in it:
        if "__row_group_done__" in r:
            msg = f"row_group {r['__row_group_done__']} done, rows={len(rows)}, examples={n_ex}, elapsed={time.time() - t0:.0f}s"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            continue
        st, tr = analyze(r)
        rows.append(st)
        if n_ex < a.examples and st["cond_ok"] and st["cond_has_verif"] and st["cond_calls"] <= 10 and st["edit_fail"] <= 2:
            ex_fh.write(json.dumps({"stats": st, "trajectory": tr, "model_patch": r["model_patch"]}, ensure_ascii=False) + "\n")
            n_ex += 1
        if a.limit and len(rows) >= a.limit:
            break
    ex_fh.close()
    pq.write_table(pa.Table.from_pylist(rows), out / "census.parquet")
    print(f"wrote {len(rows)} rows to {out / 'census.parquet'}; examples {n_ex}; total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
