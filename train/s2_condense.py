"""S2.1 / S2.2 of the E4B accuracy plan (docs/PLAN_E4B_SFT.md): choose teacher trajectories and condense each one to its critical path.

  python -I -X utf8 train/s2_condense.py --n 50 --out D:/data/s2/pilot/jobs.jsonl [--seed 1] [--per-repo 1]
  python -I -X utf8 train/s2_condense.py --n 0 --per-repo 10 --out D:/data/s2/full/jobs.jsonl      # every selected instance, at most 10 per repository

Input : the census table (selected_f10.csv), the nebius/SWE-rebench-OpenHands trajectories (Qwen3-Coder-480B-A35B-Instruct teacher, CC-BY-4.0)
        and the SWE-rebench task table (base_commit, problem_statement, licence).
Output: one JSON line per instance: what the replay (train/s2_replay.py) needs to re-run the critical path on a real checkout.
        {instance_id, repo, base_commit, license, problem_statement, patch_paths,
         disc:  {kind: issue|bash|dirview, command, terms},      # how the teacher found the file
         edits: [{path, old, new, group}], reads: [{group, path}],# group = the teacher's grounding view of the edit
         verif: {command, output} | null, teacher: {...}}
Rule (plan S2.2): keep every successful str_replace on the patch's source files; for each edit the latest earlier view of that file whose text
contains the edit's old string; the first call that revealed the path unless the issue names the file; one verification after the last edit.
Reject: insert / undo_edit, new modules, files other than source, tests and docs, an edit not grounded in a view, an old or new string with a tab
(the teacher's editor expands tabs before matching, ours does not).
"""
from __future__ import annotations

import argparse
import json
import posixpath
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_census_nebius as C  # noqa: E402

NEBIUS = r"D:\data\ext_traj\nebius_openhands\trajectories.parquet"
SELECTED = r"D:\data\ext_traj\census\selected_f10.csv"
REBENCH = [r"D:\data\ext_traj\swe_rebench\data\filtered-00000-of-00001.parquet",
           r"D:\data\ext_traj\swe_rebench\data\test-00000-of-00002.parquet",
           r"D:\data\ext_traj\swe_rebench\data\test-00001-of-00002.parquet"]
# Plan F2: repositories of the evaluation pools (public tasks and mined unseen). Matched on owner/name and on the name alone (forks).
EVAL_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx", "litestar-org/litestar", "pydantic/pydantic-settings",
              "pycqa/isort", "urllib3/urllib3", "marshmallow-code/marshmallow", "scrapy/scrapy"}
EVAL_NAMES = {r.split("/")[1] for r in EVAL_REPOS}
FOOTER = re.compile(r"\n?\[(?:The command completed with exit code \d+\.|Current working directory: [^\]]*|Python interpreter: [^\]]*|"
                    r"Command finished with exit code \d+)\]", re.M)
TEACHER = "Qwen/Qwen3-Coder-480B-A35B-Instruct (OpenHands 0.54.0), nebius/SWE-rebench-OpenHands-Trajectories, CC-BY-4.0"


def is_eval_repo(repo: str) -> bool:
    low = (repo or "").lower()
    return low in EVAL_REPOS or low.split("/")[-1] in EVAL_NAMES


def runs_patched_test(cmd: str, patched_tests: list[str]) -> bool:
    """True if the command runs a test file the teacher itself changed or added (or a directory holding one). The recorded output of such a run
    lists tests that do not exist in the checkout we replay on, so it cannot be used as the observation."""
    for tok in re.split(r"[\s'\"]+", cmd):
        tok = tok.split("::")[0].strip().rstrip("/")
        if not tok or tok.startswith("-") or tok in {".", "python", "python3", "pytest"}:
            continue
        tok = re.sub(r"^(?:\./|/workspace/[^/]+/)", "", tok)
        for p in patched_tests:
            if p == tok or p.startswith(tok + "/") or (tok.endswith(".py") and p.rsplit("/", 1)[-1] == tok.rsplit("/", 1)[-1]):
                return True
    return False


def plan_for(events, patch_paths, issue, created):
    """critical_path() of the census, returning the kept events. (plan, why): plan is None when the trajectory is rejected."""
    created_rel = {C.rel_path(p) for p in created}
    scratch = {p for p in created_rel if "/" not in p or C.SCRATCH_NAME.search(posixpath.basename(p))}
    edited_ok: dict[str, list[dict]] = {}
    for ev in events:
        if ev["name"] == "str_replace_editor" and ev["cmd"] in ("insert", "undo_edit"):
            return None, "insert_or_undo"
        if ev["name"] == "str_replace_editor" and ev["cmd"] == "str_replace" and not any(s in ev["result"] for s in C.EDIT_FAIL):
            edited_ok.setdefault(C.rel_path(ev["args"].get("path", "")), []).append(ev)
    src, other = [], []
    for p in patch_paths:
        if C.TEST_PATH.search(p) or p in scratch or C.DOC_FILE.search(p):
            continue
        if p.endswith(".py") and p in edited_ok:
            src.append(p)
        elif p in created_rel and p.endswith(".py"):
            return None, "new_module"
        else:
            other.append(p)
    if other:
        return None, "other_files"
    if not src or len(src) > 3:
        return None, "no_source_edit" if not src else "too_many_files"
    srcset = set(src)
    kept = sorted((ev for p in srcset for ev in edited_ok[p]), key=lambda e: e["i"])
    if len(kept) > 4:
        return None, "edit_count"
    grounded: dict[int, dict] = {}
    group_of: dict[int, int] = {}
    for ed in kept:
        old, new = ed["args"].get("old_str") or "", ed["args"].get("new_str") or ""
        if "\t" in old or "\t" in new:
            return None, "tab_in_edit"
        rp = C.rel_path(ed["args"].get("path", ""))
        hit = None
        for ev in events[: ed["i"]]:
            if ev["name"] == "str_replace_editor" and ev["cmd"] == "view" and C.rel_path(ev["args"].get("path", "")) == rp:
                if old.expandtabs() in C.CAT_N.sub("", ev["result"]).expandtabs():
                    hit = ev
        if hit is None:
            return None, "ungrounded_edit"
        grounded[hit["i"]] = hit
        group_of[ed["i"]] = hit["i"]
    first_view = min(grounded)
    file_in_issue = all((p in issue or posixpath.basename(p) in issue) for p in srcset)
    disc, kind = None, "issue"
    if not file_in_issue:
        for ev in events[:first_view]:
            if ev["name"] == "execute_bash" and any(p in ev["result"] for p in srcset):
                disc, kind = ev, "bash"
                break
        if disc is None:
            for ev in events[:first_view]:
                if ev["name"] == "str_replace_editor" and any(posixpath.basename(p) in ev["result"] for p in srcset):
                    disc, kind = ev, "dirview"
                    break
        if disc is None:
            return None, "no_discovery"
    last_edit = max(e["i"] for e in kept)
    patched_tests = [p for p in patch_paths if C.TEST_PATH.search(p)]
    verif, failing = None, 0
    for ev in events[last_edit + 1:]:
        if ev["name"] != "execute_bash":
            continue
        cmd = str(ev["args"].get("command", ""))
        if (C.VERIF_CMD.search(cmd) and "exit code 0" in ev["result"] and not any(cr and cr in cmd for cr in created_rel)
                and not runs_patched_test(cmd, patched_tests)):
            out = strip_command_echo(cmd, ev["result"])
            if shows_failure(out):          # exit code 0 is not success for a script that catches its own exception; do not teach "submit anyway"
                failing += 1
                continue
            verif = dict(ev, result=out)
            break
    if verif is None:
        return None, "failing_verification" if failing else "no_verification"
    return {"kind": kind, "disc": disc, "views": [grounded[k] for k in sorted(grounded)], "edits": kept, "group_of": group_of, "verif": verif}, ""


FAILURE = re.compile(r"Traceback \(most recent call last\)|^\s*(?:\w+\.)*\w*(?:Error|Exception)\b|\bFAIL(?:ED|URE)?\b|\b[1-9]\d* (?:failed|errors?)\b|[✗❌✘]", re.M)


def shows_failure(output: str) -> bool:
    """True if the recorded output visibly reports a failure although the command's exit code was 0 (a script that prints its own exception, a
    check mark that is a cross). Such a run as the last step before `submit_patch` would teach the model to submit regardless."""
    return bool(FAILURE.search(output))


def strip_command_echo(cmd: str, out: str) -> str:
    """OpenHands' terminal prints a multi-line command back before its output, blank lines dropped and `cd /workspace/<repo> && ` in front of the
    first line (11% of the recorded verification outputs); the scorer's run_command returns the output alone. Single-line commands are not echoed.
    The output is returned unchanged unless it starts with every non-blank line of the command, in order."""
    want = [l.rstrip() for l in cmd.splitlines() if l.strip()]
    if len(want) < 2:
        return out
    got = out.splitlines()
    j = 0
    while j < len(got) and not got[j].strip():
        j += 1
    for k, w in enumerate(want):
        while k and j < len(got) and not got[j].strip():
            j += 1
        if j >= len(got) or not (got[j].rstrip().endswith(w) if k == 0 else got[j].rstrip() == w):
            return out
        j += 1
    return "\n".join(got[j:])


def tidy_command(cmd: str) -> str:
    """The teacher's command for our workspace: the repository is /workspace itself here, not /workspace/<dir>."""
    cmd = re.sub(r"^\s*cd\s+/workspace/[^\s&;]+\s*&&\s*", "", cmd)
    cmd = re.sub(r"/workspace/[^/\s'\"]+/", "", cmd)
    cmd = re.sub(r"/workspace/[^/\s'\"]+(?=[\s'\"]|$)", ".", cmd)
    return cmd.replace("/opt/conda/envs/testbed/bin/", "").strip()


WIDTH = 80
BANNER = re.compile(r"^([=_\-])\1{2,} (.+?) \1{3,}$", re.M)               # `===== test session starts =====`, `____ test_x ____`
PURE_BANNER = re.compile(r"^([=_\-])\1{99,}$", re.M)
PROGRESS = re.compile(r"^(.*?\S) {10,}(\[ *\d+%\])$", re.M)               # `tests/test_a.py ....        [ 50%]`


def pytest_at_80_columns(text: str) -> str:
    """The teacher's terminal was about 500 columns wide (67% of the recorded verification outputs have banners of 491 characters), so its pytest
    banners and right-aligned progress markers are hundreds of characters apart; the scorer's pytest writes to a pipe and uses 80. Re-emit them the
    way pytest does (TerminalWriter.sep, and `msg.rjust(fullwidth - len(line) - 1)` for the marker). Lines that already fit are left alone, so the
    function does nothing to an output that was recorded at 80 columns. Other width-dependent output (wrapped dot lines) is not touched."""
    def banner(m):
        if len(m.group(0)) <= WIDTH:
            return m.group(0)
        ch, title = m.group(1), m.group(2)
        n = max((WIDTH - len(title) - 2) // 2, 1)
        line = f"{ch * n} {title} {ch * n}"
        return line + ch if len(line) + 1 <= WIDTH else line

    def progress(m):
        prefix, msg = m.group(1), " " + m.group(2)
        fill = WIDTH - 1 - len(prefix)
        return prefix + (msg.rjust(fill) if fill > len(msg) else msg)

    text = BANNER.sub(banner, text)
    text = PURE_BANNER.sub(lambda m: m.group(1) * WIDTH, text)
    return PROGRESS.sub(progress, text)


def tidy_output(text: str) -> str:
    text = FOOTER.sub("", text)
    text = re.sub(r"/workspace/[^/\s'\"]+/", "/workspace/", text)
    text = re.sub(r"/workspace/[^/\s'\"]+(?=[\s'\"]|$)", "/workspace", text)
    text = text.replace("/opt/conda/envs/testbed", "/usr/local")                 # interpreter and site-packages paths of the teacher's image
    return pytest_at_80_columns(text).rstrip("\n")


def job_from(row: dict, task: dict) -> tuple[dict | None, str]:
    tr = row["trajectory"]
    events, _c, _ct, _cc = C.build_events(tr)
    created = [str(ev["args"].get("path", "")) for ev in events if ev["name"] == "str_replace_editor" and ev["cmd"] == "create"]
    paths = C.patch_stats(row.get("model_patch") or "")["p_paths"]
    m = C.ISSUE.search(tr[1].get("content") or "") if len(tr) > 1 else None
    issue = m.group(1) if m else ""
    plan, why = plan_for(events, paths, issue, created)
    if plan is None:
        return None, why
    disc = plan["disc"]
    terms = []
    if disc is not None and disc["name"] == "execute_bash":
        terms = [t for t in C.TOKEN.findall(str(disc["args"].get("command", ""))) if t.lower() not in C.STOP_TOKENS]
    gid = {g: n for n, g in enumerate(sorted(set(plan["group_of"].values())))}
    job = {
        "instance_id": row["instance_id"], "trajectory_id": row["trajectory_id"], "repo": task["repo"], "base_commit": task["base_commit"],
        "license": task["license_name"], "problem_statement": task["problem_statement"], "teacher": TEACHER,
        "patch_paths": [p for p in paths if not C.TEST_PATH.search(p)],
        "model_patch": row.get("model_patch") or "",
        "disc": {"kind": plan["kind"], "command": tidy_command(str(disc["args"].get("command", ""))) if disc is not None and disc["name"] == "execute_bash" else None,
                 "terms": terms},
        "reads": [{"group": gid[v["i"]], "path": C.rel_path(v["args"].get("path", ""))} for v in plan["views"]],
        "edits": [{"path": C.rel_path(e["args"].get("path", "")), "old": e["args"].get("old_str") or "", "new": e["args"].get("new_str") or "",
                   "group": gid[plan["group_of"][e["i"]]]} for e in plan["edits"]],
        "verif": {"command": tidy_command(str(plan["verif"]["args"].get("command", ""))), "output": tidy_output(plan["verif"]["result"])},
        "stats": {"teacher_events": len(events), "kept_calls": len(plan["edits"]) + len(plan["views"]) + (1 if disc else 0) + 1},
    }
    return job, ""


def load_tasks(ids: set[str]) -> dict[str, dict]:
    import pyarrow.dataset as ds
    out: dict[str, dict] = {}
    cols = ["instance_id", "repo", "base_commit", "problem_statement", "license_name"]
    for p in REBENCH:
        t = ds.dataset(p).to_table(columns=cols, filter=ds.field("instance_id").isin(sorted(ids)))
        for r in t.to_pylist():
            out.setdefault(r["instance_id"], r)
    return out


def iter_rows(ids: list[str]):
    """The wanted rows only, one at a time. A row group of the file is about 1 GB compressed, so it is read in small batches and only the
    row groups that hold a wanted id are opened at all."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want, n_found = set(ids), 0
    value_set = pa.array(sorted(want))
    pf = pq.ParquetFile(NEBIUS)
    cols = ["trajectory_id", "instance_id", "repo", "trajectory", "model_patch", "resolved"]
    for g in range(pf.metadata.num_row_groups):
        present = set(pf.read_row_group(g, columns=["trajectory_id"]).column(0).to_pylist()) & want
        if not present:
            continue
        for batch in pf.iter_batches(batch_size=64, row_groups=[g], columns=cols):
            hit = pc.is_in(batch.column("trajectory_id"), value_set=value_set)
            if pc.any(hit).as_py():                   # filter in Arrow: converting every row to Python objects is what is slow
                for r in batch.filter(hit).to_pylist():
                    n_found += 1
                    yield r
        if n_found >= len(want):
            break


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-repo", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import pandas as pd
    sel = pd.read_csv(SELECTED)
    sel = sel[~sel["repo"].map(is_eval_repo)]
    rng = random.Random(a.seed)
    order = list(sel.index)
    rng.shuffle(order)
    per, picked = Counter(), []
    for ix in order:
        repo = sel.loc[ix, "repo"]
        if per[repo] < a.per_repo:
            per[repo] += 1
            picked.append(ix)
    # ask for more than n: some are rejected by the stricter checks below (n = 0 means every selected instance)
    want = sel.loc[picked[: int(a.n * 1.6) + 10] if a.n else picked]
    tasks = load_tasks(set(want["instance_id"]))
    by_tid, why = {}, Counter()
    for row in iter_rows(list(want["trajectory_id"])):             # condense as the rows stream by; the big rows are not kept
        task = tasks.get(row["instance_id"])
        if task is None:
            why["missing_task"] += 1
            continue
        job, reason = job_from(row, task)
        if job is None:
            why[reason] += 1
        else:
            by_tid[row["trajectory_id"]] = job
    out = [by_tid[t] for t in want["trajectory_id"] if t in by_tid]
    if a.n:
        out = out[: a.n]
    missing = sum(1 for t in want["trajectory_id"] if t not in by_tid) - sum(why.values())
    if missing > 0:
        why["row_not_found"] = missing
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for j in out:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(json.dumps({"written": len(out), "rejected": dict(why), "kinds": dict(Counter(j["disc"]["kind"] for j in out)),
                      "repos": len({j["repo"] for j in out}), "calls": dict(Counter(j["stats"]["kept_calls"] for j in out))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
