"""Command line: run the grader as a control or on a patch file.

  python -m grader.cli --mode gold  --task-id httpx_3672 --out results/gold.jsonl
  python -m grader.cli --mode empty --n 5
  python -m grader.cli --mode file  --patch my.diff --task-id rich_3061

gold  = apply the reference patch (MUST resolve)
empty = apply nothing          (MUST NOT resolve)
Resumable: tasks already present in --out are skipped, except those whose saved result was a grader error.
After changing grader code, delete --out so every verdict is recomputed.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

from . import config, verify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tasks", default=str(config.TASKS_JSONL))
    ap.add_argument("--snapshots", default=str(config.SNAPSHOTS))
    ap.add_argument("--task-id", action="append", default=None)
    ap.add_argument("--n", type=int, default=0, help="first N tasks")
    ap.add_argument("--mode", choices=["gold", "empty", "file"], default="gold")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="keep the run directory for inspection")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    tasks = verify.load_tasks(a.tasks)
    if a.task_id:
        want = set(a.task_id)
        tasks = [t for t in tasks if t["instance_id"] in want]
    if a.n:
        tasks = tasks[:a.n]
    done = set()
    if a.out and Path(a.out).exists():
        # Resume: keep finished results, but re-run any task whose record is a GRADER error (a defect on our
        # side, since fixed or not) so a stale failure is never reported again. Genuine "no" verdicts are kept.
        old = [json.loads(l) for l in open(a.out) if l.strip()]
        keep = [r for r in old if not r.get("grader_error")]
        done = {r["instance_id"] for r in keep}
        if len(keep) != len(old):
            print(f"re-running {len(old) - len(keep)} task(s) whose saved result was a grader error")
            with open(a.out, "w") as fh:
                for r in keep:
                    fh.write(json.dumps(r) + "\n")
    todo = [t for t in tasks if t["instance_id"] not in done]
    print(f"{len(tasks)} selected, {len(done)} already in {a.out}, {len(todo)} to run", flush=True)

    def patch_for(t):
        if a.mode == "gold":
            return t["patch"]
        if a.mode == "empty":
            return ""
        return Path(a.patch).read_text()

    def one(t):
        return verify.grade(t, patch_for(t), snapshots=Path(a.snapshots), keep=a.keep, verbose=a.verbose)

    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            r["mode"] = a.mode
            results.append(r)
            flag = "RESOLVED" if r["resolved"] else ("passes-w/o-patch" if r.get("passes_without_patch") else "no")
            print(f"[{i}/{len(todo)}] {flag:16s} {r['instance_id']:24s} stage={r.get('stage')} "
                  f"apply={r.get('apply_pass')} {r.get('seconds', '?')}s  "
                  f"{str(r.get('error') or r.get('detail'))[:90]}", flush=True)
            if a.out:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
    n_res = sum(bool(r["resolved"]) for r in results)
    n_err = sum(bool(r.get("grader_error")) for r in results)
    print(f"\nmode={a.mode}: resolved {n_res}/{len(results)}; grader errors {n_err}")
    if a.mode == "gold":
        print("EXPECTED: all resolved. Any 'no' is a grader defect or an unsound task.")
    if a.mode == "empty":
        print("EXPECTED: none resolved. 'passes-w/o-patch' means the task is unsound, or the tests do not run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
