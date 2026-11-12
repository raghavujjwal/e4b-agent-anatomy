"""Gold and empty controls for a mined dataset, with the interpreter ladder. Run inside WSL (the grader needs Linux).

  python3 mine/controls.py --data /mnt/d/data/mined/data [--workers 2] [--repos click,falcon]

1. gold under 3.13; tasks that do not resolve are retried under 3.12, then 3.11 (as for the public tasks).
2. empty patch under the interpreter where gold resolved; it must NOT resolve.
Outputs in <data>/results/: gold_<py>.jsonl, empty.jsonl, env_map.json (task -> interpreter) and usable.json
({"usable": [...], "status": {task: reason}}). A task is usable only if gold resolves and the empty patch does not.
Resumable: each grader call skips what its output file already holds. After changing grader code, delete results/.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

LADDER = ("3.13", "3.12", "3.11")


def load(p: Path) -> dict:
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["instance_id"]] = r
    return out


def grade(data: Path, mode: str, ids: list[str], out: Path, env_map: Path, workers: int) -> None:
    if not ids:
        return
    cmd = [sys.executable, "-m", "grader.cli", "--mode", mode, "--tasks", str(data / "tasks.jsonl"),
           "--snapshots", str(data / "snapshots"), "--workers", str(workers), "--out", str(out)]
    for i in ids:
        cmd += ["--task-id", i]
    env = dict(os.environ, GRADER_ENV_MAP=str(env_map))
    subprocess.run(cmd, env=env, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--repos", default="", help="comma list of repo short names to limit to")
    a = ap.parse_args()
    data = Path(a.data)
    res = data / "results"
    res.mkdir(exist_ok=True)
    tasks = [json.loads(l) for l in (data / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = {x for x in a.repos.split(",") if x}
    ids = [t["instance_id"] for t in tasks if not keep or t["instance_id"].rsplit("_", 1)[0] in keep]

    py_ok: dict[str, str] = {}
    last: dict[str, dict] = {}
    todo = list(ids)
    for py in LADDER:
        em = res / f"envmap_{py}.json"
        em.write_text(json.dumps({i: py for i in todo}))
        out = res / f"gold_{py}.jsonl"
        grade(data, "gold", todo, out, em, a.workers)
        got = load(out)
        for i in todo:
            if i in got:
                last[i] = got[i]
        done = [i for i in todo if got.get(i, {}).get("resolved")]
        py_ok.update({i: py for i in done})
        todo = [i for i in todo if i not in py_ok]
        print(f"=== gold {py}: {len(done)} resolved, {len(todo)} left", flush=True)
        if not todo:
            break

    env_map = res / "env_map.json"
    env_map.write_text(json.dumps(py_ok, indent=1))
    ok_ids = [i for i in ids if i in py_ok]
    grade(data, "empty", ok_ids, res / "empty.jsonl", env_map, a.workers)
    empty = load(res / "empty.jsonl")

    status, usable = {}, []
    for i in ids:
        if i not in py_ok:
            r = last.get(i, {})
            status[i] = "gold fails: " + str(r.get("error") or r.get("detail") or "no result")[:140]
        elif empty.get(i, {}).get("resolved") or empty.get(i, {}).get("passes_without_patch"):
            status[i] = "empty patch passes (unsound)"
        elif i not in empty:
            status[i] = "empty control missing"
        elif empty[i].get("grader_error"):
            status[i] = "grader error on empty control"
        else:
            status[i] = "usable"
            usable.append(i)
    (res / "usable.json").write_text(json.dumps({"usable": usable, "status": status}, indent=1))
    by: dict[str, list[int]] = {}
    for i in ids:
        by.setdefault(i.rsplit("_", 1)[0], [0, 0])[0] += 1
        by[i.rsplit("_", 1)[0]][1] += i in usable
    print(f"\nusable {len(usable)}/{len(ids)}")
    for k, (n, u) in sorted(by.items()):
        print(f"  {k:20s} {u:3d}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
