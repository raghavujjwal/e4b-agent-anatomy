"""Pack the usable mined tasks for Colab (notebook 03, POOL "tune" or "unseen"). One zip on Drive, no git push needed.

  python -I -X utf8 mine/make_lab_pack.py --out D:/data/lab/lab_pack.zip

Contents: tasks.jsonl and snapshots/<id>.tgz of every usable mined task, env_map.json (task id -> interpreter), usable_tasks.json
(the ids, and the two pools by the fixed repository rule of docs/EVAL_PROTOCOL.md), submissions/<folder>/ for the arm folders the
notebook copies into agent/ (an arm that exists only on the PC needs no push), and lab_manifest.json with counts and digests.
It reads no agent output. A usable task with no snapshot, no task row or no interpreter entry stops the build.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = ("submission_e4b", "submission_e4b_v2rec", "submission_e4b_v2rec_th0", "submission_e4b_v2rec_th512", "submission_e4b_v2rec_t07",
                "submission_e4b_v2", "submission_e4b_rec", "submission_e4b_v2rec_noth", "submission_e4b_v2rec_sft2",
                "submission_e4b_pipe_s1", "submission_e4b_pipe_s2", "submission_e4b_pipe_s3")
SPLIT_RULE = 'sha256("gemma4-split-v1/" + "<owner>/<repo>"), even = tune, odd = unseen (docs/EVAL_PROTOCOL.md)'
UNSEEN_POLICY = "unseen: never read, aggregate counts only, milestone comparisons only (docs/EVAL_PROTOCOL.md rules 3 and 13)"


def side(repo: str) -> str:
    """The pool of a repository ("owner/name"), by the rule written down before any result existed."""
    return "tune" if int(hashlib.sha256(("gemma4-split-v1/" + repo).encode()).hexdigest(), 16) % 2 == 0 else "unseen"


def arm_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def arm_digest(folder: Path) -> str:
    h = hashlib.sha256()
    for p in arm_files(folder):
        h.update(p.relative_to(folder).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def pack(mined: Path, usable: list[str], env_map: dict, arms: list[Path], out: Path) -> dict:
    rows = {}
    for line in (mined / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            rows[t["instance_id"]] = t
    problems = [f"{i}: {why}" for i in usable for why, ok in (("no task row", i in rows), ("no snapshot", (mined / "snapshots" / f"{i}.tgz").exists()),
                                                              ("no interpreter in env_map", i in env_map)) if not ok]
    if problems:
        raise SystemExit("cannot pack, usable tasks with missing parts:\n  " + "\n  ".join(problems[:20]))
    for a in arms:
        if not (a / "agent.yaml").exists():
            raise SystemExit(f"arm folder without agent.yaml: {a}")
    pools = {"tune": [], "unseen": []}
    for i in usable:
        pools[side(rows[i]["repo"])].append(i)
    tasks_text = "".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in usable)
    manifest = {"created": date.today().isoformat(), "tasks": len(usable), "pools": {k: len(v) for k, v in pools.items()},
                "split_rule": SPLIT_RULE, "unseen_policy": UNSEEN_POLICY,
                "repos": {k: sorted({rows[i]["repo"] for i in v}) for k, v in pools.items()},
                "arms": {a.name: arm_digest(a) for a in arms},
                "sha256_tasks_jsonl": hashlib.sha256(tasks_text.encode()).hexdigest()}
    try:
        manifest["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        manifest["git_head"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("lab_manifest.json", json.dumps(manifest, indent=1) + "\n")
        z.writestr("tasks.jsonl", tasks_text)
        z.writestr("env_map.json", json.dumps({i: env_map[i] for i in usable}, indent=1) + "\n")
        z.writestr("usable_tasks.json", json.dumps({"usable": usable, "pools": pools}, indent=1) + "\n")
        for i in usable:
            z.write(mined / "snapshots" / f"{i}.tgz", f"snapshots/{i}.tgz", compress_type=zipfile.ZIP_STORED)
        for a in arms:
            for p in arm_files(a):
                z.write(p, f"submissions/{a.name}/{p.relative_to(a).as_posix()}")
    manifest["bytes"] = out.stat().st_size
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mined", default="D:/data/mined/data", help="folder with tasks.jsonl and snapshots/ of the mined tasks")
    ap.add_argument("--usable", default=str(ROOT / "mine" / "usable.json"))
    ap.add_argument("--env-map", default=str(ROOT / "mine" / "env_map.json"))
    ap.add_argument("--arm", action="append", help="arm folder under agent/ (repeatable); default: the lab folders of DEFAULT_ARMS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    usable = json.loads(Path(a.usable).read_text(encoding="utf-8"))["usable"]
    env_map = json.loads(Path(a.env_map).read_text(encoding="utf-8"))
    arms = [ROOT / "agent" / n for n in (a.arm or DEFAULT_ARMS)]
    m = pack(Path(a.mined), usable, env_map, arms, Path(a.out))
    print(json.dumps({k: m[k] for k in ("tasks", "pools", "bytes", "git_head", "sha256_tasks_jsonl")}, indent=1))
    print("arms:", ", ".join(f"{k} {v[:8]}" for k, v in m["arms"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
