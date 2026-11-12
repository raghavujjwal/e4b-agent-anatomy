"""Drive the edit_lines skill through the real tool path (run_skill_script) with the gold change, then grade.

  python agent/check_lines_skill.py --task-id httpx_3672 --task-id requests_7309      # run in WSL/Linux, repository root

For each file the gold patch changes, the file is read from the sandbox, the gold result is computed with `git apply`, and the
difference is replayed as `show` + `replace` skill calls (bottom-up, so earlier line numbers stay valid). The submitted patch must
resolve exactly the tasks that `--backend gold` resolves. No model is involved.
"""
from __future__ import annotations
import argparse
import difflib
import json
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent" / "vendor"))

from grader import config, container, verify as gverify   # noqa: E402
from tools import Budget, Tools                            # noqa: E402  (agent/vendor/tools.py)

SKILLS = str(REPO / "agent" / "submission_e4b_lines" / "skills")


def files_of(patch: str) -> list[str]:
    import re
    return re.findall(r"^\+\+\+ b/(\S+)$", patch, re.M)


def one(task, snapshots) -> dict:
    c = container.make_factory()()
    try:
        container.make_setup(task, snapshots)(c)
        tools = Tools(c, Budget(tool_calls=200, time_minutes=30), skills_root=SKILLS)
        calls = 0
        pt = task["patch"]
        c.put_text("/tmp/_gold.patch", pt if pt.endswith("\n") else pt + "\n")
        for fp in files_of(task["patch"]):
            before = c.get_text(f"/workspace/{fp}")
            rc, so, se = c.sh(f"cd /workspace && git apply /tmp/_gold.patch --include={shlex.quote(fp)} && cat {shlex.quote(fp)} "
                              f"&& git checkout -- {shlex.quote(fp)}")
            if rc != 0:
                return {"instance_id": task["instance_id"], "error": f"git apply failed: {(so + se)[:200]}"}
            after = so
            a, b = before.split("\n"), after.split("\n")
            ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
            for tag, i1, i2, j1, j2 in reversed(ops):
                text = "\n".join(b[j1:j2])
                if j2 > j1 or i2 > i1:
                    args = {"action": "replace", "file": fp, "start": i1 + 1, "end": i2, "text": text}
                    if i2 > i1:
                        args["expect"] = a[i1]
                    r = json.loads(tools.call("run_skill_script", {"skill_name": "edit_lines", "file_path": "scripts/edit_lines.py",
                                                                   "args": args}))
                    calls += 1
                    if r.get("status") != "ok" or not str(r.get("stdout", "")).startswith("OK"):
                        return {"instance_id": task["instance_id"], "error": f"skill call failed on {fp}: {json.dumps(r)[:300]}"}
        r = json.loads(tools.call("submit_patch", {}))
        patch = tools.submitted_patch
        res = container.make_verifier(snapshots)(task, patch)
        return {"instance_id": task["instance_id"], "skill_calls": calls, "patch_bytes": len(patch),
                "resolved": bool(res.get("resolved")), "error": res.get("error")}
    finally:
        c.stop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tasks", default=str(config.TASKS_JSONL))
    ap.add_argument("--snapshots", default=str(config.SNAPSHOTS))
    ap.add_argument("--task-id", action="append", required=True)
    a = ap.parse_args(argv)
    tasks = [t for t in gverify.load_tasks(a.tasks) if t["instance_id"] in set(a.task_id)]
    bad = 0
    for t in tasks:
        out = one(t, a.snapshots)
        print(json.dumps(out), flush=True)
        bad += not out.get("resolved")
    print(f"\n{len(tasks) - bad}/{len(tasks)} resolved through the skill (gold resolves all of these)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
