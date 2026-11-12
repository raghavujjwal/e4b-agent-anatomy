"""The grader: grade(task, agent_patch) -> result. A pure function of (snapshot, patch, test_patch).

Mirrors HARNESS_README 8.2 step for step (see docs/GRADER_DESIGN.md section 5):
  fresh workspace -> eval_baseline commit -> strip protected hunks -> 4-pass patch application ->
  reset protected + test_patch paths -> apply test_patch -> hermetic pytest with JUnit XML ->
  every required node passed, unskipped.
Needs no model. Validate it with gold (must resolve) and empty (must not) patches before trusting it.
"""
from __future__ import annotations
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from . import config, envs, patching
from .sandbox import Sandbox, SandboxTimeout


def load_tasks(path: Path | str | None = None) -> list[dict]:
    p = Path(path or config.TASKS_JSONL)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pristine_dir(task: dict, snapshots: Path | None = None) -> Path:
    """Extract the snapshot once; every run copies from this tree. `--no-same-owner` matters: tar as
    root restores the archive's recorded uid and git then refuses to work ('dubious ownership')."""
    d = config.ROOT / "pristine" / task["instance_id"]
    lockdir = config.ROOT / "pristine"
    lockdir.mkdir(parents=True, exist_ok=True)
    lock = open(lockdir / f"{task['instance_id']}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if (d / ".extracted").exists():
            return d
        tgz = Path(snapshots or config.SNAPSHOTS) / f"{task['instance_id']}.tgz"
        if not tgz.exists():
            raise FileNotFoundError(f"snapshot missing: {tgz}")
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        r = subprocess.run(["tar", "--no-same-owner", "-xzf", str(tgz), "-C", str(d)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"snapshot extraction failed: {r.stderr[:300]}")
        if not any(d.iterdir()):
            raise RuntimeError(f"snapshot extracted to an EMPTY tree: {tgz}")
        (d / ".extracted").write_text("ok")
        # the marker lives in the pristine dir only; it is not copied into workspaces (see below)
        return d
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def bootstrap(sb: Sandbox, task: dict, pristine: Path, py: str, log: dict) -> None:
    """HARNESS_README 4.2 steps 1-7, in order. pytest.ini and conftest.py are written BEFORE the
    baseline commit so they land in HEAD and never show up in the agent's diff."""
    t0 = time.time()
    sb.copy_tree_in(pristine)
    marker = sb.ws / ".extracted"
    if marker.exists():
        marker.unlink()
    # circular / dangling symlinks: the real harness strips them
    sb.sh(r"find . -maxdepth 6 -type l ! -exec test -e {} \; -delete 2>/dev/null || true", timeout=120)
    rc, _, _ = sb.sh("git rev-parse --is-inside-work-tree", timeout=60)
    if rc != 0:
        sb.sh("git init -q && git add -A && git commit -qm init --allow-empty", timeout=300)
    log["copy_s"] = round(time.time() - t0, 1)

    # environment: template per signature, private copy per run
    t0 = time.time()
    runtime, extras = envs.requirement_specs(sb.ws, task.get("repo", ""))
    pypi_first = envs.envs_norm_repo(task) not in envs.PUBLIC_REPOS
    asof = envs.asof_date(task)
    sig = envs.signature(py, runtime + ["--"] + extras + (["@pypi-first", f"@asof-{asof}"] if pypi_first else []))
    template, elog = envs.build_template(runtime, extras, py, sig, pypi_first=pypi_first, exclude_newer=asof)
    log["env_sig"], log["python"] = sig, py
    log["env_missing"] = elog.get("missing", [])
    log["env_pypi"] = sorted(k for k, v in elog.get("sources", {}).items() if v == "pypi")
    venv = envs.make_run_venv(template, sb.root / "venv")
    sb.venv = venv
    log["env_s"] = round(time.time() - t0, 1)

    # step 3 + 4 + 5 + 6 via the official helpers
    off = envs.official()
    off.setup_git_exclude(sb.ws)
    ok, tail = envs.editable_install(venv, sb.ws)
    log["editable"] = "ok" if ok else f"failed: {tail}"
    off.execute_fast_path(sb.ws, repo=task.get("repo", ""), site_packages_dir=envs.site_packages(venv))
    sb.fix_perms(sb.root)

    rc, so, se = sb.sh("git add -A && git commit -qm eval_baseline --allow-empty", timeout=300)
    if rc != 0:
        raise RuntimeError(f"baseline commit failed (rc={rc}): {(so or se)[:300]}")
    rc, so, se = sb.sh("git rev-parse HEAD", timeout=60)
    log["baseline_sha"] = so.strip()


def grade(task: dict, agent_patch: str, *, snapshots: Path | None = None, keep: bool = False,
          py: str | None = None, verbose: bool = False, pytest_timeout: int = config.PYTEST_TIMEOUT,
          user: str | None = config.SANDBOX_USER, block_network: bool = config.BLOCK_NETWORK) -> dict:
    t_all = time.time()
    res = {"instance_id": task["instance_id"], "resolved": False, "error": None, "stage": None,
           "apply_pass": None, "detail": None, "log": {}}
    py = py or envs.python_for(task["instance_id"])
    sb = None
    try:
        res["stage"] = "pristine"
        pristine = pristine_dir(task, snapshots)
        sb = Sandbox("grade", user=user, block_network=block_network)
        res["stage"] = "bootstrap"
        bootstrap(sb, task, pristine, py, res["log"])
        if verbose:
            print("   bootstrap:", json.dumps(res["log"])[:300])

        stripped: list[str] = []
        if agent_patch.strip():
            res["stage"] = "apply_agent_patch"
            kept, stripped = patching.strip_protected(agent_patch)
            res["log"]["stripped_protected"] = stripped
            if not kept.strip():
                res["error"] = "agent patch is empty after stripping protected files"
                return res
            ok, which = patching.apply_patch(sb, kept)
            res["apply_pass"] = which
            if not ok:
                res["error"] = "Failed to apply agent patch"
                return res
        else:
            res["error"] = "empty agent patch"
            # An empty patch is still graded (the empty control needs the verdict of the bare task).

        # reset protected paths the agent touched, plus every path test_patch will write
        reset = sorted(set(patching.patch_files(task["test_patch"]) +
                           [p for p in patching.patch_files(agent_patch) if patching.is_protected(p)]))
        if reset:
            q = " ".join(shlex.quote(p) for p in reset)
            sb.sh(f"tracked=$(git ls-tree -r --name-only HEAD -- {q} 2>/dev/null); "
                  f"[ -n \"$tracked\" ] && git ls-tree -r -z --name-only HEAD -- {q} | xargs -0 git checkout HEAD -- "
                  f"2>/dev/null || true", timeout=120)
            sb.sh(f"git clean -f -- {q} 2>/dev/null || true", timeout=120)

        res["stage"] = "apply_test_patch"
        ok, which = patching.apply_patch(sb, task["test_patch"], name="test.patch")
        if not ok:
            res["error"] = "Failed to apply test_patch (harness-side)"
            return res

        # refresh pytest.ini / conftest.py / import paths after test_patch, as the harness does
        off = envs.official()
        off.execute_fast_path(sb.ws, repo=task.get("repo", ""), site_packages_dir=envs.site_packages(sb.venv))
        sb.fix_perms(sb.ws)

        res["stage"] = "pytest"
        targets = patching.pytest_targets(task["test_patch"])
        if not targets:
            res["error"] = "no pytest targets in test_patch"
            return res
        xml = sb.tmp / "junit.xml"
        cmd = ("PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python -s -m pytest "
               + " ".join(shlex.quote(t) for t in targets)
               + f" --junitxml={shlex.quote(str(xml))} -p no:anyio -o timeout=0 "
               + "-o python_classes='Test* *Test' -q")
        t0 = time.time()
        try:
            rc, so, se = sb.sh(cmd, timeout=pytest_timeout)
        except SandboxTimeout:
            res.update(stage="pytest_timeout", resolved=False,
                       error=f"pytest exceeded {pytest_timeout}s (the scorer's verification timeout)")
            res["pytest_s"] = round(time.time() - t0, 1)
            return res
        res["pytest_s"] = round(time.time() - t0, 1)
        res["exit_code"] = rc
        required, req_src = patching.test_functions(task["test_patch"])
        res["required"], res["required_source"] = required, req_src
        ok, why = patching.validate_junit(sb.read(xml) or "", required)
        res["detail"] = f"{why} [required via {req_src}]"
        # an empty agent patch is graded but can never be a "resolution" by the harness's own rule
        res["resolved"] = bool(rc == 0 and ok) and bool(agent_patch.strip())
        res["passes_without_patch"] = bool(rc == 0 and ok) and not agent_patch.strip()
        if not res["resolved"]:
            res["error"] = res["error"] or f"exit={rc}; {why}"
            res["pytest_tail"] = (so + se)[-1200:]
            if verbose:
                print("   pytest tail:", res["pytest_tail"][-600:])
        else:
            res["error"] = None
        return res
    except Exception as e:                      # a grader defect must be visible, never a silent 'no'
        res["error"] = f"{type(e).__name__}: {e}"
        res["grader_error"] = True
        return res
    finally:
        res["seconds"] = round(time.time() - t_all, 1)
        if sb is not None and not keep:
            sb.stop()
        elif sb is not None:
            res["kept_dir"] = str(sb.root)
