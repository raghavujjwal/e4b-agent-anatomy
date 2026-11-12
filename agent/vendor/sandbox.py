"""Thin docker wrapper + Container-A/B bootstrap, mirroring HARNESS_README section 4.2.

Deliberately uses the docker CLI rather than the SDK so this has no dependencies beyond the
standard library, matching the constraint the real harness operates under.
"""
from __future__ import annotations
import os, re, shlex, subprocess, tarfile, tempfile, time, uuid

DEFAULT_IMAGE = "swebench-sandbox:latest"


# DIVERGENCE 5 (see localeval/README.md): a supplementary wheelhouse mounted at /extra.
# /wheels is missing h11 entirely, and httpcore requires it, so httpx cannot be installed
# offline at all -- which left starlette's TestClient dead and made 65 of 67 fastapi tasks
# local blind spots. The real harness sidesteps this with a pre-baked site-packages that
# evidently contains h11. We recover the missing transitive dependencies instead, and record
# the divergence rather than pretending /wheels was sufficient.
EXTRA_WHEELS = os.environ.get("SWE_EXTRA_WHEELS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "extra_wheels")


def current_image():
    """Read SWE_IMAGE at CONTAINER CREATION time, not at import time. Binding it at import
    silently ignored every per-task image selected from env_map.json, so Container A and B
    both always ran the default -- which is how fastapi_14372 passed standalone under py311
    and failed end to end."""
    return os.environ.get("SWE_IMAGE") or DEFAULT_IMAGE

# HARNESS_README 4.2 step 3
GIT_EXCLUDE = ["__pycache__/", "*.pyc", ".pytest_cache/", "*.egg-info/", "build/", "dist/",
               ".coverage"]

# HARNESS_README 8.2 step 3: _is_protected_test_or_config_path
PROTECTED_NAMES = {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg",
                   ".pytest.ini", "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py"}
PROTECTED_RE = re.compile(
    r"(^|/)test_[^/]*\.py$"          # test_*.py
    r"|_test\.py$"                   # *_test.py
    r"|(^|/)(tests|test|testing)/.*\.py$"   # .py under tests/ test/ testing/
)


def is_protected(path: str) -> bool:
    if path.endswith(".pth"):
        return True
    if os.path.basename(path) in PROTECTED_NAMES:
        return True
    return bool(PROTECTED_RE.search(path))


def patch_files(patch: str) -> list[str]:
    """Files a unified diff touches. These diffs have no `diff --git` headers (see B-007)."""
    out = []
    for m in re.finditer(r"^\+\+\+ b/(\S+)$", patch, re.M):
        if m.group(1) not in out:
            out.append(m.group(1))
    for m in re.finditer(r"^--- a/(\S+)$", patch, re.M):
        if m.group(1) not in out and m.group(1) != "/dev/null":
            out.append(m.group(1))
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)", patch, re.M):
        if m.group(2) not in out:
            out.append(m.group(2))
    return [p for p in out if p != "/dev/null"]


def test_functions(test_patch: str) -> tuple[list[str], str]:
    """Required-to-pass nodes, and which signal produced them.

    The public tasks.jsonl carries no FAIL_TO_PASS field, so the harness's documented fallback
    is "test functions extracted from test_patch". Two signals, in order of trust:
      added-def  -- the patch adds `def test_x`, so test_x must pass
      hunk-ctx   -- the patch only EDITS inside an existing test (18% of tasks extend a
                    parametrize list and add no def at all); git names the enclosing
                    definition in the hunk header, which is suggestive but not reliable
    When neither yields anything the verdict rests on exit code 0 with at least one pass and
    no failures or errors -- which is why the empty-patch control is mandatory, not optional.
    """
    added = []
    for line in test_patch.splitlines():
        if line.startswith("+"):
            m = re.match(r"\+\s*(?:async\s+)?def\s+(test_\w+)", line)
            if m and m.group(1) not in added:
                added.append(m.group(1))
    if added:
        return added, "added-def"
    ctx = []
    for m in re.finditer(r"^@@[^@]*@@\s*(?:async\s+)?def\s+(test_\w+)", test_patch, re.M):
        if m.group(1) not in ctx:
            ctx.append(m.group(1))
    if ctx:
        return ctx, "hunk-ctx"
    return [], "exit-code-only"


class Container:
    def __init__(self, name_hint="swe", memory="4g", cpus="2", image=None):
        # `image` is passed EXPLICITLY, never read from the process environment, because
        # SWE_IMAGE is process-global and eval.py --parallel runs tasks in threads. A thread
        # that set it for its own task could have it overwritten before its container was
        # created, so a task would silently run in its neighbour's interpreter. rich_3472
        # caught exactly that: it failed under --parallel with
        #   'tests.test_pretty....<locals>.Foo' object has no attribute 'bar'
        # against an expected "'Foo' object ...", which is a newer CPython's AttributeError
        # wording -- i.e. the wrong python, not a wrong patch. Same defect as EXP-009's
        # "IMAGE bound at module import", one layer up.
        self.name = f"{name_hint}_{uuid.uuid4().hex[:10]}"
        cmd = ["docker", "run", "-d", "--name", self.name, "--network", "none",
               "--memory", memory, "--cpus", cpus, "-w", "/workspace",
               "-e", "TEST_TMPDIR=/tmp"]
        if os.path.isdir(EXTRA_WHEELS):
            cmd += ["-v", f"{os.path.abspath(EXTRA_WHEELS)}:/extra:ro"]
        cmd += [image or current_image(), "sleep", "infinity"]
        subprocess.run(cmd, check=True, capture_output=True)

    def sh(self, cmd: str, timeout=600):
        """Run through bash -c inside /workspace, exactly as run_command does."""
        r = subprocess.run(["docker", "exec", "-w", "/workspace", self.name,
                            "/bin/bash", "-lc", cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr

    def put_text(self, path: str, content: str):
        with tempfile.TemporaryDirectory() as td:
            local = os.path.join(td, os.path.basename(path) or "f")
            with open(local, "w") as fh:
                fh.write(content)
            self.sh(f"mkdir -p {shlex.quote(os.path.dirname(path) or '/')}")
            subprocess.run(["docker", "cp", local, f"{self.name}:{path}"],
                           check=True, capture_output=True)

    def get_text(self, path: str):
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(["docker", "cp", f"{self.name}:{path}", td],
                               capture_output=True)
            if r.returncode != 0:
                return None
            local = os.path.join(td, os.path.basename(path))
            if not os.path.exists(local):
                return None
            return open(local, errors="replace").read()

    def get_bytes(self, path: str):
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(["docker", "cp", f"{self.name}:{path}", td], capture_output=True)
            local = os.path.join(td, os.path.basename(path))
            if r.returncode != 0 or not os.path.isfile(local):
                return None
            with open(local, "rb") as fh:
                return fh.read()

    def put_bytes(self, path: str, data: bytes):
        with tempfile.TemporaryDirectory() as td:
            local = os.path.join(td, os.path.basename(path) or "f")
            with open(local, "wb") as fh:
                fh.write(data)
            self.sh(f"mkdir -p {shlex.quote(os.path.dirname(path) or '/')}")
            subprocess.run(["docker", "cp", local, f"{self.name}:{path}"], check=True, capture_output=True)

    def exec_scorer(self, cmd: str, timeout):
        """run_command's execution as swegemma's ContainerManager.exec does it: `/bin/bash -c` (not a login
        shell) around `timeout -k 5s N /bin/bash -c <cmd>`; exit 124 after about N seconds is a timeout."""
        wrapped = f"timeout -k 5s {timeout} /bin/bash -c {shlex.quote(cmd)}"
        t0 = time.time()
        r = subprocess.run(["docker", "exec", "-w", "/workspace", self.name, "/bin/bash", "-c", wrapped],
                           capture_output=True, timeout=timeout + 30)
        if r.returncode == 124 and time.time() - t0 >= max(0.0, timeout - 1.0):
            raise subprocess.TimeoutExpired(cmd, timeout)
        return r.returncode, r.stdout.decode("utf-8", errors="replace"), r.stderr.decode("utf-8", errors="replace")

    def put_archive(self, tgz_path: str, dest="/workspace"):
        self.sh(f"mkdir -p {dest}")
        subprocess.run(["docker", "cp", tgz_path, f"{self.name}:/tmp/snap.tgz"],
                       check=True, capture_output=True)
        # --no-same-owner matters: tar run as root restores the uid recorded in the archive,
        # which is the host builder's uid, and git then refuses every repository operation with
        # "detected dubious ownership". That failure was silent behind `|| true` and made the
        # baseline commit, submit_patch extraction AND Container B's anti-tamper reset no-ops.
        rc, so, se = self.sh(
            f"tar --no-same-owner -xzf /tmp/snap.tgz -C {dest} && rm -f /tmp/snap.tgz",
            timeout=600)
        if rc != 0:
            raise RuntimeError(f"snapshot extraction failed (rc={rc}): {(so or se)[:300]}")
        rc, so, _ = self.sh(f"ls -A {dest} | wc -l")
        if rc != 0 or int((so or '0').strip() or 0) == 0:
            raise RuntimeError(f"snapshot extracted to an EMPTY {dest}")

    def stop(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def bootstrap(c: Container, tgz: str, repo: str, baseline_msg: str, reqs=None):
    """HARNESS_README 4.2 steps 1-7, in order. The ORDER matters: pytest.ini and conftest.py
    are written BEFORE the baseline commit, so they land in HEAD and do not show up in diffs."""
    log = {}
    # belt and braces: also tell git not to care, in case an archive carries odd ownership
    c.sh("git config --global --add safe.directory '*' && "
         "git config --global --add safe.directory /workspace")
    c.put_archive(tgz)
    # some snapshots contain circular symlinks; the real harness strips them
    c.sh(r"find /workspace -maxdepth 6 -type l ! -exec test -e {} ; -delete 2>/dev/null || true")
    c.sh("git -C /workspace rev-parse --is-inside-work-tree || (git -C /workspace init -q && "
         "git -C /workspace add -A && git -C /workspace commit -qm init --allow-empty)")
    c.sh("mkdir -p /workspace/.git/info && printf '%s\\n' " +
         " ".join(shlex.quote(p) for p in GIT_EXCLUDE) + " >> /workspace/.git/info/exclude")
    # DIVERGENCE FROM THE REAL HARNESS, recorded deliberately.
    # The harness installs with --no-deps and then streams a pre-baked unpacked site-packages
    # cache (README 4.2 step 5). We have no such cache, only /wheels. And /wheels ships BOTH
    # cp312 and cp313 builds of a few packages while the image is python 3.13, so
    # sandbox/setup.py's deduplicate_wheels picks by version without checking ABI tags, hands
    # pip one incompatible wheel, and -- because it installs all resolved wheels in a single
    # command -- the whole dependency install aborts. That is how `requests` tasks ended up
    # with no urllib3.
    # So we hand pip the requirement NAMES via an editable install WITH dependencies and let
    # pip's own resolver choose ABI-compatible wheels from --find-links. Falls back to the
    # harness's exact --no-deps form if that cannot be satisfied offline.
    if reqs:
        # unseen52 (Q-051): each task ships the exact frozen requirement set its gold check
        # passed with. Install exactly that, --no-deps, from its own wheelhouse; then the repo
        # --no-deps; then setup.py --fast-path, which only writes the grader's pytest.ini /
        # conftest.py and import paths (--no-fast-path would re-resolve deps against /wheels
        # and could overwrite the frozen versions). Our testdeps heuristics are skipped too.
        rc, so, se = c.sh("pip install --no-index --find-links=/wheels_u52 --no-deps "
                          + " ".join(shlex.quote(r) for r in reqs) + " 2>&1 | tail -3", timeout=1800)
        log["frozen_reqs"] = "rc=%d " % rc + (so or se)[-300:]
        rc, so, se = c.sh("pip install --no-index --find-links=/wheels_u52 --no-build-isolation "
                          "--no-deps -e /workspace 2>&1 | tail -3", timeout=1800)
        log["editable_install"] = "FROZEN rc=%d " % rc + (so or se)[-300:]
        rc, so, se = c.sh(f"python3 /sandbox/setup.py --fast-path {shlex.quote(repo)} 2>&1 | tail -6",
                          timeout=1800)
        log["setup_py"] = (so or se)[-500:]
        return _baseline(c, baseline_msg, log)
    rc, so, se = c.sh("pip install --no-index --find-links=/wheels --find-links=/extra "
                      "--no-build-isolation -e /workspace 2>&1 | tail -4", timeout=1800)
    log["editable_install"] = ("WITH-DEPS rc=%d " % rc) + (so or se)[-300:]
    if rc != 0:
        rc2, so2, se2 = c.sh("pip install --no-index --find-links=/wheels --find-links=/extra "
                             "--no-build-isolation --no-deps -e /workspace 2>&1 | tail -3 "
                             "|| true", timeout=900)
        log["editable_install"] += " | FELL BACK TO --no-deps rc=%d " % rc2 + (so2 or se2)[-200:]
    # --no-fast-path is the SLOW path, and the slow path is the one that actually resolves
    # the repo's declared dependencies against /wheels and pip-installs them. --fast-path only
    # injects import paths and writes pytest.ini/conftest.py, which left urllib3 missing and
    # made requests tasks fail at collection.
    rc, so, se = c.sh(f"python3 /sandbox/setup.py --no-fast-path {shlex.quote(repo)} 2>&1 | tail -6",
                      timeout=1800)
    log["setup_py"] = (so or se)[-500:]
    # TEST dependencies, best effort and one pip call per package. `pip install -e .` brings
    # only runtime deps, so starlette's TestClient had no httpx and every fastapi test errored
    # at collection -- 65 of 67 fastapi tasks were blind spots for want of one package.
    here = os.path.dirname(os.path.abspath(__file__))
    c.put_text("/tmp/_testdeps.py", open(os.path.join(here, "testdeps.py")).read())
    rc, so, se = c.sh("python3 /tmp/_testdeps.py /workspace 2>&1 | tail -4", timeout=1800)
    log["testdeps"] = (so or se)[-400:]
    return _baseline(c, baseline_msg, log)


def _baseline(c: Container, baseline_msg: str, log: dict):
    rc, so, se = c.sh(f"cd /workspace && git add -A && "
                      f"git commit -qm {shlex.quote(baseline_msg)} --allow-empty")
    if rc != 0:
        raise RuntimeError(f"baseline commit failed (rc={rc}): {(so or se)[:300]}")
    rc, so, se = c.sh("cd /workspace && git rev-parse HEAD")
    sha = so.strip()
    if rc != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"no baseline commit after bootstrap: rc={rc} {(so or se)[:300]}")
    log["baseline_sha"] = sha
    # prove the index is usable, since every later step depends on it
    rc, so, se = c.sh("cd /workspace && git status --porcelain >/dev/null")
    if rc != 0:
        raise RuntimeError(f"git index unusable after bootstrap: {(so or se)[:300]}")
    return log


def apply_patch(c: Container, patch: str):
    """HARNESS_README 8.2 step 2: four resilient passes. Returns (ok, which_pass)."""
    if not patch.strip():
        return False, "empty"
    c.put_text("/tmp/agent.patch", patch if patch.endswith("\n") else patch + "\n")
    attempts = [
        ("p1", "git apply --unsafe-paths -p1 /tmp/agent.patch"),
        ("p1-3way", "git apply --unsafe-paths -p1 -3 /tmp/agent.patch"),
        ("p1-ws", "git apply --unsafe-paths -p1 --ignore-space-change --ignore-whitespace /tmp/agent.patch"),
        ("p1-recount", "git apply --unsafe-paths -p1 --recount /tmp/agent.patch"),
        ("p0", "git apply --unsafe-paths -p0 /tmp/agent.patch"),
        ("gnu-p1", "patch -p1 --batch --forward --dry-run < /tmp/agent.patch && patch -p1 --batch --forward < /tmp/agent.patch"),
        ("gnu-p0", "patch -p0 --batch --forward --dry-run < /tmp/agent.patch && patch -p0 --batch --forward < /tmp/agent.patch"),
        ("gnu-p1-l", "patch -p1 -l --batch --forward --dry-run < /tmp/agent.patch && patch -p1 -l --batch --forward < /tmp/agent.patch"),
    ]
    for name, cmd in attempts:
        rc, _, _ = c.sh(f"cd /workspace && {cmd}", timeout=300)
        if rc == 0:
            return True, name
    return False, "all_passes_failed"
