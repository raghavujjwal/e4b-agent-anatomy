"""The teammates' `Container` interface (agent/vendor/sandbox.py) on our Docker-free `Sandbox`.

Their agent code (tools.py, runner.py, eval.py) talks to a container through five methods and hard-codes
`/workspace`. A subprocess sandbox has no such directory, so paths are mapped:
  * commands and file paths:  /workspace  ->  <run dir>/workspace
  * command output:           <run dir>/workspace  ->  /workspace   (so the agent sees the scorer's paths)
The mapping is textual, which is enough for the tool surface in HARNESS_README section 6. It is not a
chroot: a command that spells out another path reaches it. See docs/GRADER_DESIGN.md section 7.

`make_setup` and `make_verifier` plug into `eval.run_one(container=, setup=, verifier=)`.
"""
from __future__ import annotations
import os
import re
import shlex
import subprocess
from pathlib import Path

from . import config, envs, verify
from .sandbox import Sandbox, SandboxTimeout

_WS = re.compile(r"/workspace(?![\w.\-])")


class LocalContainer:
    def __init__(self, name_hint: str = "evalA", image=None, **sandbox_kw):
        # `image` is accepted and ignored: the interpreter comes from the task's environment template.
        self.sb = Sandbox(name_hint, **sandbox_kw)
        self.name = self.sb.root.name
        self.log: dict = {}

    # ---- path mapping --------------------------------------------------------------------------
    def _real(self, s: str) -> str:
        return _WS.sub(str(self.sb.ws), s)

    def _virtual(self, s: str) -> str:
        return s.replace(str(self.sb.ws), "/workspace")

    # ---- the five Container methods ------------------------------------------------------------
    def sh(self, cmd: str, timeout=600):
        """bash -c inside the workspace, like run_command. Raises subprocess.TimeoutExpired on timeout,
        which is what tools.run_command expects."""
        try:
            rc, so, se = self.sb.sh(self._real(cmd), timeout=timeout)
        except SandboxTimeout:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return rc, self._virtual(so), self._virtual(se)

    def put_text(self, path: str, content: str):
        real = Path(self._real(path))
        self.sb.sh(f"mkdir -p {shlex.quote(str(real.parent))}")        # created as the sandbox user
        real.write_text(content, encoding="utf-8", newline="\n")
        self.sb.fix_perms(real)

    def get_text(self, path: str):
        return self.sb.read(self._real(path))

    def get_bytes(self, path: str):
        """The file exactly as stored (None if absent). tools.Tools uses this so edit_file keeps CRLF files CRLF,
        as the scorer's docker-cp round trip does."""
        try:
            return Path(self._real(path)).read_bytes()
        except OSError:
            return None

    def put_bytes(self, path: str, data: bytes):
        real = Path(self._real(path))
        self.sb.sh(f"mkdir -p {shlex.quote(str(real.parent))}")
        real.write_bytes(data)
        self.sb.fix_perms(real)

    def put_archive(self, tgz_path: str, dest="/workspace"):
        real = Path(self._real(dest))
        real.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["tar", "--no-same-owner", "-xzf", tgz_path, "-C", str(real)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"snapshot extraction failed: {r.stderr[:300]}")
        if not any(real.iterdir()):
            raise RuntimeError(f"snapshot extracted to an EMPTY {dest}")
        self.sb.fix_perms(real)

    def stop(self):
        self.sb.stop()


def make_factory(**sandbox_kw):
    """container= hook: a fresh LocalContainer per task, with the isolation settings of config."""
    return lambda: LocalContainer("evalA", **sandbox_kw)


def make_setup(task: dict, snapshots=None, py: str | None = None):
    """setup= hook: the same bootstrap the grader uses (environment template, editable install,
    pytest.ini, baseline commit), then the tag that tools.submit_patch diffs against."""
    def setup(c: LocalContainer):
        pristine = verify.pristine_dir(task, Path(snapshots) if snapshots else None)
        verify.bootstrap(c.sb, task, pristine, py or envs.python_for(task["instance_id"]), c.log)
        rc, so, se = c.sb.sh("git tag -f _swegemma_baseline HEAD")
        if rc != 0:
            raise RuntimeError(f"could not tag the baseline: {(so or se)[:200]}")
    return setup


def make_verifier(snapshots=None, **grade_kw):
    """verifier= hook: grade(task, patch) already returns the keys eval.run_one reads
    (resolved, error, pytest_tail, required_source)."""
    return lambda task, patch: verify.grade(task, patch, snapshots=Path(snapshots) if snapshots else None,
                                            **grade_kw)
