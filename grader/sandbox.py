"""A local, Docker-free stand-in for the harness's per-task container.

Replaces `Container` from localeval/sandbox.py on branch `main`. The official harness has the same
shape in its `--sandbox subprocess` backend (HARNESS_README 4.1): an isolated temp directory with
`workspace/`, `tmp/` and a virtual environment, commands run through `bash -c` in a new process
group that is killed whole on timeout.

What this gives, and what it does not:
  * a fresh directory per run, a scrubbed environment (no tokens are ever passed in), a timeout that
    kills the whole process group, an optional address-space limit;
  * optional: run as an unprivileged user (so a command cannot touch Drive or secrets) and in a
    network namespace with only loopback up (like `--network none`). Both need root and are probed
    by `grader.probe`; when unavailable the sandbox says so instead of pretending.
It is NOT a security boundary against a hostile program. See docs/GRADER_DESIGN.md section 7.
"""
from __future__ import annotations
import os
import pwd
import resource
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from . import config


class SandboxTimeout(Exception):
    def __init__(self, cmd: str, seconds: float):
        super().__init__(f"command exceeded {seconds:.0f}s: {cmd[:120]}")
        self.cmd, self.seconds = cmd, seconds


def lookup_user(name: str | None):
    if not name:
        return None
    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        return None
    return pw.pw_uid, pw.pw_gid


class Sandbox:
    """One run directory: <root>/workspace, <root>/tmp, <root>/home, and a venv."""

    def __init__(self, name_hint: str = "run", venv: Path | None = None, base: Path | None = None,
                 user: str | None = config.SANDBOX_USER, block_network: bool = config.BLOCK_NETWORK,
                 mem_gb: float | None = 8):
        base = Path(base or (config.ROOT / "runs"))
        self.root = base / f"{name_hint}_{uuid.uuid4().hex[:10]}"
        self.ws = self.root / "workspace"
        self.tmp = self.root / "tmp"
        self.home = self.root / "home"
        for d in (self.ws, self.tmp, self.home):
            d.mkdir(parents=True, exist_ok=True)
        self.venv = Path(venv) if venv else None
        self.ids = lookup_user(user) if os.geteuid() == 0 else None
        if user and os.geteuid() == 0 and self.ids is None:
            raise RuntimeError(f"sandbox user {user!r} does not exist (create it first)")
        self.block_network = block_network
        self.mem_gb = mem_gb
        self._write_gitconfig()
        self.fix_perms(self.root)

    # ---- environment -------------------------------------------------------------------------
    def env(self, extra: dict | None = None) -> dict:
        """Built from scratch, so no secret in the parent environment can leak into a command."""
        path = [str(self.venv / "bin")] if self.venv else []
        path += ["/usr/local/bin", "/usr/bin", "/bin"]
        e = {
            "PATH": ":".join(path),
            "HOME": str(self.home),
            "TMPDIR": str(self.tmp),
            "TEST_TMPDIR": str(self.tmp),          # Dockerfile sets TEST_TMPDIR=/tmp
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_GLOBAL": str(self.home / ".gitconfig"),
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if self.venv:
            e["VIRTUAL_ENV"] = str(self.venv)
        e.update(extra or {})
        return e

    def _write_gitconfig(self):
        (self.home / ".gitconfig").write_text(
            "[user]\n\temail = agent@eval\n\tname = Agent\n[init]\n\tdefaultBranch = main\n"
            "[safe]\n\tdirectory = *\n", encoding="utf-8")

    def fix_perms(self, path: Path | str):
        """Hand a path to the sandbox user. No-op when not running as root."""
        if self.ids is None:
            return
        uid, gid = self.ids
        p = Path(path)
        os.chown(p, uid, gid)
        if p.is_dir():
            for dp, dns, fns in os.walk(p):
                for n in dns + fns:
                    try:
                        os.chown(os.path.join(dp, n), uid, gid, follow_symlinks=False)
                    except OSError:
                        pass

    # ---- running commands --------------------------------------------------------------------
    def _argv(self, cmd: str) -> list[str]:
        drop = []
        if self.ids is not None:
            uid, gid = self.ids
            drop = ["setpriv", f"--reuid={uid}", f"--regid={gid}", "--clear-groups", "--"]
        if self.block_network:
            # `--network none` with loopback up, as in the real container. Measured on Colab (probe, 2026-10-28):
            # plain `unshare -n` cannot bring lo up as root, but a user namespace created BY the already
            # unprivileged user can. So drop privileges first, then
            #   unshare --user --map-current-user --keep-caps --net
            # keeps the sandbox uid (no write access to Drive) and gets CAP_NET_ADMIN inside the namespace.
            # Without a sandbox user, `unshare -rn` is the fallback (runs as namespace root = host uid).
            # If lo cannot come up the command is NOT run: a silently broken network is worse than an error.
            ip = shutil.which("ip") or "/usr/sbin/ip"
            lo = (f"{shlex.quote(ip)} link set lo up || "
                  "{ echo 'sandbox: could not bring loopback up' >&2; exit 126; }; "
                  "exec setpriv --ambient-caps=-all -- bash -c \"$0\"")      # then drop the namespace caps
            ns = (["unshare", "--user", "--map-current-user", "--keep-caps", "--net"] if self.ids is not None
                  else ["unshare", "-rn"])
            return drop + ns + ["bash", "-c", lo, cmd]
        return drop + ["bash", "-c", cmd]

    def _limits(self):
        def apply():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if self.mem_gb:
                b = int(self.mem_gb * 2**30)
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
        return apply

    def sh(self, cmd: str, timeout: float = 600, cwd: Path | None = None,
           env: dict | None = None) -> tuple[int, str, str]:
        """Run through `bash -c` in the workspace, as run_command does. Returns (rc, stdout, stderr).
        On timeout the whole process group is killed and SandboxTimeout is raised."""
        # Bytes, decoded as docker exec output is (utf-8, errors replaced), NOT text=True: universal-newline mode
        # would turn the "\r" in CRLF files and progress bars into "\n", which the scorer's tool results keep.
        p = subprocess.Popen(self._argv(cmd), cwd=str(cwd or self.ws), env=self.env(env),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=self._limits())
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.communicate()
            raise SandboxTimeout(cmd, timeout)
        return p.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")

    # ---- files -------------------------------------------------------------------------------
    def read(self, path: Path | str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def copy_tree_in(self, src: Path, dest: Path | None = None):
        """Copy a pristine tree into the workspace (cp -a keeps modes, links and timestamps)."""
        dest = dest or self.ws
        r = subprocess.run(["cp", "-a", f"{src}/.", str(dest)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"copy into workspace failed: {r.stderr[:300]}")
        self.fix_perms(dest)

    def stop(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()
