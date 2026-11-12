"""Step 0 of docs/GRADER_DESIGN.md section 11: what can this machine do?

  python -m grader.probe            # prints a JSON report; paste it into docs/MEASUREMENTS.md

Run it first on a fresh Colab CPU runtime. It changes nothing outside a scratch directory.
"""
from __future__ import annotations
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import config


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def main() -> int:
    rep: dict = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
                 "platform": platform.platform(), "euid": os.geteuid(), "nproc": os.cpu_count()}
    rc, out = sh("free -g | awk '/Mem:/{print $2\" total, \"$7\" available GB\"}'")
    rep["memory"] = out
    root = Path(config.ROOT)
    root.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(root)
    rep["disk_free_gb"] = round(du.free / 1e9)
    for tool in ("git", "patch", "tar", "bash", "unshare", "setpriv", "ip", "uv"):
        rep.setdefault("tools", {})[tool] = shutil.which(tool) or (
            str(Path.home() / ".local/bin/uv") if tool == "uv" and (Path.home() / ".local/bin/uv").exists() else None)
    rc, out = sh("git --version")
    rep["git"] = out
    if rep["tools"].get("uv"):
        rc, out = sh(f"{rep['tools']['uv']} python list --only-installed 2>&1 | head -8")
        rep["uv_pythons"] = out.splitlines()
    # namespaces
    rc, out = sh("unshare -n true")
    rep["unshare_net_as_current_user"] = rc == 0
    rc, out = sh("unshare -rn true")
    rep["unshare_user_net"] = rc == 0
    if os.geteuid() == 0:
        rc, out = sh("unshare -n bash -c 'ip link set lo up && ip -o addr show lo | head -1'")
        rep["netns_loopback"] = out if rc == 0 else f"failed: {out[:100]}"
    # unprivileged user, and whether it can see Drive
    user = config.SANDBOX_USER or "gsandbox"
    if os.geteuid() == 0:
        rc, out = sh(f"id {user} >/dev/null 2>&1 || useradd -M -s /bin/bash {user}; id {user}")
        rep["sandbox_user"] = out if rc == 0 else f"cannot create {user}: {out[:100]}"
        rc, out = sh(f"setpriv --reuid=$(id -u {user}) --regid=$(id -g {user}) --clear-groups id -u")
        rep["setpriv_works"] = rc == 0
        drive = "/content/drive"
        if os.path.isdir(drive):
            rc, out = sh(f"setpriv --reuid=$(id -u {user}) --regid=$(id -g {user}) --clear-groups ls {drive}")
            rep["sandbox_user_can_list_drive"] = rc == 0
        else:
            rep["sandbox_user_can_list_drive"] = "Drive not mounted here (mount it to test this)"
    else:
        rep["sandbox_user"] = "not root: user switching not testable"
    # speed of the operations the grader depends on
    tmp = Path(tempfile.mkdtemp(prefix="probe_", dir=root))
    try:
        t0 = time.time()
        rc, out = sh(f"cd {tmp} && git init -q x && cd x && for i in $(seq 1 300); do echo $i > f$i.txt; done "
                     f"&& git add -A && git -c user.name=probe -c user.email=probe@local commit -qm c && cp -a . ../y",
                     timeout=120)
        rep["git_300files_commit_and_cp_s"] = round(time.time() - t0, 2) if rc == 0 else f"failed: {out[:100]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
