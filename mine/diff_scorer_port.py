"""Differential test: agent/vendor/scorer_port.py + tools.py against the scorer's own modules.

Run inside the scorer venv (it imports swegemma, adk_eval_core and adk_submission, which are not dependencies of
this repo):

    D:\\envs\\scorer\\Scripts\\python.exe -I mine\\diff_scorer_port.py [--cases N] [--seed S]

Exit status 0 means every compared value was identical; any mismatch prints the case and exits 1. The cases are
random but seeded, so a failure reproduces. What is compared:

  edit matcher    adk_eval_core.editing.apply_replacement and swegemma.edit.apply_replacement (result or exception text)
  tools           swegemma.tools.workspace.{read_file,edit_file,write_file}, execution.{run_command,submit_patch,
                  get_status} against Tools.* bound to a container over the same files; result strings and the
                  files left on disk
  budget          check_budget refusals and budget_warning over a full 24-call session
  patch           verification.strip_protected_files_from_patch on random multi-file diffs
  prompt          agent_runner.build_agent_prompt and build_workspace_tree_command
  generation      GenerationConstraints.validate_config + apply_thinking_config_to_model
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cases", type=int, default=3000)
ap.add_argument("--seed", type=int, default=20261104)
args = ap.parse_args()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agent" / "vendor"))
import scorer_port as P  # noqa: E402
import tools as T  # noqa: E402

import adk_eval_core.editing as core_edit  # noqa: E402
import swegemma.edit as real_edit  # noqa: E402
from adk_eval_core.sandbox import ExecutionResult  # noqa: E402
from adk_submission.resolvers.generation import apply_thinking_config_to_model  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from swegemma.budget import EvaluationBudget, HarnessLimits  # noqa: E402
from swegemma.config import build_submission_limits  # noqa: E402
from swegemma.context import SwegemmaContext  # noqa: E402
from swegemma.harness import agent_runner, verification  # noqa: E402
from swegemma.models import Task  # noqa: E402
from swegemma.tools import execution as real_exec  # noqa: E402
from swegemma.tools import workspace as real_ws  # noqa: E402

rng = random.Random(args.seed)
FAILS: list[str] = []
COUNTS: dict[str, int] = {}
FAIL_KINDS: dict[str, int] = {}


def _brief(v):
    if isinstance(v, dict) and all(isinstance(k, str) for k in v):
        return {k: (x if not isinstance(x, bytes) or len(x) < 80 else x[:80] + b"...") for k, x in v.items()}
    return v


def check(kind: str, got, want, case: str) -> None:
    COUNTS[kind] = COUNTS.get(kind, 0) + 1
    if got != want:
        FAIL_KINDS[kind] = FAIL_KINDS.get(kind, 0) + 1
        if isinstance(got, dict) and isinstance(want, dict):
            keys = [k for k in set(got) | set(want) if got.get(k) != want.get(k)]
            got, want = {k: got.get(k) for k in keys}, {k: want.get(k) for k in keys}
        FAILS.append(f"[{kind}] {case[:400]}\n    port: {str(_brief(got))[:600]}\n    real: {str(_brief(want))[:600]}")


# --------------------------------------------------------------------------------------------- generators
WORDS = ["alpha", "beta", "gamma", "value", "result", "items", "config", "self", "node", "path", "count", "name"]
OPS = ["==", "!=", "<=", ">=", "->", ":=", "+=", "-=", "**", "//", "(", ")", "[", "]", "{", "}", ",", ":", "+", "-", "*"]


def rand_line(indent: int) -> str:
    kind = rng.random()
    pad = " " * indent
    if kind < 0.25:
        return f"{pad}def {rng.choice(WORDS)}_{rng.randint(0, 3)}({rng.choice(WORDS)}, {rng.choice(WORDS)}=None):"
    if kind < 0.55:
        return f"{pad}{rng.choice(WORDS)} = {rng.choice(WORDS)}[{rng.randint(0, 9)}] + {rng.randint(0, 99)}"
    if kind < 0.7:
        return f"{pad}if {rng.choice(WORDS)} == {rng.randint(0, 9)}:"
    if kind < 0.8:
        return f'{pad}print("{rng.choice(WORDS)} \\n caf\u00e9 \\"q\\" \\u00e9")'
    if kind < 0.9:
        return f"{pad}return {rng.choice(WORDS)}"
    if kind < 0.95:
        return ""
    return f"{pad}# {rng.choice(WORDS)}\t{rng.choice(WORDS)}  "


def rand_content() -> str:
    n = rng.randint(1, 40)
    lines = []
    indent = 0
    for _ in range(n):
        indent = max(0, min(12, indent + rng.choice([-4, 0, 0, 0, 4])))
        lines.append(rand_line(indent))
    if rng.random() < 0.2:                       # repeat a block so duplicates exist
        k = rng.randint(1, min(3, len(lines)))
        lines = lines + lines[:k]
    sep = "\r\n" if rng.random() < 0.15 else "\n"
    text = sep.join(lines)
    if rng.random() < 0.8:
        text += sep
    if rng.random() < 0.05:
        text = text.replace("\n", "\n\x0c", 1)
    return text


def rand_edit(content: str):
    """(old, new, allow_multiple) crafted to land in each tier."""
    lines = content.splitlines()
    nonblank = [i for i, l in enumerate(lines) if l.strip()]
    mode = rng.choice(["exact", "exact", "flex_r", "flex_s", "regex", "json", "dup", "missing", "empty", "same", "multi"])
    allow = rng.random() < 0.3
    if mode == "empty":
        return "", "x", allow
    if not nonblank:
        return "zz", "yy", allow
    i = rng.choice(nonblank)
    k = rng.randint(1, 3)
    chunk = lines[i:i + k]
    new_chunk = [l.replace(rng.choice(WORDS), rng.choice(WORDS)) for l in chunk]
    if rng.random() < 0.3:
        new_chunk = new_chunk + [" " * rng.choice([0, 4, 8]) + "extra_line = 1"]
    if rng.random() < 0.1:
        new_chunk = []
    new = "\n".join(new_chunk)
    if mode == "same":
        old = "\n".join(chunk)
        return old, old, allow
    if mode == "missing":
        return "this text is not in the file " + str(rng.randint(0, 99)), new, allow
    if mode == "exact":
        return "\n".join(chunk), new, allow
    if mode == "dup" or mode == "multi":
        return chunk[0].strip() if chunk[0].strip() else "x", new, allow
    if mode == "flex_r":
        return "\n".join(l + rng.choice(["", " ", "  ", "\t"]) for l in chunk), new, allow
    if mode == "flex_s":
        return "\n".join(rng.choice(["", " ", "    "]) + l.strip() for l in chunk), new, allow
    if mode == "regex":
        old = "\n".join(chunk)
        old = old.replace(" = ", "=").replace(", ", ",").replace("(", " ( ")
        return old, new, allow
    if mode == "json":
        old = "\n".join(chunk).replace("\n", "\\n").replace('"', '\\"').replace("\u00e9", "\\u00e9")
        return old, new.replace('"', '\\"'), allow
    return "\n".join(chunk), new, allow


def run_real_edit(content, old, new, allow, name="f.py"):
    try:
        r = real_edit.apply_replacement(content, old, new, filepath=name, allow_multiple=allow)
        return ("ok", r.new_content, r.diff, r.occurrences, r.strategy)
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__, str(e))


def run_port_edit(content, old, new, allow, name="f.py"):
    try:
        r = P.apply_replacement(content, old, new, filepath=name, allow_multiple=allow)
        return ("ok", r.new_content, r.diff, r.occurrences, r.strategy)
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__, str(e))


# ------------------------------------------------------------------------------------------ 1. the matcher
for n in range(args.cases):
    content = rand_content()
    old, new, allow = rand_edit(content)
    case = f"edit#{n} old={old!r} new={new!r} allow={allow} content={content[:300]!r}"
    check("edit.wrapper", run_port_edit(content, old, new, allow), run_real_edit(content, old, new, allow), case)
    a = P.apply_replacement_core(content, old, new, allow)
    b = core_edit.apply_replacement(content, old, new, allow)
    check("edit.core", (a.new_content, a.occurrences, a.strategy, a.error_message),
          (b.new_content, b.occurrences, b.strategy, b.error_message), case)


# --------------------------------------------------------------------------------- 2. tools over a workspace
class RealDocker:
    """The sandbox manager interface the real tools use, over a temp directory."""

    def __init__(self, root: Path, diff_text: str = "", cmd_result: ExecutionResult | None = None):
        self.root, self.diff_text, self.cmd_result = root, diff_text, cmd_result
        self.cmds: list[str] = []

    def _map(self, src):
        ws = Path("/workspace").resolve()
        return self.root / Path(str(src).rstrip("/\\")).relative_to(ws)

    def copy_from(self, cid, src, local):
        real = self._map(src)
        if not real.exists():
            raise FileNotFoundError(f"no such file: {src}")
        shutil.copyfile(real, local)

    def copy_to(self, cid, local, dest_dir):
        d = self._map(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, d / Path(local).name)

    def exec(self, cid, command, timeout=None):
        self.cmds.append(command)
        if command.startswith("mkdir -p"):
            return ExecutionResult(status="ok", exit_code=0)
        if "git diff" in command:
            return ExecutionResult(status="ok", stdout=self.diff_text, exit_code=0)
        if command.startswith("cd /workspace && git add"):
            return ExecutionResult(status="ok", exit_code=0)
        return self.cmd_result or ExecutionResult(status="ok", stdout="ok\n", exit_code=0)

    def get_sandbox(self, *a, **k):
        return None


class PortContainer:
    """The five container methods tools.Tools uses, over the same kind of directory."""

    def __init__(self, root: Path, diff_text: str = "", cmd_result: ExecutionResult | None = None):
        self.root, self.diff_text, self.cmd_result = root, diff_text, cmd_result
        self.cmds: list[str] = []

    def _p(self, path):
        return self.root / path[len("/workspace/"):]

    def get_bytes(self, path):
        p = self._p(path)
        return p.read_bytes() if p.is_file() else None

    def put_bytes(self, path, data):
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def sh(self, cmd, timeout=600):
        self.cmds.append(cmd)
        if cmd.startswith("mkdir -p"):
            return 0, "", ""
        if "git diff" in cmd:
            return 0, self.diff_text, ""
        if cmd.startswith("cd /workspace && git add"):
            return 0, "", ""
        r = self.cmd_result or ExecutionResult(status="ok", stdout="ok\n", exit_code=0)
        if r.status == "timeout":
            import subprocess
            raise subprocess.TimeoutExpired(cmd, timeout)
        return r.exit_code, r.stdout, r.stderr


def make_ctx(docker, tool_calls=24, time_minutes=5.5, turns=40, cmd_timeout=90):
    task = Task(instance_id="d__r-1", repo="d/r", base_commit="abc", problem_statement="p", hints_text="")
    ctx = SwegemmaContext(docker_manager=docker, container_id="c1", problem_statement="p", hints_text="", task=task,
                          repo="d/r", budget=EvaluationBudget(time_minutes=time_minutes, tool_calls=tool_calls, turns=turns),
                          harness=HarnessLimits(command_timeout_seconds=cmd_timeout, max_stdout_chars=5000,
                                                max_file_lines=150, max_file_chars=10000))
    ctx.start_agent_session()
    return ctx


def make_port(container, tool_calls=24, time_minutes=5.5, turns=40, cmd_timeout=90):
    b = T.Budget(tool_calls=tool_calls, time_minutes=time_minutes, turns=turns, command_timeout=cmd_timeout)
    b.start()
    return T.Tools(container, b)


def tree_bytes(root: Path) -> dict:
    return {str(p.relative_to(root)).replace("\\", "/"): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def seed_files() -> dict[str, bytes]:
    files = {"a.py": rand_content().encode("utf-8"), "pkg/b.py": rand_content().encode("utf-8"),
             "empty.py": b"", "nonl.txt": b"one\ntwo", "crlf.txt": b"x\r\ny\r\nz\r\n",
             "long.py": ("\n".join(f"line {i} " + "x" * rng.randint(0, 120) for i in range(rng.randint(1, 400)))).encode(),
             "wide.txt": ("w" * rng.randint(9000, 30000)).encode(), "uni.py": "NAME = 'caf\u00e9 \u2192 ok'\n".encode(),
             "ctrl.txt": b"a\x0cb\x1cc\x85d\xe2\x80\xa8e\n", "bad.bin": b"ok \xff\xfe bad\n"}
    return files


ODD_INTS = [None, 0, 1, 2, 5, -3, 10, 150, 151, 400, 10 ** 6, "7", " 8 ", "none", "", "x1", 3.0, 4.7, True, [1], {"a": 1}]
PATHS = ["a.py", "pkg/b.py", "empty.py", "nonl.txt", "crlf.txt", "long.py", "wide.txt", "uni.py", "ctrl.txt", "bad.bin",
         "/workspace/a.py", "./a.py", "pkg/../a.py", "../etc/passwd", "/etc/passwd", "", ".", "/workspace", "pkg//b.py", None, 7]

for n in range(max(300, args.cases // 4)):
    files = seed_files()
    d1, d2 = Path(tempfile.mkdtemp(prefix="dr_")), Path(tempfile.mkdtemp(prefix="dp_"))
    try:
        write_tree(d1, files)
        write_tree(d2, files)
        ctx = make_ctx(RealDocker(d1))
        tools = make_port(PortContainer(d2))
        op = rng.choice(["read", "read", "edit", "edit", "write"])
        fp = rng.choice(PATHS)
        if op == "read":
            s, e = rng.choice(ODD_INTS), rng.choice(ODD_INTS)
            got = tools.read_file(fp, s, e)
            if fp in ("pkg//b.py",):
                pass
            want = real_ws.read_file(ctx, fp, s, e)
            # a missing file is reported by docker itself; compare the rest
            if "no such file" in want:
                check("tools.read.missing", json.loads(got)["error_type"], "FileReadError", f"read {fp!r}")
            else:
                check("tools.read_file", got, want, f"read {fp!r} {s!r} {e!r}")
        elif op == "edit":
            base = files.get(fp if isinstance(fp, str) else "a.py", b"").decode("utf-8", "replace")
            old, new, allow = rand_edit(base)
            allow = rng.choice([allow, "true", "False", "yes", "0", 1, 0, None])
            got = tools.edit_file(fp, old, new, allow)
            want = real_ws.edit_file(ctx, fp, old, new, allow)
            if "no such file" in want:
                check("tools.edit.missing", json.loads(got)["error_type"], "FileEditError", f"edit {fp!r}")
            else:
                check("tools.edit_file", got, want, f"edit {fp!r} old={old!r} new={new!r} allow={allow!r}")
            check("tools.edit.files", tree_bytes(d2), tree_bytes(d1), f"edit-files {fp!r} old={old!r}")
        else:
            content = rng.choice(["x\n", "caf\u00e9 \u2192\n", "", "a\r\nb\n", "tab\there"])
            got = tools.write_file(fp, content)
            want = real_ws.write_file(ctx, fp, content)
            check("tools.write_file", got, want, f"write {fp!r} {content!r}")
            real_tree = tree_bytes(d1)
            if os.linesep != "\n":
                # the scorer is Linux; on this Windows host Path.write_text turns "\n" into "\r\n", which it would not
                real_tree = {k: (v.replace(b"\r\n", b"\n") if "\r" not in content and files.get(k) != v else v)
                             for k, v in real_tree.items()}
            if os.linesep == "\n" or "\r" not in content:
                check("tools.write.files", tree_bytes(d2), real_tree, f"write-files {fp!r}")
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)

# ------------------------------------------------------------------------------------------- run_command
RESULTS = [
    ExecutionResult(status="ok", stdout="hello\n", stderr="", exit_code=0),
    ExecutionResult(status="ok", stdout="out", stderr="warn caf\u00e9\n", exit_code=0),
    ExecutionResult(status="error", stdout="", stderr="boom\n", exit_code=1),
    ExecutionResult(status="error", stdout="only stdout", stderr="", exit_code=2),
    ExecutionResult(status="error", stdout="o" * 7000, stderr="e" * 6000, exit_code=127),
    ExecutionResult(status="ok", stdout="y" * 12000, stderr="", exit_code=0),
    ExecutionResult(status="timeout", stdout="", stderr="", exit_code=124),
]
for r in RESULTS:
    for cmd_timeout, minutes in [(90, 5.5), (300, 60.0), (None, 5.5), (90, None)]:
        d = Path(tempfile.mkdtemp(prefix="rc_"))
        try:
            ctx = make_ctx(RealDocker(d, cmd_result=r), time_minutes=minutes if minutes is not None else 5.5,
                           cmd_timeout=cmd_timeout)
            if minutes is None:
                ctx.budget = EvaluationBudget(time_minutes=None, tool_calls=24, turns=40)
            t = make_port(PortContainer(d, cmd_result=r), time_minutes=minutes if minutes is not None else 5.5,
                          cmd_timeout=cmd_timeout)
            if minutes is None:
                t.b.time_minutes = None
            time.sleep(0.03)        # the scorer's clock is perf_counter; time.time() can still read 0.0 elapsed here
            got = t.run_command("echo hi")
            want = real_exec.run_command(ctx, "echo hi")
            check("tools.run_command", got, want, f"run {r.status} {r.exit_code} timeout={cmd_timeout} min={minutes}")
        finally:
            shutil.rmtree(d, ignore_errors=True)

# -------------------------------------------------------------------------------------- submit / get_status
DIFFS = [
    "",
    "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n x\n-y\n+z\n",
    "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1 @@\n-a\n+b\n"
    "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n",
    "diff --git a/conftest.py b/conftest.py\nnew file mode 100644\n--- /dev/null\n+++ b/conftest.py\n@@ -0,0 +1 @@\n+x\n",
]
for dtext in DIFFS:
    d = Path(tempfile.mkdtemp(prefix="sp_"))
    try:
        ctx = make_ctx(RealDocker(d, diff_text=dtext))
        t = make_port(PortContainer(d, diff_text=dtext))
        got, want = t.submit_patch(), real_exec.submit_patch(ctx)
        check("tools.submit_patch", got, want, f"submit {dtext[:60]!r}")
        check("tools.submit_patch.state", (t.submitted_patch, t.patch_submitted), (ctx.submitted_patch, ctx.patch_submitted),
              f"submit-state {dtext[:60]!r}")
        # get_status: same keys in the same order; compare everything but the two clock fields
        g, w = json.loads(t.get_status()), json.loads(real_exec.get_status(ctx))
        check("tools.get_status.keys", list(g), list(w), "status keys")
        for k in w:
            if k not in ("time_seconds_remaining", "agent_elapsed_seconds"):
                check("tools.get_status.values", g[k], w[k], f"status {k}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
d = Path(tempfile.mkdtemp(prefix="gs_"))
try:
    ctx = make_ctx(RealDocker(d))
    t = make_port(PortContainer(d))
    check("tools.get_status.fresh", list(json.loads(t.get_status())), list(json.loads(real_exec.get_status(ctx))), "fresh keys")
finally:
    shutil.rmtree(d, ignore_errors=True)

# ----------------------------------------------------------------------------- budget walk and the warning
for calls, turns in [(24, 40), (10, 40), (20, 40), (30, 5)]:
    d = Path(tempfile.mkdtemp(prefix="bw_"))
    try:
        write_tree(d, {"a.py": b"x = 1\n"})
        ctx = make_ctx(RealDocker(d), tool_calls=calls, turns=turns)
        t = make_port(PortContainer(d), tool_calls=calls, turns=turns)
        wrapped = real_ws.read_file
        for i in range(calls + 4):
            ctx.llm_calls_used = min(i + 1, turns)      # one model response per call, as in the scorer
            t.b.turns_used = min(i + 1, turns)
            got = t.call("read_file", {"filepath": "a.py"})
            want = wrapped(ctx, "a.py")
            check("budget.walk", got, want, f"calls={calls} turns={turns} i={i}")
        got, want = t.call("get_status", {}), real_exec.get_status(ctx)
        check("budget.free_tool_after_exhaustion", json.loads(got).get("status"), json.loads(want).get("status"), "status after")
    finally:
        shutil.rmtree(d, ignore_errors=True)

# ----------------------------------------------------------------------------------------- patch stripping
PATH_POOL = ["src/pkg/mod.py", "tests/test_mod.py", "pkg/test_x.py", "pkg/x_test.py", "conftest.py", "setup.cfg",
             "pyproject.toml", "docs/index.md", "pkg/Tests/helper.py", "pkg/testing/util.py", "a.pth", "data/c\u00e9.py",
             "src/test/inner.py", "README.rst", "pkg/testing.py", "test_top.py"]


def fake_diff(paths: list[str]) -> str:
    out = []
    for p in paths:
        kind = rng.choice(["mod", "mod", "new", "del", "rename", "binary"])
        q = f'"{p}"' if any(ord(c) > 127 for c in p) else p
        qq = p.encode("unicode_escape").decode().replace("\\\\", "\\") if any(ord(c) > 127 for c in p) else p
        if any(ord(c) > 127 for c in p):
            q = '"' + p.encode("utf-8").decode("latin-1").encode("unicode_escape").decode() + '"'
        if kind == "binary":
            out.append(f"diff --git a/{q} b/{q}\nindex 111..222 100644\nBinary files a/{q} and b/{q} differ\n")
            continue
        if kind == "rename":
            out.append(f"diff --git a/{q} b/{p}.new\nsimilarity index 90%\nrename from {q}\nrename to {p}.new\n")
            continue
        a_old = [f"l{j}" for j in range(rng.randint(1, 5))]
        a_new = [x if rng.random() < 0.6 else x + "!" for x in a_old] + (["added"] if rng.random() < 0.5 else [])
        body = list(difflib.unified_diff([x + "\n" for x in a_old], [x + "\n" for x in a_new], f"a/{q}", f"b/{q}", n=1))
        head = f"diff --git a/{q} b/{q}\n"
        if kind == "new":
            head += "new file mode 100644\n"
            body = [b.replace(f"a/{q}", "/dev/null") if b.startswith("--- ") else b for b in body]
        if kind == "del":
            head += "deleted file mode 100644\n"
            body = [b.replace(f"b/{q}", "/dev/null") if b.startswith("+++ ") else b for b in body]
        txt = head + "index 111..222 100644\n" + "".join(body)
        if rng.random() < 0.15:
            txt += "\\ No newline at end of file\n"
        out.append(txt)
    return "".join(out)


for n in range(max(400, args.cases // 3)):
    paths = rng.sample(PATH_POOL, rng.randint(1, 5))
    patch = fake_diff(paths)
    if rng.random() < 0.1:
        patch = patch.replace("diff --git ", "", 1) if rng.random() < 0.5 else patch + "\n\n"
    extra = rng.sample(PATH_POOL, rng.randint(0, 2)) if rng.random() < 0.3 else []
    check("patch.strip", P.strip_protected_files_from_patch(patch, extra),
          verification.strip_protected_files_from_patch(patch, extra), f"strip#{n} paths={paths} extra={extra}\n{patch}")
from swegemma.models.task import extract_test_files_from_patch as real_extract  # noqa: E402

for n in range(max(200, args.cases // 6)):
    patch = fake_diff(rng.sample(PATH_POOL, rng.randint(1, 5)))
    check("patch.extract_test_files", P.extract_test_files_from_patch(patch), real_extract(patch), f"extract#{n}\n{patch}")
for pth in PATH_POOL + ["x/tests/y.py", "tests.py", "Testing/a.py", "pkg/.pytest.ini", "a/b/sitecustomize.py"]:
    check("patch.is_protected", P.is_protected_path(pth), verification._is_protected_test_or_config_path(pth), pth)

# --------------------------------------------------------------------------------------------- the prompt
check("prompt.tree_command", P.build_workspace_tree_command(), agent_runner.build_workspace_tree_command(), "tree")
check("prompt.tree_command.args", P.build_workspace_tree_command(50, "/x"), agent_runner.build_workspace_tree_command(50, "/x"), "tree2")
gd = Path(tempfile.mkdtemp(prefix="gr_"))
try:
    (gd / "graphs").mkdir()
    (gd / "emb").mkdir()
    (gd / "graphs" / "r_abc.json").write_text("x" * 200)
    (gd / "emb" / "r_abc.npz").write_text("x" * 200)
    for n in range(300):
        repo = rng.choice(["d/r", "owner/proj-x", "a/b"])
        stmt = rng.choice(["Fix the thing.", "multi\nline\n\nstatement with `code`\n", "caf\u00e9 \u2192 bug", ""])
        hints = rng.choice(["", "  ", "use foo()\n", "hint  with trailing  \n\n"])
        bud = EvaluationBudget(time_minutes=rng.choice([None, 5.5, 5, 60.0, 1]), tool_calls=rng.choice([None, 24, 10, 100]),
                               turns=rng.choice([None, 40, 500]), cost_usd=rng.choice([None, None, 1.5]))
        har = HarnessLimits(command_timeout_seconds=rng.choice([None, 90, 300]), max_stdout_chars=rng.choice([None, 5000]),
                            max_file_lines=rng.choice([None, 150]), max_file_chars=rng.choice([None, 10000]))
        declared = rng.choice([None, {"read_file", "run_command"}, {"run_command"}, {"read_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph"}, set()])
        tree = rng.choice(["", "./a.py\n./pkg\n./pkg/b.py"])
        sandbox = rng.choice([True, False])
        graph = rng.choice([True, False])
        task = types.SimpleNamespace(repo=repo, problem_statement=stmt, hints_text=hints, base_commit="abc" if graph else None)
        cfg = types.SimpleNamespace(budget=bud, harness=har, graph_dir=str(gd / "graphs") if graph else None,
                                    embeddings_dir=str(gd / "emb") if graph else None, enable_sandbox_testing=sandbox)
        if graph:
            task.repo = "d/r"
        want = agent_runner.build_agent_prompt(task, cfg, tree, enable_sandbox_testing=sandbox, declared_tools=declared)
        got = P.build_agent_prompt(task.repo, stmt, hints, time_minutes=bud.time_minutes, tool_calls=bud.tool_calls,
                                   turns=bud.turns, cost_usd=bud.cost_usd,
                                   command_timeout_seconds=har.command_timeout_seconds, max_stdout_chars=har.max_stdout_chars,
                                   max_file_lines=har.max_file_lines, max_file_chars=har.max_file_chars,
                                   workspace_tree=tree, enable_sandbox_testing=sandbox, declared_tools=declared,
                                   graph_available=graph)
        check("prompt.build_agent_prompt", got, want, f"prompt#{n} graph={graph} declared={declared}")
finally:
    shutil.rmtree(gd, ignore_errors=True)

# ----------------------------------------------------------------------------------------- generation config
_, constraints = build_submission_limits()
GRID = []
for budget in [None, -1, 0, 1, 512, 4096, 32768, 5.5]:
    for inc in [None, True, False]:
        for level in [None, "low", "HIGH", "none", "minimal"]:
            for seed in [None, 7]:
                tc = {k: v for k, v in (("thinking_budget", budget), ("include_thoughts", inc), ("thinking_level", level)) if v is not None}
                cfg = {"temperature": 1.0}
                if tc:
                    cfg["thinking_config"] = tc
                if seed is not None:
                    cfg["seed"] = seed
                GRID.append(cfg)
GRID += [{}, {"max_output_tokens": 8192}, {"thinking_config": {"include_thoughts": True}}]
for cfg in GRID:
    try:
        want_merged = constraints.validate_config(copy.deepcopy(cfg), "a")
    except Exception as e:  # noqa: BLE001
        want_merged = ("err", type(e).__name__)
    got_merged = P.merge_generation_config(copy.deepcopy(cfg))
    if isinstance(want_merged, tuple):
        continue                                                  # out-of-range values are rejected before the run
    check("gen.merge", got_merged, want_merged, f"merge {cfg}")
    model = LiteLlm(model="openai/m", api_base="http://x/v1", api_key="k", num_retries=5)
    cloned = apply_thinking_config_to_model(model, copy.deepcopy(want_merged))
    args_real = cloned._additional_args
    inc_real = cloned.__dict__.get("_adk_include_thoughts")
    out = P.thinking_request_fields(got_merged)
    want_extra = {k: v for k, v in args_real.get("extra_body", {}).items()}
    check("gen.extra_body", out["extra_body"], want_extra, f"extra_body {cfg}")
    check("gen.reasoning_effort", out.get("reasoning_effort"), args_real.get("reasoning_effort"), f"effort {cfg}")
    check("gen.seed", out.get("seed"), args_real.get("seed"), f"seed {cfg}")
    check("gen.include_thoughts", out["include_thoughts"], inc_real, f"include_thoughts {cfg}")

# ------------------------------------------------------------------------------------------------ summary
print("compared:", ", ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
if FAILS:
    print("mismatches by kind:", FAIL_KINDS)
    print(f"\n{len(FAILS)} MISMATCHES (showing up to 12):")
    for f in FAILS[:12]:
        print(f[:1500])
        print()
    sys.exit(1)
print("OK: no mismatches")
