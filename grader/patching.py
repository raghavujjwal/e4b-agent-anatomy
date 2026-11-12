"""Patch handling shared by the verifier and the agent sandbox.

Ported from localeval/sandbox.py and localeval/verify.py on branch `main`, with two changes:
  * protected hunks are STRIPPED from the agent patch before it is applied (HARNESS_README 8.1.2 and
    8.2.2), instead of applied and then reset;
  * nothing here depends on Docker. Functions that run commands take a Sandbox.
"""
from __future__ import annotations
import os
import re
import shlex
import xml.etree.ElementTree as ET

# HARNESS_README 8.2 step 3 (_is_protected_test_or_config_path)
PROTECTED_NAMES = {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg",
                   ".pytest.ini", "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py"}
PROTECTED_RE = re.compile(
    r"(^|/)test_[^/]*\.py$"                      # test_*.py
    r"|_test\.py$"                               # *_test.py
    r"|(^|/)(tests|test|testing)/.*\.py$"        # .py under tests/ test/ testing/
)

# HARNESS_README 4.2 step 3
GIT_EXCLUDE = ["__pycache__/", "*.pyc", ".pytest_cache/", "*.egg-info/", "build/", "dist/", ".coverage"]


def is_protected(path: str) -> bool:
    if path.endswith(".pth"):
        return True
    if os.path.basename(path) in PROTECTED_NAMES:
        return True
    return bool(PROTECTED_RE.search(path))


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _strip_prefix(p: str) -> str:
    p = p.split("\t")[0].strip()
    return p[2:] if p[:2] in ("a/", "b/") else p


def split_patch(patch: str) -> list[dict]:
    """Split a unified diff into per-file blocks: [{"path", "old", "new", "text"}].

    Handles both `diff --git` headers and the bare `---`/`+++` form that the competition's test_patch
    uses. Hunks are consumed by their declared line counts, so a content line that happens to start
    with `--- ` cannot be mistaken for a new file header.
    """
    lines = patch.splitlines(keepends=True)
    blocks: list[dict] = []
    cur: dict | None = None
    i = 0
    old_left = new_left = 0

    def start(first_line: str) -> dict:
        b = {"old": None, "new": None, "path": None, "lines": [first_line]}
        blocks.append(b)
        return b

    while i < len(lines):
        ln = lines[i]
        in_hunk = old_left > 0 or new_left > 0
        if in_hunk:
            cur["lines"].append(ln)
            if ln.startswith("\\"):
                pass                                    # "\ No newline at end of file"
            elif ln.startswith("-"):
                old_left -= 1
            elif ln.startswith("+"):
                new_left -= 1
            else:
                old_left -= 1
                new_left -= 1
            i += 1
            continue
        m = _HUNK.match(ln)
        if m and cur is not None:
            cur["lines"].append(ln)
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
            i += 1
            continue
        if ln.startswith("diff --git "):
            cur = start(ln)
            g = re.match(r"diff --git a/(\S+) b/(\S+)", ln)
            if g:
                cur["old"], cur["new"] = g.group(1), g.group(2)
        elif ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if cur is None or cur.get("seen_old"):
                cur = start(ln)
            else:
                cur["lines"].append(ln)
            cur["seen_old"] = True
            cur["old"] = _strip_prefix(ln[4:])
            cur["lines"].append(lines[i + 1])
            cur["new"] = _strip_prefix(lines[i + 1][4:])
            i += 2
            continue
        else:
            if cur is None:
                cur = start(ln)                          # preamble before the first header
            else:
                cur["lines"].append(ln)
        i += 1
    out = []
    for b in blocks:
        new = b["new"] if b["new"] not in (None, "/dev/null") else None
        old = b["old"] if b["old"] not in (None, "/dev/null") else None
        out.append({"path": new or old, "old": old, "new": new, "text": "".join(b["lines"])})
    return out


def patch_files(patch: str) -> list[str]:
    seen: list[str] = []
    for b in split_patch(patch):
        for p in (b["new"], b["old"]):
            if p and p not in seen:
                seen.append(p)
    return seen


def strip_protected(patch: str) -> tuple[str, list[str]]:
    """Drop every file block that touches a protected path. Returns (kept patch, removed paths)."""
    kept, removed = [], []
    for b in split_patch(patch):
        paths = [p for p in (b["new"], b["old"]) if p]
        if not paths:                                    # preamble text only
            continue
        if any(is_protected(p) for p in paths):
            removed.append(b["path"])
        else:
            kept.append(b["text"] if b["text"].endswith("\n") else b["text"] + "\n")
    return "".join(kept), removed


def pytest_targets(test_patch: str) -> list[str]:
    return [f for f in patch_files(test_patch) if f.endswith(".py")]


def test_functions(test_patch: str) -> tuple[list[str], str]:
    """Required-to-pass test names and which signal produced them. The public tasks carry no
    FAIL_TO_PASS field, so (as documented in HARNESS_README 8.2) the fallback is the test functions
    extracted from test_patch:
      added-def      the patch adds `def test_x`
      hunk-ctx       the patch only edits inside an existing test; git names it in the hunk header
      exit-code-only nothing extractable; the verdict rests on exit 0 with >= 1 pass, which is why
                     the empty-patch control is mandatory.
    """
    added: list[str] = []
    for line in test_patch.splitlines():
        if line.startswith("+"):
            m = re.match(r"\+\s*(?:async\s+)?def\s+(test_\w+)", line)
            if m and m.group(1) not in added:
                added.append(m.group(1))
    if added:
        return added, "added-def"
    ctx: list[str] = []
    for m in re.finditer(r"^@@[^@]*@@\s*(?:async\s+)?def\s+(test_\w+)", test_patch, re.M):
        if m.group(1) not in ctx:
            ctx.append(m.group(1))
    if ctx:
        return ctx, "hunk-ctx"
    return [], "exit-code-only"


def validate_junit(xml_text: str, required: list[str]) -> tuple[bool, str]:
    """passed > 0, failures == 0, errors == 0, and every required node present and not skipped."""
    if not xml_text:
        return False, "no junit xml"
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        return False, f"unparseable junit xml: {e}"
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    cases, failures, errors = [], 0, 0
    for s in suites:
        failures += int(s.get("failures") or 0)
        errors += int(s.get("errors") or 0)
        cases += list(s.iter("testcase"))
    if not cases:
        return False, "junit xml contains no testcases"
    if failures or errors:
        who = []
        for c in cases:
            for ch in c:
                if ch.tag in ("failure", "error"):
                    msg = (ch.get("message") or "").strip().replace("\n", " ")[:90]
                    who.append(f"{c.get('classname', '')}::{c.get('name', '')} [{ch.tag}: {msg}]")
                    break
        return False, (f"junit reports failures={failures} errors={errors}: " + "; ".join(who[:3])
                       + (f" (+{len(who) - 3} more)" if len(who) > 3 else ""))
    status = {}
    for c in cases:
        name = c.get("name") or ""
        bad = [ch.tag for ch in c if ch.tag in ("failure", "error", "skipped")]
        status[name] = bad[0] if bad else "passed"
    passed = sum(1 for v in status.values() if v == "passed")
    if passed == 0:
        return False, "no passing testcases"
    for req in required:
        hits = [v for k, v in status.items() if k == req or k.startswith(req + "[")]
        if not hits:
            return False, f"required test {req} never ran"
        if any(h != "passed" for h in hits):
            return False, f"required test {req} did not pass ({set(hits)})"
    return True, f"{passed} passed, all {len(required)} required node(s) ok"


# HARNESS_README 8.2 step 2: four resilient passes (git apply variants, symlink-normalised, -p0, GNU patch)
APPLY_ATTEMPTS = [
    ("p1", "git apply --unsafe-paths -p1 {p}"),
    ("p1-3way", "git apply --unsafe-paths -p1 -3 {p}"),
    ("p1-ws", "git apply --unsafe-paths -p1 --ignore-space-change --ignore-whitespace {p}"),
    ("p1-recount", "git apply --unsafe-paths -p1 --recount {p}"),
    ("p0", "git apply --unsafe-paths -p0 {p}"),
    ("gnu-p1", "patch -p1 --batch --forward --dry-run < {p} && patch -p1 --batch --forward < {p}"),
    ("gnu-p0", "patch -p0 --batch --forward --dry-run < {p} && patch -p0 --batch --forward < {p}"),
    ("gnu-p1-l", "patch -p1 -l --batch --forward --dry-run < {p} && patch -p1 -l --batch --forward < {p}"),
]


def apply_patch(sb, patch: str, name: str = "agent.patch") -> tuple[bool, str]:
    """Apply `patch` in the sandbox workspace. Returns (ok, which attempt worked)."""
    if not patch.strip():
        return False, "empty"
    f = sb.tmp / name
    f.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8", newline="\n")
    sb.fix_perms(f)
    p = shlex.quote(str(f))
    for label, cmd in APPLY_ATTEMPTS:
        rc, _, _ = sb.sh(cmd.format(p=p), timeout=300)
        if rc == 0:
            return True, label
    return False, "all_passes_failed"
