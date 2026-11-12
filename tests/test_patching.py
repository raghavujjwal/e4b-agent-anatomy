"""Unit tests for grader.patching. Pure Python: runs anywhere, no sandbox needed.

  python -m pytest tests/test_patching.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grader import patching as P  # noqa: E402

# the competition's test_patch style: bare ---/+++ headers, new file created with an `--- a/` header
BARE = """--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,3 +1,4 @@
 import os
+import sys
 x = 1
 y = 2
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -0,0 +1,2 @@
+def test_x():
+    assert True
"""

GIT = """diff --git a/pkg/core.py b/pkg/core.py
index 111..222 100644
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1,2 @@
 a
+b
diff --git a/conftest.py b/conftest.py
new file mode 100644
--- /dev/null
+++ b/conftest.py
@@ -0,0 +1 @@
+import foo
"""

# a removed line that looks like a header ("-- x" becomes "--- x") followed by an added "++ y" ("+++ y")
TRICKY = """--- a/docs.md
+++ b/docs.md
@@ -1,2 +1,2 @@
--- x
+++ y
 keep
--- a/pkg/real.py
+++ b/pkg/real.py
@@ -1 +1,2 @@
 z
+w
"""


def test_is_protected():
    for p in ["tests/test_a.py", "pkg/test_b.py", "pkg/b_test.py", "conftest.py", "sub/conftest.py",
              "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg", "x.pth", "testing/helpers.py",
              "sitecustomize.py"]:
        assert P.is_protected(p), p
    for p in ["pkg/core.py", "pkg/testing_utils.txt", "pkg/contest.py", "docs/tests.md", "src/requests/api.py"]:
        assert not P.is_protected(p), p


def test_split_bare_headers():
    blocks = P.split_patch(BARE)
    assert [b["path"] for b in blocks] == ["pkg/core.py", "tests/test_core.py"]
    assert P.patch_files(BARE) == ["pkg/core.py", "tests/test_core.py"]


def test_split_git_headers_and_new_file():
    blocks = P.split_patch(GIT)
    assert [b["path"] for b in blocks] == ["pkg/core.py", "conftest.py"]
    assert blocks[1]["old"] is None and blocks[1]["new"] == "conftest.py"


def test_hunk_counts_beat_header_lookalikes():
    blocks = P.split_patch(TRICKY)
    assert [b["path"] for b in blocks] == ["docs.md", "pkg/real.py"]


def test_strip_protected_drops_tests_and_config():
    kept, removed = P.strip_protected(BARE)
    assert removed == ["tests/test_core.py"]
    assert "pkg/core.py" in kept and "test_core" not in kept
    kept, removed = P.strip_protected(GIT)
    assert removed == ["conftest.py"] and "conftest" not in kept and "pkg/core.py" in kept


def test_strip_protected_only_protected_gives_empty():
    only_test = BARE.split("--- a/tests/test_core.py")[1]
    kept, removed = P.strip_protected("--- a/tests/test_core.py" + only_test)
    assert kept.strip() == "" and removed == ["tests/test_core.py"]


def test_strip_keeps_text_byte_for_byte_for_kept_blocks():
    kept, _ = P.strip_protected(BARE)
    assert kept == BARE.split("--- a/tests/test_core.py")[0]


def test_functions_signals():
    assert P.test_functions("+def test_a():\n+    pass\n+async def test_b():\n") == (["test_a", "test_b"], "added-def")
    ctx = "@@ -10,3 +10,4 @@ def test_old(x):\n     assert x\n+    assert y\n"
    assert P.test_functions(ctx) == (["test_old"], "hunk-ctx")
    assert P.test_functions("@@ -1 +1 @@\n-a\n+b\n") == ([], "exit-code-only")


def _xml(cases, failures=0, errors=0):
    body = "".join(cases)
    return f'<testsuite failures="{failures}" errors="{errors}">{body}</testsuite>'


def test_validate_junit_decisions():
    ok = _xml(['<testcase name="test_a"/>', '<testcase name="test_b[1]"/>'])
    assert P.validate_junit(ok, ["test_a", "test_b"])[0]
    skipped = _xml(['<testcase name="test_a"><skipped/></testcase>', '<testcase name="test_z"/>'])
    assert not P.validate_junit(skipped, ["test_a"])[0]                      # required test skipped
    assert not P.validate_junit(ok, ["test_missing"])[0]                     # never ran
    fail = _xml(['<testcase classname="c" name="test_a"><failure message="boom"/></testcase>'], failures=1)
    ok2, why = P.validate_junit(fail, [])
    assert not ok2 and "test_a" in why                                       # names the failing test
    assert not P.validate_junit("", [])[0]
    assert not P.validate_junit(_xml([]), [])[0]                             # no testcases
