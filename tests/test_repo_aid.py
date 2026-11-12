"""agent/submission_e4b_aid/skills/repo_aid: find_files.py and edit_retry.py, run the way the harness runs them (python3 script --flag value, from the workspace)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "agent" / "submission_e4b_aid2" / "skills" / "repo_aid" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE / "mine"))
import find_files as ff  # noqa: E402
import rank_files_probe as rp  # noqa: E402


def run(script, cwd, **flags):
    cmd = [sys.executable, str(SCRIPTS / script)]
    for k, v in flags.items():
        cmd += [f"--{k}", str(v)]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout


BS = chr(92)    # a backslash, written this way so the test file holds no escape sequences to get lost

SRC = '''def alpha(x):
    return x + 1


class Router:
    def route(self, path):
        message = "no route for %s" % path
        return message


def omega():
    return None
'''


def test_exact_edit_and_ambiguous(tmp_path):
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    rc, out = run("edit_retry.py", tmp_path, filepath="m.py", old_string="return x + 1", new_string="return x + 2")
    assert rc == 0 and "matched by: exact" in out and "return x + 2" in (tmp_path / "m.py").read_text()
    rc, out = run("edit_retry.py", tmp_path, filepath="m.py", old_string="return", new_string="yield")
    assert rc == 1 and "occurs" in out


def test_escape_sequences_are_turned_into_real_characters(tmp_path):
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    rc, out = run("edit_retry.py", tmp_path, filepath="m.py", old_string='message = \\"no route for %s\\" % path', new_string='message = "missing %s" % path')
    assert rc == 0 and "matched by: escape" in out and 'message = "missing %s" % path' in (tmp_path / "m.py").read_text()


def test_indentation_difference_is_repaired_and_new_string_reindented(tmp_path):
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    old = "def route(self, path):\n    message = \"no route for %s\" % path\n    return message"        # indented 4 too little
    new = "def route(self, path):\n    return None"
    rc, out = run("edit_retry.py", tmp_path, filepath="m.py", old_string=old, new_string=new)
    assert rc == 0 and "matched by: space" in out
    assert "    def route(self, path):\n        return None\n" in (tmp_path / "m.py").read_text()


def test_near_miss_is_applied_and_far_miss_shows_closest_lines(tmp_path):
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    rc, out = run("edit_retry.py", tmp_path, filepath="m.py", old_string='        message = "no route for %s" % pth', new_string='        message = "nope"')
    assert rc == 0 and "matched by: similar" in out and 'message = "nope"' in (tmp_path / "m.py").read_text()
    (tmp_path / "n.py").write_text(SRC, encoding="utf-8")
    rc, out = run("edit_retry.py", tmp_path, filepath="n.py", old_string="def completely_different_function(argument):\n    pass", new_string="x = 1")
    assert rc == 1 and "ERROR" in out and (tmp_path / "n.py").read_text() == SRC


def test_refuses_an_edit_that_breaks_python_and_keeps_crlf(tmp_path):
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    rc, out = run("edit_retry.py", tmp_path, filepath="m.py", old_string="return x + 1", new_string="return (x +")
    assert rc == 1 and "no longer parse" in out and (tmp_path / "m.py").read_text() == SRC
    (tmp_path / "w.py").write_bytes(SRC.replace("\n", "\r\n").encode())
    rc, out = run("edit_retry.py", tmp_path, filepath="w.py", old_string="return x + 1", new_string="return x + 3")
    assert rc == 0 and b"return x + 3\r\n" in (tmp_path / "w.py").read_bytes() and b"\r\r" not in (tmp_path / "w.py").read_bytes()


def test_outside_the_workspace_is_refused(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "out.py").write_text(SRC, encoding="utf-8")
    rc, out = run("edit_retry.py", tmp_path / "ws", filepath="../out.py", old_string="return x + 1", new_string="return 0")
    assert rc == 1 and "outside" in out


def test_find_files_lists_the_matching_file_first_and_skips_tests(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "routing.py").write_text(SRC, encoding="utf-8")
    (tmp_path / "pkg" / "other.py").write_text("def unrelated():\n    return 3\n", encoding="utf-8")
    (tmp_path / "tests" / "test_routing.py").write_text(SRC, encoding="utf-8")
    rc, out = run("find_files.py", tmp_path, query="Router route raises no route message for missing path")
    assert rc == 0 and out.splitlines()[1].startswith("1. pkg/routing.py") and "test_routing" not in out
    rc, out = run("find_files.py", tmp_path, query="of the")
    assert rc == 1 and "ERROR" in out


def test_find_files_ranks_like_the_offline_probe():
    files = {"a/routing.py": SRC, "a/other.py": "def unrelated():\n    return 3\n", "a/render.py": "class Table:\n    def render(self):\n        return 'x'\n"}
    q = "Table render output is wrong when the router has no route"
    paths, bm = rp.rankers(files)
    expected = rp.ranked(paths, bm["symbols"].scores(rp.tokens(q)))[:5]
    got = [p for p, _ in ff.candidates(files, q)]
    assert got[:3] == expected[:3]


def test_new_string_with_literal_backslash_sequences_is_rescued(tmp_path):
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    # the model wrote the line break inside new_string as the two characters backslash-n
    rc, out = run("edit_retry.py", tmp_path, filepath="m.py", old_string="return x + 1", new_string="y = x" + BS + "n    return y + 1")
    assert rc == 0 and "real characters" in out and "    y = x\n    return y + 1\n" in (tmp_path / "m.py").read_text()


def test_similar_search_also_tries_the_unescaped_old_string(tmp_path):
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    old = '    def route(self, path):' + BS + 'n        mesage = "no route for %s" % path'          # literal backslash-n and one typo
    rc, out = run("edit_retry.py", tmp_path, filepath="m.py", old_string=old, new_string='    def route(self, path):\n        message = "none"')
    assert rc == 0 and "matched by: similar" in out and 'message = "none"' in (tmp_path / "m.py").read_text()
