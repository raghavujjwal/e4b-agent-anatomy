"""edit_lines skill script: show, replace, insert, delete, refusals. Run: python -m pytest tests/test_edit_lines.py"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "agent" / "submission_e4b_lines" / "skills" / "edit_lines" / "scripts" / "edit_lines.py"
SRC = "import os\n\n\ndef f(x):\n    y = x + 1\n    return y\n\n\ndef g():\n    return 2\n"


def run(cwd, **flags):
    cmd = [sys.executable, str(SCRIPT)]
    for k, v in flags.items():
        cmd += [f"--{k}", str(v)]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout


def setup(tmp_path, text=SRC, name="m.py", newline=None):
    (tmp_path / name).write_bytes(text.encode() if newline is None else text.replace("\n", newline).encode())
    return tmp_path


def test_show(tmp_path):
    setup(tmp_path)
    rc, out = run(tmp_path, action="show", file="m.py", start=4, end=6)
    assert rc == 0 and "4| def f(x):" in out and "6|     return y" in out and "of 10" in out


def test_replace_middle(tmp_path):
    setup(tmp_path)
    rc, out = run(tmp_path, action="replace", file="m.py", start=5, end=6, text="    z = x * 2\n    return z",
                  expect="y = x + 1")
    assert rc == 0 and out.startswith("OK: replaced 2 line(s) with 2")
    assert (tmp_path / "m.py").read_text() == SRC.replace("    y = x + 1\n    return y", "    z = x * 2\n    return z")


def test_insert_and_delete(tmp_path):
    setup(tmp_path)
    rc, _ = run(tmp_path, action="replace", file="m.py", start=5, end=4, text="    assert x is not None")
    assert rc == 0 and "def f(x):\n    assert x is not None\n    y = x + 1" in (tmp_path / "m.py").read_text()
    rc, _ = run(tmp_path, action="replace", file="m.py", start=5, end=5, text="")
    assert rc == 0 and (tmp_path / "m.py").read_text() == SRC


def test_syntax_error_is_refused(tmp_path):
    setup(tmp_path)
    rc, out = run(tmp_path, action="replace", file="m.py", start=5, end=5, text="    y = (x + 1")
    assert rc == 1 and "syntax error" in out and "nothing was written" in out
    assert (tmp_path / "m.py").read_text() == SRC


def test_expect_mismatch_is_refused(tmp_path):
    setup(tmp_path)
    rc, out = run(tmp_path, action="replace", file="m.py", start=4, end=4, text="def f(x, k):", expect="return y")
    assert rc == 1 and "numbers are off" in out and (tmp_path / "m.py").read_text() == SRC


def test_crlf_and_no_final_newline_preserved(tmp_path):
    setup(tmp_path, text="a = 1\nb = 2", newline="\r\n")
    rc, _ = run(tmp_path, action="replace", file="m.py", start=2, end=2, text="b = 3")
    assert rc == 0 and (tmp_path / "m.py").read_bytes() == b"a = 1\r\nb = 3"


def test_non_python_file_is_not_parsed(tmp_path):
    setup(tmp_path, text="x: [\n", name="n.yaml")
    rc, _ = run(tmp_path, action="replace", file="n.yaml", start=1, end=1, text="x: (")
    assert rc == 0 and (tmp_path / "n.yaml").read_text() == "x: (\n"


def test_outside_workspace_and_missing(tmp_path):
    sub = tmp_path / "ws"
    sub.mkdir()
    (tmp_path / "o.py").write_text("a = 1\n")
    rc, out = run(sub, action="show", file="../o.py", start=1)
    assert rc == 1 and "outside the workspace" in out
    rc, out = run(sub, action="show", file="nope.py", start=1)
    assert rc == 1 and "does not exist" in out


def test_out_of_range(tmp_path):
    setup(tmp_path)
    rc, out = run(tmp_path, action="replace", file="m.py", start=40, end=41, text="x = 1")
    assert rc == 1 and "outside the file" in out


def test_force_and_unparseable_original(tmp_path):
    setup(tmp_path)
    rc, out = run(tmp_path, action="replace", file="m.py", start=5, end=5, text="    y = (x + 1", force="true")
    assert rc == 0 and "y = (x + 1" in (tmp_path / "m.py").read_text()
    # the original no longer parses with this interpreter, so further edits are not refused
    rc, _ = run(tmp_path, action="replace", file="m.py", start=5, end=5, text="    y = (x + 1))")
    assert rc == 0


def test_non_ascii_output_is_utf8(tmp_path):
    setup(tmp_path, text="name = 'Zażółć gęślą jaźń'\n")
    cmd = [sys.executable, str(SCRIPT), "--action", "show", "--file", "m.py", "--start", "1"]
    p = subprocess.run(cmd, cwd=tmp_path, capture_output=True, env={**__import__("os").environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
    assert p.returncode == 0 and "Zażółć".encode("utf-8") in p.stdout
