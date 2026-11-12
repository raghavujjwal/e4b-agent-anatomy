"""Line-range editor: show numbered lines of a file, or replace a range of lines. Standard library only.

  --action show     --file F --start S --end E          print lines S..E with their numbers
  --action replace  --file F --start S --end E --text T replace lines S..E (inclusive) with T
                    --end S-1 inserts T before line S; empty --text deletes the range
  Optional on replace: --expect "first line of the range as it is now" (refuses if line S does not match, ignoring indentation)

A Python file is parsed before it is written; if the result is not valid Python the file is left untouched and the error is printed.
Paths are relative to the workspace (the harness runs scripts from there). Line endings and the final newline are preserved.
"""
import argparse
import ast
import os
import sys

CONTEXT = 3


def fail(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def numbered(lines, first, last):
    width = len(str(max(last, 1)))
    return "\n".join(f"{n:>{width}}| {lines[n - 1]}" for n in range(first, last + 1))


def parses(text, name):
    try:
        ast.parse(text, filename=name)
        return True
    except SyntaxError:
        return False


def main():
    # Source files are read and written as UTF-8; print as UTF-8 too, whatever the locale of the process that launched us.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", default="")
    ap.add_argument("--action", required=True, choices=["show", "replace"])
    ap.add_argument("--file", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--expect", default=None)
    a = ap.parse_args()
    a.force = str(a.force).lower() in ("1", "true", "yes")

    path = os.path.realpath(a.file)
    root = os.path.realpath(os.getcwd())
    if os.path.commonpath([path, root]) != root:
        fail("file is outside the workspace")
    if not os.path.isfile(path):
        fail(f"{a.file} does not exist (use write_file to create a new file)")
    raw = open(path, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    body = raw.replace("\r\n", "\n")
    ends_nl = body.endswith("\n")
    lines = body.split("\n")
    if ends_nl:
        lines.pop()
    total = len(lines)

    s = a.start
    e = a.end if a.end is not None else s
    if a.action == "show":
        if s < 1 or s > max(total, 1):
            fail(f"--start {s} is outside the file ({total} lines)")
        e = min(max(e, s), total)
        e = min(e, s + 199)
        print(f"{a.file}: lines {s}-{e} of {total}")
        print(numbered(lines, s, e))
        return

    if a.text is None:
        fail("--text is required for replace (use an empty string to delete the range)")
    if s < 1 or s > total + 1:
        fail(f"--start {s} is outside the file ({total} lines)")
    if e < s - 1 or e > total:
        fail(f"--end {e} must be between {s - 1} and {total}")
    if a.expect is not None and e >= s and lines[s - 1].strip() != a.expect.strip():
        fail(f"line {s} is {lines[s - 1].strip()!r}, not {a.expect.strip()!r}; "
             f"the numbers are off, run show again\n" + numbered(lines, max(1, s - CONTEXT), min(total, s + CONTEXT)))

    new = a.text.replace("\r\n", "\n")
    if new.endswith("\n"):
        new = new[:-1]
    new_lines = [] if a.text == "" else new.split("\n")
    out = lines[:s - 1] + new_lines + lines[e:]
    result = "\n".join(out) + ("\n" if ends_nl or not out else "")

    if path.endswith(".py") and not a.force and parses(body, a.file):
        # Only checked when this interpreter parses the file as it is now, so newer syntax in the repository cannot cause a refusal.
        try:
            ast.parse(result, filename=a.file)
        except SyntaxError as exc:
            where = f"line {exc.lineno}" if exc.lineno else "unknown line"
            fail(f"the edit would leave {a.file} with a syntax error ({exc.msg}, {where}); nothing was written. "
                 f"Check indentation, brackets and quotes in --text. If the syntax is valid, repeat with force set to true.")
    if crlf:
        result = result.replace("\n", "\r\n")
    open(path, "wb").write(result.encode("utf-8"))

    first = max(1, s - CONTEXT)
    last = min(len(out), s + len(new_lines) - 1 + CONTEXT) if new_lines else min(len(out), s + CONTEXT)
    print(f"OK: replaced {e - s + 1 if e >= s else 0} line(s) with {len(new_lines)}; {a.file} now has {len(out)} lines")
    if out:
        print(numbered(out, first, last))


if __name__ == "__main__":
    main()
