"""Apply an edit whose old_string did not match exactly. Same fields as edit_file. Standard library only.

  --filepath F --old_string OLD --new_string NEW [--allow_multiple true]

Looks for ONE place in the file that old_string was meant to be:
  exact    old_string occurs as written (once, or several times with --allow_multiple)
  escape   with backslash sequences (backslash-n, backslash-quote, ...) turned into the real characters it occurs exactly once
  space    the same lines up to indentation and trailing blanks, exactly one place
  similar  the lines of the file that look most like old_string (similarity >= 0.9, clearly ahead of the next best place)
The replacement is re-indented if the match differs in indentation. A Python file is parsed before it is written; if the result would not parse, nothing is written.
When there is no such place the closest lines of the file are printed with their numbers, to copy from.
"""
import argparse
import ast
import difflib
import os
import re
import sys

MIN_SIM, MARGIN, SHOW_NEAR, MAX_LINES, LINE_LEVEL_CHARS = 0.9, 0.05, 12, 150, 600
UNESC = [("\\n", "\n"), ("\\t", "\t"), ('\\"', '"'), ("\\'", "'"), ("\\\\", "\\")]


def fail(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def unescape(t):
    for a, b in UNESC:
        t = t.replace(a, b)
    return t


def squash(t):
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.strip().split("\n"))


def parses(text, name):
    try:
        ast.parse(text, filename=name)
        return True
    except SyntaxError:
        return False


def indent_of(text):
    for ln in text.split("\n"):
        if ln.strip():
            return len(ln) - len(ln.lstrip(" \t")), ln[:len(ln) - len(ln.lstrip(" \t"))]
    return 0, ""


def reindent(new, old_indent, win_indent):
    """Shift every non-empty line of new by the indentation difference between the matched lines and old_string."""
    delta = win_indent[0] - old_indent[0]
    if delta == 0:
        return new
    out = []
    for ln in new.split("\n"):
        if not ln.strip():
            out.append(ln)
        elif delta > 0:
            out.append(win_indent[1][:delta] + ln if win_indent[1] else " " * delta + ln)
        else:
            cut = min(-delta, len(ln) - len(ln.lstrip(" \t")))
            out.append(ln[cut:])
    return "\n".join(out)


def line_level_windows(old, lines, k, limit):
    """For long old_strings: compare line by line (lines equal up to blanks count as equal), which is fast; a window with one wrong line of sixty still scores 0.98."""
    norm = lambda ln: re.sub(r"[ \t]+", " ", ln).strip()
    o = [norm(ln) for ln in old.split("\n")]
    wanted = set(o)
    sq = [norm(ln) for ln in lines]
    bounds = []
    for i in range(0, max(len(lines) - k + 1, 1)):
        shared = sum(1 for ln in sq[i:i + k] if ln in wanted)
        if shared >= 0.5 * k:
            bounds.append((shared, i))
    bounds.sort(key=lambda x: (-x[0], x[1]))
    res = [(difflib.SequenceMatcher(None, o, sq[i:i + k], autojunk=False).ratio(), i) for _s, i in bounds[:limit]]
    res.sort(key=lambda x: (-x[0], x[1]))
    return res


def best_windows(old, lines, limit=60):
    """([(similarity, start line index)] best first, k) for windows with as many lines as old.
    Every window gets the cheap character-count bound (quick_ratio); the exact ratio is computed only for the `limit` best bounds, so a long old_string in a long file stays fast."""
    k = max(len(old.split("\n")), 1)
    if k > MAX_LINES:
        return [], k
    if len(old) > LINE_LEVEL_CHARS:
        return line_level_windows(old, lines, k, limit), k
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(old)
    bounds = []
    for i in range(0, max(len(lines) - k + 1, 1)):
        win = "\n".join(lines[i:i + k])
        sm.set_seq1(win)
        if sm.real_quick_ratio() < 0.5:
            continue
        q = sm.quick_ratio()
        if q >= 0.5:
            bounds.append((q, i, win))
    bounds.sort(key=lambda x: (-x[0], x[1]))
    res = []
    for _q, i, win in bounds[:limit]:
        sm.set_seq1(win)
        res.append((sm.ratio(), i))
    res.sort(key=lambda x: (-x[0], x[1]))
    return res, k


def locate(old, text):
    """(strategy, start char, end char, extra): the place to replace, or strategy None with the best (similarity, line) for a message, or "ambiguous" with the count."""
    n = text.count(old)
    if n == 1:
        i = text.index(old)
        return "exact", i, i + len(old), None
    if n > 1:
        return "ambiguous", 0, 0, n
    un = unescape(old)
    if un != old and text.count(un) == 1:
        i = text.index(un)
        return "escape", i, i + len(un), None
    lines = text.split("\n")
    k = max(len(old.strip("\n").split("\n")), 1)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln) + 1)
    sq = squash(old)
    sq_lines = [re.sub(r"[ 	]+", " ", ln).strip() for ln in lines]
    hits = [i for i in range(0, max(len(lines) - k + 1, 1)) if "\n".join(sq_lines[i:i + k]).strip() == sq]
    if len(hits) == 1:
        i = hits[0]
        return "space", starts[i], starts[min(i + k, len(lines))] - 1, None
    ranked, k = best_windows(old, lines)
    if ranked and ranked[0][0] >= MIN_SIM:
        # a runner-up that overlaps the best window (shifted by a line or two) is the same place, not a rival
        rivals = [r for r in ranked[1:] if abs(r[1] - ranked[0][1]) >= k]
        if not rivals or ranked[0][0] - rivals[0][0] >= MARGIN:
            i = ranked[0][1]
            return "similar", starts[i], starts[min(i + k, len(lines))] - 1, None
    return None, 0, 0, ranked[0] if ranked else None


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--filepath", required=True)
    ap.add_argument("--old_string", required=True)
    ap.add_argument("--new_string", default="")
    ap.add_argument("--allow_multiple", default="")
    a = ap.parse_args()
    multi = str(a.allow_multiple).lower() in ("1", "true", "yes")
    path = os.path.realpath(a.filepath)
    root = os.path.realpath(os.getcwd())
    if os.path.commonpath([path, root]) != root:
        fail("file is outside the workspace")
    if not os.path.isfile(path):
        fail(f"{a.filepath} does not exist (use write_file to create a new file)")
    raw = open(path, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    old, new = a.old_string.replace("\r\n", "\n"), a.new_string.replace("\r\n", "\n")
    if not old.strip():
        fail("old_string must not be empty")
    trailing = old.endswith("\n")
    if trailing:
        old = old[:-1]
        new = new[:-1] if new.endswith("\n") else new
    strat, s, e, extra = locate(old, text)
    if strat == "ambiguous":
        if not multi:
            fail(f"old_string occurs {extra} times; add more surrounding lines to make it unique, or pass allow_multiple true")
        result = text.replace(old, new)
        strat, matched = "exact (all occurrences)", old
    elif strat is None:
        lines = text.split("\n")
        if extra:
            sim, i = extra
            k = max(len(old.split("\n")), 1)
            lo, hi = max(i - 1, 0), min(i + k + 1, len(lines))
            shown = "\n".join(f"{n + 1:>5}| {lines[n]}" for n in range(lo, min(hi, lo + SHOW_NEAR)))
            fail(f"no place in {a.filepath} matches old_string closely enough (best similarity {sim:.2f}). The closest lines are:\n{shown}\n"
                 "Copy old_string from these lines exactly and call edit_file again, or edit by a different, shorter fragment.")
        fail(f"no place in {a.filepath} resembles old_string. Read the file with read_file and copy the lines to replace.")
    else:
        matched = text[s:e]
        if strat in ("space", "similar"):
            new = reindent(new, indent_of(old), indent_of(matched))
        result = text[:s] + new + text[e:]
    if a.filepath.endswith(".py") and parses(text, a.filepath) and not parses(result, a.filepath):
        fail("the file would no longer parse as Python after this edit, so nothing was written. Check the indentation and quotes of new_string.")
    with open(path, "wb") as fh:
        fh.write((result.replace("\n", "\r\n") if crlf else result).encode("utf-8"))
    diff = list(difflib.unified_diff(text.split("\n"), result.split("\n"), lineterm="", n=1))[2:]
    print(f"OK: edit applied to {a.filepath} (matched by: {strat}).")
    if strat not in ("exact", "exact (all occurrences)"):
        print("old_string was not an exact copy of the file; the text that was replaced is shown below.")
    print("\n".join(diff[:40]))


if __name__ == "__main__":
    main()
