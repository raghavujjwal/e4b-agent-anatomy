"""Build supervised data for the edit step: given the issue and a view of the file, produce the exact edit call.

  python train/make_edit_data.py --out D:/data/edit_sft --commits-per-repo 300

Sources (training repositories only, see docs/EVAL_PROTOCOL.md rule 6):
  history  small commits of the mined *tune* repositories (click, falcon, pytest, black, pygments, packaging, werkzeug, tornado,
           loguru); the instruction is the commit message
  task     the gold patches of the usable mined tasks of those repositories; the instruction is the issue text
Never used: the public tasks (dev66 stays a clean comparison), rich, and the mined unseen repositories.

Every change hunk becomes one training sample per interface:
  edit_file  messages = system, issue, read_file call + result (a window the real tool would return, hunk visible), TARGET edit_file call
  lines      the same prefix, then  TARGET show call;   and prefix + show call + numbered result, TARGET replace call
The target is accepted only if the real tool code applies it and reproduces the expected text exactly: `apply_replacement` from
agent/vendor/tools.py for edit_file, the real skill script (run as a subprocess) for show/replace. Samples that fail are dropped.
Records carry no reasoning text; the trainer decides how to treat the model's thinking (see train/README.md).
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent" / "vendor"))
sys.path.insert(0, str(REPO))
import runner as R                                                       # noqa: E402
from tools import apply_replacement, MAX_FILE_LINES, MAX_FILE_CHARS      # noqa: E402

SKILL = REPO / "agent" / "submission_e4b_lines" / "skills" / "edit_lines" / "scripts" / "edit_lines.py"
PROMPTS = {"edit_file": REPO / "agent" / "submission_e4b" / "prompts" / "flat_search.md",
           "lines": REPO / "agent" / "submission_e4b_lines" / "prompts" / "flat_search.md"}
TOOLS = {"edit_file": ["read_file", "edit_file", "write_file", "run_command", "get_status", "submit_patch"],
         "lines": ["read_file", "write_file", "run_command", "get_status", "submit_patch", "run_skill_script", "load_skill_resource"]}
TUNE = ["pallets/click", "falconry/falcon", "pytest-dev/pytest", "psf/black", "pygments/pygments", "pypa/packaging",
        "pallets/werkzeug", "tornadoweb/tornado", "Delgan/loguru"]
CLONES = Path("D:/data/mined/clones")
MAX_HUNK_LINES = 30          # a hunk with more removed+added lines than this is not a "small edit"
SKIP_PATH = re.compile(r"(^|/)(tests?|testing|docs?|examples?|benchmarks?|scripts|\.github)(/|$)|(^|/)(setup|conftest|noxfile)\.py$|(^|/)test_[^/]*\.py$|_test\.py$")
SKIP_SUBJECT = re.compile(r"^(merge|bump|release|version|pre-commit|\[pre-commit|update dependenc|revert|changelog|docs?:)", re.I)


# ---------------------------------------------------------------------------------------------------- text helpers
def hunks(before: list[str], after: list[str]):
    """Change groups as (i1, i2, new_lines): before[i1:i2] becomes new_lines. Context lines are not part of a group."""
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    return [(i1, i2, after[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def tool_read(lines: list[str], s: int, e: int | None) -> dict:
    """Same result as Tools.read_file for lines (1-indexed inclusive range)."""
    total = len(lines)
    e = e or total
    sel = lines[s - 1:e]
    trunc = False
    if len(sel) > MAX_FILE_LINES:
        sel, trunc = sel[:MAX_FILE_LINES], True
    content = "\n".join(sel)
    if len(content) > MAX_FILE_CHARS:
        cut = content[:MAX_FILE_CHARS].rsplit("\n", 1)[0]
        sel = cut.split("\n")
        content, trunc = cut, True
    return {"status": "ok", "filepath": None, "content": content, "start_line": s, "end_line": s + len(sel) - 1,
            "total_lines": total, "is_truncated": trunc}


def pick_window(rng: random.Random, lines: list[str], i1: int, i2: int):
    """A read_file call (args, result) whose output contains the changed region [i1, i2) plus some context."""
    total = len(lines)
    ok = tool_read(lines, 1, None)
    if not ok["is_truncated"] and rng.random() < 0.35:
        return {}, ok                                                # whole file, no range given
    for _ in range(20):
        s = max(1, i1 + 1 - rng.randint(4, 70))
        e = min(total, max(i2, i1 + 1) + rng.randint(4, 70))
        res = tool_read(lines, s, e)
        if res["start_line"] <= i1 + 1 and res["end_line"] >= max(i2, i1 + 1):
            return {"start_line": s, "end_line": e}, res
    return None, None


def anchor(lines: list[str], i1: int, i2: int, new: list[str]):
    """Smallest unique old_string (whole lines) covering the change; returns (old, new_string) or None."""
    text = "\n".join(lines)
    n = len(lines)
    tries = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 3), (4, 4)]
    if i1 == i2:                       # pure insertion: needs a neighbour line to attach to
        tries = [t for t in tries if t != (0, 0)]
    for b, a in tries:
        lo, hi = i1 - b, i2 + a
        if lo < 0 or hi > n:
            continue
        old_lines = lines[lo:hi]
        old = "\n".join(old_lines)
        if not old.strip() or text.count(old) != 1:
            continue
        new_lines = lines[lo:i1] + new + lines[i2:hi]
        return old, "\n".join(new_lines)
    return None


# ---------------------------------------------------------------------------------------------------- skill script
def run_skill(path: Path, **flags) -> tuple[int, str]:
    cmd = [sys.executable, str(SKILL)]
    for k, v in flags.items():
        cmd += [f"--{k}", str(v)]
    p = subprocess.run(cmd, cwd=path.parent, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout or ""


def skill_text(new: list[str]) -> str:
    t = "\n".join(new)
    return t + "\n" if new and new[-1] == "" else t


# ---------------------------------------------------------------------------------------------------- records
def cid(rng):
    return "chatcmpl-tool-" + "".join(rng.choice("0123456789abcdef") for _ in range(16))


def call(rng, name, args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"type": "function", "id": cid(rng), "function": {"name": name, "arguments": json.dumps(args)}}]}


def tool_msg(prev, content):
    return {"role": "tool", "tool_call_id": prev["tool_calls"][0]["id"], "content": json.dumps(content)}


def wrap_issue(repo, statement):
    return f"You are evaluating a software engineering task for repository {repo}.\n\nProblem Statement:\n{statement}"


def samples_for_file(rng, repo, fp, before_text, after_text, statement, systems, src_id, split):
    """All samples (both interfaces) for the hunks of one file, hunks applied top to bottom."""
    cur = before_text.split("\n")
    out = []
    delta = 0
    for i1, i2, new in hunks(before_text.split("\n"), after_text.split("\n")):
        s, e = i1 + delta, i2 + delta
        delta += len(new) - (i2 - i1)
        if len(new) + (e - s) > MAX_HUNK_LINES or (e == s and not new):
            cur[s:e] = new
            continue
        expected = cur[:s] + new + cur[e:]
        meta = {"repo": repo, "file": fp, "source": src_id, "removed": e - s, "added": len(new)}
        win_args, win_res = pick_window(rng, cur, s, e)
        if win_res is None:
            cur = expected
            continue
        win_res["filepath"] = fp
        issue = {"role": "user", "content": wrap_issue(repo, statement)}
        rf = call(rng, "read_file", {"filepath": fp, **win_args})
        prefix_tail = [rf, tool_msg(rf, win_res)]

        # ---- interface 1: edit_file ------------------------------------------------------------------
        a = anchor(cur, s, e, new)
        if a:
            old, newstr = a
            res, n, strat = apply_replacement("\n".join(cur), old, newstr, False)
            if res == "\n".join(expected) and strat == "exact":
                tgt = call(rng, "edit_file", {"filepath": fp, "old_string": old, "new_string": newstr})
                out.append({"id": f"{src_id}:{fp}:{s}:edit_file", "format": "edit_file", "step": "edit", "split": split, "meta": meta,
                            "messages": [{"role": "system", "content": systems["edit_file"]}, issue, *prefix_tail, tgt]})

        # ---- interface 2: lines skill ----------------------------------------------------------------
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "f.py"
            f.write_bytes("\n".join(cur).encode("utf-8"))
            lo, hi = max(1, s + 1 - 3), min(len(cur), max(e, s + 1) + 3)
            rc, shown = run_skill(f, action="show", file="f.py", start=lo, end=hi)
            if rc != 0:
                cur = expected
                continue
            shown = shown.replace("f.py:", f"{fp}:", 1)
            rargs = {"action": "replace", "file": "f.py", "start": s + 1, "end": e, "text": skill_text(new)}
            if e > s:
                rargs["expect"] = cur[s]
            rc, _ = run_skill(f, **rargs)
            ok = rc == 0 and f.read_bytes().decode("utf-8") == "\n".join(expected)
            if ok:
                show_call = call(rng, "run_skill_script", {"skill_name": "edit_lines", "file_path": "scripts/edit_lines.py",
                                                           "args": {"action": "show", "file": fp, "start": lo, "end": hi}})
                rep_call = call(rng, "run_skill_script", {"skill_name": "edit_lines", "file_path": "scripts/edit_lines.py",
                                                          "args": {**rargs, "file": fp}})
                base = [{"role": "system", "content": systems["lines"]}, issue, *prefix_tail]
                out.append({"id": f"{src_id}:{fp}:{s}:show", "format": "lines", "step": "show", "split": split, "meta": meta,
                            "messages": [*base, show_call]})
                out.append({"id": f"{src_id}:{fp}:{s}:replace", "format": "lines", "step": "replace", "split": split, "meta": meta,
                            "messages": [*base, show_call, tool_msg(show_call, {"status": "ok", "stdout": shown, "exit_code": 0}), rep_call]})
        cur = expected
    return out


# ---------------------------------------------------------------------------------------------------- sources
def git(clone: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else ""


def history_items(repo: str, n_commits: int, rng: random.Random):
    clone = CLONES / repo.replace("/", "__")
    log = git(clone, "log", "--no-merges", "--format=%H%x1f%ct%x1f%s%x1f%b%x1e", "-n", "6000")
    commits = []
    for rec in log.split("\x1e"):
        parts = rec.strip("\n").split("\x1f")
        if len(parts) < 4:
            continue
        h, ct, subj, body = parts[0], int(parts[1]), parts[2], parts[3]
        if len(subj) < 12 or SKIP_SUBJECT.match(subj):
            continue
        commits.append((ct, h, subj, body))
    commits.sort()
    cut = int(len(commits) * 0.9)                                  # the newest 10% of commits is validation
    order = list(range(len(commits)))
    rng.shuffle(order)
    items = []
    for idx in order:
        if len(items) >= n_commits:
            break
        ct, h, subj, body = commits[idx]
        stat = git(clone, "show", "--numstat", "--format=", "--diff-filter=M", h)
        files = []
        for line in stat.splitlines():
            p = line.split("\t")
            if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
                files.append((p[2], int(p[0]) + int(p[1])))
        src = [f for f, _ in files if f.endswith(".py") and not SKIP_PATH.search(f)]
        if not src or len(src) > 3:
            continue
        if sum(c for f, c in files if f in src) > 60:
            continue
        body = re.sub(r"(?im)^(signed-off-by|co-authored-by):.*$", "", body).strip()
        statement = subj + (("\n\n" + body) if body else "")
        pairs = []
        for f in src:
            b, a = git(clone, "show", f"{h}^:{f}"), git(clone, "show", f"{h}:{f}")
            if b and a and b != a:
                pairs.append((f, b, a))
        if pairs:
            items.append((f"hist-{h[:10]}", "validation" if idx >= cut else "train", statement, pairs))
    return items


def task_items(repo: str, usable: set[str], tasks: dict):
    clone = CLONES / repo.replace("/", "__")
    items = []
    for tid, t in tasks.items():
        if t["repo"] != repo or tid not in usable:
            continue
        files = re.findall(r"^\+\+\+ b/(\S+)$", t["patch"], re.M)
        pairs = []
        for f in files:
            b = git(clone, "show", f"{t['base_commit']}:{f}")
            a = apply_patch_text(b, t["patch"], f)
            if b and a and a != b:
                pairs.append((f, b, a))
        if pairs:
            items.append((f"task-{tid}", "train", t["problem_statement"][:6000], pairs))
    return items


def apply_patch_text(before: str, patch: str, fp: str) -> str | None:
    """Apply the hunks of `patch` for file fp to `before` (context-checked); None if they do not fit."""
    m = re.search(r"^diff --git a/" + re.escape(fp) + r" b/.*?(?=^diff --git |\Z)", patch, re.M | re.S)
    if not m:
        return None
    lines = before.split("\n")
    out, pos = [], 0
    for h in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@.*\n((?:[ +\-\\].*\n?)*)", m.group(0), re.M):
        start = int(h.group(1)) - (1 if h.group(2) != "0" else 0)
        out += lines[pos:start]
        pos = start
        for l in h.group(3).split("\n"):
            if l.startswith("\\") or l == "":
                continue
            tag, txt = l[0], l[1:]
            if tag == " ":
                if pos >= len(lines) or lines[pos] != txt:
                    return None
                out.append(txt); pos += 1
            elif tag == "-":
                if pos >= len(lines) or lines[pos] != txt:
                    return None
                pos += 1
            elif tag == "+":
                out.append(txt)
    out += lines[pos:]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commits-per-repo", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repos", nargs="*", default=TUNE)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    systems = {k: p.read_text(encoding="utf-8") for k, p in PROMPTS.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, names in TOOLS.items():
        (out / f"tools_{k}.json").write_text(json.dumps(R.schemas_for(names), indent=1), encoding="utf-8")
    tasks = {}
    for l in open("D:/data/mined/data/tasks.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l); tasks[r["instance_id"]] = r
    usable = set(json.load(open(REPO / "mine" / "usable.json"))["usable"])

    allrecs, stats = [], {}
    for repo in a.repos:
        items = task_items(repo, usable, tasks) + history_items(repo, a.commits_per_repo, rng)
        n_before = len(allrecs)
        for src_id, split, statement, pairs in items:
            for fp, b, af in pairs:
                allrecs += samples_for_file(rng, repo, fp, b, af, statement, systems, src_id, split)
        stats[repo] = (len(items), len(allrecs) - n_before)
        print(f"{repo:22s} sources {len(items):4d}  samples {len(allrecs) - n_before:5d}", flush=True)
    for name in ("train", "validation"):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for r in allrecs:
                if r["split"] == name:
                    fh.write(json.dumps(r) + "\n")
    chars = [sum(len(json.dumps(m)) for m in r["messages"]) for r in allrecs]
    by = {}
    for r in allrecs:
        by.setdefault((r["split"], r["format"], r["step"]), 0)
        by[(r["split"], r["format"], r["step"])] += 1
    print("\nsamples by (split, format, step):")
    for k in sorted(by):
        print("  ", k, by[k])
    if chars:
        chars.sort()
        print(f"context size in characters: median {chars[len(chars) // 2]}, 95th pct {chars[int(len(chars) * .95)]} (about 3.2 characters per token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
