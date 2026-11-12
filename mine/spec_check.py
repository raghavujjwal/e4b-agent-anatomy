"""Static spec-sufficiency check: flag tasks whose tests need a new public name that the issue never mentions.

  python mine/spec_check.py --tasks D:/data/mined/data/tasks.jsonl --out D:/data/mined/data/results/spec_flags.json
  python mine/spec_check.py --tasks D:/tools/dataset-files/tasks.jsonl --out <path>

A task is flagged when the gold source patch adds a public `def`/`class` (or a whole new module) whose name appears in the
lines the test patch adds, but not in the problem statement. An agent reading only the issue cannot know that name, so the
task measures guessing, not fixing. It is a heuristic: it misses names introduced through assignments or attributes and can
flag a moved function; read the list before relying on it. It does not look at any agent output.
"""
from __future__ import annotations
import argparse
import collections
import json
import re
import sys
from pathlib import Path


def added_names(patch: str) -> set[str]:
    """Public names the source patch introduces: added def/class, plus the module name of every new file."""
    names = {m.group(1) for m in re.finditer(r"^\+\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", patch, re.M)
             if not m.group(1).startswith("_")}
    for f in re.findall(r"^--- /dev/null\n\+\+\+ b/(\S+)", patch, re.M):
        names.add(f.rsplit("/", 1)[-1].replace(".py", ""))
    return names


def used_by_tests(test_patch: str, name: str) -> bool:
    added = "\n".join(l[1:] for l in test_patch.splitlines() if l.startswith("+") and not l.startswith("+++"))
    return re.search(r"\b" + re.escape(name) + r"\b", added) is not None


def check(task: dict) -> list[str]:
    """Names the tests use that the problem statement does not contain (empty list = sufficiently specified)."""
    need = [n for n in added_names(task["patch"]) if used_by_tests(task["test_patch"], n)]
    return sorted(n for n in need if not re.search(r"\b" + re.escape(n) + r"\b", task["problem_statement"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", help="json {instance_id: [unspecified names]} for flagged tasks only")
    a = ap.parse_args(argv)
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    flags = {t["instance_id"]: n for t in tasks if (n := check(t))}
    per_repo = collections.Counter(t["repo"] for t in tasks)
    per_repo_flag = collections.Counter(t["repo"] for t in tasks if t["instance_id"] in flags)
    print(f"{len(flags)} of {len(tasks)} tasks flagged")
    for repo in sorted(per_repo):
        print(f"  {repo:28s} {per_repo_flag[repo]:3d} / {per_repo[repo]:3d}")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(sorted(flags.items())), indent=1) + "\n", encoding="utf-8")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
