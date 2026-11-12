"""Step 1 of the task miner: list merged PRs of a repository that look like benchmark tasks.

  python mine/candidates.py --repo pallets/click --since 2025-02-01 --out D:/data/mined/click.cands.jsonl

One GraphQL search call returns 50 PRs with their changed files, the issues they close and the merge commit's
parent, so a repository with a few hundred PRs costs a handful of calls. Filters mirror the public pipeline
(docs/../rules-raw/data-description.md, "Dataset Generation"): source .py and test files both change, no
documentation-only or mechanical changes, issue linked, small source patch. See mine/README.md.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

QUERY = """
query($q: String!, $after: String) {
  search(query: $q, type: ISSUE, first: 25, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number title body createdAt mergedAt isDraft
      author { login }
      baseRefName
      mergeCommit { oid parents(first: 3) { nodes { oid } } }
      files(first: 100) { totalCount nodes { path additions deletions changeType } }
      closingIssuesReferences(first: 5) { nodes { number title body url repository { nameWithOwner } } }
    } }
  }
}
"""

BOT = re.compile(r"(\[bot\]|dependabot|pre-commit-ci|github-actions|renovate)", re.I)
TITLE_SKIP = re.compile(r"^\s*(docs?|ci|chore|build|bump|release|deps?|pre-commit|refactor|style|typing|types?|"
                        r"revert|merge|update|📝|⬆|🔧|🎨|♻)\b", re.I)
IGNORABLE_EXT = {".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".cff", ".lock", ".in", ".png",
                 ".svg", ".gif", ".jpg", ".css"}
NON_SOURCE_DIRS = {"docs", "doc", "examples", "example", "benchmarks", "benchmark", "scripts", "tools", "tasks",
                   "bench", ".github", "changes", "news", "changelog.d", "docs_src"}

MAX_SRC_FILES, MAX_SRC_LINES, MIN_SRC_LINES, MAX_FILES = 3, 80, 2, 15


def gh_graphql(query: str, variables: dict, retries: int = 4) -> dict:
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        if v is not None:
            cmd += ["-f", f"{k}={v}"]
    for attempt in range(retries):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            return json.loads(r.stdout)
        msg = (r.stderr or r.stdout)[:300]
        if "rate limit" in msg.lower() or "abuse" in msg.lower() or "502" in msg or "timeout" in msg.lower():
            time.sleep(20 * (attempt + 1))
            continue
        raise RuntimeError(f"gh graphql failed: {msg}")
    raise RuntimeError("gh graphql: retries exhausted")


def is_test(path: str) -> bool:
    parts = path.lower().split("/")
    base = parts[-1]
    return (any(p in ("test", "tests", "testing") for p in parts[:-1]) or base.startswith("test_")
            or base.endswith("_test.py") or base == "conftest.py")


def classify(files: list[dict]):
    """-> (source_py, tests, ignored, other) lists of file dicts."""
    src, tests, ignored, other = [], [], [], []
    for f in files:
        p = f["path"]
        ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p.rsplit("/", 1)[-1] else ""
        parts = set(p.lower().split("/")[:-1])
        if is_test(p):
            tests.append(f)
        elif ext in IGNORABLE_EXT or parts & NON_SOURCE_DIRS:
            ignored.append(f)
        elif ext == ".py":
            src.append(f)
        else:
            other.append(f)
    return src, tests, ignored, other


def judge(pr: dict) -> tuple[bool, str]:
    if pr.get("isDraft"):
        return False, "draft"
    if BOT.search((pr.get("author") or {}).get("login") or ""):
        return False, "bot author"
    if TITLE_SKIP.search(pr["title"]):
        return False, "title looks like docs/chore/refactor"
    mc = pr.get("mergeCommit")
    if not mc or not mc["parents"]["nodes"]:
        return False, "no merge commit"
    files = pr["files"]["nodes"]
    if pr["files"]["totalCount"] > MAX_FILES or len(files) != pr["files"]["totalCount"]:
        return False, "too many files"
    src, tests, ignored, other = classify(files)
    if other:
        return False, "non-python, non-doc file changed"
    if not (1 <= len(src) <= MAX_SRC_FILES):
        return False, f"{len(src)} source files"
    if not any(f["path"].endswith(".py") for f in tests):
        return False, "no test .py file"
    lines = sum(f["additions"] + f["deletions"] for f in src)
    if not (MIN_SRC_LINES <= lines <= MAX_SRC_LINES):
        return False, f"{lines} source lines"
    if sum(f["additions"] for f in tests) < 3:
        return False, "almost no test lines"
    if any(f["changeType"] in ("DELETED", "RENAMED") for f in src):
        return False, "source file deleted/renamed"
    if not pr["closingIssuesReferences"]["nodes"]:
        return False, "no closing issue"
    return True, f"{len(src)} src files, {lines} lines"


def date_windows(since: str, until: str, days: int = 120):
    import datetime as dt
    a, b = dt.date.fromisoformat(since), dt.date.fromisoformat(until)
    while a <= b:
        e = min(a + dt.timedelta(days=days - 1), b)
        yield a.isoformat(), e.isoformat()
        a = e + dt.timedelta(days=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--since", default="2025-02-01")
    ap.add_argument("--until", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    seen, kept, reasons, total = set(), [], {}, 0
    for lo, hi in date_windows(a.since, a.until):
        after = None
        while True:
            q = f"repo:{a.repo} is:pr is:merged merged:{lo}..{hi} linked:issue"
            data = gh_graphql(QUERY, {"q": q, "after": after})
            if "errors" in data:
                raise RuntimeError(str(data["errors"])[:300])
            s = data["data"]["search"]
            for pr in s["nodes"]:
                if not pr or pr["number"] in seen:
                    continue
                seen.add(pr["number"])
                total += 1
                ok, why = judge(pr)
                if not ok:
                    reasons[why.split(",")[0].split(" ")[0] + " " + " ".join(why.split(" ")[1:3])] = \
                        reasons.get(why.split(",")[0].split(" ")[0] + " " + " ".join(why.split(" ")[1:3]), 0) + 1
                    continue
                kept.append({"repo": a.repo, "number": pr["number"], "title": pr["title"], "body": pr["body"] or "",
                             "created_at": pr["createdAt"], "merged_at": pr["mergedAt"],
                             "merge_oid": pr["mergeCommit"]["oid"],
                             "parent_oids": [n["oid"] for n in pr["mergeCommit"]["parents"]["nodes"]],
                             "files": pr["files"]["nodes"], "why": why,
                             "issues": [{"number": i["number"], "title": i["title"], "body": i["body"] or "",
                                         "repo": i["repository"]["nameWithOwner"]}
                                        for i in pr["closingIssuesReferences"]["nodes"]]})
            if not s["pageInfo"]["hasNextPage"]:
                break
            after = s["pageInfo"]["endCursor"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for k in kept:
            fh.write(json.dumps(k) + "\n")
    print(f"{a.repo}: {total} merged issue-linked PRs since {a.since}, {len(kept)} candidates kept -> {a.out}")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
        print(f"   rejected {v:4d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
