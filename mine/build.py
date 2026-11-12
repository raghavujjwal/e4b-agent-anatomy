"""Step 2 of the task miner: candidates -> tasks.jsonl + snapshots/<id>.tgz in the public dataset's shape.

  python mine/build.py --cands D:/data/mined/click.cands.jsonl --out D:/data/mined/data [--limit 30]

For each candidate the repository is cloned once, the diff between the merge commit's first parent (the repo
state just before the fix) and the merge commit is taken, and its file list is checked against the PR's file
list. A mismatch (rebase-merged PRs, or a parent that is not the real base) drops the candidate. The source
.py files become `patch`, the test files `test_patch`; documentation and other ignorable files are left out,
as in the public tasks. The snapshot is `git archive` of the parent (no history; the grader runs `git init`).
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import classify  # noqa: E402


def git(repo: Path, *args: str, check=True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)[:80]} failed: {r.stderr[:200]}")
    return r.stdout


def ensure_clone(repo_name: str, clones: Path) -> Path:
    d = clones / repo_name.replace("/", "__")
    if (d / ".git").exists():
        subprocess.run(["git", "-C", str(d), "fetch", "--no-tags", "-q", "origin"], capture_output=True)
        return d
    clones.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "clone", "--no-tags", "-q", f"https://github.com/{repo_name}.git", str(d)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"clone failed: {r.stderr[:300]}")
    return d


def clean_text(s: str, limit: int = 8000) -> str:
    s = re.sub(r"<!--.*?-->", "", s or "", flags=re.S)
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s[:limit]


def problem_statement(c: dict) -> str:
    parts = []
    for i in c["issues"][:3]:
        parts.append(f"{i['title'].strip()}\n\n{clean_text(i['body'])}".strip())
    return "\n\n---\n\n".join(parts) if parts else f"{c['title']}\n\n{clean_text(c['body'])}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True, help="dataset-shaped folder: tasks.jsonl + snapshots/")
    ap.add_argument("--clones", default="D:/data/mined/clones")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    out = Path(a.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    tasks_path = out / "tasks.jsonl"
    have = set()
    if tasks_path.exists():
        have = {json.loads(l)["instance_id"] for l in tasks_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    cands = [json.loads(l) for l in open(a.cands, encoding="utf-8") if l.strip()]
    if not cands:
        print("no candidates")
        return 0
    clone = ensure_clone(cands[0]["repo"], Path(a.clones))
    short = cands[0]["repo"].split("/")[1].lower()

    made, skipped = 0, {}
    for c in cands:
        iid = f"{short}_{c['number']}"
        if iid in have:
            continue
        if a.limit and made >= a.limit:
            break
        parent, merge = c["parent_oids"][0], c["merge_oid"]
        try:
            changed = set(git(clone, "diff", "--name-only", parent, merge).split("\n")) - {""}
        except RuntimeError as e:
            skipped["git error"] = skipped.get("git error", 0) + 1
            continue
        if changed != {f["path"] for f in c["files"]}:
            skipped["diff != PR files"] = skipped.get("diff != PR files", 0) + 1
            continue
        src, tests, _, _ = classify(c["files"])
        patch = git(clone, "diff", "--no-color", "--no-ext-diff", parent, merge, "--", *[f["path"] for f in src])
        test_patch = git(clone, "diff", "--no-color", "--no-ext-diff", parent, merge, "--",
                         *[f["path"] for f in tests])
        if not patch.strip() or not test_patch.strip():
            skipped["empty patch"] = skipped.get("empty patch", 0) + 1
            continue
        tgz = out / "snapshots" / f"{iid}.tgz"
        r = subprocess.run(["git", "-C", str(clone), "archive", "--format=tar.gz", "-o", str(tgz), parent],
                           capture_output=True, text=True)
        if r.returncode != 0 or not tgz.exists():
            skipped["archive failed"] = skipped.get("archive failed", 0) + 1
            continue
        rec = {"instance_id": iid, "repo": c["repo"], "base_commit": parent, "patch": patch, "test_patch": test_patch,
               "problem_statement": problem_statement(c), "hints_text": "", "created_at": c["created_at"],
               "pr_number": c["number"], "merged_at": c["merged_at"], "source": "mined"}
        with open(tasks_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec) + "\n")
        made += 1
    print(f"{cands[0]['repo']}: built {made} tasks (already had {len([h for h in have if h.startswith(short + '_')])}); "
          f"skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
