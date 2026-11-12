# Task miner

Builds extra evaluation and training tasks from public GitHub repositories, in the shape of the competition's
`tasks.jsonl` + `snapshots/<id>.tgz`, so the grader (`grader/`) and the agent driver (`agent/run_eval.py`) run on them unchanged.

Why: the 129 public tasks come from four repositories, and the data description says the hidden test set "was curated from a
set of private repositories" (`rules-raw/data-description.md`, last line). Whether those are the same four repositories is unknown.
Mined tasks give (a) tasks never read while developing the agent, (b) other repositories, and (c) fresh tasks after the model's
training cutoff. Rules section 6 allows public external data.

## Steps
```
python mine/candidates.py --repo pallets/click --out D:/data/mined/click.cands.jsonl     # GitHub GraphQL via `gh`
python mine/build.py --cands D:/data/mined/click.cands.jsonl --out D:/data/mined/data --limit 15
bash D:/tools/run-mined-controls.sh /mnt/d/data/mined/data                               # WSL: gold + empty controls
```
A task is **usable** only if the gold patch resolves and the empty patch does not.

## Selection rules (mirroring the public pipeline)
Merged PR that closes an issue; 1 to 3 changed source `.py` files and at least one test `.py` file with 3 or more added lines;
2 to 80 changed source lines; at most 15 files; no docs, CI, dependency, refactor or typing titles; no bot authors;
no deleted or renamed source file; the base commit is the merge commit's first parent and the parent-to-merge diff must list
exactly the PR's files (this drops rebase-merged and squash-oddity PRs). `patch` = source files, `test_patch` = test files;
documentation and other ignorable files are left out, as in the public tasks. `problem_statement` = the closing issue(s) text.

## Spec check
`python mine/spec_check.py --tasks <tasks.jsonl> --out <flags.json>` flags a task when the gold source patch adds a public
`def`/`class` (or new module) whose name appears in the added test lines but not in the problem statement; a model reading only the
issue cannot know that name. Heuristic: it misses names added through assignments and can flag a moved function, so read the list.
Result on the mined set: 12 of 213 flagged, 7 of the 122 usable (`spec_flags_mined.json`). On the public set: 15 of 129
(`spec_flags_public.json`; fastapi 10, rich 5, requests 0, httpx 0). Flagged tasks stay in `usable.json`; report them separately.

## Differences from the competition tasks (known, do not hide)
- The organizers' filters are not public. In particular the hidden set also requires a frontier model to pass or nearly pass each
  case; ours has no such check, so some mined tasks may be harder or less well specified.
- Environments are built from the repository's declared dependencies using the public wheelhouse first, then PyPI as a recorded
  fallback. The scorer uses its own wheelhouse; versions can differ.
- `problem_statement` is the raw GitHub issue, not whatever text the organizers used. `hints_text` is empty, as in the public set.
- Tasks added to a repository's history after the issue was filed may include PR-only fixes that never changed behaviour users hit.
- Scores on mined tasks are our own ranking tool. They say nothing about the leaderboard.

## Hygiene
Fix which repositories are for tuning and which stay unseen **before** looking at any agent output on them
(`docs/EVAL_PROTOCOL.md`). Mined task ids are `<repo>_<pr number>`; the PR number is the PR, not the issue.
