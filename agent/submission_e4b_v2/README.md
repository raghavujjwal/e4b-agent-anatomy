Prompt v2 for E4B. Identical to `../submission_e4b` (same sampling, budgets and tools) except `prompts/flat_search.md`.
Changes, each aimed at a failure measured in baseline 1 (docs/MEASUREMENTS.md) and none naming a task or repository:
1. Edit hygiene: copy old_string verbatim from a fresh read_file, 1-3 unique lines, re-read after a failed edit, no allow_multiple, no
   source rewrites with write_file or shell (15 tasks had 4+ failed edits; 3 big rewrites of applications.py).
2. A required one-line import check after the last edit (8 patches left fastapi unimportable; no run checked anything).
3. pyproject.toml and the other config files named as off limits (2 runs edited pyproject.toml).
4. Removed "finish early" and the shared-pool time pressure (median 9 of 24 calls, 10-65 s of 330 s used).
5. Removed the two paragraphs about a candidate list that this arm never provides.
Test as one bundle against the baseline on 3 seeds each; ablate single rules only if the bundle helps (docs/EVAL_PROTOCOL.md).
