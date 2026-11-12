Aid arm. Copy of `../submission_e4b` (baseline prompt, temperature 0.2, budgets) with two changes:
1. The skill `skills/repo_aid` (attached under `skills:`, `run_skill_script` is added by the harness) with two scripts:
   `find_files.py` ranks the workspace's non-test Python files against the issue text (BM25, path + defined names and file text; top 5 of each view, at most 10 files) and
   `edit_retry.py` applies an `edit_file` call that failed with "old_string not found" by locating the one place the text was meant to be (unescaped backslash sequences, indentation/blank differences,
   a >= 0.9 similar window clearly ahead of the next one), re-indenting new_string, refusing an edit that would leave a Python file unparsable, and otherwise printing the closest lines to copy from.
2. The prompt: the dangling "Choose a file from this list" section (the baseline never provided the list) now tells the model to call `find_files` first, and the editing paragraph says what to do after "old_string not found".
`edit_file` stays declared; the retry skill is a fallback after a failure, so the interface the model already uses does not change (the `lines` arm replaced `edit_file` and scored 5/1/0).
Evidence: offline the ranker puts the gold file in its list for 72% of single-file public tasks (64% of short-issue ones; the agent alone saw it in 57% / 21%) and `edit_retry`'s locate step finds a place for
35% (shown text only) to 48% (whole base-commit file) of the recorded "not found" failures (docs/MEASUREMENTS.md). Pre-registered rules for the run: below.

Pre-registered rules for `aid1` (one seed, 66 dev tasks, `fitted`; baseline 7, 7, 8; fixed before the run):
- Compliance: `find_files` is called in at least 80% of the task runs, and in the first 3 calls.
- Mechanism 1 (localization), single-file tasks: gold file seen in at least 65% of runs (baseline 56.7%), and in at least 40% on issues under 300 characters (baseline 21%).
- Mechanism 2 (edit retry): after an `edit_file` call that failed with "old_string not found", `edit_retry` is called within the next 2 calls in at least 50% of the cases, and at least 40% of those calls print "OK: edit applied".
- Outcome: 10 or more of 66 resolved is the bar for running two more seeds. One seed is inside the +-3-4 noise, so the mechanism numbers decide whether the idea works; the count only decides whether it is worth 8 more units.
- Collapse guard: interrupt the run if more than 8 of the first 20 tasks end in `tool_budget`, or more than 12 of the first 25 end with an empty patch.
