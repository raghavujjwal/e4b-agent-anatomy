Localization arm. Copy of `../submission_e4b_v2rec` with ONE change: the "Finding the right place to change" section of `prompts/flat_search.md`. Sampling, tools
and budgets are v2rec's.
Why (docs/MEASUREMENTS.md, "Where the baseline runs lose tasks"): in 42% of single-file task runs the gold file was never opened; it was opened in 21% of runs when
the issue text is under 300 characters (15 of 45 tasks) against 74% for longer text, and in 10 of 45 tasks in none of 6 runs. Once the gold file was opened an
edit attempt followed 95% of the time. The new section tells the model to take two to four distinctive words from the title and body and to rank files with one
command that counts how many of the words each file contains (tests, docs and .git excluded), then open the top file. The command is tested on a toy tree.
Mechanism check (from `mine/localization_funnel.py`): share of runs that open a gold file, overall and for the short-issue tasks (v2rec/base: 64% overall,
21% for short issues). Decision rule fixed before the run: the arm did its job if the short-issue share rises to 40% or more without the overall share falling.
Resolved counts are read against `base1-3` (7, 7, 8) and `v2rec1-3` (12, 7, 9) of 66, noise about 3-4 tasks per run.
Risk: the one-liner contains quotes, `$w` and pipes inside a prompt; if run_command calls fail with syntax errors in the traces, the arm is not testing localization.
