Contract arm, adapted from the teammates' 31B bundle H (`main`, `arms/H_frontier024`, public 0.18 / 0.17 / 0.22 / 0.18 on the same bytes). Copy of `../submission_e4b_v2rec` with ONE change: two
short sections added at the top of `prompts/flat_search.md`, "Budget" (24 calls, about 5 minutes, the tree is graded as it is when a limit hits) and "Before any tool call" (write down the exact
names, signatures, messages and return values the problem statement requires and use them verbatim). Sampling, tools and budgets are v2rec's.
Why (docs/MEASUREMENTS.md, "Where the baseline runs lose tasks"): in 28% of single-file task runs an edit applied and the tests still failed, the largest loss after localization, and the hidden
tests assert on exact strings. H's other distinctive rule, "a source file must be edited by call 12", is left out on purpose: in the 396 baseline runs the median first edit attempt is call 5 and
365 runs attempt an edit by call 10, so the rule would hardly bind here.
Mechanism check (from `mine/trace_analysis.py` and the result files): the share of single-file runs where an edit applied but the task is not resolved (v2rec/base: 28%), and the number of
runs that stop on the budget (v2rec: 3, 4, 5).
Decision rule fixed before the run: the arm is worth three seeds if v3h1 resolves 11 or more of 66 (v2rec 12, 7, 9; base 7, 7, 8; noise about 3-4 tasks per run). It counts as a gain only if the
three-seed mean is 11 or more AND budget stops are not above 8 in any run AND empty patches are not above 9 in any run (v2rec: 3, 9, 9).
Risk: more text before the first tool call means more thinking tokens per task; check the median seconds per task (v2rec 33-41 s) so the 5.5 minute cap is not threatened.
