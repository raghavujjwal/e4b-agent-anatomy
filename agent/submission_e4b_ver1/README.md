Ver1 arm. Copy of `../submission_e4b_aid2` (baseline + repo_aid skill with the aid2 fixes) with two further changes:
1. The "Finishing" section of the prompt, which told the model to finish early and keep its reasoning to a few sentences (written for the 31B, where time is scarce), is replaced by "Check your work before you finish":
   write /tmp/repro.py from the issue, run it, edit, run it again, run the one covering test file, then submit.
2. `eval_config.yaml`: max_tool_calls 24 -> 30 and max_turns 40 -> 50 (the repro routine needs about 4 more calls). Time cap (5.5 min) and timeout unchanged.
Why: the E4B runs are short (aid1: median 6 calls, 22 s of the 330 s cap), so the budget is unused; in the 396 baseline/v2rec runs 15.5% of those that ran a test or script resolved against 10.2% of those that did not
(confounded by task difficulty). Whether making the model check its work helps is untested.

Pre-registered rules for `ver1_1` (one seed, 66 dev tasks, `fitted`; baseline 7, 7, 8; aid1 7):
- Routine followed: /tmp/repro.py (or another script) written and run BEFORE the first edit in at least 60% of the runs, and run again after the first applied edit in at least 50%.
- Budget: median tool calls at most 20, runs ending in `tool_budget` at most 12 of 66, empty patches at most 6 (the extra checking must not eat the budget or the context).
- Outcome: 11 or more of 66 resolved to run two more seeds. Fewer: the verification idea is dropped.
- Collapse guard: interrupt if more than 8 of the first 20 tasks end in `tool_budget`, or more than 12 of the first 25 end with an empty patch.
