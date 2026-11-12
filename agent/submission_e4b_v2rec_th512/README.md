One-factor arm on `../submission_e4b_v2rec` (the shipped configuration): thinking_budget 512 (v2rec has 4096). Everything else, the prompt included, is identical.

Why this change: Thoughts are 31% of the prompt in the control runs (mean 3,846 of 12,283 tokens) and 43% of the episodes hit the 14,336-token compaction line; compacted episodes resolved 5.3% against 14.7% (confounded by episode length). A smaller budget keeps the history shorter and the episode faster. It may also make the model think too little at the step that matters.

Decision rule, written before any run (2026-11-07): three seeds on dev66 next to a fresh `v2rec` control seed in the same session (the corrected-harness controls `ctl_v2rec1/2` resolved 7 of 66 each).
The arm is a candidate only if its three-seed mean is at least 3 tasks above the control mean and its share of empty patches is not higher. One seed of one arm has a spread of about 2.5 tasks, so this is a weak bar:
a pass is a reason to run more seeds, not a finding. dev66 is a fitted pool; any look at `rich` or the mined unseen repositories goes in `docs/run_ledger.csv`.
No run of this arm has been made.
