One-factor arm on `../submission_e4b_v2rec` (the shipped configuration): thinking off (`thinking_budget: 0`, `include_thoughts: false`; v2rec has a 4096 budget). Everything else, the prompt included, is identical.

Why this change: The extreme of the same idea as th512: no thought channel at all, so the history holds only calls and results. With adk-submission 0.2.13 a budget of 0 sends `enable_thinking: false`; the older bridge (0.2.11) mapped `include_thoughts: false` to thinking off, so both are set. Which version the hidden scorer runs is unverified (docs/MEASUREMENTS.md).

Decision rule, written before any run (2026-11-07): three seeds on dev66 next to a fresh `v2rec` control seed in the same session (the corrected-harness controls `ctl_v2rec1/2` resolved 7 of 66 each).
The arm is a candidate only if its three-seed mean is at least 3 tasks above the control mean and its share of empty patches is not higher. One seed of one arm has a spread of about 2.5 tasks, so this is a weak bar:
a pass is a reason to run more seeds, not a finding. dev66 is a fitted pool; any look at `rich` or the mined unseen repositories goes in `docs/run_ledger.csv`.
No run of this arm has been made.
