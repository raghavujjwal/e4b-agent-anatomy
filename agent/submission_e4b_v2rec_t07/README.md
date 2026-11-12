One-factor arm on `../submission_e4b_v2rec` (the shipped configuration): temperature 0.7 (v2rec has 1.0). Everything else, the prompt included, is identical.

Why this change: 49% of the edit calls in the 132 control episodes were refused by the matcher and 93% of those 'not found' edits had the longest line of their old_string visible in the prompt: copy errors on text the model had just read. 1.0 is what the checkpoint's generation_config.json recommends; the baseline's 0.2 is the 31B campaign's. 0.7 is the one-step move toward exact copying that keeps most of the variety.

Decision rule, written before any run (2026-11-07): three seeds on dev66 next to a fresh `v2rec` control seed in the same session (the corrected-harness controls `ctl_v2rec1/2` resolved 7 of 66 each).
The arm is a candidate only if its three-seed mean is at least 3 tasks above the control mean and its share of empty patches is not higher. One seed of one arm has a spread of about 2.5 tasks, so this is a weak bar:
a pass is a reason to run more seeds, not a finding. dev66 is a fitted pool; any look at `rich` or the mined unseen repositories goes in `docs/run_ledger.csv`.
No run of this arm has been made.
