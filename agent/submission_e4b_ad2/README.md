Baseline E4B (../submission_e4b) plus the edit-step LoRA `edit_v1` at LoRA scale 0.5: the only difference is `adapter: edit_v1_s050` in agent.yaml. Prompt, sampling (temperature 0.2) and
budgets are the baseline's, so this is a clean test of the adapter against base1-3 (7, 7, 8 of 66) and the same settings the step check was run with.
Why scale 0.5: edit_v1 was trained at rank 32, alpha 64 = scale 2.0 and collapsed in the agent (`ad1`: 64 of 65 tasks used all 24 calls on failing edit_file calls). The step-type check
(notebook 05, 2026-11-02, 120 prefixes of the base model's own runs) failed scale 1.0 (read_file agreement 9% against 73% for the base; it chose edit_file at 25 of 33 read steps) and PASSED
scale 0.5 (read_file 70% vs 73%, run_command 62% vs 71%, submit_patch 87% vs 74%, no_call 0.8% vs 1.7%). Offline, scale 0.5 applies 78% of edits against 51% for the base (89% at scale 2.0).
The adapter (train/train_lora.py via notebook 04: rank 32, 4,000 edit_file samples, 266 MiB) is NOT in git. It lives on Drive at MyDrive/gemma4-adapters/edit_v1. Notebook 03 copies it to
`/content/adapters/edit_v1_s050` and rewrites `lora_alpha` to 16 in that copy (notebook config `ALPHAS`), then into `adapters/edit_v1_s050/` here. To pack a submission zip, put those two files
in `adapters/edit_v1_s050/` (with alpha 16) and run `python agent/make_submission_zip.py agent/submission_e4b_ad2 --out ...`.
Mechanism checks (mine/trace_analysis.py), fixed before the run: edit_file failure rate (base/v2rec: 56%) must fall to 35% or lower; share of edit_file calls on a file that was not read earlier in the
run (base: 42 of 454 "not found" failures were of this kind) must not rise sharply, because the step check showed the adapter edits earlier than the base (14.8% of non-edit steps became edit_file
against 4.5% for the base).
Collapse guard (the ad1 failure): any run with more than 8 tool_budget stops, more than 9 empty patches or a median above 14 calls is a failure whatever the resolved count.
Decision rule: ad2_1 resolving 10 or more of 66 earns seeds 2 and 3; it counts as a gain only if the three-seed mean is 10 or more AND the mechanism and collapse checks hold.
Resolved counts are read against base1-3 (7, 7, 8) and v2rec1-3 (12, 7, 9), noise about 3-4 tasks per run.
Open risk, not testable here: the scorer's vLLM is 0.19.1 and this was served on 0.32.0; whether the scorer loads a Gemma 4 LoRA at start-up is unverified, and a LoRA bundle that fails there errors the whole submission.
