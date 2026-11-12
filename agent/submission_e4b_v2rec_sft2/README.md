`../submission_e4b_v2rec` plus the S3 adapter `e4b_sft_v2`: the only difference is `adapter: e4b_sft_v2` in agent.yaml.

The adapter (notebook 07: rank 64, alpha 64 = LoRA scale 1.0, 191 steps, 1,606 samples) is NOT in git. It is on Drive at `MyDrive/gemma4-adapters/e4b_sft_v2_for_submission`, and notebook 03 copies it in when ADAPTERS names it:
`ADAPTERS = {"e4b_sft_v2": "/content/drive/MyDrive/gemma4-adapters/e4b_sft_v2_for_submission"}`, `ALPHAS = {}`. The server then starts with `--enable-lora`, so run a same-session v2rec control on that server too (protocol rule 9).

What is known before the run (docs/PLAN_E4B_SFT.md, docs/PLAN_EDIT_PAIRS.md): the S4 gate G3 FAILED (read_file agreement 30% against 70% for the base), the adapter's thought rate is 0%, and after a failed edit it gives up (submit 22% against 6%). This arm is the direct measurement that the gates stood in for. The prior that it beats the control is low; lab session 1 showed that an arm that reasons less makes more failing edits and more empty patches.

Rule, written before any run (2026-11-08): no collapse first (empty patches at most the control's plus 5 points, mean calls within 30% of the control's, loops of 4 or more identical failing calls in at most 10% of the runs), then `docs/EVAL_PROTOCOL.md` rule 16: at least 4 tasks more than the same-session control on the mined tune pool, and a pass buys two more seeds, not a claim.
No run of this arm has been made.
