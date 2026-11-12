# Plan: on-policy edit pairs (applied against failed), pre-registered

Written 2026-11-07, before any pair has been sampled and before any adapter of this study exists. Everything under "Rules" is fixed by this note; a change after the first sample is
listed under "Deviations" at the end with its date and the reason, and the paper reports it as a deviation. Code: `train/edit_pairs.py` (states, sampling, labels, pairs, held-out evaluation,
`decide`), `train/pref_lora.py` (the three arms), `train/format_probe.py`, `train/prompt_spec_v2rec.json`, tests in `tests/test_edit_pairs.py`, `tests/test_pref_lora.py`,
`tests/test_format_probe.py` and `tests/test_notebooks_pairs.py`. The run is notebook 09 (sample, pairs, train) and notebook 10 (serve, evaluate, gates).

## 1. Question

Does preference training on the model's own `edit_file` calls, labelled by the harness's real matcher (a call that applies against one that does not), make Gemma 4 E4B copy the
text it was shown more reliably, **without** the step-choice collapse that ended the two earlier adapters?

Why this might work where the earlier ones failed, and why it might not:
- `edit_v1` (SFT on gold edits, 6,526 samples) raised the offline apply rate from 53.3% to 88.7%, but 31% of the base's post-read steps choose another tool and the adapter edited at
  100% of them (`wrong_tool` 92 -> 0), wrote no thoughts (1,494 -> 158 tokens) and collapsed in the agent (47 of 88 episodes used all 24 calls at scale 1.0; 14.8% at scale 0.5).
  `e4b_sft_v2` (S3) failed gate G3 and, on the per-generation files, gave up after failed edits (submit 22% against the base's 6%; G4b was defined from that).
  Both trained on text the base model would not have written (gold edits, teacher traces).
- Here the training text is the base model's own, sampled at the shipped settings after one shared thought, and the label comes from the tool. In DPO the tokens two calls share cancel
  in the margin, so the loss does not ask for `edit_file` over `read_file`; it acts where the calls differ (a copied line, a quote style). Nothing guarantees this holds in practice:
  that is what the gates test.
- It may still fail for reasons the design cannot remove: the labels come from the shown window (not the whole file) and do not measure whether the edit is the right fix; the states
  are short synthetic prefixes, not the long thought-laden prompts of real episodes; apply rate already is not the bottleneck on resolved tasks (7 of 66 resolved at 40% applied edits).

## 2. What the control runs say about edits (corrected harness, 132 episodes, 501 edit calls with a known outcome)

Applied 199 (39.7%), refused by the matcher 247 (49.3%), rejected by the framework before the tool ran 55 (11.0%). By the number of earlier failed edits visible in the prompt:

| earlier failed edits | calls | applied | refused by matcher | rejected by framework |
|---|---|---|---|---|
| 0 | 254 | 58.3% | 22.0% | 19.7% |
| 1 | 66 | 28.8% | 66.7% | 4.5% |
| 2 | 49 | 12.2% | 85.7% | 2.0% |
| 3 or more | 132 | 19.7% | 79.5% | 0.8% |

Rejected calls cascade: with an earlier rejected call in the prompt the next edit is rejected 46.7% of the time (75 calls), otherwise 4.7% (426 calls). Of the 55 rejected calls, 30 are what the
labeller below calls `malformed` (Python-style keyword arguments `old_string="..."`, a mandatory argument missing, or a call that never closes; the framework refused 30 of 30 such calls) and 25
are other unreadable styles (JSON quotes after a space, triple quotes, bare values): of the 38 calls of those styles the framework ran 13 (3 applied, 10 refused by the matcher). 93% of the failed "not found" edits have their longest line visible in the prompt: copy errors. Scripts and outputs:
`D:\data\e4b_plan\analysis\` (logged in `D:\data\e4b_plan\logs\LOG.md`).

## 3. Rules

**States.** The 6,526 `edit_file` prefixes of `D:\data\edit_sft\train.jsonl` (9 tune repositories; 6,227 from historical commits, 299 from the tune tasks), each a read of a window
around the gold hunk. They are re-rendered the way the scorer renders the shipped agent: the v2rec system text with ADK's agent line, the six real tool declarations and the real user
wrapper (budget, rules, workspace tree) from `tests/fixtures/scorer_requests_v2rec.json`, with `enable_thinking`. (The edit_sft prompt is the older recipe with shorter tool
descriptions; edit_v1 was trained on it.) Gaps that remain: the workspace tree is a stand-in built from the file path; the history has no thought for the read turn; the window is
about 40 lines, not the file. 2,400 prefixes are drawn with seed 0.

**Sampling.** Shipped settings (temperature 1.0, top_p 0.95, top_k 64), token ids sent to `/v1/completions`. One full generation (up to 6,144 tokens; a state whose first three
generations do not end in a call is dropped); its thought is kept and k = 8 calls are drawn after it (the first generation's call and 7 more in one request). **Round 2:** for every state
that has a `not_found` or `ambiguous` call, the first such call is added to the history with its thought and the harness's own error text, and that state is sampled the same way.

**Labels** (agent/vendor/scorer_port.py, the scorer's four-tier matcher, applied to the window):
- chosen: the call applies at the exact tier, changes the window, and its changed lines are within 3 lines of the gold edit's (`localized`); same file as the gold edit;
- rejected: `not_found`, `ambiguous`, `unchanged` (old equals new), `empty_old`, or `malformed` (Python-style keyword arguments, a mandatory argument missing, a call never closed);
- no preference: applied at another tier, applied elsewhere, `unparsed`, `wrong_tool`, `wrong_file`, `multi_call`, `no_call`.
`unparsed` is excluded by default because the template's reader refuses calls the server's parser runs; `--unparsed-as-rejected` exists and is a labelled variant, not the main result.
Check against the control runs: `malformed` is 30 of 30 rejected by the framework; no call classed readable was rejected.

**Pairs.** At most 2 per state: distinct chosen calls, rejected calls taken round-robin over their statuses, no call over 1,024 tokens, no duplicate text. No length matching (the mean
call length of each side is reported). Both rounds are pooled. 1,600 pairs (random, seed 0) train every arm; 5% of commits (hash of `source`) are held out for pair accuracy.

**Arms** (same pairs, LoRA rank 32 / alpha 64 on the language model's attention and MLP projections, 1 epoch, 8 pairs per step, cosine to 10%, 5% warm-up, seed 0, loss on the call tokens only):
`dpo` (beta 0.1, lr 5e-5), `dpo_nll` (the same + 1.0 x the chosen call's mean token cross entropy), `rft` (cross entropy on the chosen calls, lr 1e-4). Order if units run short: dpo, rft, dpo_nll.
The target of every call is its text plus, for a call that closes, the stop token the model ends on (`<|tool_response>`, eos id 50, as in the SFT data); the shared thought is part of the prompt and
carries no loss. The DPO reference is the same network with the adapter switched off, computed on the fly.
Checkpoints every 50 steps (last 3 kept). One learning-rate correction is allowed per arm: if the mean reward accuracy over the last 25 steps is below 0.60, rerun once at twice the lr,
and only if at least 15 units of the cap remain. The reported checkpoint is the last; if it fails a gate, the one at the half of the steps is tried once.

**Offline evaluation** (`train/edit_pairs.py eval`): 300 prefixes of `validation.jsonl` (seed 0; held-out commits of the same repositories, 101 of its 137 files also occur in training),
2 independent generations each, shipped settings. Per model: `good_%` (a chosen-class call, per generation), `edit_call_%`, `applied_any_tier_%`, `not_found_%`, `malformed_%`,
`wrong_tool_%`, `reproduces_%`. Paired bootstrap over prefixes (2,000 resamples) for the difference to the base.
- **Primary outcome:** `good_%` rises by at least 10 points against the base with the 95% interval above 0.
- **Guards** (an arm that fails one is called unusable whatever its primary result): `reproduces_%` not more than 2 points below the base's; `edit_call_%` within 10 points of the base's
  in both directions (a rise is the edit-eagerness of edit_v1, a fall is avoiding edits); `malformed_%` not above the base's.

**Gates** (train/s4_gates.py, thresholds as in `docs/PLAN_E4B_SFT.md` and its note of 2026-11-07, unchanged): G3 step types, G4 repeat and re-read after a failed edit, G4b submit and
recover where no applied edit is visible, G5 format. Check files rebuilt with the current `build_replay_data.py` from the control runs. Only PASS passes; INSUFFICIENT does not. Report
only: the share of well-formed calls when each of the 55 rejected prefixes is resampled 8 times (the format probe).

**Agent level** (only for an arm with the primary outcome and all four gates PASS): dev66 with 3 seeds, next to a third control seed in the same session (the corrected-harness
controls are 7 and 7 of 66). An improvement is a mean at least 3 tasks above the control mean with the empty-patch share not higher. Anything less is reported as no evidence of a
gain. dev66 is a fitted pool: it cannot confirm, only fail to refute. Any run on `rich` or the mined unseen repositories is a look and goes in `docs/run_ledger.csv`.

**Budget.** At most 60 Colab units for this study, agent runs outside the cap; the 40-unit reserve is not touched. Estimates, not measurements (about 10 units an hour): notebook 09 is setup 2.5,
sampling 5-10, speed test 1, training 25-32 (dpo and dpo_nll about 60-80 minutes each, rft about 30); notebook 10 is setup 2.5 and evaluation with gates 5-7: 42-56 in all. The learning-rate rerun
needs 15 units of the cap, so it is possible only if the first pass comes in under 45. Nothing here is pushed, trained or run on a GPU without the owner's go.

## 4. What each outcome will be called

| result | what the paper says |
|---|---|
| primary outcome met, guards and gates pass | on-policy pairs raised the edit success rate without the step-choice collapse in this setting; agent-level effect as measured |
| primary met, a guard or gate fails | the offline gain did not survive; which gate caught it, and that the on-policy objective did not by itself prevent the edit_v1 / sft_v2 pattern |
| primary not met | pair training did not move the edit success rate beyond noise at this size; training curves and pair statistics reported |
| arms differ (dpo against rft against dpo_nll) | the comparison is the contribution: what the rejected calls add over the chosen ones |

Author's priors, recorded now: primary outcome met by at least one arm 65%; at least one arm passes all four gates 40%; dev66 at least 3 tasks above the control 15%. They do not change the
odds of the 0.20 target.

## 5. Limits stated in advance

Labels come from a 40-line window, so a call counted as applying can still fail on the whole file (uniqueness), and the label does not say the edit is the right fix; `localized` uses the
gold edit and a valid fix elsewhere is not credited. States are post-read prefixes without earlier thoughts. One seed per arm. Validation shares files with training. The tune
repositories are permissive (to be verified before any release of pair data). A pair set is released as a resource only after that check.
The calls are sampled from the W4A16 checkpoint the scorer serves, and the adapters are trained on its unquantized QAT sibling in bf16 (as `edit_v1` and S3 were): the DPO reference is the bf16
network, not the one that sampled. The server's vLLM is 0.32.0, the scorer's is 0.19.1 (patched); whether 0.19.1 loads these adapters is checked only by the competition's own submission test, not
here. The `unparsed` calls (the template's reader refuses them, the server's parser sometimes runs them) are excluded from the main result; a variant that uses them as rejected calls is a
labelled extra, not a replacement.

## Deviations

(none yet)
