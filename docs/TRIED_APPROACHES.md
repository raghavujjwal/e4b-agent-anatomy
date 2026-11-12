# Tried approaches

Everything we have tried on the E4B (small model) side of the Kaggle Gemma 4 Developer Agent competition, what came out, and what it tells us. Details and raw numbers are in
`docs/MEASUREMENTS.md`. Last updated 2026-11-03.

How to read the numbers:
- **Dev66** = the 66 usable public tasks without `rich` (60 fastapi, 5 requests, 1 httpx), run in our own Docker-free grader on Colab. Every score from it is **`fitted`**: we read these
  tasks in full before designing arms, and local scores run about **5x higher** than the Kaggle public score (E4B: local 12 of 66 = 18% against 0.03 on Kaggle).
- One run of 66 tasks is noisy: the same arm moved by 3-5 tasks between seeds (v2rec: 12, 7, 9). A single seed only shows a large effect. We therefore fix a decision rule in each arm's README
  before the run.
- All arms are E4B (`gemma-4-e4b-it-qat-w4a16-ct`) with a 24-call, 5.5-minute budget (`ver1`: 30 calls). Kaggle's public split is 58 tasks, so one task is 0.017.

## Summary

| # | Approach | Result on dev66 (resolved of 66) | Verdict |
|---|---|---|---|
| 1 | Baseline: the teammates' 31B prompt with only the model name changed | 7, 7, 8 (11.1%) | Reference |
| 2 | `v2rec`: edit-hygiene prompt + Google's sampling (temperature 1.0, top_k 64) | 12, 7, 9 (14.1%); **Kaggle 0.03** | Inside the noise; the only arm submitted |
| 3 | `edit_v1` LoRA (edit-only fine-tuning, scale 2.0) | 1 of 65 (64 of 65 tasks used all 24 calls) | **Collapsed** |
| 4 | `lines` arm (edit_file replaced by a line-range skill) | 5, 1, 0 | Negative |
| 5 | `v3fid`: stricter "Making an edit" prompt | 7 | Failed its rule |
| 6 | `v3loc`: file-ranking localization prompt | 9 | Failed its rule |
| 7 | `v3h`: budget + exact-names prompt from the 31B bundle H | 9 | Failed its rule |
| 8 | `edit_v1` at LoRA scale 1.0 / 0.5 (step-type check, then `ad2` at 0.5) | 1.0 failed the check; 0.5 passed the check, then `ad2` collapsed (0 of 14 tasks) | **Collapsed** |
| 9 | Self-teacher probe (SDPO idea, arXiv 2601.20802), no training | A 47% (need 65%), B +0.0007 nats (need -2.0) | **FAIL**, SDPO-style training dropped |
| 10 | Forgiving edit and file ranker, offline replays on saved traces (no model) | edit_retry finds a place for 48% of 463 failed edits; ranker lists the gold file for 72% of single-file public tasks (64% on short issues) | Built into the `aid` arms |
| 11 | `aid1`: baseline + `repo_aid` skill (`find_files`, `edit_retry`) | 7 | Failed its rules (gold file opened 60%, retry applied 1 of 12) |
| 12 | `aid2`: aid1 + fixes for what aid1 showed (unescape new_string, firmer prompt) | 9 | Failed its rules (retry applied 4 of 14, short-issue gold file opened 20%) |
| 13 | `ver1`: aid2 + reproduce-edit-recheck routine instead of "finish early", 30 calls | 9 | Failed its rules (repro script written only after the first edit) |
| 14 | `rich` milestone run (unseen pool, 43 tasks): baseline, v2rec, aid2 | not yet measured; first start failed (0 tool calls in every task, cause being checked) | Pending |

The team's 0.22 on the public leaderboard comes from the 31B "H" bundle (prompt and configuration only, no training), not from any of the above. The same bytes scored 0.18, 0.17, 0.22 and
0.18, so the 0.22 is mostly a lucky draw (their log EXP-052). For the small-model leaderboard we saw a top of 0.06 on Oct 20 and a median of 0.05 with a top of 0.10 on Oct 30.

## 1. Baseline (`agent/submission_e4b`)
- **What:** the 31B campaign's flat-search prompt, tools and budgets with only `model:` changed to E4B; temperature 0.2.
- **Result:** 7, 7, 8 of 66 (22 of 198). Median 8-10 tool calls and 25-30 s per task. First run: 7 empty patches, 2 tool-budget stops.
- **What it showed:** 15 tasks with four or more failed `edit_file` calls never resolved; 8 patches left fastapi unimportable; about half the runs ended a turn without a tool call right after an edit and
  needed the harness's nudge.

## 2. `v2rec` = prompt v2 + recommended sampling (`agent/submission_e4b_v2rec`)
- **What:** a prompt that tells the model to copy `old_string` exactly, re-read after a failed edit, run one import check, and not run the full test suite; plus the checkpoint's own sampling
  (temperature 1.0, top_p 0.95, top_k 64). The two changes were never separated: `v2` (prompt only) and `rec` (sampling only) exist as arms but were not run on their own.
- **Result:** 12, 7, 9 of 66 (28 of 198, +6 over the baseline). The gain comes from about ten tasks that move by one or two runs each; the seed range (7 to 12) is wider than the gap. The six reliable
  tasks are the same in both arms. Temperature 1.0 raised calls with keys outside the tool's schema from 4-5 to 11-19 per run.
- **Kaggle:** submitted on 2026-10-31 (submission 57843763, no adapter, no skills). It scored **0.03** (2 of 58 public tasks); it was still pending 13.5 hours after submission and the finish time is not shown by the API.

## 3. `edit_v1` LoRA at scale 2.0 (`agent/submission_e4b_ad1`)
- **What:** rank 32, alpha 64 (scale 2.0), 4,000 `edit_file` training samples from tune-repo commit history and mined gold patches, empty thoughts, one epoch (about 10-11 Colab units). Training loss
  fell from 0.834 to 0.374.
- **Offline edit benchmark** (300 held-out edit prefixes): 53% of the base model's edits apply against 89% for the adapter. Conditional on actually making an edit call, 81% against 89%. The share that reproduces
  the gold edit stays at 7%: the adapter learned the form and exact copying, not which fix to make.
- **In the agent: collapse.** 64 of 65 tasks used all 24 calls on `edit_file` calls that failed; the adapter never read, searched or submitted.
- **Cause:** every training sample was an "edit now" decision, so the model learned to edit at every step. The offline benchmark cannot detect this because every prefix in it is an edit prefix.
- **Kept from it:** gate every adapter on a step-type check (item 8) before any agent run.

## 4. `lines` arm (`agent/submission_e4b_lines`)
- **What:** `edit_file` replaced by an `edit_lines` skill (replace a line range), motivated by the model rebuilding `old_string` instead of copying it.
- **Result:** 5, 1, 0 of 66 (6 of 198, 3.0%) against 22 of 198 for the baseline; 25-40 empty patches per run against 3-7.
- **Cause not established.** Candidates: the skill call itself errors, edits refused by the syntax check, or the skill not being called. The `lines` traces were not copied to the PC, so we could not tell.

## 5-7. Prompt-only arms from the trace analysis (2026-11-02, one seed each, about 25 min and 4 units per run)
Each arm is `v2rec` with one prompt change and a pass rule fixed in its README before the run.

| Arm | Change | Rule | Measured | Resolved |
|---|---|---|---|---|
| `v3fid` | Edit one short fragment (15-100 chars); read_file output is JSON so use real newlines and quotes; never resend a failed `old_string` | edit failure rate 40% or lower (was 56%) | **56.9%** (204 edits) | 7 |
| `v3loc` | Pick 2-4 distinctive words and rank files by how many each contains; open the top file | gold-file open rate on short issues 40% or more (was 21%), overall not lower | **13.3%** (2 of 15), overall 53% against 57% | 9 |
| `v3h` | Add the call/time budget and "write down the exact names and messages the issue requires" (from H) | 11 or more of 66 | 9 | 9 |

- `v3fid` changed what the model writes (25% of `old_string`s under 40 characters against 16%; identical repeats 12 against about 22 per run) but not how often it fails: short fragments still fail 60%, and
  anything with a backslash fails 98%. "Not found" is not mainly about length or copying care.
- `v3loc`: the ranking command ran without shell errors in 32 of 66 tasks, but the model chose generic words ("id", "event", "Response", "pydantic"), so large common files ranked first. It also had 8 empty patches.
- `v3h`: edit failure 47.6%, a bit lower, but not a pre-registered metric and inside the noise.
- **Conclusion:** three different prompt changes leave edit failures at 48-57%. The prompt is not the lever for the edit step.

## 8. `edit_v1` at lower LoRA scales (notebook 05, then `agent/submission_e4b_ad2`)
- **Why:** to test whether the collapse depends on the LoRA scale, we served the same weights at scale 1.0 and 0.5 by rewriting `lora_alpha` in a local copy of `adapter_config.json`.
- **Step-type check** (120 prefixes of the base model's own runs; agreement % with the tool the base model chose): scale 1.0 **failed** (read_file 9% against 73% for the base; it chose `edit_file` at 25 of 33 read
  steps). Scale 0.5 **passed** (read_file 70% against 73%, run_command 62% against 71%, submit_patch 87% against 74%).
- **Offline benchmark:** edits apply 51% (base), 86% (scale 1.0), 78% (scale 0.5), 89% (scale 2.0); none reproduces the gold edit more often than the base.
- **In the agent (`ad2` = baseline + `edit_v1` at 0.5): collapsed again.** Stopped by hand after 14 of 66 tasks (9 tool-budget stops, 4 empty patches, 0 resolved). Of 308 tool calls, 283 were `edit_file`,
  none were `read_file`, and 89% of the edits repeated the call just before them.
- **What it taught us:** the step-type check scores single steps on prefixes of the base model's own runs, where no edit has failed yet. It cannot see a loop that starts after a failed edit. A gate for any
  next adapter must include prefixes that end in a failed edit and measure how often the model repeats the failed call (base: 12% of edit calls).
- **A bug found on the way:** the first run of the step check had every request refused (`Connection refused`) and still printed no gate failures, because a base model at 0% agreement cannot be undercut.
  `train/step_check.py` now marks a run with request errors INVALID.

## 9. Self-teacher probe (SDPO idea; run 2026-11-02): FAIL
- **Idea** (arXiv 2601.20802, "Reinforcement Learning via Self-Distillation"): the same model, with the tool's error message in its context, re-scores its own attempt; tokens it now finds unlikely get a negative
  advantage. This is a training signal for exactly our failure (resending a failed edit without reacting to the error), and it is zero where the feedback adds nothing, unlike edit-only fine-tuning.
- **Limits from the paper:** gains grow with model size (SDPO beat GRPO at Qwen3-8B and 4B, little at 1.7B and 0.6B, and lost at Qwen2.5-1.5B); full online training needs fresh agent rollouts per step and
  is far too costly for us (our estimate: on the order of 100+ Colab units).
- **Probe (no training, about 11 minutes of GPU):** `train/build_sdpo_probe.py` + `train/sdpo_probe.py` + notebook 06, 382 edit turns, rule fixed before the run.
- **Result: FAIL.** With the "not found" text in context E4B does not rate the mistaken characters of its own old_string any lower than the correct ones (d < 0 in 47% of samples, the same 47% with another
  edit's error, needed 65% / 55%). A repeated failed call keeps its probability (median sum of advantages +0.0007 nats, needed -2.0). Details and the caveat on test B (the student prompt of a repeat already
  contains an earlier error) are in `docs/MEASUREMENTS.md`. Consistent with the paper's finding that the effect needs larger models. SDPO-style training on recorded turns is dropped.

## 10-13. The `aid` line and `ver1` (2026-11-02 to 11-03; `agent/submission_e4b_aid`, `_aid2`, `_ver1`)
- **Offline replays first (no GPU):** a forgiving `edit_file` would find a place for 35% (text shown to the model) to 48% (whole base-commit file) of the 463 failed edits; a BM25 ranker over file path, defined
  names and file text puts the gold file in its list of up to 10 files for 72% of single-file public tasks. Pre-registered rule for the ranker (recall@5 at least 75% overall and 50% on short issues) came out
  borderline (63.3% / 56.4%). Value of finding the file: single-file runs resolve 29.4% when the gold file was seen, 3.4% when not.
- **`aid1`** (skill with both tools, `edit_file` kept): 7 of 66. The model called `find_files` in 83% of runs but sent a short query (median 70 characters, asked for up to 800), opened the gold file in 60%
  (short issues 27%), and followed a failed edit with `edit_retry` in 27% of cases; 11 of 12 retry calls failed (7 refused by the parse guard because the model's own new_string held literal backslash sequences).
  A first version of `edit_retry` was also too slow (82 s at 60 lines, 616 s at 140; the harness kills a skill at 90 s) and was fixed.
- **`aid2`** (unescape rescue, firmer prompt): 9 of 66. First read went to the top-listed file more often (29 of 51 runs), retry applied 4 of 14, but the query stayed short (median 76) and the retry was still
  followed in only 31% of failures.
- **`ver1`** (reproduce, edit, recheck; 30 calls): 9 of 66. The model wrote and ran `/tmp/repro.py` in 75 runs but never before its first edit. Median calls 10.5, no budget stops, 5 empty patches.
- **Reading:** the tools work as built, E4B uses them too little and too late, and the count did not move beyond the 7-12 band of every other arm.

## Analyses behind the arms (no model training)
- **Where runs lose tasks** (396 baseline and v2rec runs): the gold file is never opened in 42% of single-file runs (21% for issues under 300 characters, against 74% for longer ones); 56% of 1,077 `edit_file`
  calls fail; multi-file gold tasks are solved in 0.8% of runs. Tasks with no failed edit resolve 38% of the time, with 1-2 failed edits 26%, with 3 or more 7%. E4B makes its first edit attempt early
  (median call 5; 365 of 396 runs by call 10), which is why H's "edit by call 12" rule was left out of `v3h`.
- **Edit failure causes** (454 "not found"): near miss 126, far from anything in the file 113, escape sequences copied 78, partial 71, file not read 42, old_string actually in the file 24. The failure rate
  is 96% when `old_string` contains a backslash and 83% with triple quotes.
- **Spec check:** 7 of 66 dev tasks need a name the issue never mentions; the 31B solves none of them either.
- **Headroom across 12 dev66 runs** (base1-3, v2rec1-3, v3fid1, v3loc1, v3h1, aid1, aid2_1, ver1_1): mean 8.3 resolved per run; 21 of 66 tasks solved by at least one run, 3 by all 12, 7 by at least 6, 4 only once.
  A perfect chooser among 3 attempts would score about 14.
- **The model's own checks do not predict success:** over 457 runs with an applied edit, resolved 18% when the last python/pytest command after the edit exited 0, 15% when it did not, 13% when no check ran.
  So retrying or choosing between attempts on that signal would not help.
- **Task pool:** dev66 is not selected by failure type; it is the usable public tasks minus `rich` (91% fastapi). The 43 usable `rich` tasks and the mined tune/unseen repositories are held out
  (`docs/EVAL_PROTOCOL.md`); no arm has been measured on them yet.

## Infrastructure built along the way
- A Docker-free patch grader that runs on Colab (109 of the 129 public tasks usable) and the agent harness vendored from `main` (`agent/`, `grader/`).
- Notebooks 01-06: load E4B, grader on Colab, serve and run arms, train a LoRA, serve an adapter and benchmark it, self-teacher probe.
- Replay data and a step-type check (`train/build_replay_data.py`, `train/step_check.py`), mined held-out tasks (`mine/`), and the trace analyses (`mine/trace_analysis.py`, `mine/localization_funnel.py`).

## Not tried yet
- Repair training / rejection-sampling fine-tuning of the adapter on all step types from the model's own successful runs (about 40 units to generate fresh runs on held-out tasks, train and check). Risks: collapse as with
  `edit_v1`, and the unverified question below.
- Retry after an empty patch / use of idle time budget (the teammates have an experiment on `main`; E4B runs are short, median 6-10 calls and 22-33 s of a 330 s cap).
- The scorer's built-in code-graph tools (`search_similar_code`, `get_code_neighbors`): not available in our local runs, so untested.
- Unverified: whether the scorer's vLLM (0.19.1) loads a Gemma 4 LoRA at start-up. Everything above was served on vLLM 0.32.0 in Colab.
