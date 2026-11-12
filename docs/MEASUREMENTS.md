# Measurements

Facts measured on Colab. Everything here was observed; nothing is a scorer number.

## Colab CPU runtime (notebook 02, 2026-10-28)
Root, 2 CPUs, 12 GB RAM, 219 GB disk, Python 3.13.18, glibc 2.39. `unshare -n` works but cannot bring `lo` up; the working
isolation is `setpriv` (drop to an unprivileged user) then `unshare --user --map-current-user --keep-caps --net` plus `lo` up.
Use at most 2 grader workers. Probe JSON: Drive `gemma4-data/results/probe.json`.

## Colab GPU runtime (notebook 03, 2026-10-29)
GPU: NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB, driver 580.82.07, CUDA 13.0.

Serving the competition checkpoint `gemma-4-e4b-it-qat-w4a16-ct` (11.5 GB, compressed-tensors):
- vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, in a Python 3.12 venv (`uv pip install vllm`).
- Flags: `--max-model-len 32768 --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser gemma4
  --reasoning-parser gemma4`, one GPU (the scorer uses tensor parallel 4 on L4s).
- **First start failed** at the final warm-up: `FileNotFoundError: 'ninja'`. vLLM wanted to compile a FlashInfer sampling
  kernel and Colab has no `nvcc`/`ninja`. Fix: run `vllm download-kernels` once (installs prebuilt kernels, including
  `sm120f` for this GPU), then start again. Second start was ready in 80 s.
- E4B has head dims 256 (sliding) and 512 (full attention), so vLLM forces the Triton attention backend (slower, same output).
- KV cache 71.3 GiB = 2,038,500 tokens; 62x concurrency at 32,768 tokens.
- Smoke test: plain chat correct (`17*23 -> 391`, 146 reasoning tokens, separated by the reasoning parser); a request with one
  tool returned a parsed `tool_calls` entry with finish reason `tool_calls`. The checkpoint's own `chat_template.jinja` works;
  no custom template is needed.

## First agent run (baseline submission `agent/submission_e4b`, one task, one sample)
`httpx_3672`: not resolved. 11 tool calls, 13 turns, 9,959 completion tokens, 61.9 s agent time (73.7 s with grading), patch
564 bytes in `src/httpx/_parsers.py`. The tests needed a new `HTTPParser.reset` method; the patch did not add it
(5 tests failed with `AttributeError ... no attribute 'reset'`).
Behaviour worth tracking across tasks: after its first successful edit the model ended a turn with empty text (the harness
nudged it), then made 5 consecutive `edit_file` calls that failed with "old_string not found" before submitting.
One task is an anecdote; see the pilot and baseline runs for rates.

## Pilot 1 (baseline submission, 12 dev tasks, one sample each, scores are `fitted`)
Drive: `gemma4-data/results/e4b/pilot1.jsonl` and `pilot1_trace.jsonl`. Whole run about 8.5 minutes.
Tasks: 1 httpx, 4 requests, 7 fastapi, drawn at random from the usable dev tasks (rich held out).
Result: 4 of 12 resolved: requests 4/4 (6644, 7205, 7309, 7315), fastapi 0/7, httpx 0/1.
The sample over-represents requests (4 of the 5 usable requests tasks; the dev split is 60 fastapi, 5 requests, 1 httpx),
so 4/12 overstates the dev-split rate. Do not quote it.
Per task: 5-14 tool calls (limit 24), 9-65 s agent time (budget 330 s), 1.5k-10k completion tokens. No run executed a test.
Failure patterns seen (counts are over 12 runs, anecdotal until the full baseline):
- 7 runs ended a turn with empty or short text after an edit and needed the harness nudge before submitting.
- fastapi_15280: `edit_file` with `allow_multiple` replaced 131 occurrences, 93,637-byte patch, SyntaxError.
- httpx_3672 (twice): 5 consecutive `edit_file` calls failed with "old_string not found".
- fastapi_14419: changed a method into a `cached_property`, callers still called it (`'set' object is not callable`).
- fastapi_9555: patch left `cm` unassigned on one branch (`UnboundLocalError`).
- fastapi_15661: tests import a module that does not exist yet; the model edited `scripts/lint.sh` instead of creating it.

## Baseline 1 (`base1`): E4B, submission `agent/submission_e4b` unchanged, all 66 dev tasks, one sample each (`fitted`)
Drive: `gemma4-data/results/e4b/base1.jsonl` and `base1_trace.jsonl`. Wall time about 40 minutes (2026-10-29 23:49 to 2026-10-30 00:29 UTC).
**7 of 66 resolved (10.6%)**; a 95% interval for one run of 66 is roughly 5% to 21%. By repo: fastapi 6/60, requests 1/5, httpx 0/1.
Outcome classes: 7 resolved, 50 non-empty patch that failed the tests, 7 empty patch, 2 hit the 24-call cap, 0 hit the time cap.
Resolved tasks were all small edits (patch 404 to 783 bytes, 2 to 16 tool calls).
**Run-to-run noise is large.** The pilot and base1 ran the same configuration. requests_6644, 7205 and 7315 resolved in the pilot and did not in
base1 (requests went from 4/4 to 1/5); the fastapi pilot tasks were 0/7 both times. One run per task cannot rank two versions that differ by a few tasks.
Failure patterns (counts over the 66 runs, read from the traces; they describe behaviour, not causes):
- 15 tasks had 4 or more failed `edit_file` calls ("old_string not found" or "matched N places"); none of those 15 resolved (7 of the other 51 did).
  Five of the 7 empty patches are failed-edit loops (requests_7427, 7205, fastapi_14458, 14459, 13537).
- 8 patches left `fastapi` unimportable (SyntaxError or NameError when the tests import it): 15588, 15589, 15030, 15280, 14986, 13713, 15800, 13786.
  Three of those are 69-144 KB patches to `applications.py` made by multi-place replacement or a shell rewrite.
- About 31 of 66 runs ended a turn with no tool call right after an edit and needed the harness nudge to submit.
- No sign of any test or import check in the traces; commands are listings and greps (and one shell rewrite).
- 2 runs edited `pyproject.toml` (protected) despite the prompt (fastapi_14492 became an empty patch, 14964 lost that file).
- Tool calls: median about 9 of 24; agent time mostly 10 to 65 s of 330 s.
Evaluation protocol note: all 66 dev traces were read in full before any train/validation split was fixed, so the dev split can no longer serve as an unseen check.
Remaining unseen sets: `rich` (43 usable tasks, never run) and any newly mined tasks.

## Spec check on the dev tasks (2026-10-30, static, `mine/spec_check.py`; no agent output used to flag)
7 of the 66 dev tasks need a new name that the issue never mentions (fastapi_14099, 14609, 14953, 15030, 15280, 15661, 15800).
In the teammates' pooled 31B runs (about 28 runs per task, `fitted`) these 7 are solved 0% of the time, against a mean of 35% on
the other 59; E4B base1 solved none of them. Reading base1 on the 59 tasks that can be solved from the issue: **7 of 59 (11.9%)**
instead of 7 of 66. Both numbers stay in the record; the 66-task figure is the one comparable with earlier rows.
Mined tasks: 7 of the 122 usable are flagged (5.7%), against 7 of 66 dev tasks (10.6%) and 15 of 129 public tasks (11.6%).
The check is a heuristic (see `mine/README.md`), so these counts are approximate.

## Combined arm vs baseline, 3 seeds each (2026-10-30/31, 66 dev tasks, `fitted`, 2 workers)
Colab RTX PRO 6000, notebook 03, order v2rec1, base2, v2rec2, base3, v2rec3 (base1 is the earlier 1-worker run). Whole notebook
(install, kernels, checkpoint copy, five runs): **22 compute units**. Wall time per run: v2rec 27.6 / 24.6 / 24.9 min, base 21.2 / 17.2 min
(base1, one worker: about 40 min). `generation_config.json` print-out not yet recorded.

| Arm | Resolved per seed | Total | Rate | Empty patch per seed | Median calls | Median agent s |
|---|---|---|---|---|---|---|
| baseline `submission_e4b` | 7, 7, 8 | 22/198 | 11.1% | 7, 6, 3 | 8-10 | 25-30 |
| `submission_e4b_v2rec` (prompt v2 + temp 1.0, top_p .95, top_k 64) | 12, 7, 9 | 28/198 | 14.1% | 3, 8, 9 | 12 | 33-41 |
On the 59 unflagged tasks (the 7 flagged are never solved by either arm): 22/177 = 12.4% vs 28/177 = 15.8%.

Per task (solved in k of 3 runs; baseline / v2rec):
- Solved by both in 3/3: fastapi_14786, 14301, 14303. Near-stable: fastapi_14463 (2 / 3), 9425 (2 / 2), 14873 (1 / 1), requests_7309 (2 / 2).
- v2rec better: requests_6644 (1 / 3), fastapi_14492 (0 / 2), 14616 (0 / 2), requests_7315 (1 / 2), fastapi_14372 (0 / 1), 14458 (0 / 1).
- Baseline better: requests_7205 (2 / 0), requests_7427 (1 / 0), fastapi_13537 (1 / 0).
- Tasks ever solved: baseline 12, v2rec 13. Four tasks were solved only by v2rec (14492, 14372, 14458, 14616); three only by the baseline (7205, 7427, 13537).
Reading: +6 solves in 198 runs comes from about ten tasks that move by one or two runs each; the seed range of v2rec (7 to 12) is wider than the gap,
and v2rec1 alone (12) accounts for most of it. This does not show the combined arm helps, and does not show it hurts. The six reliable
tasks are unchanged. Tool-budget stops fell (baseline 3, 1; v2rec 0, 1, 0) but empty patches did not (v2rec 8 and 9 in seeds 2 and 3).
Next: run `submission_e4b_v2` and `submission_e4b_rec` for 3 seeds each, which completes a 2x2 design (prompt x sampling, 6 runs per side).

## Why edits fail (traces of base2, base3, v2rec1-3; `fitted`; raw files copied to D:\data\e4b_runs, not in the repo)
Counted from the traces of 66 tasks per run. `edit_file` is the weak step in every arm and prompt v2 did not change it.
| | base2 | base3 | v2rec1 | v2rec2 | v2rec3 |
|---|---|---|---|---|---|
| edit_file ok / failed | 80 / 117 | 80 / 89 | 97 / 116 | 103 / 127 | 88 / 131 |
| tasks with 4+ failed edits | 15 | 12 | 12 | 16 | 19 |
| ...of which resolved | 0 | 0 | 0 | 0 | 0 |
| tasks with no successful edit | 12 | 6 | 3 | 9 | 8 |
| tool calls with keys outside the tool's schema | 5 | 4 | 19 | 13 | 11 |
| tasks with `python -c "import ..."` after the last edit | 0 | 0 | 60 | 54 | 53 |
| tasks that ran pytest | 0 | 0 | 1 | 3 | 0 |
- 53-60% of all `edit_file` calls fail in every arm. No task with 4 or more failed edits was ever resolved (5 runs).
- Every empty patch except one (v2rec3 `fastapi_13713`) is a task where no edit succeeded: the model never changed anything the tool would accept.
- Prompt v2 is followed (import check after the last edit in 53-60 tasks, against none for the baseline) and no run tests behaviour (pytest in 0-3 tasks).
  The import check did not buy resolutions; the failed-edit counts did not fall.
- Temperature 1.0 / top_k 64 raised calls with keys outside the schema from 4-5 to 11-19 per run.
- The tool is already tolerant (exact, then whitespace-insensitive line match, then token regex; `agent/vendor/tools.py`), so a failure means
  the text is not in the file. For failed "old_string not found" edits, comparing every stripped line of `old_string` with everything the model had
  seen in that task (file reads and command output): all lines seen but not contiguous: base 6 and 4, v2rec 30, 19, 13; no line seen at all: base 31 and 16,
  v2rec 24, 24, 42; some lines seen: base 42 and 42, v2rec 19, 34, 30. Samples show multi-line strings written with literal `\n`, lines skipped
  inside a signature, and text from memory of the library. The model rebuilds the text instead of copying it.
- `read_file` accepts `start_line`/`end_line` and truncates large files (applications.py, routing.py); the model often edits text from beyond the part it read.
Consequence for design: an exact multi-line `old_string` is the wrong interface for this model. Candidate lever (untested): a skill script that replaces a line
range (`--file --start --end --text`), run through `run_skill_script` (it runs `python3` in `/workspace`; flags are rendered as `--key value`).

## The checkpoint's generation_config.json (read 2026-10-30 from the file on Drive)
`temperature 1.0, top_k 64, top_p 0.95, do_sample true`, eos ids 1, 106, 50 (transformers 5.8.0.dev0). By default vLLM uses the model's
generation config for any sampling field a request leaves out (its start-up log says "Default sampling parameters have been
overridden by the model's Hugging Face generation config"; not yet confirmed in our server log). The baseline sends temperature 0.2
and top_p 0.95 but no top_k, so the baseline probably already ran with top_k 64. If so, the `rec` and `v2rec` arms differ from the
baseline in temperature only (0.2 vs 1.0), not in top_k. The scorer's vLLM may or may not apply the same default; unknown.
`base1.jsonl` (7/66, 50 non-empty failures, 7 empty, 2 tool-budget) was re-read from the copy downloaded on 2026-10-30 and matches the table above.

## Edit-step LoRA `edit_v1`: training and offline edit benchmark (2026-11-01)

Training (notebook 04, `train/train_lora.py`): unquantized QAT E4B in bf16, LoRA rank 32 / alpha 64 on q,k,v,o,gate,up,down of the language model,
4,000 `edit_file` samples from `train/make_edit_data.py` (6,227 from tune-repo commit history, the rest gold patches of mined tune tasks; thought
channel empty), 1 epoch = 250 optimizer steps of 16, lr 1e-4. Validation loss on 100 held-out samples 0.834 -> 0.374. Adapter 266 MiB, 516 tensors,
all keys match the pattern the scorer's vLLM loaded for the public canary. Whole notebook (setup, 16 GB copy, smoke test, training): about 10-11 Colab units.

Loads and serves: the competition checkpoint (W4A16) started in vLLM with `--enable-lora --max-loras 1 --max-lora-rank 32 --lora-modules edit_v1=...`
accepts the adapter and answers requests with it (notebook 05; Colab's vLLM, not the scorer's 0.19.1, one GPU, not TP=4).

Offline benchmark (`train/edit_bench.py`, 300 held-out validation prefixes of tune repos, the model sees the right file window, thinking on, temperature 0.2,
one sample each; the edit is applied to the window with the harness's `apply_replacement`):

| | base (W4A16) | + edit_v1 |
|---|---|---|
| calls edit_file with valid keys | 198 + 10 bad keys | 299 + 1 bad keys |
| called another tool instead (`wrong_tool`) | 92 | 0 |
| applies / all 300 | 53.3% | 88.7% |
| applies / its edit_file calls | 160/198 = 81% | 266/299 = 89% |
| applies at the strict (exact copy) tier / its edit_file calls | 155/198 = 78% | 257/299 = 86% |
| reproduces the gold edit | 7.0% | 7.3% (paired: 12 vs 11) |
| mean completion tokens / reasoning chars | 1,494 / 5,073 | 158 / 0 |
| wall time for 300 requests | 239 s | 95 s |

Reading it:
- The headline 53% vs 89% mostly measures that the base model often chose a different tool at this step (31%) while the adapter always edits; in a real run
  a read or search at that point can be right, so this is not a like-for-like failure rate. Conditional on making an edit call, 81% vs 89%.
- These windows are the easy case (the hunk is inside what the model was shown); the agent runs saw 53-60% failed edits on real files, so absolute rates do not transfer.
- `reproduces` is equal: the adapter learned the form and exact copying, not which fix to make (expected; the data are gold edits).
- The adapter does not think at this step (0 reasoning chars, 9x fewer tokens, 2.5x faster). That saves grader time (about 25 tok/s on 4 x L4) but
  could also lower fix quality or make the model edit before it has looked enough; only the full agent can show it.
- Not measured: whether it changes behaviour at search or test steps; resolve rate (next: `agent/submission_e4b_ad1` x3 seeds on dev66 via notebook 03).

### `ad1` (baseline + edit_v1) in the full agent: collapse (2026-11-01, run stopped after 65 of 66 tasks of `ad1_1`)
64 of 65 tasks ended `tool_budget` (24 calls, 24 turns), 1 resolved (fastapi_14786, also with 24 calls); baseline `base1-3` resolve 7, 7, 8 of 66.
The traces of the first three tasks show 24 consecutive `edit_file` calls each, every one failing (`old_string` not found / path error), the first call being
an edit with no read or search before it. The adapter always emits `edit_file`, whatever the conversation state: it never reads, searches, submits or stops,
and does not react to 20 consecutive errors. Cause: every training sample was one decision point whose target was an `edit_file` call (data from
`train/make_edit_data.py`, thought channel empty); no sample had another action as the correct next step. The offline benchmark showed the first sign
(`wrong_tool` 92 -> 0) but cannot detect this, because every benchmark prefix is an "edit now" prefix.
Lesson: an adapter trained on one step type must be gated on a step-type preservation check (does it choose the same tool as the base model at real
trace prefixes where the base reads, searches or submits?) before any agent run. Do not read this as a result about LoRA in general.
Data available for a repair: all 396 traces of base1-3 and v2rec1-3 carry exact prompt and completion token ids per turn (`turn_tokens`) and the thoughts:
about 1,177 run_command, 1,077 edit_file (about 40-47% applied), 1,014 read_file, 377 submit_patch turns.

## `lines` arm (edit_file replaced by the edit_lines skill), 3 seeds (2026-11-01, 66 dev tasks, `fitted`, 2 workers)
| run | resolved | empty patch | tool_budget | calls (median) | agent s (median) |
|---|---|---|---|---|---|
| lines1 | 5 | 25 | 3 | 15 | 41 |
| lines2 | 1 | 28 | 4 | 14 | 38 |
| lines3 | 0 | 40 | 1 | 14 | 41 |
Total 6 of 198 (3.0%) against 22 of 198 (11.1%) for `base1-3` (empty patches there: 7, 6, 3). Negative result, and it does not answer whether better
edits would help: the new edit method itself fails (the arm's README predicted "if the model cannot use the skill the arm will show it as empty patches").
Cause not yet established (traces of lines1-3 not yet on this PC). Candidates: a `run_skill_script` call that errors, edits refused by the syntax or `expect` check, or the skill not being called.

## Where the baseline runs lose tasks (2026-11-01; traces of base1-3 and v2rec1-3 = 396 task runs; `fitted`; `mine/trace_analysis.py`, `mine/localization_funnel.py`)
Gold files come from the competition's `tasks.jsonl` (downloaded to D:\data\kaggle_comp, 129 tasks), non-test files of the gold patch.
**Single-file gold tasks (270 task runs):** gold file never opened 113 (42%) | opened but no edit on it 15 (6%) | edit attempted, none applied 18 (7%) |
edit applied, tests still fail 75 (28%) | resolved 49 (18%). Multi-file gold tasks (126 runs): 1 resolved (0.8%).
- Localization is the largest single loss. The gold file is opened in 21% of runs for short issues (< 300 characters, 15 of 45 single-file tasks) against 74% for
  longer ones; 86% when the issue has a code block against 46%. 10 of the 45 tasks never had their gold file opened in any of the 6 runs. 4 of the 50
  resolved runs never opened a gold file (alternative fixes exist but are rare). Opening the file and trying an edit there is reliable (95%).
- Edit failures: 1,077 edit_file calls, 56.2% failed: old_string not found 454, ambiguous 85, empty old_string 12, mangled argument keys 54 (5%, keys
  contain code text). Failure rate by what old_string contains: backslash 96% (218 calls)
  against 46% without; triple quote 83% (90 calls) against 54%; longer than 400 characters 75% against 53%; multi-line 55% against 58% (no effect).
  Why "not found" (454 calls, compared with the file text the model was shown): near miss, similarity >= 0.9: 126; far from anything in the file: 113;
  escape sequences copied: 78; partial 0.7-0.9: 71; file not read in the conversation: 42; old_string is in the file as shown: 24.
  135 calls repeat the previous call exactly and 132 of those fail again.
- Effect on the outcome, single-file tasks where the gold file was opened (153 runs): 0 failed edits in the run: 38% resolved (96 runs), 1-2 failed: 26% (27),
  3+ failed: 7% (30). (Unconditioned, tasks with 4+ failed edits resolve 0 of 76, but that mixes in multi-file tasks.)
- Empty patches (36 runs, baseline and v2rec): all 36 ended with submit_patch; 32 had edit calls and none applied, 3 made no edit attempt, 1 had an applied edit.
  So an empty patch is the model submitting after its edits failed.
- Task killers on the scorer: no call to an undeclared tool in any of the 396 runs.

## Prompt-only arms v3fid, v3loc, v3h, one seed each (2026-11-02, 66 dev tasks, `fitted`, 2 workers; about 25 min and 4 Colab units per run)
Each arm is `v2rec` with one prompt change; the decision rules are in the arm READMEs and were fixed before the runs. Single-file tasks only for the open rates (45 tasks, 15 short).
| run | resolved | empty patch | tool_budget | calls (med) | agent s (med) | edit_file calls | edit failure % | gold file opened, all / short issues |
|---|---|---|---|---|---|---|---|---|
| base1-3 / v2rec1-3 (pooled) | 7,7,8 / 12,7,9 | | | | | 1,077 (6 runs) | 56.2 | 56.7% / 21.1% |
| v3fid1 (edit fidelity) | 7 | 3 | 3 | 11 | 33 | 204 | 56.9 | 48.9% / 13.3% |
| v3loc1 (localization) | 9 | 8 | 0 | 10 | 33 | 164 | 54.9 | 53.3% / 13.3% |
| v3h1 (budget + exact names, from the 31B bundle H) | 9 | 4 | 1 | 11 | 40 | 170 | 47.6 | 60.0% / 13.3% |
Verdicts against the pre-registered rules: **v3fid fails** (edit failure rate had to fall to 40% or lower; 56.9%), **v3loc fails** (short-issue open rate had to reach 40%;
2 of 15 = 13.3%, overall 53.3% against 56.7%), **v3h fails** (needed 11 or more of 66; 9). No arm gets more seeds. All three resolved counts are inside the noise of base and v2rec.
- v3fid changed what the model writes but not how often it fails: 25% of old_strings are shorter than 40 characters (baseline 16%) and identical repeats fell to 12
  in the run (about 22 per run in the baseline), but short old_strings still fail 60%, and old_strings with a backslash fail 98% (21% of calls, as in the baseline). So
  "not found" is not mainly about length or care in copying.
- v3loc: the ranking command ran without shell errors in 32 of 66 tasks (no quoting problem). The model picked generic words ("id", "event", "Response", "pydantic"),
  so large common files (routing.py, utils.py) came first. It also had 8 empty patches.
- v3h: edit failure 47.6% is lower, but it is not a pre-registered metric and a single run cannot separate it from noise.
Reading: three different prompt changes leave edit failures at 48-57%, so the prompt is not the lever for the edit step.

## `edit_v1` at LoRA scales 1.0 and 0.5: step-type check and edit benchmark (notebook 05, 2026-11-02, vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0)
`edit_v1` was trained at rank 32, alpha 64 (scale 2.0) and collapsed in the agent (`ad1`). Here the same weights are served at scale 1.0 and 0.5 by rewriting `lora_alpha` (32, 16) in a local copy of
`adapter_config.json`; both adapters load next to the W4A16 checkpoint (`--enable-lora --max-loras 2 --max-lora-rank 32`, ready after 160 s). A first 6b run the same day had every request
refused (Connection refused, 360 of 360) and the gate still printed no failures: with the base at 0% agreement nothing can fall 10 points below it. `step_check.py` now marks a run with request errors
above 2% INVALID (43743cb); the numbers below are from the second session where no request failed.

Step-type check: 120 prefixes of the base model's own runs (temperature 0.2), tool chosen by the generated call against the tool recorded. Agreement % by recorded tool:
| model | edit_file (n 32) | read_file (33) | run_command (24) | submit_patch (31) | no call | non-edit steps that became edit_file | gate |
|---|---|---|---|---|---|---|---|
| base | 84.4 | 72.7 | 70.8 | 74.2 | 1.7% | 4 of 88 (4.5%) | |
| edit_v1, scale 1.0 | 96.9 | **9.1** | **16.7** | 80.6 | 0% | 47 of 88 (53%) | **FAIL** |
| edit_v1, scale 0.5 | 96.9 | 69.7 | 62.5 | 87.1 | 0.8% | 13 of 88 (14.8%) | **PASS** |
Scale 1.0 reproduces the collapse (it chooses edit_file at 25 of 33 read_file steps). Scale 0.5 passes the gate, but with 24-33 prefixes per tool the binomial spread is about 8 points, so run_command
(-8.3) is within noise and also close to the 10-point line, and the adapter edits earlier than the base (14.8% against 4.5%).

Edit benchmark (200 held-out edit prefixes, `train/edit_bench.py`, thinking on, temperature 0.2). Every prefix is an edit step in the recorded runs, so the base choosing another tool counts against it:
| model | edit_file call valid | edit applies | exact-copy tier | reproduces the gold edit | old_string equals gold | mean completion tokens | reasoning chars |
|---|---|---|---|---|---|---|---|
| base | 65.0% (58 steps called another tool) | 51.0% | 49.5% | 7.0% | 10.0% | 1,474 | 4,920 |
| scale 1.0 | 98.5% | 86.0% | 82.5% | 7.0% | 10.0% | 205 | 0 |
| scale 0.5 | 99.5% | 78.0% | 76.5% | 7.5% | 6.5% | 294 | 0 |
(Scale 2.0, earlier run: 89% applies.) Lowering the scale costs little offline: 89% -> 86% -> 78%. The adapter fixes copy fidelity ("not found" failures), not what to change: the share that reproduces the
gold edit does not move. It also drops the thinking in the benchmark (0 reasoning characters, as in its training data). Not tested: the scorer's vLLM is 0.19.1 and whether it loads a Gemma 4 LoRA at start-up.
Next: agent run `ad2` (agent/submission_e4b_ad2: baseline + edit_v1 at scale 0.5), decision rule and collapse guard in its README.

## `ad2` (baseline + `edit_v1` at LoRA scale 0.5): collapsed, stopped after 14 of 66 tasks (2026-11-02)
The run was interrupted by hand after 14 tasks: 9 `tool_budget` (24 calls), 4 empty patches, 1 non-empty patch that failed to import, 0 resolved. That already breaks the collapse guard in the arm README
(more than 8 budget stops). Traces of the 14 tasks (D:\data\e4b_runs\ad2_1*): 308 tool calls, of which **283 are edit_file, 20 run_command, 5 submit_patch and none read_file**. 252 of the 283 edits
(89%) are identical to the edit call just before, and 280 failed (one applied). The usual shape is one `run_command`, then the same failing `edit_file` until the budget ends; the first error seen
was a guessed path (`fastapi/fastapi/responses.py does not exist or is empty`). Hardly any assistant message carries a thought (1 of 309). So `edit_v1` at scale 0.5 still loops, it never reads the
file it edits, and it does not react to the error message. Same signature as `ad1` at scale 2.0, a little later in the run.
Lesson for the gate: the step-type check passed scale 0.5 (read_file 70% vs 73% for the base), but it scores single steps on prefixes of the base model's own runs, where no failed edit has been seen
yet. It cannot see a loop that starts after a failed edit. A gate for the next adapter must include prefixes that end in a failed edit_file result and measure how often the generated call repeats
the failed one (base: 135 identical consecutive repeats in 1,077 edit calls, 12%).

## Self-teacher probe `sdpo_probe_v1`: FAIL (notebook 06, 2026-11-02, transformers 5.22.1, torch 2.11.0+cu130, 677 s)
No training. 382 edit turns of the base model's own runs (base1-3, v2rec1-3): notfound 76, repeat_fail 76, first_fail 130, ok_edit 100. Each turn scored twice with the unquantized E4B: log p(output | prompt) and
log p(output | prompt + a user turn with the tool's answer). A_t = teacher minus student per token. Rule fixed before the run (docstring of `train/sdpo_probe.py`); result files in D:\data\sdpo_result_dl.
- **A, credit assignment (76 near-miss "not found" edits, known-wrong tokens against other old_string tokens):** d < 0 in 47.4% of samples with the real error text and 47.4% with another edit's error.
  Needed at least 65% and at most 55%. Median d = 7e-7, median advantage on the wrong tokens -1.4e-6 and on the right tokens -1.0e-6. The teacher does not single out the mistaken characters.
- **B, loop suppression (76 repeated failed calls):** median sum of A over the call tokens = +0.0007 nats with the real error (needed at most -2.0). Successful edits: -0.02 (inside the band). Mismatched
  error text moved the repeated call more (-0.014) than the real one. first_fail median -0.09.
- **Verdict: FAIL** (neither A nor B). The numbers are small but not zero (|call_sum| median 0.04-0.26, max 53), so the teacher prompt was read; it changes little because E4B sampled these tokens at
  probability close to 1 both with and without the error.
- **Caveat on B:** in a repeat_fail turn the student prompt already contains the previous failed edit's error (the model saw one error and resent the call). So B asks whether a second copy of the error
  changes anything; the stronger statement from the data is that the model, with a "not found" result in context, still gives the failed call probability near 1. A is the cleaner test (the student has not
  seen the error) and it is flat. Prompt format of the teacher turn (user turn rather than a tool-result block) was a design choice and is untested against alternatives.

## Would a forgiving edit_file have rescued the failed edits? (replay of base1-3 and v2rec1-3, 2026-11-02, `mine/fuzzy_edit_replay.py`, no model)
463 `edit_file` calls failed with "old_string not found" (42 on files never read). A tool that applies the edit when ONE place in the shown file matches (backslash sequences unescaped: 63; same
text up to blanks and line-end whitespace: 29; best window of the same line count with similarity >= 0.9 and a margin of 0.05 over the second best: 72) would have found a place for **164 (35%)**.
- Those 164 against what the model did next on the same file: **93 never got any edit onto that file applied** (a real rescue if the place is right), 36 later replaced the same region, 35 later replaced a
  different region (the forgiving tool might have edited the wrong place, or the model changed its plan; 49% of the 71 with a later success).
- Rescued failures are as often on a gold-patch file as all failures (67% against 66%): the tool fixes copying, not localization.
- Per run: 56 of the 346 unresolved runs (16%) and 2 of 50 resolved runs had a rescuable failure; of the 36 empty-patch runs 8 did.
- Upper bound: if a fifth of those 56 runs ended up resolved that is about 11 of 396 runs, +1 to +2 tasks of 66, inside the +-3-4 noise of one dev66 run. Not a large lever; cheap to build though.
Limit: file text is what read_file showed, earlier edits are not replayed, "same region" is a proxy for correct. Output in D:\data\analysis\fuzzy_edit.json.

## Localization: what the gold file is worth, and an offline file ranker (2026-11-02, `mine/localization_funnel.py` output and `mine/rank_files_probe.py`, no model)
**Value of finding the file** (396 runs, single-file gold = 270): resolved **29.4%** (45 of 153) when the gold file was seen, **3.4%** (4 of 117) when not. Short issues (under 300 characters, 90 runs):
file seen in 21% (6 of 19 seen runs resolved, 4 of 71 not seen); long issues (180 runs): seen in 74%, 29% of those resolved. Once seen, short and long issues convert alike (32% / 29%), so the whole short-issue
gap is finding the file. Multi-file gold: 1 of 126 resolved. Ceiling for a perfect finder: +5.6 tasks of 66; raising "seen" from 57% to 80%: about +2 to +3.

**Offline ranker** (non-test .py files of the repo at base_commit, median 506 candidates; 128 public tasks, 90 single-file; query = issue text; BM25 over identifier tokens; three rankers fixed beforehand):
| ranker | recall@1 | @3 | @5 | @10 | @5 short issues (n=39) | @5 long (n=51) | @5 dev66 (n=44) |
|---|---|---|---|---|---|---|---|
| content (file text) | 28.9 | 43.3 | 54.4 | 72.2 | 38.5 | 66.7 | 59.1 |
| path only | 20.0 | 32.2 | 36.7 | 46.7 | 30.8 | 41.2 | 27.3 |
| symbols (path + names of defined classes/functions/constants) | 45.6 | 57.8 | **63.3** | 70.0 | **56.4** | 68.6 | 54.5 |
Pre-registered rule: build if recall@5 >= 75% overall AND >= 50% on short issues; drop below 60% overall. Result: 63.3% / 56.4% -> **BORDERLINE** (short-issue condition met, overall missed).
Post-hoc (not pre-registered): union of the top 5 of `symbols` and of `content` (up to 10 files) 72.2% overall, 64.1% short, 78.4% long, 68.2% dev66.
Reading: the agent on its own saw the gold file in 56.7% of single-file runs (21% short, 74% long), so the ranker adds a lot on short issues (56% in the top 5 against 21% seen in a whole run) and nothing on long ones
(68.6% against 74%). The gain would be limited to short-issue tasks, about 19 of the 44 single-file dev tasks: roughly 6-7 more tasks with the gold file seen, about +2 resolved of 66 at 31% conversion, if the
model opens a listed file. Inside run-to-run noise. Whether the model uses a listed file is untested (v3loc showed it does not follow a procedure it has to carry out itself).

## `edit_retry` (aid arm) replayed on the recorded failures (2026-11-02, no model)
The skill's own `locate` step, run on the full base-commit file for each of the 463 "old_string not found" `edit_file` calls of base1-3 and v2rec1-3: it finds a place for **224 (48%)**: 134 similar
(>= 0.9 and clearly ahead of the next place), 85 escape sequences, 5 blanks/indentation only; 199 have no close match; 40 have old_string present in the base file (an earlier edit of the run had changed it).
Higher than the 35% of the first replay, which only had the text read_file had shown. Wrong-place risk is not measured here (49% of the first replay's rescued edits with a later success went to a
different region than the model finally edited); the tool prints the diff it applied. Speed: a first version took 82 s at 60 lines and 616 s at 140 lines in a 4,500-line file (the harness kills a skill
script at 90 s); long old_strings (over 600 characters) are now compared line by line, under 1 s at every size tested. `find_files` takes about 1.3 s on a 685-file repo.

## `aid1`: baseline + repo_aid skill (find_files, edit_retry), one seed, 66 dev tasks (2026-11-02, `fitted`, notebook 03, `mine/aid_metrics.py`)
**Resolved 7 of 66** (baseline 7, 7, 8). 3 empty patches, median 6 calls, no collapse. Rules from `agent/submission_e4b_aid/README.md`:
| rule | needed | measured | |
|---|---|---|---|
| `find_files` called, and in the first 3 calls | 80% | 83.3% / 83.3% | met |
| single-file gold file opened (baseline 56.7%) | 65% | 60.0% (27 of 45) | missed |
| same, issues under 300 characters (baseline 21%; base1 13.3%) | 40% | 26.7% (4 of 15) | missed |
| `edit_retry` called within 2 calls of an "old_string not found" | 50% | 26.8% (15 of 56) | missed |
| `edit_retry` calls that applied | 40% | 8.3% (1 of 12) | missed |
| resolved | 10 | 7 | missed |
What happened:
- The list contained the gold file in 53.3% of single-file runs (short issues 60%; offline on all public tasks 72% / 64%). The model sent a very short query: median **70 characters**, although the prompt asked for
  up to 800, so the ranker had less text than in the offline test.
- Listing is not opening: of the 9 short single-file runs with the gold file in the list, the model opened it in 4. The first `read_file` went to rank 1 of the list in 28 of 74 runs, to a file outside the list in 18.
- `edit_retry` failed in 11 of 12 calls. 7 times "the file would no longer parse as Python" (the model's own new_string was broken: it contained literal backslash-n or backslash-quote sequences, so the parse guard
  refused it), 4 times no place within 0.9 similarity (best 0.89 in one case). Only 1 edit applied (escape strategy, fastapi_15588), and no resolved run used a successful retry.
- Only 15 of the 56 "not found" edits were followed by a retry call, so the prompt paragraph is followed about a quarter of the time.
Reading: the harness-side help works as built (the skill runs, the ranker lists the file in half the runs, the retry tool finds places offline), but E4B does not make use of it well enough to move the count.
Per the pre-registered plan, two more seeds are not justified (7 < 10).

## `aid2_1`: aid with the aid1 fixes, one seed, 66 dev tasks (2026-11-03, `fitted`, `mine/aid_metrics.py`)
**Resolved 9 of 66** (baseline 7, 7, 8; aid1 7; v2rec 12, 7, 9). 6 empty patches (aid1 3), median 7 calls, 26 s.
| rule | needed | aid1 | aid2_1 |
|---|---|---|---|
| `find_files` called, first 3 calls | 80% | 83.3% | **84.8%** (met) |
| single-file gold file opened (baseline 56.7%) | 65% | 60.0% | 60.0% (missed) |
| same, issues under 300 characters (baseline 21%) | 40% | 26.7% | 20.0% (3 of 15, missed) |
| `edit_retry` within 2 calls of "old_string not found" | 50% | 26.8% | 31.4% (16 of 51, missed) |
| `edit_retry` calls that applied | 40% | 8.3% (1 of 12) | **28.6% (4 of 14)** (missed, up) |
| resolved | 10 | 7 | 9 (missed by 1) |
- The "word for word" wording did not change the query: median 76 characters, 14% of queries are 300 characters or longer (aid1 median 70). The gold file was in the list for 60% of single-file runs (short issues 66.7%).
- The "read the first listed file next" wording worked: the first read_file went to rank 1 of the list in 29 of 51 runs (aid1: 28 of 74). Short issues still opened the gold file in only 3 of 15 although 10 of 15 had it listed.
- `edit_retry` failures: 3 refused by the parse guard (down from 7), 8 "no place" (the model's old_string was too far from the file, 4 files); 4 applied. The prompt asks for `edit_retry` as the next action after a failed edit, yet it followed only 31% of them.
- 6 empty patches: each tried to edit, failed, and submitted anyway.
Per the plan, no further aid seeds (9 < 10). The count is inside the noise of the baseline (7-8) and v2rec (7-12).

## `ver1_1`: aid2 + reproduce-edit-recheck routine instead of "finish early", 30 calls (2026-11-03, one seed, 66 dev tasks, `fitted`, `mine/ver_metrics.py`)
**Resolved 9 of 66** (aid2_1 9, baseline 7, 7, 8). 5 empty patches, median 10.5 calls (aid2_1 7), 33 s (aid2_1 26 s), no tool_budget stop, all 66 stopped by `submit_patch`.
| rule | needed | measured |
|---|---|---|
| reproduction script written and run BEFORE the first edit | 60% | **0%** (missed) |
| something re-run after the first applied edit | 50% | 79.3% (met) |
| median calls / tool_budget stops / empty patches | <= 20 / <= 12 / <= 6 | 10.5 / 0 / 5 (met) |
| resolved | 11 | 9 (missed) |
The model did write and run `/tmp/repro.py` (75 `python /tmp/...` commands, 19 pytest runs) but only after its first edit, never before: before the first edit the 66 runs held 131 read_file, 75 find_files and 16 grep calls and no run of a script.
The routine therefore cost 3.5 calls and 7 s and changed nothing visible in the count. Runs that ran any python/pytest resolved 8 of 49, those that did not 1 of 17 (small numbers, confounded by task).
Verdict: the verification idea is dropped (rule: fewer than 11).

## Headroom across all 12 dev66 runs, and whether the model's own checks tell good patches from bad (2026-11-03)
- Runs: base1-3, v2rec1-3, v3fid1, v3loc1, v3h1, aid1, aid2_1, ver1_1. Resolved per run 7 to 12, mean 8.3. **21 of the 66 tasks were solved by at least one run**; 3 by all 12 runs, 7 by at least 6, 4 only once, 4 only twice.
  Union of 3 random runs 14.0, of 6 runs 17.7, of all 12 runs 21.0. So a perfect chooser among 3 attempts would score about 14 instead of 8, but only if it could pick.
- The model's own checks do not pick: over 457 runs with an applied edit (8 runs sets), resolved 18% (27 of 151) when the last python/pytest command after the edit exited 0, 15% (9 of 61) when it exited non-zero,
  13% (32 of 245) when no check ran. The signal is too weak to choose between attempts or to decide on a retry.

## Scorer fidelity audit (2026-11-04: the scorer's packages read and run against a mock model server; no model involved)
**What was read.** The organizers' wheelhouse (Kaggle dataset `metric/gemma-4-developer-agent-wheelhouse`, 41 wheels), unpacked as text into `D:\data\wheelhouse_src` and run only inside the isolated venv `D:\envs\scorer`. The scorer's own modules were driven by `mine/capture_scorer_requests.py` (mock OpenAI server, fake sandbox over `/workspace`) to record the exact requests it sends, and by `mine/diff_scorer_port.py` (seeded random inputs) to check `agent/vendor/scorer_port.py` against them.
| package | file size | sha256 (first 16) | newest entry in the zip |
|---|---|---|---|
| swegemma 0.2.11 | 119,670 | 9ad87658 1f3acde8 | 2026-10-12 |
| adk-submission 0.2.13 | 66,528 | f156f0bd 84580ccb | 2026-10-12 |
| adk-eval-core 0.1.0 | 90,060 | e614e1b2 fe8d886a | 2026-10-12 |
| google-adk 1.36.1 | 2,877,731 | 1a2f6868 c509e315 | (same file as in the kit) |
| vLLM 0.19.1 (organizer build) | 433,132,704 | 73c69baf ca24fbda | 2026-10-12 (3 files; the rest 2026-04-18) |
| transformers 5.13.1 | 11,503,977 | 53f0ea8a a397e292 | (same file as in the kit) |
**The community verification kit (`external/gemma-4-serving-and-lora-verification-kit/official_wheel_hashes.csv`) lists older builds**: swegemma 0.2.7, adk-submission 0.2.11, a different adk-eval-core 0.1.0 (89,306 bytes), a different vLLM wheel (433,132,506 bytes, sha 6b29fdc2). Only compressed-tensors, google-adk and transformers match. So everything below describes the 2026-10-12 build of the wheelhouse. That the hidden scorer runs that build is likely (it is the organizers' published stack, three weeks old, and the leaderboard has been scoring since) **but not verified**; the kit's older build is the alternative.

**The fidelity table (plan S0.2).** "Was" is what the local harness did before the 2026-11-04 port (`pre-fidelity-203e80f`).
| # | Question | Answer | Was |
|---|---|---|---|
| 1 | How is the prompt built? | System message = the submission's instruction text (our `flat_search.md` has no placeholders) + `"\n\n"` + ADK's identity sentence `You are an agent. Your internal name is "swe_agent". The description about you is "Resolves an issue with one minimal source edit.".` (google-adk `flows/llm_flows/identity.py`). The issue is in the **first user message**, built by `swegemma.harness.agent_runner.build_agent_prompt`: problem statement, hints, Task Budget, Execution Environment Rules, five numbered instructions, an optional Code Intelligence Tools block, and a Workspace Layout tree read from the container after setup (`find` depth 3 with the exclusions in `scorer_port.PRUNED_TOP_DIRS`, 200 lines). | two lines of text; no identity sentence |
| 2 | Tool declarations | The six tools our submissions declare, in YAML order, with the full docstring descriptions and JSON schemas; captured in `agent/vendor/fixtures/scorer_tools.json`. The request has no `tool_choice`. | hand-written block; `edit_v1` training data used another |
| 3 | Is `reasoning_content` sent back? | Yes, with adk-submission 0.2.13: its LiteLLM patch mirrors `reasoning_content` into `reasoning` (vLLM 0.19.1 reads only `reasoning`). Assistant history messages carry `tool_calls`, `reasoning_content` and `reasoning`, and no `content` key when empty. The competition's original template renders an earlier thought only for an assistant turn that has tool calls and comes after the last user message. `include_thoughts: false` strips them. | `reasoning_content` only, `content: null` always present |
| 4 | The real `edit_file` | Four tiers: exact, flexible (line by line, whitespace-insensitive, re-indented), regex (tokens joined by `\s*`), JSON-unescaped (the escape-removal fallback). Errors: "Failed to replace: old_string not found. Ensure you're not escaping content incorrectly and check whitespace, indentation, and context.", "Target string occurs N times in F. Please provide more surrounding context.", "File 'F' is empty. Use write_file to populate an empty or newly created file." Reads and writes keep CRLF. | three tiers, own texts, CRLF turned into LF |
| 5 | Result envelope | A `tool` message whose content is one level of `json.dumps(..., ensure_ascii=False)`; ok results start `{"status": "ok", ...}`, errors are `{"status": "error", "error_type", "error_message"[, "details"]}`. ADK's own errors (unknown tool, missing argument) have different shapes and are not counted as calls. | similar, but own `error_message` texts and rounded status floats |
| 6 | Truncation caps | `stdout` and `stderr` 5,000 characters each; `read_file` 150 lines or 10,000 characters, `end_line` not recomputed after a cut; `get_status` floats unrounded. | caps the same; `end_line` recomputed |
| 7 | Compaction | HARNESS_README 7.2 gives `EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14,336, event_retention_size=5)` and `ContextCacheConfig(min_tokens=2,048, ttl_seconds=1,800, cache_intervals=10)`. These are set in `scripts/inference.py`, which is **not in the wheelhouse**; the wheel's `swegemma/config.py` only carries the empty fields. The teammates' code says `compaction_interval=5`; it hardly matters, since that window counts invocations and a task is one. The summary call is built by ADK's `LlmEventSummarizer` without a config, so it sends **no sampling fields** (vLLM falls back to the checkpoint's `generation_config.json`: 1.0 / 0.95 / 64); but the thinking settings and the seed sit on the model object (`apply_thinking_config_to_model`), so they go out with it. It sees only the text parts of the events, not tool calls or results. | threshold 14,336, retention 5 and the unsampled summary call were already modelled (the teammates had read ADK for this); the summary call lacked the thinking budget and the seed; `compaction_interval=5` in a comment |
| 8 | Defaults and budget enforcement | Organizer generation defaults `{max_output_tokens: 16384, thinking_config: {thinking_budget: 4096}}`, merged under the submission's values key by key; `thinking_budget` is limited to 0 to 32,768. Budgets come from the submission's `eval_config.yaml` (ours: 24 calls, 5.5 minutes, 40 turns, 90 s command timeout). The call is checked **before** it runs and counted before it runs; `submit_patch` and `get_status` are free of the call and turn limits but obey the clock; the low-budget warning is attached to counted tools only, from call 14 of 24; `run_command` gets `min(90 s, max(5 s, time left))` inside `timeout -k 5s N`. | pre-call check existed; error texts, free-tool clock, warning scope and `get_status` differed |

**Findings beyond the eight questions.**
1. **Thinking is controlled by the submission** (adk-submission 0.2.13, `resolvers/generation.py` and `limits.py`): `thinking_budget` 0 (or level "none") sends `chat_template_kwargs: {enable_thinking: false}`; a positive budget sends `enable_thinking: true` and `thinking_token_budget: N`; `include_thoughts` only decides when there is no budget. Captured: `think4096` and `think0` requests differ exactly in those fields. HARNESS_README section 2.4 (our copy, saved 2026-10-28) documents it. The teammates' `runner.py` assumed the grader ignores `thinking_config` (it read only google-adk's LiteLlm adapter); with this build that is wrong. The first version of the accuracy plan (written before this audit) made the same assumption.
   The bridge changed twice (wheelhouse versions): v25 / adk-submission 0.2.11 forwarded neither `thinking_budget` nor `seed` and mapped `include_thoughts: false` to thinking off; 0.2.12 fixed the forwarding (745059); 0.2.13 (wheelhouse v29/v30) checks the budget before `include_thoughts`, so `include_thoughts: false` no longer turns thinking off and `thinking_budget: 0` does (747914). An organizer (Ryan Holbrook, staff) answered that 0.2.13 "was intended", and a participant writes that submissions have been scored with thinking on since Oct 8. So the 0.2.13 bridge is very likely what scores us now. That is not confirmed by a look at the scorer, and the private scoring on Dec 2 could in principle run something else: the shipped mode is chosen by gate M1, and the first canary shows the speed difference.
   A positive budget makes the request carry `thinking_token_budget`, and vLLM refuses that with a 400 ("thinking_token_budget is set but reasoning_config is not configured") unless the server was started with `--reasoning-config`. The scorer's `VllmServer` adds that flag itself for `--reasoning-parser gemma4`. **Our notebooks 03 and 05 did not** (they relied on the old runner never sending the budget). Both now pass `--reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}'`, and `runner.py` raises a readable error on that 400. Not yet run on a real server; vLLM 0.32.0 (what the notebooks install) may word the flag differently from 0.19.1.
2. **The scorer serves the competition model's original chat template** (sha256 `33204f1acb5bd0002713e16a593847f24ceeafe711ed88bda2a352dc996a3373`, identical to the Kaggle model copy), not the July canonical one (`0a2c8073...`). `enable_thinking` only adds `<|think|>` to the first system turn and the generation prompt is a bare `<|turn>model\n`. A completion with no thought channel (T0) renders exactly the same under thinking off, and nearly so under thinking on; hence the 50/50 mix in the data plan. `train/served.py` refuses any other template.
3. **A request has**: `model, messages, max_completion_tokens, temperature, tools, top_p, chat_template_kwargs, thinking_token_budget` (when thinking is on with a budget) and `top_k` (as the float 64.0). Tool messages carry `role, tool_call_id, content`.
4. **The session does not end at `submit_patch`**: the result goes back to the model and its closing text-only turn ends the run. The local runner stopped one call earlier.
5. **The organizer's vLLM build is stock 0.19.1 plus Gemma 4 LoRA support** (`D:\data\vllm_diff`, three files): `Gemma4ForConditionalGeneration` gains `SupportsLoRA` and an expert-weight name mapping; `gemma4.py` stops registering shared modules twice (so LoRA wrapping works); `lora/model_manager.py` skips resetting a module another name already loaded. Nothing in it changes sampling or the template.
6. Missing-file reads return docker-py's `NotFound` text in the scorer (container id and API version inside). Ours is an approximation; training data must skip those samples.
7. **The scorer's server** (`adk_submission.server.VllmServer.build_cmd`, with the parameters HARNESS_README 3.3 names: `tool_call_parser='gemma4'`, `reasoning_parser='gemma4'`, `max_model_len=32768`; the driver script `scripts/inference.py` is not in the wheelhouse): `vllm.entrypoints.openai.api_server --max-model-len 32768 --dtype auto --gpu-memory-utilization 0.90 [--tensor-parallel-size 4] --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --reasoning-config {"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"} --no-scheduler-reserve-full-isl`, plus `--enable-lora --max-loras N --max-lora-rank R` (R = the largest adapter rank rounded up to a vLLM bucket) when the bundle has adapters. A `--default-chat-template-kwargs` flag is **not** in the README's list; the teammates' comment attributes a server default `enable_thinking: true` to README line 198, which I could not find in the copy we have. It does not matter while every request carries `enable_thinking` (they do under 0.2.13).

**Gate results so far** (gates are in `docs/PLAN_E4B_SFT.md`).
| gate | result | detail |
|---|---|---|
| G0 | **passed on both scripted scenarios, one exception** | `tests/test_scorer_port.py::test_requests_match_scorer_end_to_end`: the 6 request bodies of the basic scenario are identical to the scorer's (every key, only the local `return_token_ids` aside). `test_tool_results_match_scorer_envelopes`: 7 of 9 tool results are byte-identical; get_status differs only in its clock fields (excluded by the gate); the missing-file text cannot be compared, because the capture's fake sandbox is not Docker (the error type is compared). The differential run: 8,000 seeded cases, about 40,000 comparisons of matcher, tools, patch stripping, prompts and generation config against the scorer's own modules, 0 mismatches. |
| D0 | not run | no training sample exists yet |
| M1 | not run | needs an adapter |
Not done: the plan's check of 10 real tasks through `swegemma eval --sandbox subprocess`; it needs Linux paths and is folded into the first replay pilot (S2.3). Not exercised: a real vLLM server (tool-call parser, streaming), the graph and skill tools' envelopes, the old adk-submission 0.2.11 behaviour.

## Multi-agent trees against the scorer's own code (2026-11-09: the pipeline ladder; no model involved)
The audit above covered one agent. The ladder of lab session 2 (`agent/submission_e4b_pipe_s1` .. `_s3`, written by `mine/make_pipeline_arms.py`) is a SequentialAgent of two LlmAgents, `investigate` (no edit tools) and `fix` (`include_contents: none`), so the S2 tree was run through the scorer's stack (swegemma 0.2.11, adk-submission 0.2.13, google-adk 1.36.1, LiteLLM 1.82.4) with `mine/capture_scorer_requests.py` (new flag `--outer-loop` = the scorer's continuation loop, `swegemma/harness/agent_runner.py`) and six scripted conversations (`mine/fidelity_scripts/script_pipeline*.json`; `mine/record_pipe_fixtures.py` records them into `tests/fixtures/scorer_requests_pipe_s2*.json`). What the requests show (google-adk `flows/llm_flows/contents.py`):
1. **One event list per session; each agent builds its prompt from it.** The user's events and the agent's own go out as they are (own thoughts as `reasoning_content`). Another agent's events become USER messages of text parts, `For context:` first and then one part each: `[<agent>] said: <text>`, ``[<agent>] called tool `<name>` with parameters: <args dict, Python repr>``, ``[<agent>] `<name>` tool returned result: <response dict, Python repr>``. Its thoughts are left out. One message per event, so a reply with two tool calls is one message of three parts.
2. **`include_contents: none` starts at the latest event of the user or of another agent** and takes everything after it. After a stage that ended with a note, the next stage's whole conversation is `For context: / [investigate] said: <note>`; the first message of the task is not in it. The old vendored runner started such a stage with an empty history and the fallback message `Handle the requests as specified in the System Instruction.`: wrong.
3. **A reply without visible text leaves no event** (nothing at all, or thoughts only, as after a token-limit cut inside the thinking): `output_key` is not set (`{findings?}` renders empty; a required `{findings}` would raise `KeyError` and the stage would never run), and the next `include_contents: none` stage starts from the latest event before it, e.g. the first message, or the tool result a stage ended on.
4. **A nudge restarts a SequentialAgent root at its first stage.** The scorer's outer loop sends the nudge as a new user message of the same session after any pass that ended without a submitted patch (at most 3 in a row without a tool call in the pass). The root is not an LlmAgent, so `Runner._find_agent_to_run` does not resume the stage that spoke last. On pass 2 `investigate` sees the whole session (its own history, `fix`'s calls and results as `For context` messages, the nudge, which asks for `submit_patch`, a tool it does not have) and `fix` starts again from the new note alone.
5. **Accounting.** One budget over the whole tree; a call to a tool the stage was not given is answered by ADK (`Tool 'edit_file' not found. Available tools: ...`) and not counted. `tool_calls_used` was equal in all captures; the turn count differs by one where a stage ends with an empty reply (the scorer does not count it).
**Base rate of the restart.** In the flat control (lab session 1, tune pool) 32 of 70 episodes had at least one nudge (most were plain continuations of a single agent) and 3 of its 9 resolved tasks were among them (th0 4 of 6, baseline 3 of 9). In a pipeline each of those becomes a second pass, so the two stage prompts carry a paragraph for it; its effect is unmeasured.
**Result.** `agent/vendor/runner.py` now keeps `state["_contents"]` as that session log (messages tagged `_author`; compaction runs on the log; a nudge is appended to it as a user message) and gives each agent a projection of it (`runner.view`). The request bodies of the vendored runner equal the scorer's for all six scenarios (first pass; empty note; thoughts-only; error-only stage 1; multi-call steps with thoughts on both stages and a nudge; the nudge pass), and the flat-agent test still passes unchanged. Not modelled: events compaction inside a multi-agent tree beyond "summarise the log", parallel branches' branch filter beyond the copy-and-merge the runner already did, the scorer's turn count for empty replies.

## S2: the SFT training set `full3` (2026-11-04; `train/s2_*.py`; CPU only, no model involved; data card `D:/data/s2/full3/DATA_CARD.md`)
Every number is computed from the files by `train/s2_card.py` and `train/s2_audit.py`.
- **Funnel.** Census (S1) about 2,545 issues in 1,095 repositories -> condensed 2,202 jobs from 1,067 repositories (rejected 69: no verification 55, visibly failing verification 13, tab in an edit 1) -> replayed on real checkouts and kept 1,721 (78.2%) -> rendered and audited **1,606 samples from 888 repositories** (93.3% of the replayed). Most samples from one repository: 10.
- **Replay drops (481).** No search term near the edit 325, read window over the 150-line cap 67, edit text not unique or absent in the file 30, edit outside the window the sample read 19, download timed out even at 400 s 17, commit gone (404) 11, tarball over 150 MB 7 (curl 63) plus one repository too large, edits under a `build/` directory 2, other mismatches with the teacher's diff 2. The replay is download-bound: GitHub builds each archive on the fly at 100 to 450 KB/s per connection.
- **Render drops (115).** Verification that neither runs a test runner nor names an edited file 66, no declared licence 34, teacher path left in the prompt 4 or in a tool result 1, evaluation-pool repository in the prompt 4 or in a tool result 2, over 12,000 tokens 4. No sample was dropped for an unstable prefix.
- **Size.** Median 5,859 tokens, maximum 11,574; loss on 9.0% of the tokens; calls per sample (submit included) 4: 126, 5: 1,241, 6: 68, 7: 139, 8: 15, 9: 17; thinking on in 790, off in 816; validation by repository: 54 repositories, 83 samples held out, 1,523 train. Tool mix of this set against E4B's own turns (pre-fidelity harness): read_file 21% against 36%, run_command 37% against 41%, edit_file 23% against 11%, submit_patch 19% against 13%; there are no turns of E4B's own in this set (S2.5 needs a run of the corrected harness).
- **Audit (`train/s2_audit.py`), plan conditions D0 to D5, all hold on the final 1,606:** median 5,859 and maximum 11,574 tokens (D5: 7,000 / 12,000); exactly the control tokens each episode implies (one `<bos>`, three turns, one call, stop token, response and response end per call) in every sample; one loss run per call, each exactly `<|tool_call>` through the stop token, then the closing text through `<turn|>`; stored ids equal the tokenization of the stored text; every call valid against the declared schemas; every edit `exact` with one occurrence on one to three files (D2); read windows at most 150 lines and 10,000 characters, command output at most 5,000 characters, every result `ok` (D3); keys in the served alphabetical order; no teacher path and no evaluation-pool repository (D4); no teacher or benchmark name in the trained spans; the fixed text of every first user message equals the scorer capture `tests/fixtures/scorer_requests_v2rec.json` byte for byte, and the system message and tool block equal the captured request's as JSON and as the first 1,640 (thinking off) / 1,642 (on) tokens the server would send (D0); no prefix-instability drop in the render log (D1). The detectors have unit tests that fire on injected defects (an extra control token, a loss run starting mid-call, a fuzzy edit, an over-wide window, a swapped key order, a teacher path, an altered budget line).
- **Read by eye: 20 random samples end to end** (seed 20261104, drawn from the 1,678-sample intermediate set): prompt, search output, reads, edits, results and closing text are consistent in all 20. What the reading found: one sample whose "verification" was `python -c` printing that `memoryview` is a builtin (aiogram-1370). Counting it over the intermediate set: 68 of 1,678 (4.1%) ran a snippet that touches neither a test runner nor the edited module; the render now leaves those out. Also counted: 5 verifications print nothing (0.3%); 165 (9.8%) end at the 5,000-character cap, mid-line, as the scorer's plain head slice would.
- **Fidelity defects found in the teacher's recorded verification outputs, all fixed before the final jobs:** 67% had `=====` banners about 491 characters wide (the teacher's terminal was about 500 columns; the scorer's pytest writes 80), which also pushed the pass/fail summary past the 5,000-character cut; 11% began with the terminal echo of the multi-line command; about 3% visibly failed with exit code 0 (a script printing its own traceback). The verification output remains the teacher's recorded one, not a rerun (the repositories' dependencies are not installed on the replay machine); the search command is chosen by the builder; the closing sentence is a template. Said so in the data card.
- **Scorer fact seen through the replay.** The sandbox adds `build/` and `dist/` to the git excludes (`agent/vendor/sandbox.py` GIT_EXCLUDE), so an edit to a file under any directory named `build` never reaches the patch. Two ARMmbed/mbed-tools jobs (`src/mbed_tools/build/...`) produced an empty diff and were dropped.
- **Licences of the kept repositories:** MIT 764 samples, Apache-2.0 374, BSD variants about 400, Zope 21, ISC 4, Unlicense 4, CC0 2, one MPL-2.0, the rest permissive; no GPL-family licence. 34 samples from repositories with no declared licence were left out (`--drop-unlicensed`); including them is a release decision, not made here.
- **Network.** For a few minutes (about 11:44 to 11:51 local) the Wi-Fi presented a certificate issued by a Sophos SSL CA for `codeload.github.com`, which neither WSL nor Windows trusts (curl 60); 527 downloads failed, then connections verified again without any change on this machine. Certificate verification was never disabled and no CA was added.
- **Not measured:** peak GPU memory and tokens/s at 8k and 12k tokens, whether `--checkpoint layers` is needed, any adapter's behaviour or score, whether the scorer's build of 2026-12-02 equals the one read on 2026-11-04.

## S3: the `e4b_sft_v2` training run (2026-11-06; notebook 07 on a Colab RTX PRO 6000; `train/train_lora.py`; numbers copied from the notebook's output)
- **Setup.** Base: the unquantized copy of the QAT checkpoint (`gemma-4-e4b-it-qat-q4_0-unquantized`). LoRA rank 64, alpha 64, dropout 0.05; learning rate 1.5e-4, 5% warm-up, cosine down to 10%; 2 epochs over 1,523 sequences, 16 sequences per optimizer step (191 steps); `--checkpoint layers`; format `onepass`, `--thought none`; sequences up to 12,288 tokens; seed 0. Loss only on the model's own tool calls and the closing text. Validation: 83 sequences from 54 held-out repositories.
- **Run.** 191 of 191 steps in 150.1 minutes, 2,052 tokens/s (2,027 in the smoke test), exit code 0. Peak memory **36.7 GiB**; the smoke test's 27.5 GiB was a floor, as flagged then (its eight sequences stopped at 8,173 tokens). Training loss printed at steps 105 to 191: 0.088 to 0.142, mean 0.113 (19 values).
- **Validation loss** (same spans, held-out repositories): step 0 **0.4193**; step 100 **0.1688**; step 150 **0.1704**; step 191 **0.1698**. The step-50 value is in `train_report.json` on Drive and was not part of the output sent to me. One epoch is about 95 steps, so the fall happened in the first epoch; from step 100 to 191 the validation loss moved by 0.0016 while the training loss stayed between 0.09 and 0.14. The second epoch neither helped nor hurt held-out loss.
- **What the loss does and does not say.** The held-out set is the same kind of trajectory as the training set: a teacher's, every tool result `ok`, no failed edit. The fall from 0.42 to 0.17 per token is mostly the model learning the call syntax, the argument order and the teacher's habits. It does not measure resolving more issues, and it says nothing about what the model does after a mistake of its own.
- **Gates.** G1 (key audit, size): 516 tensors, 0 bad keys, rank 64, 532.3 MiB (limit 3 GiB), **pass**, as audited by the trainer itself; whether vLLM loads it is G2 and has not been run. G6: **pass** on the first clause (0.1698 is below the step-0 value 0.4193); on the second, the final is 0.6% above the lowest value I have (0.1688 at step 100), and only the last three saves (steps 100, 150, 191) were kept, so step 50 could not have been chosen in any case.
- **Files** (Drive, `MyDrive/gemma4-adapters/`): `e4b_sft_v2/` holds `adapter_model.safetensors` (558.2 MB), `adapter_config.json`, `train_report.json`, `state.json`, `state/` (optimizer, 1,116.6 MB, never ship) and the adapters of the last three saves in `ckpt-00100`, `ckpt-00150`, `ckpt-00191` (558.2 MB each, written by `save_pretrained` like the final one, so notebook 08 can serve any of them by pointing `ADAPTERS` at the folder); `e4b_sft_v2_for_submission/` holds the two files a submission takes.
- **Not known:** whether vLLM serves this adapter on the W4A16 checkpoint (G2), what it does on prefixes of the scorer's harness (G3 to G5), and anything about its score.
- **Billing note.** Notebook 07 ended without releasing the runtime, so the GPU keeps billing (about 10 units an hour) until the runtime is disconnected. Notebooks 03, 07 and 08 now end with a section that flushes Drive and releases the runtime (`RELEASE = True`).

## Corrected harness against a real server: the smoke test (2026-11-06; notebook 03, vLLM 0.32.0, arm v2rec, one task `httpx_3672`; `fitted`)
One task says nothing about the score. This checks that the ported harness (`scorer-port-1`) works against a real server and that its traces carry what S4 needs. Raw files: `ctl_smoke.jsonl` and `ctl_smoke_trace.jsonl` (Drive, `gemma4-data/results/e4b/`).
- **Server.** vLLM 0.32.0 started with `--reasoning-config` and the `gemma4` reasoning and tool-call parsers. A plain chat returned reasoning and an answer; a request with one tool returned a parsed `tool_calls` entry.
- **Run.** 14 tool calls of 24 allowed, 17 model turns, 8,237 completion tokens, 53 s of agent time (63.6 s with the tests), `unresolved_nonempty_patch` (the task's tests fail because `HTTPParser` has no attribute `reset`). Compaction fired once, after 13 calls (21 events compacted, 6 kept, 3.7 s). One edit failed (the target occurs 3 times); three re-reads and two good edits followed. After the 14th call the model ended on text before `submit_patch`; the outer loop sent one nudge and it submitted.
- **Thinking.** 16 of 17 generations open a thought channel and close it before the call; the exception is the closing text after `submit_patch`. Completion tokens per turn: 33 to 1,338.
- **Fidelity.** The system message, the tool declarations and the 1,468 characters of fixed text in the first user message equal the scorer capture (`tests/fixtures/scorer_requests_v2rec.json`) byte for byte. The token ids the server saw equal our offline render (`train/served.py`, original template) of the same messages for the first prompt and for the last prompt of each of the two runs (9 and 13 messages, with replayed thoughts, a compaction summary and a nudge message): vLLM 0.32.0 renders the harness's history exactly as the training renderer does. The scorer's vLLM 0.19.1 was not tested.
- **After a compaction the history has no user turn.** The summary (a model message) replaces the task message and the older events, and the next prompt shares only its first 1,643 tokens (the system message and the tool block) with the one before. ADK's own `_process_compaction_events` (google-adk 1.36.1) removes the events inside the compacted range and materialises the summary as a model-authored event, which is what the port does. The six captured scorer requests contain no compaction case, so the exact shape is not checked against a scorer capture. The S2 training samples (median 5,859 tokens, maximum 11,574) never reach the compaction threshold (14,336), so neither this state nor the nudge state is in the training data; the G3 prefixes include both.
- **Records.** Two `conversation` events (15 and 2 turns, the second after the nudge), each with its own token chain; `train/build_replay_data.py` now reads both. On this trace it gives 17 turns, 14 kept tool decisions (8 read_file, 3 run_command, 2 edit_file, 1 submit_patch) and the failed edit as one G4 prefix (13,341 tokens; the next call was a re-read of lines 38 to 48). The G5 parser calls 15 of 17 generations well formed; the other two are the closing text turns, which no check set contains.

## S4: `e4b_sft_v2` against the base model on the corrected-harness control prefixes (2026-11-07; notebook 08, vLLM 0.32.0; `fitted`: the prefixes come from dev66 runs)
Setup. Prefixes from the control arm (`ctl_v2rec1/2`: shipped recipe, thinking on): 201 G3 prefixes (63 tasks; 50 per tool, 1 `write_file`), 150 G4 prefixes (35 tasks), 2 samples per G3 prefix and 3 per G4 prefix, temperature 1.0, top_p 0.95, top_k 64. Base = the W4A16 checkpoint re-sampled in the same server; adapter = step 191, rank 64, scale 1.0. Verdicts and the table: `docs/PLAN_E4B_SFT.md`, third S4 note (G2 PASS, **G3 FAIL**, G4 PASS, G5 PASS). The per-generation files are on Drive (`gemma4-data/results/s4/s4_e4b_sft_v2`).

First call by the tool the recorded run chose (100 generations each):
| recorded | base re-sample | `e4b_sft_v2` |
|---|---|---|
| `read_file` | read 70, run_command 16, edit 14 | read 30, **edit 49**, run_command 13, submit 8 |
| `edit_file` | edit 75, read 20, run_command 3, submit 2 | edit 86, read 7, submit 4, run_command 3 |
| `run_command` | run_command 68, read 20, no call 5, submit 4, edit 3 | run_command 62, edit 18, submit 11, read 9 |
| `submit_patch` | submit 62, no call 13, edit 12, run_command 7, read 6 | submit 92, run_command 6, edit 2 |
Over all 402 generations: base read 29%, edit 26%, run_command 23%, submit 17%, no call 4%; adapter edit 39%, submit 29%, run_command 21%, read 11%, no call 0%.

The `read_file` drop is not an artefact of unfamiliar states. 41 of the 50 `read_file` prefixes are not after a compaction: base reads 76%, adapter 30% (edits 55%). After a nudge (4 prefixes) the adapter reads 0%. 46 of the 50 prefixes carry the base's thoughts; the 4 that do not show the adapter reading in 0 of 8 generations, so the data cannot separate thoughts as a cause (the thought-stripped re-run was not made). With no edit applied yet (39 prefixes) the base reads 78% and the adapter 36%, editing 54%. 22% of the adapter's `edit_file` calls at these prefixes (10 of 46) name an `old_string` whose longest line is nowhere in the prefix (base: 0 of 13); at `edit_file` prefixes both models are at 94% visible.

After a failed edit (the 150 G4 prefixes: 111 with nothing applied yet, 39 with an edit applied earlier). G4 counts repeat and re-read only; the rest is "other": base 65 of 450 (14%), adapter 231 (51%). Next call, nothing applied yet (333 generations each): base edit_file 64%, read_file 15%, run_command 11% (no test run), submit_patch 6%, no call 4%; adapter run_command 37% (**pytest in 18%**), read_file 23%, **submit_patch 22%**, edit_file 18%. Edit applied earlier (117 each): base edit 73%, read 21%, run_command 6%; adapter edit 38%, read 32%, run_command 19%, submit 12%. By failure kind: after "occurs N times" the base edits again in 65% of generations (more context or `allow_multiple`), the adapter in 19%. Of the adapter's 86 `submit_patch` calls after a failed edit, 72 have no applied edit visible in the history (very likely an empty patch; 59 of the 150 prefixes come after a compaction, which can hide an earlier edit, and the base is affected in the same way); the base's 20 all have none visible (4.4% of its generations, against 16% for the adapter). "Nothing applied yet" above means: no applied edit visible in the prompt. The S2 samples contain no failed edit, so the adapter has never seen an error result answered by a fix.

G5 misses (852 generations each): base 49 = 31 with no tool call, 16 with arguments that do not parse, 2 with two calls; adapter 8 = 5 arguments that do not parse, 3 with a missing parameter, none without a call. Mean tokens 519 against 137; thought channel in 93.9% against 0%.

Not known: what any of this does to the number of resolved tasks (the plan forbids an agent run after a failed G3); how the adapter does on the 55 prefixes before a rejected call (`check_rejected.jsonl`, probe not run); whether the checkpoints at steps 100 and 150, or a smaller adapter scale, pass G3; the effect of the base's thoughts in the history.
