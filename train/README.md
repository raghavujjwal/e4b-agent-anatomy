# Edit-step training data

Why: in five E4B runs 53-60% of `edit_file` calls failed because the model rebuilds `old_string` instead of copying it, and no task
with 4 or more failed edits was resolved (docs/MEASUREMENTS.md). Prompts did not change this. Weights might. This directory builds
supervised data for exactly that step, and nothing else.

`python train/make_edit_data.py --out D:/data/edit_sft --commits-per-repo 250` writes `train.jsonl`, `validation.jsonl`,
`tools_edit_file.json`, `tools_lines.json` (OpenAI-style tool schemas from `agent/vendor/runner.py`).

## What a sample is
One change hunk of a real diff. Messages: the arm's system prompt, the issue in the harness's wrapper, a `read_file` call and the result
the real tool would return (a window of at most 150 lines / 10,000 characters that contains the hunk), then the **target**: the next
assistant tool call. Two interfaces are generated from the same hunks so the choice can wait for the `lines` experiment:
- `edit_file`: target is the `edit_file` call with a unique whole-line anchor (smallest context that makes it unique).
- `lines`: target 1 is the `show` call, target 2 (context includes the numbered `show` output) is the `replace` call with `expect`.
Every target is checked by running the real code: `apply_replacement` from `agent/vendor/tools.py` for `edit_file`, the skill script itself
for `lines`; a sample is kept only if the result equals the expected text exactly. That check is also the reward for a later
rejection-sampling round (sample N calls from a model, keep those that apply and reproduce the text).

## Sources and what is kept out
- `history`: commits of the mined **tune** repositories (click, falcon, pytest, black, pygments, packaging, werkzeug, tornado, loguru), 1-3 changed
  source `.py` files, at most 60 changed lines, no merges/bumps/releases; instruction = commit message. Newest 10% of commits = `validation`.
- `task`: gold patches of the usable mined tasks of those repositories; instruction = the issue.
- Not used, by design (docs/EVAL_PROTOCOL.md rule 6): public dev66 and the other public tasks (they stay a clean comparison for the
  baseline runs base1-3 / lines1-3), `rich`, and the mined unseen repositories (litestar, pydantic-settings, isort, urllib3, marshmallow, scrapy).

## Limits (known, not hidden)
- A target is the gold change, so the model is also taught which change to make, not only how to copy. Commit messages are much shorter
  than issues. Treat the data as a copy-fidelity signal, not as fix-quality data.
- Samples are single decision points. The real run has searches before and a test or submit after; none of that is trained here.
- No reasoning text. Gemma 4 thinks before a tool call; training only on the call may reduce or change thinking. The trainer must decide whether
  to mask or keep the thought channel, and the chat template of the
  checkpoint has to render `tool_calls` and tool results; neither is tested yet.
- Hunks of one file are applied top to bottom and each sample shows the file with the earlier hunks already applied.
- Python source only; CRLF files are read with LF.
- No E4B adapter has been loaded in the scorer's vLLM by anyone on the team. Test that before spending GPU time on training.

## Training (written, tested on a tiny random model on CPU only)
- `render.py` turns a sample into token ids as the scorer's vLLM renders it (thinking on, so `<|think|>` is in the system turn; past reasoning is
  dropped from history by the template). The completion is the thought channel + the call + the `<|tool_response>` stop token. `--thought empty`
  (default) trains the call without thinking at that step; `--thought copy` makes the model first write the exact lines it will replace.
  `python train/render.py --data D:/data/edit_sft --tokenizer <checkpoint dir>` reports token lengths: median prompt 2.1k tokens, max 4.5k,
  median completion 87 tokens; no sample is dropped (798/798 validation `edit_file`).
- `train_lora.py`: LoRA rank 32 / alpha 64 on q,k,v,o,gate,up,down of the language model only, bf16 on the unquantized QAT weights, loss on
  completion tokens, 16 samples per step, lr 1e-4, saves every 50 steps and resumes, audits the saved keys against the pattern the scorer's vLLM
  loaded. `--tiny` runs a small random model on CPU (smoke test).
- `notebooks/04_train_edit_lora.ipynb` runs it on Colab: needs `edit_sft.zip` (train/validation jsonl and tool schemas, 8 MB) on Drive.
- Not tested: the real 16 GB model (memory, speed, gradient checkpointing with PEFT), and whether the adapter loads in vLLM.

## Repair after the `ad1` collapse (2026-11-01)
The first adapter (edit_v1) was trained only on "next action = edit_file" samples and called `edit_file` at every step of the agent (docs/MEASUREMENTS.md).
Three pieces, none run on a GPU yet:
- `build_replay_data.py`: the base model's own turns from saved traces, with the exact token ids (`turn_tokens`) and thoughts. Keeps every read_file /
  run_command / submit_patch turn and only the edits the tool applied (status read from the next prompt's tool response; about 43% of edit turns). Split by task
  hash into `turns_train.jsonl` and `turns_holdout.jsonl` (12 of 66 dev tasks); `check.jsonl` is a stratified sample of the holdout. From base1-3 and v2rec1-3:
  about 2,030 training turns (823 run_command, 729 read_file, 219 edit_file, 257 submit_patch), 11M prompt tokens, 201 check turns. Per docs/EVAL_PROTOCOL.md
  rule 6 this uses dev66 traces for training, so dev66 is no longer a clean comparison for an adapter trained on it: report on `rich` and the unseen mined pool.
- `step_check.py`: sends check prompts as token ids to /v1/completions and compares the tool in the generated call with the base model's recorded choice, for the
  base model (its own agreement is the ceiling) and each adapter. Gate decided in advance: for every tool but edit_file, agreement within 10 points of the base
  model's, and no-call share within 5 points. Unit-tested to fail an adapter that always edits.
- `train_lora.py --format none --replay-dir ... [--replay-edit-repeat N]` trains on the replay turns only (smoke-tested on the tiny model); notebook 04 takes
  `REPLAY_ZIP`, notebook 05 section 6b runs the step check.

## Next steps (not done)
1. Run notebook 04 (smoke test, then 4,000 samples, one epoch).
2. Serve the competition checkpoint with the adapter in vLLM (loads? output changes?) and run an offline edit benchmark: replay `validation.jsonl`
   prefixes against base and adapter, apply the generated call with the real tool code, report the share that applies and reproduces the text.
3. Only if that moves: the full agent on dev66 (3 seeds), then one round of rejection-sampling fine-tuning with the application check as reward.
4. Ablations if time: rank 8 vs 32, `--thought copy`, the `lines` interface, separate adapters.

## S2: condensed teacher trajectories (2026-11-04; plan in `docs/PLAN_E4B_SFT.md`, runbook step S2)
Turns the CC-BY-4.0 `nebius/SWE-rebench-OpenHands-Trajectories` (teacher Qwen3-Coder-480B-A35B-Instruct) into samples that look like what the scorer's agent loop sends to
E4B: at most 8 calls, real observations, the scorer's own first prompt. Three stages, then a data card; data lives outside the repo (`D:/data/s2/...`).
1. `python -I -X utf8 train/s2_condense.py --n 50 --out D:/data/s2/pilot2/jobs.jsonl` (Windows; `--n 0 --per-repo 10` for every selected instance).
   Keeps the critical path of a trajectory (successful `str_replace` edits on 1 to 3 source files, the view that grounds each edit, the call that revealed the file,
   one verification); rejects `insert`/`undo_edit`, new modules, other file types, ungrounded edits, tabs in an edit, and a verification that runs a test the teacher
   itself changed. Evaluation-pool repositories are excluded by name. The recorded verification output is made to look like the scorer's: the terminal echo of a
   multi-line command is cut off the front (11% of outputs), pytest banners and progress markers go from the teacher's ~500 columns to the 80 of a pipe (67%), and a
   verification that visibly failed although its exit code was 0 (a script that prints its own exception, a cross mark) is skipped in favour of the next clean one, or the
   job is rejected (`failing_verification`).
2. `bash train/s2_run_replay.sh /mnt/d/data/s2/pilot2 4` (WSL; stdlib only; 4 shards, 16 for the full set). The run is download-bound (GitHub builds each archive on the fly,
   a few hundred KB/s per connection), so the first pass gives up on a download after 75 s and a second pass takes the failures slowly:
   `FETCH_TIMEOUT=400 RETRY_FETCH=1 bash train/s2_run_replay.sh <dir> 3`. It can be stopped and started again at any point (it resumes, also with another number of shards).
   Per job: GitHub tarball at `base_commit`, unpacked on a tmpfs bind-mounted at
   `/workspace` in a private mount namespace (`unshare -Urm`, no sudo), the scorer's `sandbox_setup.py` pytest.ini/conftest.py/git excludes, baseline commit, then the
   scorer's first prompt (`agent/vendor/scorer_port.py`) and the real harness tools: a grep with line numbers, `read_file` windows (about 20 lines above the hit,
   at most 150 lines; whole file when short), the teacher's edits through the real `edit_file` (must be strategy `exact` and the diff must equal the teacher's change),
   `submit_patch`. Run it from a script file, not `wsl -- bash -c '...'` (quoting).
3. `python -I -X utf8 train/s2_render.py --replayed D:/data/s2/pilot2/replayed.jsonl --out D:/data/s2/pilot2/sft.jsonl` renders each conversation once with the scorer's
   chat template (`train/served.py`), half with `<|think|>` in the system turn, loss only on the calls (including the `<|tool_response>` stop token) and the closing text
   (format T0, no thought channel), and checks it (prefix stability at every assistant turn, envelope, read caps, no pool names or teacher paths). A sample over 12,000 tokens
   is left out (`--max-tokens`; the trainer's own limit is 12,288), and so is one whose command after the last edit neither runs a test runner nor names a part of an edited
   file (`unrelated_verification`: a snippet that checks how Python behaves verifies nothing about the patch; 4% of the samples). `--drop-unlicensed` also leaves out repositories
   with no declared licence (the packaged set uses it). `--workers 8` renders in parallel (about 4 s per sample per process; each worker holds a tokenizer, about 730 MB of the
   Windows commit limit: 14 workers raised MemoryError).
4. `python -I -X utf8 train/s2_audit.py --sft ... --replayed ... --render-log ...` (plan S2.7) re-derives every pass condition from the stored data instead of trusting the render
   step: lengths (median at most 7,000, max 12,000), the exact count of control tokens each episode implies (a repository file that contains the text `<tool_response|>` would
   become a real control token), one loss run per call and none other, ids = tokenization of the text, every call valid against the declared schemas, every edit `exact` with one
   occurrence on one to three files, read windows and command output within the caps, keys in the served order, no teacher path or evaluation-pool repository, no teacher or benchmark
   name in what the model is trained to say, an identical system turn in every sample, and (plan D0) the fixed text of every first user message and the system message and tool
   block equal to the recorded scorer request `tests/fixtures/scorer_requests_v2rec.json`, as text and as tokens. It exits 1 if any fails. Reading 20 random samples end to end
   is still done by eye (`D:/data/s2/full3` was read this way; the reading is what found the vacuous verifications).
5. `python -I -X utf8 train/s2_card.py --jobs ... --replayed ... --sft ... --out .../DATA_CARD.md --condense-log .../condense.log`, then
   `python -I -X utf8 train/s2_pack.py --sft ... --card ... --out .../sft_v2.zip` for Drive (the rendered text is left out; a manifest holds the counts, the sha256 and the commit).
Not real, and said so in the card: the verification output is the teacher's recorded output (the repositories' dependencies are not installed on the replay machine); the
search command is chosen by the builder from an issue term near the edit; the closing sentence is a template. Own-run turns (plan S2.5) are not included: the saved
traces came from the pre-fidelity harness. `tests/test_s2.py` covers the rules above.

## S3: training on whole trajectories (2026-11-04; plan step S3, notebook `07_train_sft_v2.ipynb`)
`train_lora.py --format onepass --data <folder holding sft.jsonl from s2_render.py>` trains on each rendered conversation in one forward pass instead of one sample per turn.
Under causal attention the two are the same loss (a test builds a tiny random Gemma 4 and checks the one-pass loss against the sum of the per-turn losses), at about a third of
the compute for a six-call trajectory. Per-turn data (`--replay-dir`, the edit-step formats) can still be mixed in; the loss function is chosen per example.
- Loss only where there is a target: `logits_to_keep` is given the tensor of target positions, so the 262,144-wide output layer is computed there and nowhere else; the model applies its
  own `final_logit_softcapping` inside `forward` (a test covers it).
- Every training forward passes `use_cache=False`. Without it the text model builds a `DynamicCache` from its own default (True) setting even though `model.config.use_cache = False`.
- `--checkpoint layers` wraps each decoder layer in `torch.utils.checkpoint` by hand (`gradient_checkpointing_enable()` is reported to save nothing on E4B); a test shows identical loss and
  gradients with and without it. `--probe-lens 4000,8000,12000` measures peak memory and tokens/s of one forward and backward pass at those lengths and exits (plan S3.0 tests 3 and 4);
  run it once per `--checkpoint` mode on the card you will train on and choose from the numbers. Neither has been run on a GPU yet.
- Validation is by repository (`train/split.py`: 5% of the repositories by a hash of the name, the same rule the data card uses), so validation loss measures code the adapter has not seen.
- The optimizer state is saved to `<out>/state/optimizer.pt` at every checkpoint and restored by `--resume`. It is about as large as the adapter again or more: copy only
  `adapter_model.safetensors` and `adapter_config.json` into a submission, never the `state` folder.
- Progress lines carry tokens/s and, on a GPU, peak memory.
Tests: `tests/test_onepass.py` (needs torch and transformers; the checkpoint test also needs peft).

## S4: offline gates before any agent run (2026-11-06; plan gates G1 to G6 and its S4 amendment; notebook `08_serve_adapter_and_gates.ipynb`)
- G1 (key audit, size) and G6 (validation loss at each saved checkpoint) come from notebook 07. G2 (vLLM loads the adapter on the W4A16 checkpoint, 10 requests) is notebook 08.
- `build_replay_data.py` now also writes `check_failed_edit.jsonl`: the turns whose prompt ends in the result of a failed `edit_file`/`write_file`, with the failed call parsed and the
  call the recorded run made next. `--g3-tasks` and `--g4-tasks` (`holdout` or `all`) choose which tasks the two check files draw from; use `all` for an adapter that has trained on none of
  the traces, `holdout` once it has.
- It also writes `check_rejected.jsonl` (2026-11-07): the prefixes before an `edit_file`/`write_file` call that the framework refused without running it (a mandatory parameter was
  missing from the arguments it could read), with that generation. An edit's status is read from the response that belongs to the call (for a generation with n calls, the n-th block
  from the end of the next turn's suffix): `ok`, `error` (FileEditError) or `rejected`. A report-only format probe is defined for the file in docs/PLAN_E4B_SFT.md (note of
  2026-11-07); the script is `format_probe.py` (below).
- `s4_gates.py` sends both check files to a vLLM server as token ids, for the base model and each adapter, at the shipped arm's sampling, and prints G3 (step types, via `step_check.py`),
  G4 (repeat and re-read after a failed edit, relative to the base model, with task-clustered standard errors), G4b (the same prefixes, restricted to those whose prompt shows no applied
  edit: the share of first calls that are `submit_patch` and the share that re-read or try another edit; needs `applied_edit_visible` in the check file, which `build_replay_data.py`
  writes since 2026-11-07, and reports INSUFFICIENT without it) and G5 (exactly one well-formed call, length hits, thought-channel share, mean
  tokens). Verdicts are PASS, FAIL, INSUFFICIENT or INVALID; only PASS passes. The tool-call parser in it is checked against the real chat template and against all 8,363 calls of the S2 set.
  Tests: `tests/test_s4_gates.py` (the end-to-end tests run against a fake server on localhost).

## Edit pairs: applied against failed, on the model's own calls (2026-11-07; pre-registered in `docs/PLAN_EDIT_PAIRS.md`; notebooks `09_edit_pairs_sample_and_train.ipynb` and `10_edit_pairs_eval_and_gates.ipynb`)

- `edit_pairs.py` has six commands. `export-spec` writes `prompt_spec_v2rec.json` from the captured scorer request (system text, user wrapper, the six tool declarations). `pack-data` zips
  the `edit_file` records that `sample` and `eval` draw from `D:\data\edit_sft` (a few MB instead of 205). `sample` re-renders each state the way the scorer renders the shipped agent, takes one
  full generation at the shipped sampling, keeps its thought, draws k-1 more calls after it (one request, `n=k-1`), labels all k with the harness's matcher (`agent/vendor/scorer_port.py`)
  against the window the model was shown, and with `--round2` repeats the step after one failed edit (the real error text is added to the history). `pairs` builds chosen / rejected pairs
  (applied exactly and near the gold edit, against refused or unreadable; at most 2 per state). `eval` gives the good-edit rate of served models on held-out prefixes with a paired bootstrap.
  `decide` applies the plan's outcome rules (primary outcome, guards, gates) to the files `eval` and `s4_gates.py` wrote.
- `pref_lora.py` trains `dpo`, `rft` and `dpo_nll` on the same pairs (loss on the call tokens, the shared thought stays in the prompt; the DPO reference is the network with the adapter
  switched off). `format_probe.py` is the report-only probe on `check_rejected.jsonl`. `s2_pack.py --pairs-code-out` packs all of it (and `prompt_spec_v2rec.json` and
  `agent/vendor/scorer_port.py`) for the notebooks.
- The labeller is checked against the 501 edit calls of the corrected control runs: the calls it calls `malformed` (Python-style keyword arguments, a missing mandatory argument, a call that
  never closes) were refused by the framework 30 times in 30; the template's reader is stricter than the server's parser, so unreadable calls of other kinds (`unparsed`) are left out of the
  pairs. Tests: `tests/test_edit_pairs.py` (against a fake server and the real chat template), `tests/test_pref_lora.py` (losses on numbers; the three arms on a tiny random Gemma 4,
  which needs peft: `D:\envs	rain-smoke`), `tests/test_format_probe.py`, `tests/test_notebooks_pairs.py` (names, flags and bundle contents of the notebooks, since a typo there costs units).
