# Where a 4B Coding Agent Fails

Code, run records and analysis scripts for the paper

> **Where a 4B Coding Agent Fails: A Fidelity-Checked Anatomy of Gemma 4 E4B on Real Repository Issues**
> *Edit copy errors, unseen files, and three fine-tuning routes that learned their targets and broke the agent*
> Ujjwal Raghav, Atharv Sharma (jetha ke jabaaz), November 2026

The paper is in [`paper/paper.md`](paper/paper.md) (figures in [`paper/figures/`](paper/figures/)).

We study Gemma 4 E4B (`gemma-4-e4b-it`, QAT W4A16, thinking on with a 4,096-token budget) as a single tool-using agent
that receives a GitHub issue and a repository snapshot, edits files through six fixed tools (`read_file`, `edit_file`,
`write_file`, `run_command`, `get_status`, `submit_patch`) under a hard budget (24 calls, 5.5 minutes, 40 turns), and is
graded by hidden tests. Everything here was built to answer two questions: what limits a 4B model in this loop, and which
interventions can move it at all.

## Main findings and where each one comes from

| Finding | Number | Evidence in this repo |
|---|---|---|
| Held-out result of the main arm `v2rec` | 2 of 58 tasks (thinking off: 1 of 58) | reported in the paper; the held-out evaluation is hosted, so no per-task file exists |
| Local proxy is optimistic | dev66 controls 14 of 132 episodes (10.6%); tune pool 9 of 70 (12.9%) | `results/dev66_controls/`, `results/tune_session1/` |
| Local harness matches the reference harness | request bodies byte-identical for a single agent and in six two-stage scenarios; 8,000 seeded cases, 0 mismatches | `agent/vendor/scorer_port.py`, `tests/test_scorer_port.py`, `mine/diff_scorer_port.py` |
| Localization comes first | 62 of 132 episodes never edit a file the reference fix touches; none of them resolves; multi-file fixes 0 of 42 | `results/episode_funnel/funnel_rows.json`, `analysis/episode_funnel/` |
| Failed edits are copy errors | 501 edit calls: 39.7% applied, 49.3% refused, 11.0% unparseable; 93% of not-found anchors are visible in the prompt | `analysis/edit_anatomy/*.out`, `analysis/edit_anatomy/` |
| Edit recovery has a low ceiling | 19 of 103 locatable refused edits lie within 10 lines of the gold hunk, about 2 tasks in 66 at best | `results/episode_funnel/recovery_ceiling.json`, `analysis/episode_funnel/recovery_ceiling.py` |
| Twelve prompt, sampling and tool arms stay inside noise | mean 8.3 resolved per dev66 run, controls span 7-12 | `docs/run_ledger.csv`, `docs/TRIED_APPROACHES.md`, `results/dev66_analysis/` |
| Three training routes learned their target and broke the agent | edit-only LoRA collapsed at scale 2.0 and 0.5; self-distillation signal absent (47% vs a 65% rule); trajectory SFT failed gate G3 | `docs/MEASUREMENTS.md`, `results/gates_s4/summary.json`, `results/sdpo_probe/stats.json` |
| Two-stage pipeline S1 | 10 of 70 against 6 for the same session's control (one seed) | `agent/submission_e4b_pipe_s1/`, `docs/MEASUREMENTS.md` |

`results/README.md` lists every result file and the paper statement it supports.

## Repository layout

| Path | What it is |
|---|---|
| `paper/` | The paper (Markdown) and its three figures |
| `agent/submission_e4b_*/` | One folder per arm: declarative agent config (`agent.yaml`, `prompts/`, `configs/sampling.yaml`, `eval_config.yaml`, optional `skills/`, `sub_agents/`) and a README saying what changed and why |
| `agent/vendor/` | The local harness: agent loop (`runner.py`), tools (`tools.py`), evaluation driver (`eval.py`), and `scorer_port.py`, the behavioural port of the reference harness |
| `agent/run_eval.py` | Runs one arm on tasks against an OpenAI-compatible server (vLLM) and grades the patches |
| `grader/` | Docker-free patch grader (unprivileged subprocess sandbox + `uv` virtual environments, network off) |
| `mine/` | Task miner for public GitHub repositories, pool tools, the fidelity capture/diff scripts, and the trace analyses |
| `train/` | Training data builders (edit-step data, SFT replay pipeline `s2_*`), LoRA trainer, the self-distillation probe, offline gates (`s4_gates.py`, `step_check.py`) |
| `notebooks/` | Colab notebooks that serve E4B, run arms, train and gate adapters |
| `analysis/` | Offline analyses behind sections 4-5 of the paper (episode funnel, edit anatomy, gates, noise), with their printed outputs; see `analysis/README.md` |
| `results/` | Run records and analysis outputs (see `results/README.md`) |
| `docs/` | Experiment logs: `MEASUREMENTS.md` (every measurement, dated), `TRIED_APPROACHES.md` (every arm, why, verdict), `EVAL_PROTOCOL.md` (task pools and the pre-registered screening rules), `PLAN_E4B_SFT.md` and `PLAN_EDIT_PAIRS.md` (pre-registrations), `GRADER_DESIGN.md`, `run_ledger.csv` |
| `tests/` | Unit and fixture tests (CPU, no model, no network) |

## Quick start (CPU only, a few minutes)

Python 3.11 or newer.

```bash
git clone https://github.com/raghavujjwal/e4b-agent-anatomy
cd e4b-agent-anatomy
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                       # tests that need torch/peft or the reference wheels skip themselves
```

Recompute the tune-pool table of the paper (session 1: control 9, baseline 9, thinking off 6 of 70; union 12 against 21.4
expected for independent runs; edit failure 56% against 72% with thinking off) from the shipped run records:

```bash
gunzip -k results/tune_session1/*_trace.jsonl.gz      # lab_report.py reads each run's sibling <run>_trace.jsonl
python -X utf8 mine/lab_report.py results/tune_session1/lab_tune_ctl1.jsonl \
    results/tune_session1/lab_tune_base1.jsonl results/tune_session1/lab_tune_th0_1.jsonl
```

Count the dev66 control results (14 of 132):

```bash
python -c "import json; rows=[json.loads(l) for f in ('results/dev66_controls/ctl_v2rec1.jsonl','results/dev66_controls/ctl_v2rec2.jsonl') for l in open(f)]; print(sum(r['resolved'] for r in rows), 'of', len(rows))"
```

## Reproducing the full pipeline

The steps below need GPUs, Linux and data that this repo does not redistribute. We ran them on Google Colab
(A100 40 GB and RTX PRO 6000 96 GB) and a Windows PC with WSL 2.

### 1. Inputs you need to obtain

| Input | Where it comes from | Used by |
|---|---|---|
| Gemma 4 E4B checkpoints: `gemma-4-e4b-it-qat-w4a16-ct` (served) and an unquantized QAT checkpoint (trained) | Google's Gemma 4 release, under the Gemma terms of use | notebooks 01, 03-08 |
| Reference harness wheels: `swegemma 0.2.11`, `adk-submission 0.2.13`, `adk-eval-core 0.1.0`, plus `google-adk 1.36.1` | The benchmark hosts' wheelhouse (not redistributed here; `google-adk` is on PyPI) | `mine/capture_scorer_requests.py`, `mine/diff_scorer_port.py` |
| Benchmark public tasks (`tasks.jsonl`, `snapshots/<id>.tgz`) and its `sandbox_setup.py` | The benchmark's public dataset (not redistributed here) | dev66 runs; the grader reads `sandbox_setup.py` from `GRADER_SETUP_PY` |
| Teacher trajectories | Hugging Face dataset `nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0) | `train/s2_*` |

Without the benchmark dataset you can still run everything on the mined pools (step 3), which come from public
repositories only.

### 2. Verify the local harness against the reference harness (CPU)

`agent/vendor/scorer_port.py` re-implements, from a reading of the reference packages, every string and rule the
model sees: the first user message, tool declarations, the four-tier `edit_file` matcher, error texts, truncation, the
thinking switch and thought replay, budgets, and multi-agent session handling.

```bash
python -m pytest -q tests/test_scorer_port.py          # against request bodies recorded from the reference packages (tests/fixtures/)
# inside a venv that has the reference wheels installed:
python mine/capture_scorer_requests.py --help         # replays scripted model turns through the reference ADK path against a mock server
python mine/diff_scorer_port.py --help                # differential test with random inputs against the reference modules
```

The scripted scenarios are in `mine/fidelity_scripts/`. `docs/MEASUREMENTS.md` (section "Scorer fidelity audit") and
`agent/vendor/README.md` list the differences the port removed.

### 3. Tasks

- **dev66**: the 66 usable public tasks of fastapi, requests and httpx (`mine/usable.json`); `rich` is held out.
- **tune (70 tasks)** and **unseen (52 tasks)**: mined from merged pull requests that close an issue, split by repository
  before any agent ran on them (`docs/EVAL_PROTOCOL.md`).

```bash
python mine/candidates.py --repo pallets/click --out data/mined/click.cands.jsonl        # GitHub GraphQL through the gh CLI
python mine/build.py --cands data/mined/click.cands.jsonl --out data/mined/data --limit 15
python mine/controls.py --help                                                         # gold and empty-patch controls (Linux/WSL)
python mine/spec_check.py --tasks data/mined/data/tasks.jsonl --out spec_flags.json
python -X utf8 mine/make_lab_pack.py --out data/lab/lab_pack.zip                       # pack the usable tasks for Colab
```

A task is usable only if its gold patch resolves and the empty patch does not. Selection rules are in `mine/README.md`.

### 4. Grade patches without Docker (Linux or WSL)

`grader/` grades a (task, patch) pair the way the reference harness does: tests run as an unprivileged user with the
network off; a task counts only if every test in the test files passes. Design: `docs/GRADER_DESIGN.md`. On Colab use
notebook 02 on a CPU runtime. Set `GRADER_SETUP_PY` to the benchmark's `sandbox_setup.py`.

```bash
python agent/run_eval.py --backend gold --task-id httpx_3672      # pipeline control: the gold patch must resolve
```

### 5. Serve E4B and run an arm

Notebook `notebooks/03_serve_and_run_e4b.ipynb` serves E4B with vLLM (we used 0.32.0) and runs arms on dev66 or a
mined pool (`POOL = "tune"` or `"unseen"`), writing one result row and one trace per episode. Locally:

```bash
python agent/run_eval.py --backend openai --base-url http://127.0.0.1:8000/v1 \
    --model gemma-4-e4b-it-qat-w4a16-ct --submission agent/submission_e4b_v2rec --task-id <id>
python -X utf8 mine/lab_report.py RUN.jsonl [RUN.jsonl ...]                # the paper's tables, overlaps and paired tests
```

Arms named in the paper:

| Paper name | Folder | Change |
|---|---|---|
| baseline | `submission_e4b` | earlier 31B prompt, temperature 0.2 |
| `v2rec` (control) | `submission_e4b_v2rec` | edit-hygiene prompt v2 + recommended sampling (T 1.0, top-p 0.95, top-k 64) |
| `v3fid`, `v3loc`, `v3h` | `submission_e4b_v3fid`, `_v3loc`, `_v3h` | short clean edit anchors; ranking files by distinctive words; budget and exact-name rules |
| `lines` | `submission_e4b_lines` | line-range edit skill instead of `edit_file` |
| `aid1`, `aid2` | `submission_e4b_aid`, `_aid2` | BM25 file finder + forgiving edit retry (skill `repo_aid`) |
| `ver1` | `submission_e4b_ver1` | reproduce, edit, recheck |
| `th0` | `submission_e4b_v2rec_th0` | thinking off |
| `edit_v1` | `submission_e4b_ad1` (scale 2.0), `_ad2` (scale 0.5) | edit-only LoRA |
| `e4b_sft_v2` | `submission_e4b_v2rec_sft2` | trajectory-SFT LoRA |
| S1, S2, S3 | `submission_e4b_pipe_s1` .. `_s3` | two-stage `investigate` -> `fix` pipeline (written by `mine/make_pipeline_arms.py`) |

`agent/make_submission_zip.py <arm folder> --out arm.zip` packs an arm into a single archive with `agent.yaml` at the root.

### 6. Failure analyses (CPU, from saved traces)

```bash
python mine/trace_analysis.py --runs RUN_DIR [RUN_DIR ...] --out trace_report.json        # edit failures, empty patches
python mine/localization_funnel.py --tasks tasks.jsonl --runs RUN_DIR ... --out funnel.json   # read -> edit -> resolve funnel
python mine/fuzzy_edit_replay.py --runs RUN_DIR ... --tasks tasks.jsonl --out fuzzy_edit.json # would a forgiving matcher help? (aim check)
python mine/copy_slip_census.py --help                                                    # near-miss anatomy of not-found anchors
```

The analyses behind the paper's funnel, edit anatomy, aim check, gate behaviour and noise numbers are in `analysis/`,
in run order, with the environment variables they read (`analysis/README.md`).

These read full dev66 traces, which are not shipped (see below); their summaries are in `results/`.

### 7. Training routes

| Route | Data | Train | Gate / evaluate |
|---|---|---|---|
| Edit-only LoRA `edit_v1` (rank 32, alpha 64) | `train/make_edit_data.py` | `train/train_lora.py`, notebook 04 | notebook 05, `train/step_check.py` |
| Self-distillation probe | `train/build_sdpo_probe.py` | none (probe only) | `train/sdpo_probe.py`, notebook 06 |
| Trajectory SFT `e4b_sft_v2` (rank 64, alpha 64, lr 1.5e-4, 2 epochs) | `train/s2_census_nebius.py` -> `s2_condense.py` -> `s2_replay.py` (WSL) -> `s2_render.py` -> `s2_audit.py`, `s2_card.py`, `s2_pack.py` | `train/train_lora.py`, notebook 07 | `train/s4_gates.py`, notebook 08 |
| Edit pairs (pre-registered, not run) | `train/edit_pairs.py` | `train/pref_lora.py`, notebook 09 | notebook 10 |

`train/train_lora.py --tiny` runs a small random model on CPU as a smoke test. The SFT data card and audit of the
1,606-sample set are in `results/sft_v2_data/`. The gates and their thresholds were written down before the adapter
produced any output (`docs/PLAN_E4B_SFT.md`).

## Protocol

Every arm had a pass rule written before it ran and was compared with a control rerun in the same session. A screening
pass needs +4 tasks on the 70-task tune pool and buys two more seeds, never a claim (`docs/EVAL_PROTOCOL.md`). One
70-task run has a standard deviation of about 2.5 tasks. Numbers measured before the harness port are labelled
pre-fidelity and are compared only with controls from the same harness.

## What is not included, and why

- **Benchmark tasks, snapshots and test output** of the public dev66 tasks, and the full dev66 traces (they embed the
  benchmark's task text). Per-episode result rows are included without the test output.
- **The reference harness wheels.** Only our re-implementation and the recorded request fixtures are here.
- **Model weights and adapters** (Gemma terms of use; sizes). The adapter configs and training scripts are here.
- **The SFT training set** (1,606 samples whose tool results contain code from 888 repositories under their own licences).
  Its data card and audit are included; `train/s2_*` rebuilds it from the public teacher dataset.
- **Lab session 2 run files** (pipeline ladder): the numbers in the paper come from the session's notebook table;
  the raw files were not kept.

## Known limitations

See section 7 of the paper. In short: the diagnosis leans on fastapi (91% of dev66); our server (vLLM 0.32.0 on one GPU)
differs from the held-out evaluator's; the prompt arms are pre-fidelity; S1 is a single seed; the gates are surrogates
for resolved tasks; mined tasks lack the held-out set's frontier-model filter.

## Citation

```bibtex
@misc{raghav2026where4b,
  title  = {Where a 4B Coding Agent Fails: A Fidelity-Checked Anatomy of Gemma 4 E4B on Real Repository Issues},
  author = {Raghav, Ujjwal and Sharma, Atharv},
  year   = {2026},
  note   = {Code and run records: https://github.com/raghavujjwal/e4b-agent-anatomy}
}
```

## License

Our code and documents are released under the Apache License 2.0 (`LICENSE`). The wheels in `grader/extra_wheels/`
are unmodified third-party packages under their own licences. Gemma 4 is subject to Google's Gemma terms of use.
