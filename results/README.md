# Results

Run records and analysis outputs behind the paper. Every number in the paper that comes from a local run can be
recomputed from these files with the scripts named below; numbers that cannot are listed at the end.

## Files

| Folder | Contents | Produced by | Paper |
|---|---|---|---|
| `dev66_controls/` | `ctl_v2rec1.jsonl`, `ctl_v2rec2.jsonl`: one row per episode of the two verified `v2rec` control runs on the 66 dev tasks (instance id, resolved, outcome class, tool calls, turns, tokens, seconds, edited files). The benchmark's test output (`pytest_tail`, `verify_error`) is removed. `ctl_analysis.json`: edit outcomes and episode statistics. | `agent/run_eval.py` (notebook 03), `analysis/episode_funnel/analyze_ctl.py` | 14 of 132 resolved (10.6%) |
| `episode_funnel/` | `funnel_rows.json`: one row per control episode (read a gold file, edited one, resolved, calls). `recovery_ceiling.json`: full-file replay of every refused edit and the aim check. `retrieval_bound.json`: BM25 bound on finding the gold file. | `analysis/episode_funnel/` | Figure 3; section 5 |
| `tune_session1/` | Lab session 1 on the 70 mined tune tasks: `lab_tune_ctl1` (`v2rec` control), `lab_tune_base1` (baseline), `lab_tune_th0_1` (thinking off). Result rows plus full traces (`*_trace.jsonl.gz`; the tasks come from public repositories). | `agent/run_eval.py` (notebook 03, `POOL = "tune"`) | 9, 9, 6 of 70; union 12 vs 21.4; edit failure 56% vs 72% |
| `dev66_analysis/` | Pre-fidelity analyses over 12 dev66 runs: `trace_report_*.json` (edit outcomes, empty patches), `funnel_*.json` (localization funnel), `fuzzy_edit.json` (forgiving-matcher replay), `copy_slip_census.json` (near-miss anatomy), `rank_files.json` (file ranking probe), `aid1.json`, `aid2_1.json` (file-finder use), `stdout_*.txt` (printed reports). | `mine/trace_analysis.py`, `mine/localization_funnel.py`, `mine/fuzzy_edit_replay.py`, `mine/copy_slip_census.py`, `mine/rank_files_probe.py`, `mine/aid_metrics.py` | prompt and tool arms (section 4); Figure 2 |
| `gates_s4/` | `summary.json`: gates G3, G4, G5 for the base model and `e4b_sft_v2` on 201 check prefixes and the failed-edit prefixes. | `train/s4_gates.py` (notebook 08) | training table: G3 fail (reads 30% vs 70%) |
| `sdpo_probe/` | `stats.json`: the probe set (failed, repeated and applied edits from six base runs). The probe's pass/fail numbers are in `docs/MEASUREMENTS.md`. | `train/build_sdpo_probe.py`, `train/sdpo_probe.py` (notebook 06) | 47% vs a 65% rule; +0.0007 nats vs a -2.0 rule |
| `sft_v2_data/` | `DATA_CARD.md` and `audit.json` of the 1,606-sample trajectory-SFT set (888 repositories, 83 samples held out). | `train/s2_card.py`, `train/s2_audit.py` | section 3 |

The run ledger of every agent run is `docs/run_ledger.csv`; the dated record of every measurement is
`docs/MEASUREMENTS.md`, and every arm with its pass rule and verdict is in `docs/TRIED_APPROACHES.md`.

## Recompute

```bash
gunzip -k results/tune_session1/*_trace.jsonl.gz
python -X utf8 mine/lab_report.py results/tune_session1/lab_tune_ctl1.jsonl \
    results/tune_session1/lab_tune_base1.jsonl results/tune_session1/lab_tune_th0_1.jsonl

python -c "import json; r=json.load(open('results/episode_funnel/funnel_rows.json')); print(len(r), sum(x['read_g'] for x in r), sum(x['edit_g'] for x in r), sum(x['res'] for x in r))"
# 132 88 70 14
```

## Numbers that do not come from files in this folder

- **Held-out evaluation** (`v2rec` 2 of 58, thinking off 1 of 58): the evaluation is hosted and returns only the score.
- **Lab session 2** (pipeline ladder: control 6 of 70, S1 10 of 70, S2 3 of the first 33 tasks): taken from the session's
  notebook table; the raw run files were not kept.
- **Training curves** (`e4b_sft_v2` held-out loss 0.419 to 0.170, `edit_v1` 0.834 to 0.374): copied from the training
  notebooks' output into `docs/MEASUREMENTS.md`.
