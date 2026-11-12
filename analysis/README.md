# Analysis scripts

Offline analyses of saved agent runs (no model, no GPU). They read the result rows and traces written by
`agent/run_eval.py` (`<run>.jsonl` and `<run>_trace.jsonl`) and, for the gold-file views, the benchmark's `tasks.jsonl`.
Paths are set with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `CTL_DIR` | folder with `ctl_v2rec1.jsonl`, `ctl_v2rec1_trace.jsonl`, `ctl_v2rec2.jsonl`, `ctl_v2rec2_trace.jsonl` | `data/ctl_files` |
| `TASKS_JSONL` | the benchmark's `tasks.jsonl` (gold patches) | `data/tasks.jsonl` |
| `REPOS_DIR` | git clones of the task repositories (for full-file replay) | `data/repos` |
| `WORK_DIR` | where intermediate files (`episodes.json`, `edits_all.json`, ...) are written | `work` |
| `TOKENIZER_DIR` | a Gemma 4 E4B tokenizer (only `gates_s4/recorded_rates.py`) | |

The dev66 traces are not shipped (they embed benchmark task text); the summaries these scripts produced are in
`results/episode_funnel/` and the `*.out` files here. Run with `python -X utf8 <script>` from the repository root.

## `episode_funnel/`: where the 132 verified control episodes are lost (paper section 5)

| Order | Script | Produces | Paper statement |
|---|---|---|---|
| 1 | `extract_episodes.py` | `WORK_DIR/episodes.json` (every call, response, read window) | input to the rest |
| 2 | `funnel.py` | `WORK_DIR/funnel_rows.json` (shipped as `results/episode_funnel/funnel_rows.json`) | 132 -> read a gold file 88 -> applied a gold-file edit 70 -> resolved 14; 62 never edit a gold file, 0 resolve; multi-file 0 of 42 |
| 3 | `statement_vs_gold.py` | printed | in 32 of the 62 the issue names the file or a distinctive identifier in it |
| 3 | `refusal_vs_outcome.py`, `not_in_gold.py`, `gold_aim_files.py` | printed | refused edits and outcome; edits in files the gold patch never touches |
| 3 | `recovery_ceiling.py` (needs `REPOS_DIR` and `WORK_DIR/edits_all.json`) | `recovery_ceiling.json` (shipped) | full-file replay agrees with 423 of 434 recorded outcomes (97%); 19 of 103 locatable refused edits within 10 lines of the gold hunk; ceiling about 2 tasks in 66 |
| 3 | `retrieval_bound.py`, `search_behaviour.py`, `failure_buckets.py`, `analyze_ctl.py` | printed / `retrieval_bound.json` (shipped) | BM25 bound, search before the first edit, why unresolved episodes fail |

## `edit_anatomy/`: what the failed edits are (paper section 5)

| Script | Output | Paper statement |
|---|---|---|
| `extract_edits.py` | `WORK_DIR/edits_all.json`, `extract_edits.out` | the edit calls of the control runs |
| `rejected_vs_prior_failures.py` | `rejected_vs_prior_failures.rerun.out` | 501 edit calls: 39.7% applied, 49.3% refused, 11.0% unparseable; apply rate 58% / 29% / 12% after 0 / 1 / 2 earlier failures |
| `escape_anatomy2.py` | `escape_anatomy2.out` | refusals: 41% ambiguity, 48% content mismatch (classes D-F), about 3% fixed by a plain JSON un-escape |
| `edit_anatomy_stats.py`, `edit_anatomy_fast.py` | printed | anchors of ten or more lines apply 21% of the time; after an ambiguity error the next edit to that file applies 10 times in 97 |
| `read_and_anchor.py` | printed | the longest line of a not-found anchor is visible in the prompt 93% of the time |
| `edit_features.py`, `rejected_styles.py`, `malformed_*.py`, `labeler_vs_real.py`, `compaction_effect.py`, `thought_share.py`, `g4b_prefixes.py` | `*.out` where kept | supporting checks (unparseable calls, compaction, thought share of the prompt) |

## `gates_s4/`: the trajectory adapter against the base model (paper section 4)

`analyze_s4.py` and `analyze_s4_g4.py` read the per-generation files of `train/s4_gates.py`: after a failed edit with
nothing applied, the adapter submits in 22% of generations against 6% for the base and retries an edit in 18% against
64%; it emits a thought channel in 0% against 94%. `recorded_rates.py` gives the base model's recorded rates on the
same check prefixes. The gate summary is `results/gates_s4/summary.json`.

## `noise/`: run-to-run variation (paper section 5)

`rerun_variance*.py`: overlap of solved tasks between control runs. Two runs of one arm on dev66 each resolved 7 tasks
but shared only 3. For the tune pool, `mine/lab_report.py` prints the union (12 of 70 against 21.4 expected for
independent runs) and the paired tests.

## Also in `mine/` (part of the main code)

`mine/trace_analysis.py`, `mine/localization_funnel.py`, `mine/fuzzy_edit_replay.py`, `mine/copy_slip_census.py`,
`mine/rank_files_probe.py`: the pre-fidelity versions of these analyses over 12 dev66 runs; their outputs are in
`results/dev66_analysis/`.
