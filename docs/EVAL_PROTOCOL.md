# Evaluation protocol (written 2026-10-30, before any agent run on mined tasks)

Purpose: keep an honest measure of whether a change to the agent helps, given that every score on the public tasks is `fitted`
and that the 66 dev tasks were read in full.

## Task pools and what each may be used for

| Pool | Tasks | Status | Use |
|---|---|---|---|
| public dev66 (non-rich usable) | 66 | traces read in full on 2026-10-29 | tuning and diagnosis only. Never quote as unseen. |
| public `rich` | 43 usable | never run, never read | unseen. Run only at milestones (unchanged baseline, best version). |
| mined, **tune** repos | click, falcon, pytest, black, pygments, packaging, werkzeug, tornado, loguru | not yet run | may be read, used for diagnosis, and as a source of training traces |
| mined, **unseen** repos | litestar, pydantic-settings, isort, urllib3, marshmallow, scrapy | not yet run | never read. Aggregate numbers only. Run only at milestones. |

The mined split is by **repository**, assigned by a fixed rule before any result existed:
`sha256("gemma4-split-v1/" + "<owner>/<repo>")`, even = tune, odd = unseen. The hidden set may use other repositories, so
repository-level separation is the check that matters; a split by task would let repo-specific habits leak.

## Rules
1. Diagnosis (reading traces, patches, test output) happens only on dev66 and mined tune repos.
2. Prompt and config changes must be general behaviours (how to edit, when to check), never repo facts, task names, or test details.
3. Unseen pools are run only for a milestone comparison: baseline vs best candidate, same seeds. Only aggregate counts are read
   from them (resolved / total, outcome classes); no patch, trace or test output is opened. If one is opened, that pool is burned
   and this file must say so with the date.
4. Because one run is noisy (the same configuration moved requests from 4/4 to 1/5), every version is run on at least 3 seeds
   and reported as mean and range, never as one number.
5. Every score is labelled with its pool: `fitted` (dev66, mined tune), `unseen-mined`, `unseen-rich`.
6. Training (RFT or LoRA) uses traces only from pools that are **not** later used to report the result: train on mined tune and
   dev66, report on mined unseen and `rich`.
7. A task enters a pool only if it is usable: gold resolves and the empty patch does not, under the grader.
8. Spec check (added 2026-10-30): tasks flagged by `mine/spec_check.py` (the tests use a new public name the issue never
   mentions) stay in the pools but are reported separately: every score is given on all tasks and on the unflagged tasks.
   The check reads only the task files, never agent output. Flag lists: `mine/spec_flags_public.json`, `mine/spec_flags_mined.json`.

## Known limits
Mined tasks are not the organizers' tasks (see `mine/README.md`). They rank our versions; they do not predict the leaderboard.

## Amendment 2026-11-04 (S0.6 of the E4B accuracy plan; gates in `docs/PLAN_E4B_SFT.md`)
Nothing above is changed. Added:
9. **Pre-fidelity numbers are a different instrument.** The local harness was brought in line with the scorer on 2026-11-04
   (`FIDELITY = "scorer-port-1"`, `agent/vendor/README.md`; tag `pre-fidelity-203e80f`). Every result produced before that, on any pool,
   is labelled `pre-fidelity` and is not compared with a result from the corrected harness. A candidate is compared with a **control rerun
   on the corrected harness**, in the same Colab session, with the same seeds, the same server build and the same flags
   (`--reasoning-config` included). Results files carry `harness_fidelity`; a file without it is pre-fidelity.
10. **Pool sizes** (the repository split rule above is unchanged). Mined pools recounted on 2026-11-04 from `mine/usable.json` (122 usable):
    mined **tune** (click 15, falcon 12, packaging 15, werkzeug 10, tornado 10, loguru 4, black 2, pygments 2; pytest has no usable task) = 70,
    may be read; mined **unseen** (pydantic-settings 13, isort 13, urllib3 10, marshmallow 9, litestar 5, scrapy 2) = 52, unseen.
    Public pools as in the table above: dev66 = 66 (`fitted`) and `rich` = 43 (unseen), taken from the Drive file `usable_tasks.json`, which is
    not on the PC that wrote this amendment (the 129 public tasks are fastapi 67, rich 48, requests 13, httpx 1). Unseen total 95.
11. **What the instrument can see.** One run of 66 tasks has a standard deviation of about 1.5 to 1.8 tasks. Three seeds against three seeds can
    show a difference of about 4 tasks (6 points) on dev66, and about 4 to 5 tasks on the 95 unseen tasks. A smaller true gain will not show
    up; that is a property of the pools, so a result under that size is reported as "not distinguishable", not as a gain or a loss.
12. **Paired comparison.** Compare control and candidate per task and bootstrap over tasks (not over runs). Report the paired counts (tasks only
    one of them solved) next to the totals.
13. **Looks are counted.** Every run on an unseen pool is listed in the log below with its purpose. The report of any unseen-pool result says how
    many looks the pool had had. A look that changes a decision (which arm to ship) is a selection, and its number is optimistic by that much.
14. **Ledger.** One row per run in `docs/run_ledger.csv` (columns in its header): date, run id, arm, adapter checkpoint, pool, seed, server
    build, tasks, resolved, empty patches, mean calls, edit failure rate, gold-file-open rate, repeat-after-failure rate, median completion
    tokens, time-cap hits, Colab units, notes. A run that is not in the ledger did not happen as far as a decision is concerned.
15. **Server build is part of the result.** Record the output of `vllm --version`, the full launch command and the model directory's
    `chat_template.jinja` sha256 (`train/served.py::template_sha256`; the scorer's is `33204f1a...3373`) next to every run. The notebooks print
    the command; copy it into the ledger's notes column.

## Amendment 2026-11-08 (lab sessions on the mined tune pool; written before the second session)
16. **Single-run screening is a screen, not a result (a carve-out of rule 4).** A lab session runs ONE seed of each arm on the 70 mined tune tasks,
    next to a control of the same session (rule 9; with two controls, one at the start and one at the end, their mean is the yardstick and their
    difference is the pool's own run-to-run noise). An arm is a *candidate* only if it resolves **at least 4 tasks more than that control**; the paired
    counts, the interval and the sign test that `mine/lab_report.py` prints are read next to it, and the mechanism numbers (empty patches, edit failure
    rate, calls) are secondary and not part of the pass condition. A candidate is given two more seeds of itself and of the control in one later session;
    only the pooled three-seed comparison may be called a result, and it needs a mean difference of at least 4 tasks with the lower end of the 90% paired
    bootstrap interval above 0 (the A2 condition of `docs/PLAN_E4B_SFT.md`). A smaller difference is "not distinguishable" (rule 11). Arms are never ranked by
    single-run counts. Why this much care: with four arms against one control and NO true difference, the chance that at least one shows +4 is about 20% if a
    run's standard deviation is 1.7 tasks (rule 11) and about 44% if it is the binomial 2.7 (simulation of 2026-11-08; against the mean of two controls about 12% and 39%), so one pass is a reason to run again,
    never a finding. Power of the same screen at a standard deviation of 1.7 with two controls: a true gain of +6 tasks passes with a chance of about 86%, +4 about 55%, +3 about 36%,
    +2 about 20%. It finds big effects and mostly misses small ones; a small true gain still matters for the ranking, which is one noisy draw of about 60 tasks, but only pooled seeds can confirm it. Session 1 (9, 6 and 9 of 70) separated nothing under this rule. The unseen pools are not screened (rule 3).

## Log of milestone runs on unseen pools
- 2026-11-03: first run on the unseen `rich` pool (43 usable tasks), notebook 03 with `POOL = "rich"`: `rich_base1` (baseline), `rich_v2rec1` (v2rec, the submitted arm), `rich_aid2_1` (aid2), one seed each.
  Only the aggregate table of the notebook (resolved / tasks, median calls, outcome classes) is read; no trace, patch or test output of these runs is opened. The pool is NOT burned unless a line here says so.
  Look 1 of `rich`. `pre-fidelity` (rule 9): these three runs cannot be compared with any run of the corrected harness.
