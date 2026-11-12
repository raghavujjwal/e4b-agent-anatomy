# Data card: S2 full set (full3)

Built by `train/s2_condense.py` (choose and condense), `train/s2_replay.py` (replay on real checkouts through the harness tool code) and `train/s2_render.py` (render and audit), for the E4B accuracy plan (`docs/PLAN_E4B_SFT.md`, step S2). Every number below is computed from the files.

## Sources and licences
- Trajectories: `nebius/SWE-rebench-OpenHands-Trajectories` (CC-BY-4.0; attribution: Nebius). The agent in them is Qwen3-Coder-480B-A35B-Instruct run in OpenHands 0.54.0; use of its outputs falls under that model's licence terms (Apache-2.0 as published on its model card; re-check before release).
- Task data (problem statements, base commits, patches): `nebius/SWE-rebench` (CC-BY-4.0). Repository sources are fetched from GitHub at each task's base commit and are used only to produce tool observations; they are not copied into the training files except as the text the tools returned.
- Repository licences of the kept samples (counts are samples; 888 repositories): MIT License 764, Apache License 2.0 374, BSD 3-Clause "New" or "Revised" License 280, BSD 2-Clause "Simplified" License 43, BSD License 41, Zope Public License 2.1 21, BSD-3-Clause 10, MIT-CMU License 9, BSD 8, New BSD License 7, Apache License 2.0 or MIT License 6, BSD 3-Clause 5, BSD 3-Clause License 5, ISC License 4, The Unlicense 4, MIT 3, MIT/Apache-2.0 Dual License 2, Creative Commons Zero v1.0 Universal 2, MIT No Attribution 2, BSD-2-Clause-Patent 2, BSD-3 2, MIT/X Consortium license 1, Modified BSD License 1, Mozilla Public License 2.0 1, BSD 2-clause license 1, PostgreSQL License 1, CC0 1.0 Universal 1, BSD-3 Clause 1, BSD-2-Clause Plus Patent License 1, Revised BSD License 1, Apache 2.0 or BSD3 1, BSD License (3 Clause) 1, 3-Clause BSD license 1.
- 34 samples from repositories with no declared licence were left out of this set (render option `--drop-unlicensed`).

## Funnel
- Condensing the selected instances (plan S1: about 2,545 issues in 1,095 repositories) wrote 2202 jobs from 1067 repositories and rejected 69: no_verification 55, failing_verification 13, tab_in_edit 1.
- Condensed jobs: 2202.
- Replayed on a real checkout and kept: 1721 (78% of 2202 attempted).
- Dropped in the replay: no_targeted_search_term 325, window_too_big 67, old_string_not_unique_or_absent 30, old_string_not_in_read_window 19, fetch:fetch failed (28): curl: (28) Operation ti 17, fetch:fetch failed (22): curl: (22) The requeste 11, fetch:fetch failed (63): curl: (63) Exceeded the 7, diff_files_differ:[] 2, exception:RuntimeError:repo_too_large 1, edit_not_exact:FileEditError 1, diff_differs_from_teacher 1.
- Left out by the render checks: verification does not touch the edit 66, repository has no declared licence 34, teacher path left in user 4, over 12000 tokens 4, evaluation-pool repository in user 4, evaluation-pool repository in tool 2, teacher path left in tool 1.
- Rendered and passed the audit: 1606 (93% of the replayed).

## Samples
- Distinct repositories: 888; most samples from one repository: 10.
- Tokens per sample: median 5859, max 11574, loss tokens 9% of all tokens.
- Tool calls per sample (counted by the harness, `submit_patch` included): 4: 126, 5: 1241, 6: 68, 7: 139, 8: 15, 9: 17.
- Calls by tool: run_command 3061, edit_file 1913, read_file 1783, submit_patch 1606.
- Share of calls by tool, this set against E4B's own turns (runs of the harness before the fidelity fixes, as kept by `build_replay_data.py`): read_file 21% against 36%; run_command 37% against 41%; edit_file 23% against 11%; submit_patch 19% against 13%. Every sample here contains an edit, so edits are over-represented; the first adapter has no turns of E4B's own to anchor the other step types (plan S2.5 needs a run of the corrected harness first), and gates G3 and G4 are what check for drift.
- Thinking: 790 samples carry `<|think|>` in the system turn, 816 do not. The model's own turns have no thought channel in either (format T0).
- Validation split (`train/split.py`, 5% of the repositories chosen by a hash of the name, so it does not move when the set is rebuilt): 54 repositories and 83 samples are held out, 1523 samples train.

## What is real and what is not
- Real (produced by running the harness tool code on the checkout): the workspace tree, the search output, every `read_file` result, every `edit_file` result (strategy `exact`, and the resulting diff equals the teacher's change to those files), the `submit_patch` result.
- Not real: the verification output (1606 samples) is the teacher's recorded output with paths rewritten, because the repositories' dependencies are not installed on the machine that replays; the search command (1455 samples) was chosen by the builder from a term in the issue that occurs near the edit, because the teacher's own searches print no line numbers. The closing sentence is a fixed template.
- Left out: hints (the prompt has none), tests and documentation changes, trajectories with `insert`, `undo_edit`, new modules or more than three source files, and samples whose verification command neither runs a test runner nor names an edited file (a snippet that checks how Python or a library behaves). Also absent: turns from E4B's own runs (plan S2.5), because the saved traces came from the harness before the fidelity fixes; they need a run with the corrected harness first.
- Bias: only small, local, correctly grounded fixes in Python; the selection favours tasks where the issue names an identifier. Repositories whose tarball did not download within the time limit (`fetch:` in the funnel) or is over 150 MB (`repo_too_large`) are missing, so very large repositories are under-represented. Held-out evaluation repositories (fastapi, rich, requests, httpx, litestar, pydantic-settings, isort, urllib3, marshmallow, scrapy) are excluded by name.

## Top repositories

- pypa/setuptools_scm: 10
- python-pillow/Pillow: 9
- pydicom/pydicom: 9
- pre-commit/pre-commit: 9
- oasis-open/cti-python-stix2: 9
- streamlink/streamlink: 9
- joke2k/faker: 9
- pytorch/ignite: 9
- marshmallow-code/apispec: 9
- tox-dev/tox: 9
