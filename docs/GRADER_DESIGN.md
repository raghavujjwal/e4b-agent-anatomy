# Patch grader design: Colab, no Docker

**Status:** design only, nothing built. **Written:** 2026-10-27.
**Sources:** `rules-raw/dataset/HARNESS_README.md` (the official spec, section numbers below refer to it),
`rules-raw/dataset/sandbox_setup.py`, `rules-raw/dataset/Dockerfile.public`, and on branch `main`:
`localeval/{sandbox,verify,tools,soundness,solve_env,testdeps}.py`, `localeval/env_map.json`,
`external/gemma-4-task-soundness-and-local-results/task_soundness.csv`.

## 1. What the grader must do

The grader is a pure function: `grade(task, agent_patch) -> {resolved, reason, seconds}`. It is used three ways:
1. **Score** an agent on the public tasks (Phase 3 and 4).
2. **Reward** for rejection sampling and RL (Phase 5): one call per rollout, so it must be cheap and parallel.
3. **Sandbox for the agent itself** (the "Container A" half): the agent's `run_command`, `read_file`, `edit_file` and
   `submit_patch` need a workspace. This design covers the workspace, not the tool code (that is ported from `main`).

Non-goals: matching the hidden scorer exactly (impossible, see section 8); security against a hostile agent
(see section 7 for what we do and do not protect).

## 2. What the official grader does (HARNESS_README section 8.2)

For each task, in a fresh container (4 GiB RAM, 2 vCPUs, no network, Python 3.13):
1. Extract the snapshot at `base_commit`; write `.git/info/exclude`; editable-install the repo (`--no-deps`);
   install cached test dependencies; write the harness `pytest.ini` and `conftest.py`; commit `eval_baseline`.
2. Strip diff hunks touching protected paths from `agent_patch`, then apply it with four fallback passes
   (`git apply` -p1/-3/-ws/--recount, symlink-normalized, -p0, GNU `patch`). If all fail, the task fails.
3. Reset every protected path the patch touched (plus every path in `test_patch`) to `eval_baseline`; apply `test_patch`.
4. Run `PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest <test files> --junitxml=... -p no:anyio -o timeout=0
   -o python_classes="Test* *Test" -q` with a 300 s timeout.
5. Resolved only if pytest exits 0, JUnit has `passed>0`, `failures==0`, `errors==0`, and every required node passed
   without being skipped.

Protected paths: `test_*.py`, `*_test.py`, any `.py` under `tests|test|testing/`, `conftest.py`, `pytest.ini`,
`pyproject.toml`, `tox.ini`, `setup.cfg`, `.pytest.ini`, `sitecustomize.py`, `usercustomize.py`, `_swegemma_stubs.py`, `*.pth`.

## 3. Facts that shape the design

| Fact | Value | Source |
|---|---|---|
| Tasks / repos | 129 public tasks: fastapi 67, rich 48, requests 13, httpx 1 | `tasks.jsonl` |
| Snapshots | 129 `.tgz`, 21.5 GB, average 167 MB | Kaggle file listing |
| Offline wheels | 124 wheels, about 30 MB total; 115 are pure-python `py3-none-any`, 8 `cp313`, 8 `cp312` | Kaggle file listing |
| Official image | Python 3.13-slim, git, patch, pytest, pytest-timeout, build backends; `imp.py` and `telnetlib.py` shims | `Dockerfile.public` |
| `test_patch` format | plain `--- a/` / `+++ b/` headers, no `diff --git` line | `tasks.jsonl` |
| Required nodes | public tasks have **no** FAIL_TO_PASS field; derive from `test_patch` | data description, `main:localeval/sandbox.py` |
| Earlier local work | `env_map.json` records the interpreter that made gold pass for 102 tasks (100 on 3.13, 1 on 3.11, 1 on 3.12); 27 tasks unmapped | `main:localeval/env_map.json` |

Consequences:
- **Only four repos.** Dependencies repeat across tasks, so environments can be built per dependency set, not per task.
- **Grading is slow (about 100 s) and CPU-bound.** It will dominate the cost of rejection sampling and RL, not generation.
- **Colab cannot run Docker,** so the sandbox is a local subprocess with its own directory, interpreter and user.
  The official harness has the same fallback (`--sandbox subprocess`, section 4.1), so this is a supported shape.

## 4. Architecture

```
tasks.jsonl + snapshots/*.tgz + wheels/        (Drive: archives; /content: working copies)
        |
  [A] Task store ----> [B] Env builder ----> venv per env signature (/content/envs/<sig>)
        |                                              |
  [C] Workspace manager  --- copy of pristine task tree ---> /content/work/<run_id>/workspace
        |                                              |
  [D] Sandbox backend  (run as unprivileged user, scrubbed env, limits, timeout)
        |
  [E] Verifier (steps 2-5 of section 2)  ---> [F] Result cache (key: task id + sha256(patch))
        |
  [G] Controls (gold / empty / defect injection)   [H] Two-tier mode (fast proxy, full official)
```

### A. Task store
- Archives (`tasks.jsonl`, `snapshots/`, `wheels/`) live on Drive. A session extracts only the tasks it needs
  to `/content/tasks/<id>/pristine` (extract with `tar --no-same-owner`; the repo learned that root-owned
  files make git refuse to work, "dubious ownership").

### B. Environment builder
- Use `uv` (fast; can fetch CPython 3.11/3.12/3.13) to create venvs. Colab has internet, so this works, but
  **fidelity comes from the dataset wheels**: install with `--find-links wheels/` and a constraints file generated
  from the wheel versions, and fall back to PyPI only for packages absent from `wheels/`.
  (The repo found `h11` missing from the wheels, which blocked `httpx` and so 65 of 67 fastapi tasks; its
  `localeval/extra_wheels/` fixed that. Reuse that list as the known-missing set.)
- **Env signature** = (repo, interpreter version, resolved dependency set). Build lazily, cache locally.
- **Interpreter ladder:** try 3.13 (official), then 3.12, then 3.11, and record the first where the gold patch passes
  in `env_map.json`. Reuse the 102 existing entries; solve the 27 missing ones.
- Install the build backends the Dockerfile lists (setuptools, wheel, poetry-core, hatchling, flit-core, pdm-backend,
  editables), `pytest`, `pytest-timeout==2.1.0`, `typer`, and copy the `imp.py`/`telnetlib.py` shims into site-packages.
- **Caching across sessions:** venvs are many small files, which is slow on Drive. Store each as one `.tar.zst` on Drive
  (a few hundred MB at most) and unpack to `/content/envs/` at session start, or just rebuild with `uv` from the
  wheel cache. Measure which is faster before choosing.

### C. Workspace manager
- Per run: copy the pristine tree to `/content/work/<run_id>/workspace` (`cp -a`), then do official bootstrap steps 1 to 7
  (section 4.2). Reuse `sandbox/setup.py` directly: its `setup_git_exclude`, `configure_workspace_pytest_ini` and
  `configure_workspace_conftest` functions are pure Python, so the harness files come out identical.
- Order matters: write `pytest.ini`/`conftest.py` **before** the baseline commit (they must land in `HEAD`), as the harness does.
- A group of rollouts on one task reuses one pristine tree, so extraction is paid once per task per session.

### D. Sandbox backend
Replaces the repo's Docker `Container` class (`sh`, `put_text`, `get_text`, `put_archive`, `stop`) with a subprocess
implementation that keeps the same method names, so `verify.py` and `tools.py` port with small changes. See section 7.

### E. Verifier
Port `main:localeval/verify.py` (214 lines). Its JUnit validation already names the failing tests, which saved a re-run
once. Keep the four-pass patch application and the protected-path stripping from `sandbox.py`.

### F. Result cache
Key `(instance_id, sha256(patch))` -> result JSON on Drive. Re-grading the same patch is free. Store the pytest log per
result so a surprising verdict can be inspected later.

### G. Controls
See section 9. They are mandatory, not optional.

### H. Two-tier mode (for RL cost)
At 100 s per grade, a group of 8 rollouts costs about 13 CPU-minutes per task. Two tiers:
- **Fast tier (proxy):** run only the required nodes (`pytest <file>::<test> ...`) with the same environment. Use it for
  rollout filtering and as an early-stop in RL.
- **Full tier (official):** the exact command from section 2, run for every reported score and for the final
  confirmation of any rollout used as training data.
Calibrate the fast tier against the full tier on the gold and empty controls and on real agent patches; report their
disagreement rate. If it disagrees often, do not use it for rewards.

## 5. Verification procedure (what `grade` runs)

1. Look up the task; skip it if it is in the exclusion list (section 9).
2. Build or reuse the venv for the task's env signature.
3. Copy the pristine tree to a fresh workspace; bootstrap (section 4C); commit `eval_baseline`.
4. `agent_patch`: if empty or `NO_PATCH` -> not resolved. Else strip protected hunks, apply with the four passes.
   On failure return `resolved=False, reason="patch did not apply"`.
5. Reset protected paths and `test_patch` paths to `eval_baseline`; clean untracked files at those paths.
6. Apply `test_patch` (plain `---/+++` headers; the repo found some create files with an `--- a/<path>` header and no
   trailing newline, which both `git apply` and GNU `patch` reject, so normalize before applying).
7. Run pytest (command in section 2) as the sandbox user, in the venv, with a 300 s timeout.
8. Parse JUnit: apply the resolution rule in section 2 step 5, with required nodes from `test_functions()`
   (`added-def`, else `hunk-ctx`, else exit-code-only). Return reason, seconds, and the log path.

## 6. Python and dependency risks already known

- pydantic 1.10.15 is broken on CPython 3.12.4+, so some old fastapi tasks need 3.11 or 3.12 (hence the ladder).
- Wrong interpreter shows up as wording differences in error messages (the repo hit this on `rich_3472`:
  a newer CPython's `AttributeError` text). Pass the interpreter per task explicitly, never through a global.
- Newer dependencies can make a task pass with **no** patch (e.g. `fastapi_14851` with a newer starlette). That is why the
  empty control decides the scoring set, not gold alone.
- `pytest.mark.anyio` tests are disabled by the official `-p no:anyio`; keep that flag so we score the same way.

## 7. Isolation and safety on Colab

Docker gave us a memory cap, 2 CPUs, no network and a throwaway filesystem. Without it:

| Concern | Plan | Confidence |
|---|---|---|
| Agent commands touching Drive or secrets | Colab runs everything as root, and **Drive is mounted writable at `/content/drive`**. Run all agent and test commands as an unprivileged user (`setpriv`/`su`), with a scrubbed environment (no `KAGGLE_API_TOKEN`, no `GITHUB_TOKEN`). A normal FUSE mount is usually not readable by other users, so Drive should be inaccessible to that user. | Must verify on Colab |
| Runaway memory / CPU | `prlimit`/`ulimit` (address space, CPU seconds, process count); pin to 2 cores with `taskset`; kill the whole process group on timeout (as the official subprocess backend does) | Standard |
| Network | Try `unshare -rn` (user + net namespace) at startup. If it fails, run without a network block and record "network not blocked" as a divergence. Tests that need the network behave differently (the soundness table shows 2 such tasks). | Probe first |
| Leftover state between runs | Fresh workspace per run; delete it after; never reuse a venv for writing | Standard |
| GPU process | The model server runs in its own process; the sandbox user cannot touch its files | Standard |

If `unshare` and unprivileged execution both fail, the fallback is a **dedicated CPU-only Colab session without Drive
mounted** for anything that executes agent-written commands, with results returned through GitHub or a download.

## 8. Fidelity ledger (known divergences from the hidden scorer)

| Divergence | Effect on scores |
|---|---|
| Python version may be 3.11/3.12 instead of 3.13 for a few tasks | Small; recorded per task in `env_map.json` |
| Dependencies resolved partly from PyPI (wheels missing transitive packages) | Can make some tasks pass without a fix; caught by the empty control |
| No prebaked site-packages; the real harness streams era-appropriate packages | Unknown size |
| Network may not be blocked | Tests needing the network may differ |
| `_swegemma_stubs.py` hardware stubs are not in the dataset | Unknown; probably none for these four repos |
| Required nodes derived from `test_patch`; the hidden scorer uses real FAIL_TO_PASS / PASS_TO_PASS | Our verdict can be slightly looser or stricter |
| CPU count, speed and memory differ from the 4 GiB / 2 vCPU container | Affects timeouts and OOM, not verdicts, mostly |
| Local wall-clock is not the 4xL4 serving speed | Never use local time to set per-task budgets; only token counts transfer |

## 9. Validation plan (before any score is trusted)

1. **Gold control:** applying `task.patch` must resolve. Target: at least 90% of the public tasks.
2. **Empty control:** an empty patch must not resolve. Target: 0 of the sound tasks. Run it for all, because 18% of tasks
   add no new test function and can pass without a fix.
3. **External oracle:** compare our gold/empty results per task with `task_soundness.csv`. Target: at least 95% agreement.
   Disagreements are listed and explained, not averaged away.
4. **Defect injection:** the grader must catch each of these (a clean first run proves nothing):
   - a patch that edits only a test file or `conftest.py` -> stripped, not resolved
   - a patch that leaves a `.orig` file (GNU `patch` leaves them; the harness does not exclude them)
   - a patch with a bad hunk -> "did not apply"
   - a gold patch with one line removed -> not resolved
   - a patch that makes a required test skip -> not resolved
5. **Determinism:** grade the same gold patch 3 times; verdicts must match. Record flaky tasks and exclude them.
6. **Concurrency:** grade the same task from several workers at once; verdicts must match the serial run
   (the earlier work found a global-variable race that ran a task in the wrong interpreter).
7. **Timing:** record seconds per task.

Report coverage the way the repo does: tasks graded, tasks excluded (with reason), and which scores are `fitted`.

## 10. Reuse from `main` (and how to fetch it)

`main` is large (820 MB, mostly traces and wheels). In Colab use a sparse clone of only what is needed:
`localeval/{sandbox,verify,tools,soundness,solve_env,testdeps}.py`, `localeval/env_map.json`, `localeval/extra_wheels/`,
and `external/gemma-4-task-soundness-and-local-results/task_soundness.csv`.

| File | Reuse |
|---|---|
| `sandbox.py` | patch application, protected paths, `test_functions`, bootstrap order; **replace `Container`** |
| `verify.py` | the verifier and JUnit check, close to as is |
| `tools.py` | the agent's tools with the official caps, for the agent sandbox |
| `soundness.py`, `solve_env.py`, `testdeps.py` | scoring-set and environment solving, adapted from Docker images to `uv` venvs |
| `env_map.json` | 102 task -> interpreter entries |
| `extra_wheels/` | the missing-transitive-package wheelhouse (check Python tags; some are `cp311`) |

## 11. Build order, each step with an exit condition

| Step | Work | Done when |
|---|---|---|
| 0 | **Probe the Colab VM** (below) | Probe results recorded in `MEASUREMENTS.md` |
| 1 | `LocalSandbox` class with the same interface as `Container`, plus the unprivileged-user wrapper | A shell command runs as the sandbox user and cannot see `/content/drive` |
| 2 | Env builder (`uv` venvs, wheel constraints, ladder) for one task per repo | Those 4 tasks' gold patches resolve |
| 3 | Port `verify.py`; run gold and empty controls on all 129 | Section 9 items 1 to 3 met, with a coverage table |
| 4 | Defect injection and determinism and concurrency checks | Section 9 items 4 to 6 pass; each injected defect is caught |
| 5 | Result cache and two-tier mode; measure agreement of the fast tier | Fast-tier disagreement rate reported |
| 6 | Agent sandbox: port `tools.py` onto `LocalSandbox` | The gold patch applied through the tool path resolves, equal to the direct verify result |

Step 3 needs no GPU. Do it on a **CPU runtime** to save compute units.

### Step 0 probe (one cell, run first)
- `python --version`, `uname -r`, `nproc`, `free -g`, `df -h /content`
- `git --version`; `pip --version`; install `uv` and run `uv python install 3.13 3.12 3.11`
- `unshare -rn true` (can we make a network namespace?)
- `id`; create a user and run `setpriv --reuid=... id`; check that user cannot `ls /content/drive`
- Time a 167 MB snapshot extraction, and a `uv venv` + install from the wheel cache

## 12. Open questions

1. Does Colab allow `unshare -rn` and user switching? If not, use the dedicated no-Drive session (section 7).
2. How long does extraction plus bootstrap take per task, and how big is the extracted tree?
3. Do the 27 unmapped tasks resolve on some interpreter, or stay excluded?
4. How many CPU cores does the Colab GPU runtime give? That caps parallel graders.
5. Is the fast tier close enough to the full tier to use as an RL reward?
