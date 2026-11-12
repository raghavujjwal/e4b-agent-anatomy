# Vendored from branch `main`, commit 98eec6e (localeval/ and submission/), then brought into line with the scorer

`tools.py`, `runner.py`, `eval.py`, `sandbox.py` and `verify.py` are the teammates' local reimplementation of the
competition harness. They use Docker. We do not use their `Container`, `bootstrap` or `verify`:
`grader/container.py` provides the same interface on our Docker-free sandbox and is passed to `eval.run_one`
through its `container=`, `setup=` and `verifier=` hooks. `sandbox.py` and `verify.py` are here only because
`eval.py` imports them at load time.

## These files are no longer unchanged copies (2026-11-04, `FIDELITY = "scorer-port-1"`)

On 2026-11-04 the organizers' scorer packages (swegemma 0.2.11, adk-submission 0.2.13, adk-eval-core 0.1.0,
google-adk 1.36.1) were read and run against a mock model server (`mine/capture_scorer_requests.py`). The harness
here differed from them in things the model can see. Every number measured before the change was measured on a
different task than the scored one; the tag `pre-fidelity-203e80f` marks the last commit with the old behaviour.
Compare a result from this code only with a control rerun on this code (rule R1 in `docs/PLAN_E4B_SFT.md`).

The teammates' version was already close in many places (pre-call budget gate, `git add -N .` + `git diff --binary`,
ADK-style `tool_calls`/`role: tool` history, the `budget_warning` rule). What differed, verified against the code at
the tag:

| Area | Before | Now |
|---|---|---|
| First user message | two lines | the scorer's `build_agent_prompt`: problem, hints, budget and environment rules, five instructions, workspace tree read after setup |
| System message | the instruction text | instruction + ADK's identity suffix |
| Tool declarations | hand-written `TOOL_DECLS` | the scorer's, as captured (`fixtures/scorer_tools.json`), in declared order |
| Request body | `max_tokens`, `tool_choice: "auto"` | `max_completion_tokens`, no `tool_choice`, `top_k` as a float, as captured |
| Thinking control | `enable_thinking: true` on every call, `thinking_config` ignored on the reading that google-adk's LiteLlm adapter drops it | adk-submission 0.2.13's bridge: `thinking_budget` 0 turns thinking off, a positive budget goes out as `thinking_token_budget`; organizer defaults (16,384 tokens, budget 4,096) merged under the submission's `sampling.yaml` |
| History | `content: null` always present; `reasoning_content` only | no `content` key when empty; `reasoning` and `reasoning_content` both, as the scorer's litellm patch sends; `include_thoughts: false` strips them |
| Budget errors | `BudgetExhausted` "No tool calls or time remaining." | `BudgetExceeded` "Tool call budget exhausted (N calls)" / "Turn budget exhausted (N turns)" / "Session time budget exhausted (X minutes)"; the clock also stops free tools |
| `budget_warning` | attached to every tool | counted tools only |
| `edit_file` | 3-tier matcher (exact, flexible, regex), own error texts, CRLF normalised to LF on write | 4-tier matcher (adds JSON-unescaped), the scorer's error texts, CRLF preserved |
| `read_file` | `split("\n")`, `end_line` recomputed after a cut, `FileNotFound` | `splitlines`, `end_line` kept as requested, `FileReadError` (docker's not-found text, approximated) |
| `run_command` | error text `exit N`; timeout text "Command exceeded Ns and was terminated." | `error_message` is the stderr (stdout when stderr is empty), `details` carry stdout, stderr and exit code; timeout text "Command timed out after N seconds" |
| `submit_patch`, `get_status` | no protected-file stripping; `files_changed` counted by `+++ b/` lines; floats rounded | protected files and the task's test files stripped; `files_changed = diff.count('diff --git')`; floats unrounded |
| Unknown tool, missing argument | `UnknownTool` | ADK's own error texts; neither counts as a tool call; unknown arguments are dropped |
| Run loop | stopped at `submit_patch` | the result goes back to the model, whose closing text ends the session (one more model call, as scored) |
| Shell output | decoded as text, `\r` folded | bytes, decoded UTF-8 with replacement, `\r` kept |

`scorer_port.py` is a re-implementation from reading the scorer's code, not a copy. It has to reproduce the strings
the model sees byte for byte, so those appear verbatim. The wheels carry no licence metadata: settle that before
this repository is made public. `mine/diff_scorer_port.py` checks that the port keeps matching: it feeds seeded
random inputs to the port and to the scorer's own modules (in `D:\envs\scorer`, built from the organizers'
wheelhouse) and compares results. The last run, 8,000 cases and about 40,000 comparisons, found no mismatch.
`tests/test_scorer_port.py` replays the recorded scorer requests and tool results through this harness and
requires them to be identical.

## What this still does not reproduce

* The scorer's `read_file` error for a missing file is docker-py's `NotFound` text, which carries a container id
  and an API version; ours is an approximation.
* Graph and skill tools (`run_skill_script`, `load_skill_resource`, the code-graph tools) are not in the scorer's
  default declaration; their envelopes are unverified.
* Whether the hidden scorer runs adk-submission 0.2.13 or later is unverified. If it is older, `thinking_budget: 0`
  may not switch thinking off. Gate M1 in `docs/PLAN_E4B_SFT.md` treats thinking mode as a measured A/B for that
  reason.
* vLLM's tool-call parser and streaming behaviour were not exercised: no real server was in the capture loop.
* The Docker-free sandbox invokes commands through its own shell wrapper, not the scorer's
  `/bin/bash -c "timeout -k 5s N /bin/bash -c ..."`; `Container.exec_scorer` does, for Docker.

`../submission_ref/` is the 31B submission from `main` (reference only; the E4B submission is written separately).
