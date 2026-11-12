# Pipeline ladder, rung S1: two-stage skeleton, diagnosis only

One rung of the ladder `ctl -> S1 -> S2 -> S3 -> ctl` of notebook 03, lab session 2 on the mined tune pool (docs/EVAL_PROTOCOL.md rule 16: a rung is a
candidate only if it resolves at least 4 tasks more than the mean of the two same-session controls, then it gets two more seeds). The folder is written by
`mine/make_pipeline_arms.py` from `../submission_e4b_v2rec`; do not edit it by hand, `tests/test_pipeline_arms.py` compares it with the script's output.

**Structure.** `agent.yaml` is a SequentialAgent of two LlmAgents (`sub_agents/`):
- `investigate`: tools `read_file`, `run_command`, `get_status` (no `edit_file`, `write_file`, `submit_patch`). It finds where the change belongs and why and
  ends with a text-only hand-over note (FILE / LINES / CAUSE / CHANGE / MATCH). The note is stored as `findings` (`output_key`).
- `fix`: `include_contents: none`, so it starts without stage 1's thoughts and tool output. Its instruction holds the problem statement (`{problem_description}`)
  and the note (`{findings?}`); tools are the control's six. It re-reads the lines, edits, checks that the code loads and calls `submit_patch`.

**Restarts.** The scorer's loop sends a nudge ("Please continue your work ... or call submit_patch ...") as a new user message of the same session whenever a pass
ends without `submit_patch`; the control's tune episodes had one in 32 of 70 and a third of its resolved tasks. A SequentialAgent root then starts over at
`investigate`, which sees the whole session (its own history, `fix`'s events as "For context: [fix] ..." user messages, the nudge), and `fix` starts again from the
new note alone. So both prompts carry a short paragraph for that case ("If this step starts again" in stage 1; the last paragraph of the note section in stage 2).
`mine/lab_report.py` counts the episodes with a nudge; the vendored runner reproduces this second pass byte for byte (tests/fixtures/scorer_requests_pipe_s2_nudge.json).

**Same as the control (`submission_e4b_v2rec`).** Model, `configs/sampling.yaml` (T 1.0, top_p 0.95, top_k 64, thinking 4096, thoughts kept) and
`eval_config.yaml` (24 tool calls, 5.5 min, 40 turns, 90 s per command) are byte copies; the budget and the clock are ONE budget shared by both stages (a
tool-call allowance the investigation uses is gone for the fix). The wording for finding the place, what a correct change looks like, making an edit,
the rules and finishing is the control's, byte for byte, split between the two stages. Calls are not capped below the control's 24: at the scorer's
speeds (about 12 s per tool call) the clock, not the call count, ends long episodes, and `mine/lab_report.py` prints how many episodes it would cut.

**What this rung adds.** The split itself. Stage 1 may only read and run commands, finishes with a note, and stage 2 starts from the issue plus the note with a short context instead of the exploration history (the control's thoughts are 31% of its context). Reference: 27% of control episodes never edit a file the gold patch touches, and none of those resolve.

**What it cannot tell.** Which part of the skeleton helped or hurt (split, fresh context, no edit tool in stage 1, the note format): they come together. A change of 1-3 tasks against the control is not readable at one seed.

**Mechanism numbers to read next to the count.** Calls per stage and the episodes in which `fix` made no call (stage 1 used the budget up) are printed by
`mine/lab_report.py`. If `fix` makes no call in more than a few of the 70 episodes, the investigation budget in the prompt is not being followed and the
count says nothing about the idea.
