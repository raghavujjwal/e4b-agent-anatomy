"""Write the three rungs of the pipeline ladder, agent/submission_e4b_pipe_s1 .. _s3, from the control arm.

  python -I -X utf8 mine/make_pipeline_arms.py            # rewrite the three folders
  python -I -X utf8 mine/make_pipeline_arms.py --check    # exit 1 if a folder differs from what this script writes

The rungs are nested bundles, each adding one thing to the one below:
  S1  the two-stage skeleton: investigate (cannot edit) -> hand-over note -> fix (fresh context), diagnosis only
  S2  S1 + reproduce before the note, rerun the reproduction after the edit (one repair round)
  S3  S2 + two worked examples (an invented project) in the prompts
All three rungs also carry a short paragraph for a restart: the scorer's loop sends a nudge, as a new user message of the same session, whenever a pass ends
without submit_patch (32 of the control's 70 tune episodes), and a SequentialAgent root then starts over at stage 1 (tests/fixtures/scorer_requests_pipe_s2_nudge.json).
Every sentence of the control prompt (agent/submission_e4b_v2rec/prompts/flat_search.md) that a stage keeps is reused byte for byte from that file, and
configs/sampling.yaml and eval_config.yaml are copied from the control, so a rung differs from the control and from the rung below only where this
file says so (tests/test_pipeline_arms.py). The prompts contain no braces except {problem_description} and {findings?}: google-adk 1.36.1 renders
every {identifier} of an instruction from the session state and raises when one is missing.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
PARENT = "submission_e4b_v2rec"
MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
RUNGS = {"s1": {"repro": False, "demos": False}, "s2": {"repro": True, "demos": False}, "s3": {"repro": True, "demos": True}}
FIND, CORRECT, EDIT, CHECK, RULES, FINISH = ("Finding the right place to change", "What a correct change looks like", "Making an edit",
                                             "Checking your edit", "Rules", "Finishing")

# ------------------------------------------------------------------------------------------------ stage 1: investigate
INV_INTRO = """You are a senior Python engineer on a two-person team. Your teammate will make a small change to a repository, but only after you have found
exactly where the change belongs and why. This step is investigation only: you can read files and run commands, and you have no tool that edits files or
submits a patch. The task message tells you to edit and to call submit_patch; those instructions are for your teammate, so skip them. Work only inside
the workspace directory."""

INV_MUST_NOT = """Do not create or change any file inside the workspace in this step: anything left there ends up in the patch. Anything you want to try out
goes under /tmp. Do not run the repository's full test suite. Never conclude that the code is already correct or that nothing needs to change: someone
will change it, so name your best judgement of where."""

INV_BUDGET = """Use at most 6 tool calls in this step; your teammate needs the rest. When the problem statement already names a file, a function or an
error message, one search and one read of the right place is often enough. As soon as you know where the change goes and why, stop searching and write
your note."""

INV_BUDGET_REPRO = """Use at most 8 tool calls in this step, at most 3 of them for the reproduction; your teammate needs the rest. When the problem
statement already names a file, a function or an error message, one search and one read of the right place is often enough. As soon as you know where
the change goes and why, and have tried the reproduction, stop and write your note."""

REPRO = """Once you know where the problem is, reproduce it, so that your teammate can check the fix. Write a short script that triggers what the problem
statement describes (use its example if it has one), create it outside the workspace and run it in the same call, for example:

cd /workspace && cat > /tmp/repro.py << 'EOF'
(a few lines that call the code and print what happens)
EOF
python /tmp/repro.py

You want to see it fail the way the problem statement says: the exception, the wrong value or the missing behaviour. Keep the script short, make it
print its result, and use no network and no test runner. If the first run does not fail the way the statement says, correct the script once. If it still
does not, or the statement gives nothing you can run (a wording change, a documentation fix, behaviour that needs a server or a terminal), stop and write
REPRO: none with a few words of reason. Spend at most three tool calls on this."""

INV_RESTART = """Sometimes this step starts again after your teammate has worked: the conversation then also holds their calls and results (messages that begin
"For context: [fix]") and a request to carry on or to call submit_patch. Ignore that request, only your teammate can submit. Do not search again from the
beginning: read once what their edits left in the file, then write the note below. If an edit_file call of theirs succeeded, begin the CHANGE line with
"ALREADY MADE:" and name what they changed; if their edits failed, say under CHANGE how to make the edit so that it matches the file exactly."""

NOTE_HEAD = """Finish by replying with the note below and no tool call; this reply ends your step. Your teammate sees only the problem statement and your
note, so make the note complete and exact. Start each line with its heading:

FILE: path of the file to change, relative to the workspace (if the change needs a second file, repeat FILE and LINES for it)
LINES: first and last line number of the function or block to change
CAUSE: two or three sentences: what the code does now, and why that is wrong
CHANGE: what the edit must do, in words, with the exact names, values or messages to use
MATCH: error types, message strings, names and type annotations in that file which the change must follow"""

NOTE_REPRO_LINE = """
REPRO: the command that runs your reproduction again, and the last lines it printed before the fix; or "none" with a few words of reason"""

NOTE_TAIL = """

If you could not find the place, give your best guess in the same format and say so under CAUSE."""

INV_DEMO = """This example comes from an invented project and only shows the shape of a good investigation; your project and your problem are different.

Problem statement: parse_size("1.5k") raises ValueError: invalid literal for int() with base 10: '1.5'. It should return 1536.

Call 1, run_command: grep -rn "def parse_size" src
  result: src/units/sizes.py:21:def parse_size(text: str) -> int:
Call 2, read_file: src/units/sizes.py, start_line 15, end_line 45
  result: the function splits the text into a number and a unit, converts the number with int() and multiplies it by SIZE_UNITS[unit]; the next
  function, parse_rate, has the same shape
Call 3, run_command: writes /tmp/repro.py, which calls parse_size("1.5k") and prints the result, and runs it
  result: Traceback (most recent call last): ... ValueError: invalid literal for int() with base 10: '1.5'
Final reply, with no tool call:

FILE: src/units/sizes.py
LINES: 21-24
CAUSE: parse_size converts the numeric part with int(), so a decimal such as 1.5 raises ValueError before the unit is applied.
CHANGE: convert the numeric part with float() and return int(value * SIZE_UNITS[unit]); leave parse_rate alone.
MATCH: this file reports bad input with ValueError and annotates its functions with (text: str) -> int.
REPRO: cd /workspace && python /tmp/repro.py   (before the fix it ends with: ValueError: invalid literal for int() with base 10: '1.5')"""

# ------------------------------------------------------------------------------------------------ stage 2: fix
FIX_INTRO = """You are a senior Python engineer on a two-person team. A teammate has investigated the problem below and left you a hand-over note. You
will make one small, correct change to the repository and then stop. Work only inside the workspace directory."""

FIX_ENV = """All source code is under /workspace, and every dependency is already installed: the sandbox is offline, so do not run pip. A single command is
killed after 90 seconds. read_file shows at most 150 lines per call, so ask for the range you need with start_line and end_line. Tool calls are limited
and your teammate has already used some of them; get_status is free and tells you how many remain."""

FIX_PROBLEM = """{problem_description}"""

FIX_NOTE = """{findings?}

The note says where the change belongs and why. It can be wrong: if the file contradicts it, trust the file. If the note is empty or says the place was
not found, find it yourself first: pull identifiers and error strings out of the problem statement and search with run_command (for example with grep
-rn), at most three searches, and then continue.

You start every step without any memory of the earlier ones. If the lines the note points at already contain the change it asks for, or the CHANGE line
starts with "ALREADY MADE", an earlier attempt of yours made that edit: do not edit those lines again, check that the code loads and call submit_patch."""

CHECK_REPRO = """After your last edit, run the reproduction from the REPRO line of the note again. What the problem statement complains about must be gone.
If it still fails the same way, your edit did not reach the cause: read the lines again, reread CAUSE in the note, and make one more edit. If the run now
shows a traceback from your own change, fix that first. Make at most one such repair round, then submit what you have. If the note says REPRO: none,
run one command to confirm the code still loads instead, for example `cd /workspace && python -c "import <package>"` where <package> is the top-level
package you edited (the first directory under src/ or at the repository root). If one test file clearly covers the function you changed, you may also
run that single file. Do not run the repository's full test suite: it is slow and much of it may already be failing for reasons unrelated to this issue."""

FIX_DEMO = """This example comes from the same invented project and only shows the shape of a good edit; your files and lines are different.

The note says FILE: src/units/sizes.py and LINES: 21-24. You read them with read_file (start_line 21, end_line 30), which returns the file's own text:

```
def parse_size(text: str) -> int:
    number, unit = _split(text)
    value = int(number)
    return value * SIZE_UNITS[unit]


def parse_rate(text: str) -> int:
    number, unit = _split(text)
    value = int(number)
    return value * RATE_UNITS[unit]
```

First attempt, old_string is the one line you want to change, copied exactly:

```
    value = int(number)
```

The tool answers: Target string occurs 2 times in src/units/sizes.py. Please provide more surrounding context.
So add the next line, which only parse_size has, and send both lines together with their exact indentation.

old_string:

```
    value = int(number)
    return value * SIZE_UNITS[unit]
```

new_string:

```
    value = float(number)
    return int(value * SIZE_UNITS[unit])
```

If the tool had answered "Failed to replace: old_string not found", the text you sent differs from the file in some character, most often the
indentation or a quote mark. Read the same lines again and copy them exactly; never retry with a guessed variation."""


# ------------------------------------------------------------------------------------------------ assembly
def sections(text: str) -> tuple[str, dict[str, str]]:
    """The control prompt split at its '## ' headings: (text before the first heading, {title: body}), bodies byte for byte."""
    intro, out, title, buf = [], {}, None, []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("## "):
            if title is not None:
                out[title] = "\n".join(buf).strip()
            title, buf = line[3:].strip(), []
        elif title is None:
            intro.append(line)
        else:
            buf.append(line)
    if title is not None:
        out[title] = "\n".join(buf).strip()
    return "\n".join(intro).strip(), out


def block(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def compose(head: str, blocks: list[tuple[str, str]]) -> str:
    return "\n".join([head.strip() + "\n"] + [block(t, b) for t, b in blocks])


def investigate_prompt(sec: dict, repro: bool, demos: bool) -> str:
    blocks = [(FIND, sec[FIND]), (CORRECT, sec[CORRECT])]
    if repro:
        blocks.append(("Reproducing the problem", REPRO))
    blocks += [("What you must not do", INV_MUST_NOT), ("Your budget", INV_BUDGET_REPRO if repro else INV_BUDGET),
               ("If this step starts again", INV_RESTART),
               ("Your final message: the hand-over note", NOTE_HEAD + (NOTE_REPRO_LINE if repro else "") + NOTE_TAIL)]
    if demos:
        blocks.append(("Example of a finished investigation", INV_DEMO))
    return compose(INV_INTRO, blocks)


def fix_prompt(sec: dict, repro: bool, demos: bool) -> str:
    blocks = [("Environment", FIX_ENV), ("The problem statement", FIX_PROBLEM), ("Hand-over note from your teammate", FIX_NOTE),
              (CORRECT, sec[CORRECT]), (EDIT, sec[EDIT])]
    if demos:
        blocks.append(("Example of an edit", FIX_DEMO))
    blocks += [(CHECK, CHECK_REPRO if repro else sec[CHECK]), (RULES, sec[RULES]), (FINISH, sec[FINISH])]
    return compose(FIX_INTRO, blocks)


ROOT_YAML = """# Pipeline ladder, rung {rung} (README.md; generated by mine/make_pipeline_arms.py, do not edit by hand).
# Two stages in one tree: investigate (no edit tools) hands a note to fix (fresh context). The budget, the clock and the thinking settings are shared with
# the control: eval_config.yaml and configs/sampling.yaml are byte copies of ../submission_e4b_v2rec.
agent_class: SequentialAgent
name: swe_pipeline
description: Investigates the issue, then makes one minimal source edit.
sub_agents:
  - config_path: sub_agents/investigate.yaml
  - config_path: sub_agents/fix.yaml
"""

INVESTIGATE_YAML = """# Stage 1. No edit_file, write_file or submit_patch: the stage cannot change the repository, so a reproduction (rung S2 on) necessarily comes before the edit.
# Its last message is stored in session state as `findings` and rendered into the instruction of the next stage.
name: investigate
model: {model}
description: Finds where and why, and writes a hand-over note.
instruction: !include ../prompts/investigate.md
output_key: findings
tools:
  - read_file
  - run_command
  - get_status
generate_content_config: !include ../configs/sampling.yaml
"""

FIX_YAML = """# Stage 2. include_contents: none, so it starts without the thoughts and tool output of stage 1; its instruction carries the problem statement and the note.
# The tools are the control's.
name: fix
model: {model}
description: Makes the edit described in the hand-over note and submits.
instruction: !include ../prompts/fix.md
include_contents: none
tools:
  - read_file
  - edit_file
  - write_file
  - run_command
  - get_status
  - submit_patch
generate_content_config: !include ../configs/sampling.yaml
"""

TITLES = {"s1": "two-stage skeleton, diagnosis only", "s2": "skeleton + reproduce and verify", "s3": "skeleton + reproduce and verify + worked examples"}

README = """# Pipeline ladder, rung {rung_up}: {title}

One rung of the ladder `ctl -> S1 -> S2 -> S3 -> ctl` of notebook 03, lab session 2 on the mined tune pool (docs/EVAL_PROTOCOL.md rule 16: a rung is a
candidate only if it resolves at least 4 tasks more than the mean of the two same-session controls, then it gets two more seeds). The folder is written by
`mine/make_pipeline_arms.py` from `../submission_e4b_v2rec`; do not edit it by hand, `tests/test_pipeline_arms.py` compares it with the script's output.

**Structure.** `agent.yaml` is a SequentialAgent of two LlmAgents (`sub_agents/`):
- `investigate`: tools `read_file`, `run_command`, `get_status` (no `edit_file`, `write_file`, `submit_patch`). It finds where the change belongs and why and
  ends with a text-only hand-over note (FILE / LINES / CAUSE / CHANGE / MATCH{repro_field}). The note is stored as `findings` (`output_key`).
- `fix`: `include_contents: none`, so it starts without stage 1's thoughts and tool output. Its instruction holds the problem statement (`{{problem_description}}`)
  and the note (`{{findings?}}`); tools are the control's six. It re-reads the lines, edits, {fix_check} and calls `submit_patch`.

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

**What this rung adds.** {adds}

**What it cannot tell.** {cannot}

**Mechanism numbers to read next to the count.** Calls per stage and the episodes in which `fix` made no call (stage 1 used the budget up) are printed by
`mine/lab_report.py`. If `fix` makes no call in more than a few of the 70 episodes, the investigation budget in the prompt is not being followed and the
count says nothing about the idea.
"""

ADDS = {
    "s1": ("The split itself. Stage 1 may only read and run commands, finishes with a note, and stage 2 starts from the issue plus the note with a short "
           "context instead of the exploration history (the control's thoughts are 31% of its context). Reference: 27% of control episodes never edit a "
           "file the gold patch touches, and none of those resolve."),
    "s2": ("S1 plus a reproduction. Stage 1 writes a short script under /tmp and runs it (or says `REPRO: none` after at most three calls), stage 2 reruns "
           "it after the edit and gets one repair round. Because stage 1 has no edit tool the reproduction necessarily comes before the edit; `ver1` asked for "
           "that order in the prompt and every repro it wrote came after the first edit."),
    "s3": ("S2 plus two worked examples from an invented project (not from any task): a finished investigation with its note, and an exact-copy edit "
           "including the two refusals `occurs N times` and `old_string not found`. Of the refused edits of the dev66 control runs 41% are the first and 48% the second."),
}
CANNOT = {
    "s1": ("Which part of the skeleton helped or hurt (split, fresh context, no edit tool in stage 1, the note format): they come together. A change of "
           "1-3 tasks against the control is not readable at one seed."),
    "s2": ("Whether a gain over S1 comes from the reproduction or from noise. A reproduction needs something to run: 44 of the 70 tune issues contain a fenced "
           "code block (63%; 32 of them, 46%, look like Python) against 15 of the 81 issues of the three dev66 repositories (19%; 12, 15%, look like Python)."),
    "s3": "Whether a gain over S2 comes from the examples; the examples may also be copied (names from the invented project appearing in edits).",
}


def render_all(parent: Path = AGENT / PARENT) -> dict[str, dict[str, bytes]]:
    """{arm folder: {relative path: file bytes}} for the three rungs."""
    _, sec = sections((parent / "prompts" / "flat_search.md").read_text(encoding="utf-8"))
    sampling, evalcfg = (parent / "configs" / "sampling.yaml").read_bytes(), (parent / "eval_config.yaml").read_bytes()
    out = {}
    for rung, flags in RUNGS.items():
        text = {"agent.yaml": ROOT_YAML.format(rung=rung.upper()),
                "sub_agents/investigate.yaml": INVESTIGATE_YAML.format(model=MODEL),
                "sub_agents/fix.yaml": FIX_YAML.format(model=MODEL),
                "prompts/investigate.md": investigate_prompt(sec, **flags),
                "prompts/fix.md": fix_prompt(sec, **flags),
                "README.md": README.format(rung_up=rung.upper(), title=TITLES[rung], adds=ADDS[rung], cannot=CANNOT[rung],
                                           repro_field=" / REPRO" if flags["repro"] else "",
                                           fix_check="reruns the reproduction (one repair round)" if flags["repro"] else "checks that the code loads")}
        files = {k: v.encode("utf-8") for k, v in text.items()}
        files["configs/sampling.yaml"], files["eval_config.yaml"] = sampling, evalcfg
        out[f"submission_e4b_pipe_{rung}"] = files
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if a folder differs from the script's output")
    a = ap.parse_args(argv)
    bad = []
    for arm, files in render_all().items():
        folder = AGENT / arm
        have = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts} if folder.exists() else set()
        if a.check:
            bad += [f"{arm}/{n}" for n in sorted(have ^ set(files))] + [f"{arm}/{n}" for n, b in files.items() if n in have and (folder / n).read_bytes() != b]
            continue
        for n, b in files.items():
            p = folder / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
        for n in sorted(have - set(files)):
            (folder / n).unlink()
        print(f"wrote {arm}: {len(files)} files")
    if bad:
        print("differs from the script's output:\n  " + "\n  ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
