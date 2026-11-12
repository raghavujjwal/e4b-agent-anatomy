Line-range editing arm. Copy of `../submission_e4b` with two changes and nothing else:
1. `edit_file` is removed from `tools:` and the skill `skills/edit_lines` is attached (`run_skill_script` is added by the harness).
   The skill shows numbered lines and replaces a line range; it refuses an edit that would leave a Python file with a syntax error
   (`skills/edit_lines/scripts/edit_lines.py`, tests in `tests/test_edit_lines.py`).
2. The paragraph "Prefer edit_file with a short, unique anchor string..." in `prompts/flat_search.md` is replaced by an "Editing" section
   that explains the skill. Everything else in the prompt, the sampling (temperature 0.2) and the budgets are the baseline's.
Why: in five E4B runs 53-60% of `edit_file` calls failed ("old_string not found") because the model rebuilds the text instead of
copying it, and no task with 4 or more failed edits was resolved (docs/MEASUREMENTS.md). The skill needs no copied text.
Edit_file is removed so that compliance is not a question; if the model cannot use the skill the arm will show it as empty patches.
Read against `base1..3` (baseline) with 3 seeds each (docs/EVAL_PROTOCOL.md).
