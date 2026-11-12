---
name: edit_lines
description: Show numbered lines of a file and replace a range of lines by number, so an edit never needs the old text copied back exactly. Refuses an edit that would leave a Python file with a syntax error.
---
# edit_lines

Use the run_skill_script tool with skill_name set to edit_lines and file_path set to scripts/edit_lines.py.
The args object takes these fields.

Show lines, with their line numbers:
- action: show
- file: path relative to the workspace
- start, end: first and last line (at most 200 lines are printed)

Replace lines:
- action: replace
- file, start, end: the inclusive range to replace, taken from the numbers printed by show
- text: the new lines, exactly as they should appear, with their indentation (an empty string deletes the range)
- expect: optional, the current text of line start; the edit is refused if line start differs, which catches wrong numbers

To insert without removing anything, set end to start minus one; text goes in before line start.

- force: optional, true to write even when the result does not parse as Python

The reply shows the changed region with new line numbers. If the new file would not parse as Python, nothing is written and the error is printed
(files this interpreter already cannot parse, for example because of newer syntax, are not checked).
