---
name: repo_aid
description: Two helpers. find_files ranks the Python files of the workspace against the issue text. edit_retry applies an edit_file call that failed with old_string not found, by locating the closest matching lines.
---
# repo_aid

Use the run_skill_script tool with skill_name set to repo_aid. The args object takes the fields below.

## Find the files to look at (first action)
file_path: scripts/find_files.py
- query: the issue text, copied as written (the title and the first sentences, up to about 800 characters)

The reply lists up to ten files, best match first, with the names inside each that match the issue. Read the first file that fits with read_file.

## Retry an edit that failed with "old_string not found"
file_path: scripts/edit_retry.py
- filepath, old_string, new_string: the same values as the failed edit_file call
- allow_multiple: optional, true to replace every occurrence

It looks for the one place in the file that old_string was meant to be (backslash sequences, indentation or blanks that differ, a small slip in the text), applies new_string there
with matching indentation, and prints what was replaced. A Python file is checked to still parse before it is written. When nothing is close it prints the closest lines of the file with
their numbers; copy old_string from them exactly and call edit_file again.
