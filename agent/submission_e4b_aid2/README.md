Aid2 arm. Copy of `../submission_e4b_aid` (the baseline plus the repo_aid skill) with the three fixes that the `aid1` run pointed to (docs/MEASUREMENTS.md, section `aid1`):
1. `edit_retry.py`: when the file would not parse after the edit and new_string holds backslash sequences (backslash-n, backslash-quote), the edit is tried again with them as real characters
   (the parse guard stays); the similarity search also tries the unescaped old_string. In aid1, 7 of 12 retry calls were refused because new_string held such sequences, 4 found no place. Replayed on those 12 calls
   against the base-commit files, 10 now apply (1 before). Applying is not the same as being right.
2. Prompt: `find_files` query "word for word, do not summarize" (aid1: the median query was 70 characters although up to 800 were asked for), and the next action after the list is read_file on the first listed file
   (aid1: the first read_file went to rank 1 of the list in 28 of 74 runs, to a file outside the list in 18; of 9 short single-file runs with the gold file listed it was opened in 4).
3. Prompt: after "old_string not found" the next action is edit_retry (aid1: followed in 15 of 56 cases).
Everything else is the baseline's: temperature 0.2, budgets, tools (edit_file stays declared).

Pre-registered rules for `aid2_1` (one seed, 66 dev tasks, `fitted`; same as aid1 so the two compare; aid1: 7 resolved, find_files 83%, gold opened 60% / 26.7% on short issues, retry followed 26.8%, retry applied 8.3%):
- Compliance: `find_files` called in at least 80% of the runs, in the first 3 calls.
- Localization, single-file tasks: gold file opened in at least 65% of runs, and at least 40% on issues under 300 characters.
- Edit retry: after "old_string not found", `edit_retry` within the next 2 calls in at least 50% of the cases, and at least 40% of its calls print "OK: edit applied".
- Outcome: 10 or more of 66 resolved to run two more seeds; fewer: stop the aid line and write the result up.
- Collapse guard: interrupt if more than 8 of the first 20 tasks end in `tool_budget`, or more than 12 of the first 25 end with an empty patch.
