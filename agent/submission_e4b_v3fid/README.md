Edit-fidelity arm. Copy of `../submission_e4b_v2rec` with ONE change: the "Making an edit" section of `prompts/flat_search.md`. Sampling, tools and budgets are v2rec's.
Why (docs/MEASUREMENTS.md, "Where the baseline runs lose tasks"): 56% of 1,077 edit_file calls failed; the failure rate was 96% when old_string contained a
backslash (against 46%), 83% with triple quotes (against 54%), 75% when longer than 400 characters (against 53%); 135 calls repeated the previous failed call
exactly; 78 of 454 "not found" failures were escape sequences copied from the JSON-encoded read_file output. v2rec's rules ("copy exactly", "reread after a
failure") already existed and were not enough. The new section asks for a short fragment of a line (15-100 characters) as old_string, tells the model that
read_file output is JSON (write real line breaks and quotes), says to anchor next to lines with backslashes or triple quotes, and forbids resending a failed old_string.
Mechanism check (less noisy than the resolved count, from `mine/trace_analysis.py`): edit_file failure rate (v2rec/base 56%), share of calls with backslash or
triple quote in old_string, identical repeats (135 in 1,077 calls). Decision rule fixed before the run: the arm did its job if the failure rate falls to 40% or lower.
Resolved counts are read against `base1-3` (7, 7, 8) and `v2rec1-3` (12, 7, 9) of 66, noise about 3-4 tasks per run.
