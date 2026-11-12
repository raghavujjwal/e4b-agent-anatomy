Combined arm: prompt v2 (`../submission_e4b_v2/prompts/flat_search.md`, see its README for the five changes) plus the model's
recommended sampling (temperature 1.0, top_p 0.95, top_k 64; `../submission_e4b_rec`). Everything else is identical to the baseline
`../submission_e4b`. This is the bundle we would ship if it beats the baseline; it cannot say which of the two changes helped.
If it does not beat the baseline, run `../submission_e4b_v2` (prompt only) next, because exact-copy edits may suffer at temperature 1.0.
