E4B baseline with the model's recommended sampling (temperature 1.0, top_p 0.95, top_k 64, from the published
generation_config.json of google/gemma-4-E4B-it). Identical to `../submission_e4b` in every other file, so a difference
between the two runs is the sampling plus run-to-run noise (the baseline moved requests from 4/4 to 1/5 between two identical runs).
Same known prompt mismatch as the baseline (flat_search.md mentions a candidate list this arm never provides).
