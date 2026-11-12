Baseline E4B (../submission_e4b) plus the edit-step LoRA `edit_v1`: the only difference is `adapter: edit_v1` in agent.yaml.

The adapter (train/train_lora.py via notebook 04: rank 32, 4,000 edit_file samples, 266 MiB) is NOT in git. It lives on Drive at
MyDrive/gemma4-adapters/edit_v1 and notebook 03 copies `adapter_config.json` and `adapter_model.safetensors` into `adapters/edit_v1/` here
before the run. To pack a submission zip, put those two files in `adapters/edit_v1/` and run `python agent/make_submission_zip.py agent/submission_e4b_ad1 --out ...`.
