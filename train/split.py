"""Which repositories are held out for validation (plan S2.6).

One rule for the data card (train/s2_card.py), the trainer (train/train_lora.py, validation loss) and the offline gates (G6), so they always agree.
The split is by repository, never by sample: a repository in validation has no sample in training, so the validation loss measures code the adapter
has not seen. The decision is a hash of the repository name, so it does not change when the data set is rebuilt or grows.
"""
from __future__ import annotations

import hashlib


def is_validation(repo: str, share: float = 0.05) -> bool:
    h = int(hashlib.sha256(("val/" + repo.lower()).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < share
