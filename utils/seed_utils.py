"""
Reproducibility: set global and per-worker seeds so multiple runs yield the same results.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set all relevant RNG seeds for reproducible runs.

    Args:
        seed: Random seed (e.g. 42).
        deterministic: If True, use deterministic cuDNN (slower, exact GPU reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # When deterministic=False we keep cudnn.benchmark=True for speed (default in train.py)


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """
    Use as DataLoader(..., worker_init_fn=lambda wid: seed_utils.worker_init_fn(wid, seed))
    so each worker has a deterministic but distinct stream when num_workers > 0.
    """
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)
