"""
Class-aware sampler for zero-shot learning (Ba et al. ICCV 2015).
Ensures each mini-batch contains diverse classes for better training.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler


class ClassAwareSampler(Sampler):
    """
    Sampler that ensures each batch contains diverse classes.

    Strategy: group samples by class, then sample from multiple classes per batch.
    This helps the model learn to distinguish between many classes during training,
    which is important for the paper's mini-batch training strategy.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 200,
        classes_per_batch: int = 50,
        seed: int = 42,
    ):
        """
        Args:
            dataset: ZeroShotDataset or similar with 'class_id' field
            batch_size: total samples per batch (paper: 200)
            classes_per_batch: target number of unique classes per batch
            seed: random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.classes_per_batch = min(classes_per_batch, len(dataset.seen_classes))
        self.seed = seed
        self.rng = random.Random(seed)

        # Group indices by class - use dataset.samples directly if available (much faster!)
        self.class_to_indices = {}
        if hasattr(dataset, 'samples'):
            # ZeroShotDataset has samples as [(path, class_id), ...]
            for idx, (_, class_id) in enumerate(dataset.samples):
                if class_id not in self.class_to_indices:
                    self.class_to_indices[class_id] = []
                self.class_to_indices[class_id].append(idx)
        else:
            # Fallback: iterate through dataset (slower)
            for idx in range(len(dataset)):
                sample = dataset[idx]
                class_id = sample['class_id']
                if class_id not in self.class_to_indices:
                    self.class_to_indices[class_id] = []
                self.class_to_indices[class_id].append(idx)

        # Filter to only seen classes (for train mode)
        if hasattr(dataset, 'seen_classes'):
            self.valid_classes = [c for c in dataset.seen_classes if c in self.class_to_indices]
        else:
            self.valid_classes = list(self.class_to_indices.keys())

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yield batches of indices with class diversity.

        Efficient strategy: pre-compute all batches once using round-robin.
        """
        # Shuffle indices within each class pool (no dataset access needed)
        pools = []
        for cls in self.valid_classes:
            indices = self.class_to_indices[cls].copy()
            self.rng.shuffle(indices)
            pools.append(indices)

        # Pre-compute all batches using round-robin (much faster than dynamic iterator)
        batches = self._round_robin_batches(pools)

        for batch in batches:
            yield batch

    def _round_robin_batches(self, pools: list[list[int]]) -> list[list[int]]:
        """
        Pre-compute batches using round-robin sampling across classes.
        This is much more efficient than dynamic iteration with StopIteration.
        """
        batches = []
        batch = []

        # Track current position in each pool
        positions = [0 for _ in pools]
        pool_order = list(range(len(pools)))

        # Keep sampling until all pools are exhausted
        while any(pos < len(pool) for pos, pool in zip(positions, pools)):
            for pool_idx in pool_order:
                if positions[pool_idx] < len(pools[pool_idx]):
                    batch.append(pools[pool_idx][positions[pool_idx]])
                    positions[pool_idx] += 1

                    if len(batch) >= self.batch_size:
                        batches.append(batch)
                        batch = []

        # Add remaining samples as final batch
        if batch:
            batches.append(batch)

        return batches

    def __len__(self) -> int:
        # Exact number of batches per epoch
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
