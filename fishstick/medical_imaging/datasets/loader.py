"""
Medical Image Data Loaders

DataLoader creation and train/validation splitting utilities.
"""

from typing import Optional, List, Tuple, Any, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> DataLoader:
    """Create a DataLoader for medical images.

    Args:
        dataset: Dataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
        sampler: Custom sampler

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
    )


def create_train_val_split(
    dataset: torch.utils.data.Dataset,
    val_ratio: float = 0.2,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[Subset, Subset]:
    """Split dataset into training and validation sets.

    Args:
        dataset: Dataset to split
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting

    Returns:
        Tuple of (train_subset, val_subset)
    """
    if seed is not None:
        torch.manual_seed(seed)
        import random

        random.seed(seed)
        import numpy as np

        np.random.seed(seed)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed) if seed else None,
    )


def create_stratified_split(
    dataset: torch.utils.data.Dataset,
    labels: List[int],
    val_ratio: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Create stratified train/val split maintaining label distribution.

    Args:
        dataset: Dataset
        labels: List of labels for stratification
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices)
    """
    from collections import defaultdict

    if seed is not None:
        torch.manual_seed(seed)

    label_to_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label, indices in label_to_indices.items():
        n_val = int(len(indices) * val_ratio)

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return train_indices, val_indices


def create_k_fold_splits(
    dataset: torch.utils.data.Dataset,
    k: int = 5,
    seed: Optional[int] = None,
) -> List[Tuple[Subset, Subset]]:
    """Create k-fold cross-validation splits.

    Args:
        dataset: Dataset
        k: Number of folds
        seed: Random seed

    Returns:
        List of (train_subset, val_subset) tuples
    """
    if seed is not None:
        torch.manual_seed(seed)

    total_size = len(dataset)
    fold_size = total_size // k

    splits = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else total_size

        val_indices = list(range(start, end))
        train_indices = list(range(0, start)) + list(range(end, total_size))

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        splits.append((train_subset, val_subset))

    return splits
