"""
Data Processing Module for fishstick

Advanced data loading, augmentation, and preprocessing for:
- Image datasets
- Text datasets
- Graph datasets
- Time series
- Custom datasets
"""

from typing import Optional, Callable, List, Union, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
from pathlib import Path
import json
import pickle


class AugmentedDataset(Dataset):
    """
    Dataset with built-in augmentation support.

    Supports multiple modalities and custom augmentation pipelines.
    """

    def __init__(
        self,
        data: Union[List, np.ndarray, torch.Tensor],
        labels: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
        transform: Optional[Callable] = None,
        augment: bool = True,
    ):
        """
        Args:
            data: Input data
            labels: Target labels
            transform: Transformation function
            augment: Whether to apply augmentations
        """
        self.data = (
            data
            if isinstance(data, torch.Tensor)
            else torch.tensor(data, dtype=torch.float32)
        )
        self.labels = (
            labels
            if labels is None
            else (
                torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
            )
        )
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Get item with optional augmentation."""
        x = self.data[idx]
        y = self.labels[idx] if self.labels is not None else None

        if self.transform and self.augment:
            x = self.transform(x)

        return (x, y) if y is not None else x


class CachedDataset(Dataset):
    """
    Dataset with disk/memory caching for large datasets.

    Loads data on first access and caches for subsequent accesses.
    """

    def __init__(
        self,
        data_source: Union[str, Path, Callable],
        cache_dir: Optional[str] = None,
        cache_in_memory: bool = True,
    ):
        """
        Args:
            data_source: Path to data or callable that loads data
            cache_dir: Directory for disk cache
            cache_in_memory: Whether to cache in RAM
        """
        self.data_source = data_source
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_in_memory = cache_in_memory
        self._cache = {}
        self._data = None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self):
        """Load data from source."""
        if isinstance(self.data_source, (str, Path)):
            # Load from file
            path = Path(self.data_source)
            if path.suffix == ".npy":
                return np.load(path)
            elif path.suffix == ".pt":
                return torch.load(path)
            elif path.suffix == ".pkl":
                with open(path, "rb") as f:
                    return pickle.load(f)
            elif path.suffix == ".json":
                with open(path, "r") as f:
                    return json.load(f)
        elif callable(self.data_source):
            return self.data_source()

        raise ValueError(f"Unknown data source: {self.data_source}")

    def __getitem__(self, idx: int):
        """Get item with caching."""
        if idx in self._cache:
            return self._cache[idx]

        # Load on first access
        if self._data is None:
            self._data = self._load_data()

        item = self._data[idx]

        if self.cache_in_memory:
            self._cache[idx] = item

        return item

    def __len__(self) -> int:
        if self._data is None:
            self._data = self._load_data()
        return len(self._data)


class StreamingDataset(Dataset):
    """
    Streaming dataset for data larger than memory.

    Loads batches on-the-fly from disk or network.
    """

    def __init__(
        self,
        file_list: List[str],
        loader_fn: Callable[[str], Any],
        buffer_size: int = 100,
    ):
        """
        Args:
            file_list: List of file paths
            loader_fn: Function to load a single file
            buffer_size: Number of items to buffer in memory
        """
        self.file_list = file_list
        self.loader_fn = loader_fn
        self.buffer_size = buffer_size
        self._buffer = {}
        self._buffer_keys = []

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """Get item with streaming buffer."""
        if idx in self._buffer:
            return self._buffer[idx]

        # Load item
        item = self.loader_fn(self.file_list[idx])

        # Manage buffer
        if len(self._buffer) >= self.buffer_size:
            # Remove oldest
            oldest = self._buffer_keys.pop(0)
            del self._buffer[oldest]

        self._buffer[idx] = item
        self._buffer_keys.append(idx)

        return item


class MixUp:
    """
    MixUp augmentation for improved training.

    From: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor
    ) -> tuple:
        """Apply MixUp to two samples."""
        lam = np.random.beta(self.alpha, self.alpha)

        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2

        return x, y


class CutMix:
    """
    CutMix augmentation for images.

    From: CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor
    ) -> tuple:
        """Apply CutMix to two image samples."""
        lam = np.random.beta(self.alpha, self.alpha)

        _, h, w = x1.shape[-3:]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x = x1.clone()
        x[..., bby1:bby2, bbx1:bbx2] = x2[..., bby1:bby2, bbx1:bbx2]

        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        y = lam * y1 + (1 - lam) * y2

        return x, y


class DataLoader(TorchDataLoader):
    """
    Enhanced DataLoader with fishstick-specific features.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Args:
            dataset: Dataset to load
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            pin_memory: Pin memory for GPU transfer
            prefetch_factor: Prefetch batches per worker
            collate_fn: Custom collate function
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=collate_fn,
            **kwargs,
        )


class DataSchema:
    """
    Schema validation for datasets.

    Ensures data conforms to expected structure and types.
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Args:
            schema: Dictionary defining expected structure
                   e.g., {"image": "tensor[3,224,224]", "label": "int"}
        """
        self.schema = schema

    def validate(self, sample: Dict[str, Any]) -> bool:
        """Validate a single sample against schema."""
        for key, expected_type in self.schema.items():
            if key not in sample:
                raise ValueError(f"Missing key: {key}")

            # Basic type checking
            if expected_type == "int" and not isinstance(
                sample[key], (int, np.integer)
            ):
                raise TypeError(f"{key} should be int, got {type(sample[key])}")
            elif expected_type == "float" and not isinstance(
                sample[key], (float, np.floating)
            ):
                raise TypeError(f"{key} should be float, got {type(sample[key])}")
            elif expected_type.startswith("tensor"):
                if not isinstance(sample[key], torch.Tensor):
                    raise TypeError(f"{key} should be tensor, got {type(sample[key])}")

        return True


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Create an optimized DataLoader.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        **kwargs: Additional arguments

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs,
    )
