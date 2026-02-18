"""
Data Loaders and Datasets Module for fishstick

Provides advanced data loading strategies including lazy loading, mapping,
chaining, and stateful iterators for efficient memory management.

Features:
- Lazy loading for large datasets
- Mapped and chained datasets
- Stateful iteration support
- Memory-efficient data handling
"""

from __future__ import annotations

from typing import (
    Optional,
    Callable,
    List,
    Union,
    Dict,
    Any,
    Tuple,
    Iterator,
    Generic,
    TypeVar,
    Sequence,
    Mapping,
    Iterable,
)
from pathlib import Path
import threading
import pickle
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
K = TypeVar("K")
V = TypeVar("V")


class LazyDataset(Dataset[T_co]):
    """
    Memory-efficient lazy loading dataset.

    Loads data on-demand rather than storing in memory, ideal for
    large datasets that don't fit in RAM.
    """

    def __init__(
        self,
        file_paths: List[str],
        loader_fn: Callable[[str], T_co],
        cache_size: int = 1000,
        preload: bool = False,
    ):
        """
        Args:
            file_paths: List of file paths to load
            loader_fn: Function to load a single item from path
            cache_size: Maximum number of items to cache
            preload: Whether to preload all data on initialization
        """
        self.file_paths = file_paths
        self.loader_fn = loader_fn
        self.cache_size = cache_size
        self._cache: Dict[int, T_co] = {}
        self._cache_order: List[int] = []
        self._lock = threading.Lock()

        if preload:
            self._preload_all()

    def _preload_all(self) -> None:
        """Preload all data into cache."""
        for i in range(len(self)):
            _ = self[i]

    def _add_to_cache(self, idx: int, item: T_co) -> None:
        """Thread-safe cache insertion with LRU eviction."""
        with self._lock:
            if len(self._cache) >= self.cache_size and idx not in self._cache:
                oldest_idx = self._cache_order.pop(0)
                del self._cache[oldest_idx]
            self._cache[idx] = item
            if idx not in self._cache_order:
                self._cache_order.append(idx)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> T_co:
        if idx in self._cache:
            return self._cache[idx]

        item = self.loader_fn(self.file_paths[idx])
        self._add_to_cache(idx, item)
        return item


class MappedDataset(Dataset[T_co], Generic[K, V]):
    """
    Key-value mapped dataset with on-the-fly transformation.

    Wraps a dataset and applies a mapping function to both keys and values.
    """

    def __init__(
        self,
        data: Mapping[K, V],
        key_transform: Optional[Callable[[K], K]] = None,
        value_transform: Optional[Callable[[V], T_co]] = None,
        keys: Optional[List[K]] = None,
    ):
        """
        Args:
            data: Input data mapping
            key_transform: Transform applied to keys
            value_transform: Transform applied to values
            keys: Specific keys to use (order matters)
        """
        self.data = dict(data)
        self.key_transform = key_transform
        self.value_transform = value_transform
        self.keys = keys or list(self.data.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[K, T_co]:
        key = self.keys[idx]
        if self.key_transform:
            key = self.key_transform(key)

        value = self.data[key]
        if self.value_transform:
            value = self.value_transform(value)

        return key, value


class ConcatDataset(Dataset[T_co]):
    """
    Smart concatenation of multiple datasets.

    Provides unified interface for combining datasets with optional
    offset tracking and label mapping.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset[T_co]],
        label_offset: bool = True,
        cumulative_sizes: Optional[List[int]] = None,
    ):
        """
        Args:
            datasets: Sequence of datasets to concatenate
            label_offset: Whether to offset labels between datasets
            cumulative_sizes: Pre-computed cumulative sizes
        """
        self.datasets = datasets
        self.label_offset = label_offset

        if cumulative_sizes is not None:
            self.cumulative_sizes = cumulative_sizes
        else:
            self.cumulative_sizes = self._get_cumulative_sizes()

    def _get_cumulative_sizes(self) -> List[int]:
        sizes = [len(d) for d in self.datasets]
        cumsum = [0]
        for s in sizes:
            cumsum.append(cumsum[-1] + s)
        return cumsum

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> T_co:
        dataset_idx, sample_idx = self._get_dataset_index(idx)

        item = self.datasets[dataset_idx][sample_idx]

        if self.label_offset and isinstance(item, tuple) and len(item) >= 2:
            data, label = item[0], item[1]
            offset = self.cumulative_sizes[dataset_idx]
            if isinstance(label, (int, np.integer)):
                label = int(label) + offset
            return data, label

        return item

    def _get_dataset_index(self, idx: int) -> Tuple[int, int]:
        for i, size in enumerate(self.cumulative_sizes[1:], 0):
            if idx < size:
                return i, idx - self.cumulative_sizes[i]
        raise IndexError(f"Index {idx} out of range")


class ChainDataset(IterableDataset[T_co]):
    """
    Chain multiple iterables into a single streaming dataset.

    Ideal for combining data from multiple sources without loading
    everything into memory.
    """

    def __init__(
        self,
        iterables: Sequence[Iterable[T_co]],
        cycle: bool = False,
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            iterables: Sequence of iterables to chain
            cycle: Whether to cycle through iterables
            weights: Sampling weights for each iterable
        """
        self.iterables = iterables
        self.cycle = cycle
        self.weights = weights

        if weights is not None:
            self.weights = [w / sum(weights) for w in weights]

    def __iter__(self) -> Iterator[T_co]:
        if self.weights is not None:
            while True:
                for i, it in enumerate(self.iterables):
                    if np.random.random() < self.weights[i]:
                        yield from it

                if not self.cycle:
                    break
        else:
            for iterable in self.iterables:
                yield from iterable
                if not self.cycle:
                    continue


class ShuffleDataset(Dataset[T_co]):
    """
    On-the-fly shuffling wrapper for datasets.

    Maintains a buffer of samples and shuffles dynamically.
    """

    def __init__(
        self,
        dataset: Dataset[T_co],
        buffer_size: int = 10000,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dataset: Base dataset to wrap
            buffer_size: Size of shuffle buffer
            shuffle: Whether to shuffle
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.buffer_size = min(buffer_size, len(dataset))
        self.shuffle = shuffle
        self.seed = seed

        self._indices = list(range(len(dataset)))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self._indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> T_co:
        if self.shuffle:
            buffer_start = max(0, idx - self.buffer_size // 2)
            buffer_end = min(len(self), buffer_start + self.buffer_size)

            if buffer_end - buffer_start < self.buffer_size:
                buffer_start = max(0, buffer_end - self.buffer_size)

            swap_idx = np.random.randint(buffer_start, buffer_end)
            idx = swap_idx

        return self.dataset[self._indices[idx]]


class StatefulDataLoader(DataLoader[T_co]):
    """
    DataLoader that remembers iteration state.

    Enables resuming from where iteration left off, useful for
    checkpointing and distributed training.
    """

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            **kwargs,
        )
        self._state: Dict[str, Any] = {
            "epoch": 0,
            "batch_idx": 0,
            "iteration_order": [],
        }

    @property
    def state(self) -> Dict[str, Any]:
        """Get current iteration state."""
        return self._state.copy()

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """Restore iteration state."""
        self._state = value.copy()

    def get_state_dict(self) -> Dict[str, Any]:
        """Get serializable state dict."""
        return {
            "epoch": self._state["epoch"],
            "batch_idx": self._state["batch_idx"],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self._state.update(state_dict)


class TensorDataset(Dataset[T_co]):
    """
    Enhanced tensor dataset with automatic type handling.

    Supports mixed-type samples and automatic collation.
    """

    def __init__(
        self,
        tensors: Union[torch.Tensor, List[torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Args:
            tensors: Single tensor or list of tensors
            labels: Optional label tensor
            transform: Optional transformation
        """
        if isinstance(tensors, torch.Tensor):
            self.tensors = [tensors]
        else:
            self.tensors = tensors

        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        items = [t[idx] for t in self.tensors]

        if self.transform:
            items = [self.transform(item) for item in items]

        if self.labels is not None:
            return tuple(items), self.labels[idx]

        return tuple(items)


class MemoryMappedDataset(Dataset[T_co]):
    """
    Memory-mapped dataset for large files.

    Uses numpy memmap for zero-copy loading of large datasets.
    """

    def __init__(
        self,
        file_path: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        offset: int = 0,
        label_file: Optional[str] = None,
    ):
        """
        Args:
            file_path: Path to memory-mapped file
            shape: Expected shape of data
            dtype: Data type
            offset: Byte offset to data start
            label_file: Optional path to labels
        """
        self.file_path = file_path
        self.shape = shape
        self.dtype = dtype
        self.offset = offset

        self.data = np.memmap(
            file_path, dtype=dtype, mode="r", offset=offset, shape=shape
        )

        if label_file and Path(label_file).exists():
            self.labels = np.load(label_file)
        else:
            self.labels = None

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        item = self.data[idx].copy()

        if self.labels is not None:
            return item, self.labels[idx]
        return item


def create_lazy_dataloader(
    file_paths: List[str],
    loader_fn: Callable[[str], Any],
    batch_size: int = 32,
    cache_size: int = 1000,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader with lazy loading.

    Args:
        file_paths: List of file paths
        loader_fn: Function to load each file
        batch_size: Batch size
        cache_size: Cache size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        **kwargs: Additional DataLoader args

    Returns:
        Configured DataLoader
    """
    dataset = LazyDataset(file_paths, loader_fn, cache_size=cache_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs,
    )


def create_streaming_dataloader(
    data_iterable: Iterable[Any],
    batch_size: int = 32,
    collate_fn: Optional[Callable] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader from an iterable.

    Args:
        data_iterable: Iterable of data
        batch_size: Batch size
        collate_fn: Custom collate function
        **kwargs: Additional DataLoader args

    Returns:
        Configured DataLoader
    """
    if collate_fn is None:
        collate_fn = default_collate

    return DataLoader(
        data_iterable,
        batch_size=batch_size,
        collate_fn=collate_fn,
        **kwargs,
    )
