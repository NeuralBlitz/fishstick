"""
Streaming Data Handling Module for fishstick

Provides streaming data handling utilities for processing large datasets
that don't fit in memory, including buffering, rate limiting, checkpointing,
and transformation capabilities.

Features:
- Infinite streaming data loaders
- Buffered streaming iterators
- Rate-limited streaming
- Checkpointed streams with recovery
- Transform streaming data
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
    Iterable,
    TypeVar,
    Generic,
)
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import queue
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class StreamState(Enum):
    """Stream processing state."""

    IDLE = "idle"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class StreamStats:
    """Statistics for stream processing."""

    items_processed: int = 0
    items_yielded: int = 0
    errors: int = 0
    total_time: float = 0.0
    last_item_time: float = 0.0
    state: StreamState = StreamState.IDLE


class StreamDataLoader(IterableDataset[T_co]):
    """
    Infinite streaming data loader.

    Provides infinite iteration over a data source with configurable
    buffering and prefetching.
    """

    def __init__(
        self,
        data_source: Iterable[T_co],
        buffer_size: int = 1000,
        prefetch_factor: int = 2,
        infinite: bool = True,
        shuffle_buffer: bool = True,
    ):
        """
        Args:
            data_source: Source iterable
            buffer_size: Size of shuffle buffer
            prefetch_factor: Number of buffers to prefetch
            infinite: Whether to loop infinitely
            shuffle_buffer: Whether to shuffle buffer
        """
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        self.infinite = infinite
        self.shuffle_buffer = shuffle_buffer

        self._stats = StreamStats()
        self._lock = threading.Lock()

    def __iter__(self) -> Iterator[T_co]:
        buffer: List[T_co] = []
        source_iter = iter(self.data_source)

        while True:
            try:
                for _ in range(self.buffer_size):
                    try:
                        item = next(source_iter)
                        buffer.append(item)
                        self._stats.items_processed += 1
                    except StopIteration:
                        if not self.infinite:
                            break
                        source_iter = iter(self.data_source)
                        break

                if not buffer:
                    break

                if self.shuffle_buffer:
                    np.random.shuffle(buffer)

                for item in buffer:
                    yield item
                    self._stats.items_yielded += 1

                buffer.clear()

            except Exception as e:
                self._stats.errors += 1
                if not self.infinite:
                    raise

    @property
    def stats(self) -> StreamStats:
        """Get stream statistics."""
        return self._stats


class BufferedIterator(Generic[T]):
    """
    Buffered streaming iterator.

    Provides buffering and lookahead capabilities for iterators.
    """

    def __init__(
        self,
        source: Iterable[T],
        buffer_size: int = 100,
        preload: bool = True,
    ):
        """
        Args:
            source: Source iterator
            buffer_size: Size of lookahead buffer
            preload: Whether to preload buffer on init
        """
        self.source_iter = iter(source)
        self.buffer_size = buffer_size
        self._buffer: List[T] = []
        self._exhausted = False
        self._lock = threading.Lock()

        if preload:
            self._fill_buffer()

    def _fill_buffer(self) -> None:
        """Fill buffer from source."""
        try:
            while len(self._buffer) < self.buffer_size:
                item = next(self.source_iter)
                self._buffer.append(item)
        except StopIteration:
            self._exhausted = True

    def peek(self) -> Optional[T]:
        """Peek at next item without consuming."""
        if not self._buffer:
            return None
        return self._buffer[0]

    def peek_n(self, n: int) -> List[T]:
        """Peek at next n items."""
        return self._buffer[:n]

    def __iter__(self) -> "BufferedIterator[T]":
        return self

    def __next__(self) -> T:
        if not self._buffer:
            if self._exhausted:
                raise StopIteration
            self._fill_buffer()

        if not self._buffer:
            raise StopIteration

        item = self._buffer.pop(0)

        if len(self._buffer) < self.buffer_size // 2 and not self._exhausted:
            self._fill_buffer()

        return item

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_exhausted(self) -> bool:
        """Check if source is exhausted."""
        return self._exhausted and len(self._buffer) == 0


class RateLimitedStream(Generic[T]):
    """
    Rate-limited streaming iterator.

    Controls the rate of item production to match target throughput.
    """

    def __init__(
        self,
        source: Iterable[T],
        max_rate: float = 100.0,
        burst_size: int = 10,
    ):
        """
        Args:
            source: Source iterator
            max_rate: Maximum items per second
            burst_size: Maximum burst size
        """
        self.source_iter = iter(source)
        self.max_rate = max_rate
        self.burst_size = burst_size
        self._min_interval = 1.0 / max_rate if max_rate > 0 else 0
        self._last_time = 0.0
        self._burst_remaining = burst_size

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        current_time = time.time()
        elapsed = current_time - self._last_time

        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            if self._burst_remaining <= 0:
                time.sleep(sleep_time)
                self._burst_remaining = self.burst_size

        if self._burst_remaining > 0:
            self._burst_remaining -= 1

        self._last_time = time.time()
        return next(self.source_iter)


class CheckpointedStream(Generic[T]):
    """
    Streaming iterator with checkpointing.

    Supports pausing and resuming from checkpoints.
    """

    def __init__(
        self,
        source: Iterable[T],
        checkpoint_dir: str = ".checkpoints",
        checkpoint_interval: int = 1000,
        restore: bool = True,
    ):
        """
        Args:
            source: Source iterator
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Items between checkpoints
            restore: Whether to restore from checkpoint
        """
        self.source_iter = iter(source)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self._checkpoint_file = self.checkpoint_dir / "stream_checkpoint.pkl"
        self._count = 0
        self._state: Dict[str, Any] = {}

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if restore and self._checkpoint_file.exists():
            self._load_checkpoint()

    def _save_checkpoint(self) -> None:
        """Save current state to checkpoint."""
        self._state = {
            "count": self._count,
            "source_state": getattr(self.source_iter, "__dict__", {}),
        }
        with open(self._checkpoint_file, "wb") as f:
            pickle.dump(self._state, f)

    def _load_checkpoint(self) -> None:
        """Load state from checkpoint."""
        try:
            with open(self._checkpoint_file, "rb") as f:
                self._state = pickle.load(f)
            self._count = self._state.get("count", 0)
        except Exception:
            pass

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        item = next(self.source_iter)
        self._count += 1

        if self._count % self.checkpoint_interval == 0:
            self._save_checkpoint()

        return item

    @property
    def progress(self) -> int:
        """Get number of items processed."""
        return self._count

    def clear_checkpoint(self) -> None:
        """Remove checkpoint file."""
        if self._checkpoint_file.exists():
            self._checkpoint_file.unlink()


class TransformStream(Generic[T, T_co]):
    """
    Streaming iterator with transformations.

    Applies transformations to streaming data without materializing.
    """

    def __init__(
        self,
        source: Iterable[T],
        transform: Optional[Callable[[T], T_co]] = None,
        filter_fn: Optional[Callable[[T], bool]] = None,
    ):
        """
        Args:
            source: Source iterator
            transform: Transform function
            filter_fn: Filter function (return True to keep)
        """
        self.source_iter = iter(source)
        self.transform = transform
        self.filter_fn = filter_fn

    def __iter__(self) -> Iterator[T_co]:
        for item in self.source_iter:
            if self.filter_fn is not None and not self.filter_fn(item):
                continue

            if self.transform is not None:
                yield self.transform(item)
            else:
                yield item

    def map(self, fn: Callable[[T_co], Any]) -> "TransformStream[T, Any]":
        """Add a transformation."""
        original_transform = self.transform

        def new_transform(x: T) -> Any:
            if original_transform:
                x = original_transform(x)
            return fn(x)

        return TransformStream(
            source=self.source_iter,
            transform=new_transform,
            filter_fn=self.filter_fn,
        )

    def filter(self, fn: Callable[[T_co], bool]) -> "TransformStream[T, T_co]":
        """Add a filter."""
        original_filter = self.filter_fn

        def new_filter(x: T) -> bool:
            if original_filter and not original_filter(x):
                return False
            return fn(x)  # type: ignore

        return TransformStream(
            source=self.source_iter,
            transform=self.transform,
            filter_fn=new_filter,
        )


class ChunkedStream(Generic[T]):
    """
    Streaming iterator that yields chunks.

    Groups consecutive items into chunks.
    """

    def __init__(
        self,
        source: Iterable[T],
        chunk_size: int = 32,
        drop_last: bool = False,
    ):
        """
        Args:
            source: Source iterator
            chunk_size: Size of each chunk
            drop_last: Whether to drop incomplete last chunk
        """
        self.source_iter = iter(source)
        self.chunk_size = chunk_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[T]]:
        chunk: List[T] = []

        for item in self.source_iter:
            chunk.append(item)

            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []

        if chunk and not self.drop_last:
            yield chunk


class BatchStream(Generic[T]):
    """
    Stream that batches items on-the-fly.

    Similar to ChunkedStream but yields tensors when possible.
    """

    def __init__(
        self,
        source: Iterable[T],
        batch_size: int = 32,
        drop_last: bool = False,
        collate_fn: Optional[Callable[[List[T]], Any]] = None,
    ):
        """
        Args:
            source: Source iterator
            batch_size: Batch size
            drop_last: Whether to drop incomplete last batch
            collate_fn: Custom collate function
        """
        self.source_iter = iter(source)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate

    def __iter__(self) -> Iterator[Any]:
        batch: List[T] = []

        for item in self.source_iter:
            batch.append(item)

            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        if batch and not self.drop_last:
            yield self.collate_fn(batch)


class WindowedStream(Generic[T]):
    """
    Streaming iterator with sliding window.

    Provides sliding window view of streaming data.
    """

    def __init__(
        self,
        source: Iterable[T],
        window_size: int = 5,
        step: int = 1,
    ):
        """
        Args:
            source: Source iterator
            window_size: Size of sliding window
            step: Step size between windows
        """
        self.source_iter = iter(source)
        self.window_size = window_size
        self.step = step
        self._buffer: List[T] = []
        self._index = 0

    def __iter__(self) -> Iterator[List[T]]:
        for item in self.source_iter:
            self._buffer.append(item)

            if len(self._buffer) >= self.window_size:
                yield self._buffer[: self.window_size]
                self._buffer = self._buffer[self.step :]


class MergedStream(Generic[T]):
    """
    Merge multiple streams.

    Combines multiple iterators into a single stream.
    """

    def __init__(
        self,
        sources: List[Iterable[T]],
        weights: Optional[List[float]] = None,
        mode: str = "round_robin",
    ):
        """
        Args:
            sources: List of source iterables
            weights: Weights for random selection
            mode: Merge mode ('round_robin', 'random', 'sequential')
        """
        self.sources = [iter(s) for s in sources]
        self.weights = weights
        self.mode = mode
        self._round_robin_idx = 0

        if weights:
            self.weights = [w / sum(weights) for w in weights]

    def __iter__(self) -> Iterator[T]:
        if self.mode == "round_robin":
            return self._round_robin_iter()
        elif self.mode == "random":
            return self._random_iter()
        else:
            return self._sequential_iter()

    def _round_robin_iter(self) -> Iterator[T]:
        while self.sources:
            active = []
            for i, it in enumerate(self.sources):
                try:
                    yield next(it)
                    active.append(i)
                except StopIteration:
                    pass

            if not active:
                break

            self._round_robin_idx = (self._round_robin_idx + 1) % len(active)

    def _random_iter(self) -> Iterator[T]:
        import random

        active = list(range(len(self.sources)))

        while active:
            idx = random.choices(active, weights=[self.weights[i] for i in active])[0]
            try:
                yield next(self.sources[idx])
            except StopIteration:
                active.remove(idx)

    def _sequential_iter(self) -> Iterator[T]:
        for source in self.sources:
            yield from source


class AsyncStream(Generic[T]):
    """
    Async wrapper for iterables.

    Provides background prefetching for iterators.
    """

    def __init__(
        self,
        source: Iterable[T],
        buffer_size: int = 10,
    ):
        """
        Args:
            source: Source iterable
            buffer_size: Size of async buffer
        """
        self.source = source
        self.buffer_size = buffer_size
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._done = False

    def _producer(self) -> None:
        """Background producer."""
        try:
            for item in self.source:
                self._queue.put(item)
        except Exception as e:
            self._queue.put(e)
        finally:
            self._queue.put(None)

    def __iter__(self) -> Iterator[T]:
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

        while True:
            item = self._queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        if self._thread:
            self._thread.join()


def create_stream_from_files(
    file_paths: List[str],
    loader_fn: Callable[[str], T],
    buffer_size: int = 100,
) -> StreamDataLoader[T]:
    """
    Create a streaming data loader from file paths.

    Args:
        file_paths: List of file paths
        loader_fn: Function to load each file
        buffer_size: Buffer size

    Returns:
        StreamDataLoader
    """

    def file_iterator():
        for path in file_paths:
            yield loader_fn(path)

    return StreamDataLoader(
        data_source=list(file_iterator()),
        buffer_size=buffer_size,
    )


def stream_batches(
    data: Iterable[T],
    batch_size: int = 32,
    drop_last: bool = False,
) -> Iterator[List[T]]:
    """
    Create batches from a stream.

    Args:
        data: Input data stream
        batch_size: Batch size
        drop_last: Whether to drop incomplete last batch

    Yields:
        Batches of data
    """
    batch: List[T] = []

    for item in data:
        batch.append(item)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch and not drop_last:
        yield batch
