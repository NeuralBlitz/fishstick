"""
Time Series Data Utilities.

Provides data loading, preprocessing, and utilities:
- Sliding window dataset
- Temporal embeddings
- Padding and masking
- Data loaders for forecasting

Example:
    >>> from fishstick.timeseries_forecast import (
    ...     TimeSeriesDataset,
    ...     SlidingWindowDataset,
    ...     TemporalEmbedding,
    ...     TimeSeriesDataLoader,
    ... )
"""

from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting.

    Args:
        data: Time series data [N, D] or [N,]
        seq_len: Input sequence length
        pred_len: Prediction horizon length
        stride: Stride between windows
        transform: Optional transform

    Example:
        >>> data = torch.randn(1000, 7)
        >>> dataset = TimeSeriesDataset(data, seq_len=96, pred_len=24)
        >>> loader = DataLoader(dataset, batch_size=32)
    """

    def __init__(
        self,
        data: Tensor,
        seq_len: int = 96,
        pred_len: int = 24,
        stride: int = 1,
        transform: Optional[Callable] = None,
    ):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.transform = transform

        self.n_samples = max(0, (len(data) - seq_len - pred_len) // stride + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            (input, target) tuple
        """
        start = idx * self.stride

        x = self.data[start : start + self.seq_len]
        y = self.data[start + self.seq_len : start + self.seq_len + self.pred_len]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


class SlidingWindowDataset(Dataset):
    """Sliding window dataset for time series.

    Args:
        data: Time series data [N, D]
        window_size: Size of each window
        step: Step size between windows
        mode: 'train', 'val', or 'test'
        train_ratio: Ratio for train split
        val_ratio: Ratio for validation split

    Example:
        >>> data = torch.randn(1000, 7)
        >>> dataset = SlidingWindowDataset(data, window_size=120)
    """

    def __init__(
        self,
        data: Tensor,
        window_size: int = 100,
        step: int = 1,
        mode: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        self.data = data
        self.window_size = window_size
        self.step = step

        n = len(data)

        if mode == "train":
            self.start = 0
            self.end = int(n * train_ratio)
        elif mode == "val":
            self.start = int(n * train_ratio)
            self.end = int(n * (train_ratio + val_ratio))
        else:
            self.start = int(n * (train_ratio + val_ratio))
            self.end = n

        self.n_samples = max(0, (self.end - self.start - window_size) // step + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tensor:
        start = self.start + idx * self.step
        return self.data[start : start + self.window_size]


class TemporalEmbedding(nn.Module):
    """Temporal embedding layer for time series.

    Encodes time information (hour, day, month, etc.) as embeddings.

    Args:
        d_model: Model dimension
        time_features: List of time features to encode

    Example:
        >>> embed = TemporalEmbedding(d_model=256, time_features=['hour', 'day'])
        >>> time = {'hour': torch.tensor([12]), 'day': torch.tensor([1])}
        >>> embedded = embed(time)
    """

    def __init__(
        self,
        d_model: int,
        time_features: Optional[List[str]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.time_features = time_features or ["hour", "day", "weekday", "month"]

        self.embeddings = nn.ModuleDict()

        feature_dims = {
            "hour": 24,
            "day": 31,
            "weekday": 7,
            "month": 12,
            "minute": 60,
            "second": 60,
        }

        for feat in self.time_features:
            if feat in feature_dims:
                self.embeddings[feat] = nn.Embedding(
                    feature_dims[feat],
                    d_model // len(self.time_features),
                )

        remaining = d_model - sum(
            d_model // len(self.time_features) for feat in self.time_features
        )
        if remaining > 0:
            self.embeddings["remaining"] = nn.Linear(1, remaining)

    def forward(self, time_dict: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Args:
            time_dict: Dictionary of time features

        Returns:
            Temporal embeddings [B, D]
        """
        embeddings = []

        for feat in self.time_features:
            if feat in time_dict and feat in self.embeddings:
                emb = self.embeddings[feat](time_dict[feat])
                embeddings.append(emb)

        if "remaining" in self.embeddings:
            remaining = torch.zeros(
                embeddings[0].shape[0],
                self.d_model - sum(e.shape[-1] for e in embeddings),
                device=embeddings[0].device,
            )
            embeddings.append(remaining)

        return torch.cat(embeddings, dim=-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for time series.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate

    Example:
        >>> pos_enc = PositionalEncoding(d_model=256, max_len=1000)
        >>> x = torch.randn(32, 100, 256)
        >>> output = pos_enc(x)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input [L, B, D] or [B, L, D]

        Returns:
            Positional encoded output
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TimeSeriesMasker:
    """Masking utilities for time series.

    Args:
        mask_value: Value to fill masked positions
    """

    def __init__(self, mask_value: float = 0.0):
        self.mask_value = mask_value

    def random_mask(
        self,
        x: Tensor,
        mask_ratio: float = 0.15,
    ) -> Tuple[Tensor, Tensor]:
        """Randomly mask positions.

        Args:
            x: Input [B, L, D]
            mask_ratio: Ratio of positions to mask

        Returns:
            (masked_x, mask) tuple
        """
        B, L, D = x.shape

        mask = torch.rand(B, L, device=x.device) < mask_ratio

        x_masked = x.clone()
        x_masked[mask] = self.mask_value

        return x_masked, mask

    def causal_mask(self, x: Tensor) -> Tensor:
        """Create causal mask.

        Args:
            x: Input [B, L, D]

        Returns:
            Causal mask [L, L]
        """
        L = x.shape[1]
        return torch.tril(torch.ones(L, L, device=x.device)).bool()


class TimeSeriesScaler(nn.Module):
    """Learnable time series scaler.

    Args:
        method: 'standard', 'minmax', or 'robust'

    Example:
        >>> scaler = TimeSeriesScaler(method='standard')
        >>> x = torch.randn(32, 100, 7)
        >>> x_scaled = scaler(x)
        >>> x_unscaled = scaler.inverse_transform(x_scaled)
    """

    def __init__(self, method: str = "standard"):
        super().__init__()
        self.method = method

        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))
        self.register_buffer("min", torch.zeros(1))
        self.register_buffer("max", torch.ones(1))

    def fit(self, x: Tensor) -> "TimeSeriesScaler":
        """Fit scaler parameters.

        Args:
            x: Training data

        Returns:
            self
        """
        if self.method == "standard":
            self.mean = x.mean(dim=(0, 1), keepdim=True)
            self.std = x.std(dim=(0, 1), keepdim=True) + 1e-8
        elif self.method == "minmax":
            self.min = x.min(dim=(0, 1), keepdim=True)[0]
            self.max = x.max(dim=(0, 1), keepdim=True)[0]
        elif self.method == "robust":
            self.mean = x.median(dim=(0, 1), keepdim=True)[0]
            q25 = x.quantile(0.25, dim=(0, 1))
            q75 = x.quantile(0.75, dim=(0, 1))
            self.std = (q75 - q25) / 2

        return self

    def forward(self, x: Tensor) -> Tensor:
        """Transform data.

        Args:
            x: Input data

        Returns:
            Scaled data
        """
        if self.method == "standard":
            return (x - self.mean) / self.std
        elif self.method == "minmax":
            return (x - self.min) / (self.max - self.min + 1e-8)
        elif self.method == "robust":
            return (x - self.mean) / (self.std + 1e-8)
        return x

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Inverse transform data.

        Args:
            x: Scaled data

        Returns:
            Original scale data
        """
        if self.method == "standard":
            return x * self.std + self.mean
        elif self.method == "minmax":
            return x * (self.max - self.min) + self.min
        elif self.method == "robust":
            return x * self.std + self.mean
        return x


class TimeSeriesDataLoader:
    """Data loader for time series with additional utilities.

    Args:
        dataset: Time series dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        drop_last: Whether to drop last incomplete batch
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)


def create_sequences(
    data: Tensor,
    seq_len: int,
    pred_len: int = 1,
    stride: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Create sequences from time series data.

    Args:
        data: Time series [N, D]
        seq_len: Input sequence length
        pred_len: Prediction length
        stride: Stride between sequences

    Returns:
        (X, y) tuple of input and target sequences

    Example:
        >>> data = torch.randn(1000, 7)
        >>> X, y = create_sequences(data, seq_len=96, pred_len=24)
        >>> # X: [N, 96, 7], y: [N, 24, 7]
    """
    sequences = []
    targets = []

    for i in range(0, len(data) - seq_len - pred_len + 1, stride):
        sequences.append(data[i : i + seq_len])
        targets.append(data[i + seq_len : i + seq_len + pred_len])

    return torch.stack(sequences), torch.stack(targets)


def temporal_train_test_split(
    data: Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split time series into train, validation, and test sets.

    Args:
        data: Time series [N, D]
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        (train, val, test) tuple

    Example:
        >>> data = torch.randn(1000, 7)
        >>> train, val, test = temporal_train_test_split(data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test


@dataclass
class DataConfig:
    """Configuration for time series data."""

    seq_len: int = 96
    pred_len: int = 24
    stride: int = 1
    batch_size: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    scaler_type: str = "standard"


class TimeSeriesCollator:
    """Custom collator for batching time series data.

    Args:
        pad_value: Value for padding sequences

    Example:
        >>> collator = TimeSeriesCollator(pad_value=0)
        >>> batch = collator(samples)
    """

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, samples: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Collate samples.

        Args:
            samples: List of (x, y) tuples

        Returns:
            Batched (x, y)
        """
        xs, ys = zip(*samples)

        max_x_len = max(x.shape[0] for x in xs)
        max_y_len = max(y.shape[0] for y in ys)

        B = len(xs)
        D_x = xs[0].shape[-1]
        D_y = ys[0].shape[-1]

        x_padded = torch.full(
            (B, max_x_len, D_x),
            self.pad_value,
            dtype=xs[0].dtype,
        )
        y_padded = torch.full(
            (B, max_y_len, D_y),
            self.pad_value,
            dtype=ys[0].dtype,
        )

        x_lens = []
        y_lens = []

        for i, (x, y) in enumerate(samples):
            x_padded[i, : x.shape[0], :] = x
            y_padded[i, : y.shape[0], :] = y
            x_lens.append(x.shape[0])
            y_lens.append(y.shape[0])

        return x_padded, y_padded


def create_time_series_loader(
    data: Tensor,
    config: DataConfig,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test loaders.

    Args:
        data: Time series data
        config: Data configuration
        shuffle: Whether to shuffle training data

    Returns:
        (train_loader, val_loader, test_loader)

    Example:
        >>> data = torch.randn(1000, 7)
        >>> config = DataConfig(seq_len=96, pred_len=24)
        >>> train_loader, val_loader, test_loader = create_time_series_loader(data, config)
    """
    train, val, test = temporal_train_test_split(
        data,
        config.train_ratio,
        config.val_ratio,
    )

    train_dataset = TimeSeriesDataset(
        train,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        stride=config.stride,
    )

    val_dataset = TimeSeriesDataset(
        val,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        stride=config.stride,
    )

    test_dataset = TimeSeriesDataset(
        test,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        stride=config.stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
    )

    return train_loader, val_loader, test_loader
