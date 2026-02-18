"""
Data Utilities for Climate and Weather Modeling

Data loading, preprocessing, and transformation utilities.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from fishstick.climate_weather.types import (
    ClimateDataSpec,
    CoordinateSystem,
    GridSpec,
    VariableType,
    WeatherState,
    create_lat_lon_grid,
)


class ClimateDataset(Dataset):
    """Dataset for climate/weather data.

    Args:
        data: Climate data array of shape (T, C, H, W)
        seq_length: Input sequence length
        forecast_horizon: Number of steps to forecast
        transform: Optional transform to apply
    """

    def __init__(
        self,
        data: Union[Tensor, np.ndarray],
        seq_length: int = 24,
        forecast_horizon: int = 6,
        transform: Optional[Callable] = None,
    ):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        self.data = data
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.transform = transform

        self.max_idx = len(data) - seq_length - forecast_horizon

    def __len__(self) -> int:
        return max(0, self.max_idx)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.data[idx : idx + self.seq_length]
        y = self.data[
            idx + self.seq_length : idx + self.seq_length + self.forecast_horizon
        ]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y


class ClimateDataLoader:
    """DataLoader wrapper for climate data with built-in scaling.

    Args:
        data: Climate data array
        batch_size: Batch size
        seq_length: Input sequence length
        forecast_horizon: Forecast horizon
        shuffle: Whether to shuffle
        scaler: Data scaler (StandardScaler, MinMaxScaler, etc.)
    """

    def __init__(
        self,
        data: Union[Tensor, np.ndarray],
        batch_size: int = 32,
        seq_length: int = 24,
        forecast_horizon: int = 6,
        shuffle: bool = True,
        scaler: Optional[Any] = None,
    ):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        self.data = data
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.shuffle = shuffle
        self.scaler = scaler

        if scaler is not None:
            self._fit_scaler()

        self.dataset = ClimateDataset(
            data=self.scaled_data,
            seq_length=seq_length,
            forecast_horizon=forecast_horizon,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _fit_scaler(self):
        """Fit the scaler to the data."""
        flat_data = self.data.reshape(self.data.shape[0], -1)
        self.scaler.fit(flat_data.numpy())
        scaled = self.scaler.transform(flat_data.numpy())
        self.scaled_data = scaled.reshape(self.data.shape)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)


class WeatherStandardScaler:
    """Standard scaler for weather data with optional temporal normalization.

    Args:
        method: Scaling method ('standard', 'minmax', 'robust')
        temporal: Whether to normalize per time step
    """

    def __init__(
        self,
        method: str = "standard",
        temporal: bool = False,
    ):
        self.method = method
        self.temporal = temporal
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None
        self.min: Optional[Tensor] = None
        self.max: Optional[Tensor] = None
        self.median: Optional[Tensor] = None
        self.iqr: Optional[Tensor] = None

    def fit(self, data: Tensor) -> "WeatherStandardScaler":
        """Fit the scaler to the data.

        Args:
            data: Input data of shape (B, C, H, W) or (B, T, C, H, W)

        Returns:
            self
        """
        if self.temporal:
            axes = tuple(range(2, len(data.shape)))
        else:
            axes = tuple(range(1, len(data.shape)))

        if self.method == "standard":
            self.mean = data.mean(dim=axes, keepdim=True)
            self.std = data.std(dim=axes, keepdim=True)
            self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        elif self.method == "minmax":
            self.min = data.min(dim=axes, keepdim=True)[0]
            self.max = data.max(dim=axes, keepdim=True)[0]
        elif self.method == "robust":
            self.median = data.median(dim=axes, keepdim=True)[0]
            q75 = torch.quantile(data, 0.75, dim=axes, keepdim=True)
            q25 = torch.quantile(data, 0.25, dim=axes, keepdim=True)
            self.iqr = q75 - q25
            self.iqr = torch.where(self.iqr == 0, torch.ones_like(self.iqr), self.iqr)

        return self

    def transform(self, data: Tensor) -> Tensor:
        """Transform the data.

        Args:
            data: Input data

        Returns:
            Scaled data
        """
        if self.method == "standard":
            return (data - self.mean) / self.std
        elif self.method == "minmax":
            return (data - self.min) / (self.max - self.min)
        elif self.method == "robust":
            return (data - self.median) / self.iqr
        return data

    def inverse_transform(self, data: Tensor) -> Tensor:
        """Inverse transform the data.

        Args:
            data: Scaled data

        Returns:
            Original scale data
        """
        if self.method == "standard":
            return data * self.std + self.mean
        elif self.method == "minmax":
            return data * (self.max - self.min) + self.min
        elif self.method == "robust":
            return data * self.iqr + self.median
        return data


def create_weather_patches(
    data: Tensor,
    patch_size: Tuple[int, int] = (8, 8),
) -> Tensor:
    """Create patches from weather data for vision-based models.

    Args:
        data: Input data of shape (B, C, H, W)
        patch_size: Size of patches (height, width)

    Returns:
        Patched data of shape (B, num_patches, C * patch_h * patch_w)
    """
    B, C, H, W = data.shape
    ph, pw = patch_size

    data = data.unfold(2, ph, ph).unfold(3, pw, pw)
    data = data.reshape(B, C, -1, ph, pw)
    data = data.permute(0, 2, 1, 3, 4)
    data = data.reshape(B, -1, C * ph * pw)

    return data


def add_calendar_features(
    timestamps: List[datetime],
) -> Tensor:
    """Add calendar-based features to timestamps.

    Args:
        timestamps: List of timestamps

    Returns:
        Calendar features tensor of shape (len(timestamps), 7)
    """
    features = []
    for ts in timestamps:
        day_of_year = ts.timetuple().tm_yday
        day_of_week = ts.weekday()
        month = ts.month
        hour = ts.hour
        is_weekend = 1.0 if day_of_week >= 5 else 0.0

        sin_doy = np.sin(2 * np.pi * day_of_year / 365)
        cos_doy = np.cos(2 * np.pi * day_of_year / 365)
        sin_hour = np.sin(2 * np.pi * hour / 24)
        cos_hour = np.cos(2 * np.pi * hour / 24)

        features.append(
            [
                sin_doy,
                cos_doy,
                sin_hour,
                cos_hour,
                month / 12,
                day_of_week / 6,
                is_weekend,
            ]
        )

    return torch.tensor(features, dtype=torch.float32)


def compute_climatology(
    data: Tensor,
    time_dim: int = 0,
    window_size: int = 30,
) -> Tuple[Tensor, Tensor]:
    """Compute climatology (long-term mean and std) from data.

    Args:
        data: Input data with time dimension
        time_dim: Time dimension index
        window_size: Window for rolling statistics

    Returns:
        Tuple of (climatology_mean, climatology_std)
    """
    mean = data.mean(dim=time_dim, keepdim=True)
    std = data.std(dim=time_dim, keepdim=True)

    return mean, std


def compute_anomaly(
    data: Tensor,
    climatology_mean: Tensor,
    climatology_std: Tensor,
) -> Tensor:
    """Compute anomaly from climatology.

    Args:
        data: Input data
        climatology_mean: Long-term mean
        climatology_std: Long-term standard deviation

    Returns:
        Anomaly data
    """
    return (data - climatology_mean) / (climatology_std + 1e-8)


def temporal_train_test_split(
    data: Tensor,
    train_ratio: float = 0.8,
) -> Tuple[Tensor, Tensor]:
    """Split time series data into train and test sets temporally.

    Args:
        data: Input data tensor
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (train_data, test_data)
    """
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def load_era5_sample(
    num_samples: int = 100,
    variables: List[str] = None,
    grid_size: Tuple[int, int] = (32, 64),
) -> Tensor:
    """Generate sample ERA5-like climate data for testing.

    This is a utility function to generate realistic-looking climate
    data for development and testing purposes.

    Args:
        num_samples: Number of time steps to generate
        variables: List of variable names
        grid_size: Grid size (lat, lon)

    Returns:
        Generated data tensor of shape (num_samples, num_vars, H, W)
    """
    if variables is None:
        variables = ["temperature", "pressure", "u_wind", "v_wind", "humidity"]

    num_vars = len(variables)
    H, W = grid_size

    data = []
    for i in range(num_samples):
        time_step = []

        base_temp = 288.0 + 10 * np.sin(2 * np.pi * i / 365)
        temp = base_temp + np.random.randn(H, W) * 5
        time_step.append(temp)

        pressure = 101325 + np.random.randn(H, W) * 500
        time_step.append(pressure)

        u_wind = 5 * np.sin(2 * np.pi * i / 100) + np.random.randn(H, W) * 3
        time_step.append(u_wind)

        v_wind = 3 * np.cos(2 * np.pi * i / 150) + np.random.randn(H, W) * 3
        time_step.append(v_wind)

        humidity = 0.5 + 0.3 * np.sin(2 * np.pi * i / 50) + np.random.randn(H, W) * 0.1
        humidity = np.clip(humidity, 0, 1)
        time_step.append(humidity)

        data.append(np.stack(time_step))

    return torch.tensor(np.stack(data), dtype=torch.float32)
