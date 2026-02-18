"""
Weather Forecasting Models

LSTM, GRU, and Transformer-based models for weather prediction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from fishstick.climate_weather.data_utils import WeatherStandardScaler
from fishstick.climate_weather.types import ForecastOutput, WeatherState


class BaseWeatherForecaster(ABC):
    """Abstract base class for weather forecasters.

    Args:
        device: Device to run computations on
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.scaler: Optional[WeatherStandardScaler] = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        epochs: int = 100,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """Train the forecaster.

        Args:
            train_data: Training data loader
            val_data: Optional validation data loader
            epochs: Number of training epochs
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Tensor,
        forecast_horizon: int,
    ) -> ForecastOutput:
        """Generate weather forecasts.

        Args:
            X: Input sequence of shape (B, T, C, H, W)
            forecast_horizon: Number of steps to forecast

        Returns:
            ForecastOutput with predictions
        """
        pass


class LSTMWeatherForecaster(nn.Module):
    """LSTM-based weather forecasting model.

    Args:
        input_channels: Number of input variables/channels
        hidden_dim: Hidden dimension size
        num_layers: Number of LSTM layers
        forecast_horizon: Number of steps to forecast
        output_channels: Number of output variables
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        forecast_horizon: int = 6,
        output_channels: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_channels = output_channels or input_channels

        self.encoder = nn.Conv2d(
            input_channels, hidden_dim // 4, kernel_size=3, padding=1
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.decoder = nn.Conv2d(
            hidden_dim, self.output_channels, kernel_size=3, padding=1
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon * self.output_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        x = self.encoder(x)
        x = F.relu(x)

        x = x.reshape(B, T, -1)
        lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1]

        forecast = self.forecast_head(last_hidden)
        forecast = forecast.reshape(
            B, self.forecast_horizon, self.output_channels, 1, 1
        )

        return forecast


class TransformerWeatherForecaster(nn.Module):
    """Transformer-based weather forecasting model.

    Args:
        input_channels: Number of input variables
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        forecast_horizon: Number of steps to forecast
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        forecast_horizon: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_pos = nn.Parameter(torch.randn(1, 1000, hidden_dim))

        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon),
        )

        self.output_proj = nn.Conv2d(forecast_horizon, forecast_horizon, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        x = self.input_proj(x)
        x = x.reshape(B, T, H * W, -1)
        x = x.permute(0, 2, 1, 3)

        seq_len = x.shape[1]
        pos = self.temporal_pos[:, :T, :].unsqueeze(2).expand(-1, -1, H * W, -1)
        x = x + pos

        x = x.reshape(B * H * W, T, -1)
        x = self.transformer(x)
        x = x.reshape(B, H * W, T, -1)

        x = x[:, :, -1, :]

        forecast = self.forecast_head(x)
        forecast = forecast.permute(0, 2, 1).reshape(B, self.forecast_horizon, 1, H, W)

        return forecast


class ConvLSTMWeatherForecaster(nn.Module):
    """Convolutional LSTM for weather forecasting.

    Args:
        input_channels: Number of input variables
        hidden_channels: List of hidden channel sizes per layer
        kernel_size: Conv LSTM kernel size
        forecast_horizon: Number of steps to forecast
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int] = [64, 128],
        kernel_size: int = 3,
        forecast_horizon: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.forecast_horizon = forecast_horizon

        layers = []
        in_ch = input_channels
        for hidden_ch in hidden_channels:
            layers.append(
                ConvLSTMCell(
                    input_channels=in_ch,
                    hidden_channels=hidden_ch,
                    kernel_size=kernel_size,
                )
            )
            in_ch = hidden_ch

        self.convlstms = nn.ModuleList(layers)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                hidden_channels[-1], hidden_channels[-1] // 2, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels[-1] // 2, forecast_horizon, kernel_size=1),
        )

        self.hidden_states: Optional[List[Tuple[Tensor, Tensor]]] = None

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        h = x[:, 0]
        for convlstm in self.convlstms:
            h, c = convlstm(h)

        forecast = self.decoder(h)
        forecast = forecast.unsqueeze(1).expand(-1, self.forecast_horizon, -1, -1, -1)

        return forecast


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell.

    Args:
        input_channels: Number of input channels
        hidden_channels: Number of hidden channels
        kernel_size: Convolution kernel size
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = torch.zeros_like(x[:, : self.hidden_channels])
        c = torch.zeros_like(x[:, : self.hidden_channels])

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class ThreeDimensionalWeatherModel(nn.Module):
    """3D weather prediction model using 3D convolutions.

    Args:
        input_channels: Number of input variables
        hidden_channels: Number of hidden channels
        forecast_horizon: Number of steps to forecast
        temporal_stride: Temporal upsampling stride
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 128,
        forecast_horizon: int = 6,
        temporal_stride: int = 4,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.forecast_horizon = forecast_horizon
        self.temporal_stride = temporal_stride

        self.encoder_3d = nn.Sequential(
            nn.Conv3d(
                input_channels,
                hidden_channels // 2,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(hidden_channels // 2),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels // 2,
                hidden_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
        )

        self.temporal_upsample = nn.ConvTranspose3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(temporal_stride, 1, 1),
            stride=(temporal_stride, 1, 1),
        )

        self.decoder_3d = nn.Sequential(
            nn.Conv3d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(hidden_channels // 2),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels // 2,
                forecast_horizon,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4)

        x = self.encoder_3d(x)
        x = self.temporal_upsample(x)

        x = self.decoder_3d(x)

        x = x.permute(0, 2, 1, 3, 4)

        return x


@dataclass
class WeatherForecasterConfig:
    """Configuration for weather forecaster training."""

    model_type: str = "lstm"
    input_channels: int = 5
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    forecast_horizon: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 100
    device: str = "cuda"
    grad_clip: float = 1.0


def create_weather_forecaster(
    config: WeatherForecasterConfig,
) -> nn.Module:
    """Create a weather forecaster model based on config.

    Args:
        config: Model configuration

    Returns:
        Weather forecasting model
    """
    if config.model_type == "lstm":
        return LSTMWeatherForecaster(
            input_channels=config.input_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            forecast_horizon=config.forecast_horizon,
            dropout=config.dropout,
        )
    elif config.model_type == "transformer":
        return TransformerWeatherForecaster(
            input_channels=config.input_channels,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            forecast_horizon=config.forecast_horizon,
            dropout=config.dropout,
        )
    elif config.model_type == "convlstm":
        return ConvLSTMWeatherForecaster(
            input_channels=config.input_channels,
            hidden_channels=[config.hidden_dim // 4, config.hidden_dim // 2],
            forecast_horizon=config.forecast_horizon,
            dropout=config.dropout,
        )
    elif config.model_type == "3d":
        return ThreeDimensionalWeatherModel(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_dim,
            forecast_horizon=config.forecast_horizon,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
