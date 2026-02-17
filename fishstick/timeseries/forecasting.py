"""
fishstick Time Series Forecasting Module

Comprehensive time series forecasting with deep learning models, feature engineering,
metrics, data utilities, and ensemble methods.

Example:
    >>> from fishstick.timeseries.forecasting import (
    ...     LSTMForecaster,
    ...     TransformerForecaster,
    ...     FeatureEngineer,
    ...     TimeSeriesMetrics,
    ...     TimeSeriesDataset,
    ...     EnsembleForecaster,
    ... )
    >>>
    >>> # Prepare data
    >>> dataset = TimeSeriesDataset(data, seq_length=24, forecast_horizon=12)
    >>> train_loader = DataLoader(dataset, batch_size=32)
    >>>
    >>> # Create and train model
    >>> model = LSTMForecaster(input_dim=10, hidden_dim=128, forecast_horizon=12)
    >>> forecaster = BaseForecaster(model)
    >>> forecaster.fit(train_loader, epochs=100)
    >>>
    >>> # Generate forecasts with intervals
    >>> predictions, intervals = forecaster.predict_interval(test_data, confidence=0.95)
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Sequence,
)
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import math


# =============================================================================
# Base Forecaster Abstract Class
# =============================================================================


class BaseForecaster(ABC):
    """Abstract base class for all time series forecasters.

    Provides a unified interface for time series forecasting with methods for
    training, prediction, evaluation, and uncertainty quantification.

    Args:
        model: The underlying PyTorch model
        device: Device to run computations on ('cuda' or 'cpu')

    Example:
        >>> class MyForecaster(BaseForecaster):
        ...     def fit(self, train_data, **kwargs):
        ...         # Training implementation
        ...         pass
        ...     def predict(self, X, **kwargs):
        ...         # Prediction implementation
        ...         pass
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.is_fitted = False
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    @abstractmethod
    def fit(
        self,
        train_data: Union[DataLoader, np.ndarray, pd.DataFrame],
        val_data: Optional[Union[DataLoader, np.ndarray, pd.DataFrame]] = None,
        **kwargs,
    ) -> "BaseForecaster":
        """Train the forecaster on historical data.

        Args:
            train_data: Training data (DataLoader, array, or DataFrame)
            val_data: Optional validation data
            **kwargs: Additional training parameters

        Returns:
            self: The fitted forecaster instance
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        forecast_horizon: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate forecasts for input data.

        Args:
            X: Input time series data
            forecast_horizon: Number of steps to forecast (overrides default)
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Forecast values as numpy array
        """
        pass

    def predict_interval(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        confidence: float = 0.95,
        num_samples: int = 100,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals.

        Args:
            X: Input time series data
            confidence: Confidence level for intervals (e.g., 0.95 for 95%)
            num_samples: Number of samples for Monte Carlo estimation
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Point forecasts
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
        """
        self.model.eval()

        # Enable dropout for uncertainty estimation
        def enable_dropout(model):
            for m in model.modules():
                if isinstance(
                    m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
                ):
                    m.train()

        enable_dropout(self.model)

        predictions_list = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.predict(X, **kwargs)
                predictions_list.append(pred)

        predictions_array = np.stack(predictions_list)
        mean_pred = predictions_array.mean(axis=0)

        alpha = (1 - confidence) / 2
        lower_bounds = np.percentile(predictions_array, alpha * 100, axis=0)
        upper_bounds = np.percentile(predictions_array, (1 - alpha) * 100, axis=0)

        return mean_pred, lower_bounds, upper_bounds

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        y_true: Union[np.ndarray, pd.DataFrame, Tensor],
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Evaluate forecasting performance.

        Args:
            X: Input time series data
            y_true: True target values
            metrics: List of metric names to compute (default: all)
            **kwargs: Additional evaluation parameters

        Returns:
            results: Dictionary of metric names to values
        """
        y_pred = self.predict(X, **kwargs)

        if metrics is None:
            metrics = ["mae", "rmse", "mape", "mase"]

        results = {}
        metric_calculator = TimeSeriesMetrics()

        for metric in metrics:
            metric_fn = getattr(metric_calculator, metric, None)
            if metric_fn:
                results[metric] = metric_fn(y_true, y_pred)

        return results

    def save(self, path: str) -> None:
        """Save forecaster to disk.

        Args:
            path: File path to save model
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "history": self.history,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: str) -> "BaseForecaster":
        """Load forecaster from disk.

        Args:
            path: File path to load model from

        Returns:
            self: The loaded forecaster instance
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        self.is_fitted = checkpoint.get("is_fitted", True)
        return self


# =============================================================================
# Deep Learning Forecasters
# =============================================================================


class LSTMForecaster(BaseForecaster):
    """LSTM-based forecaster with encoder-decoder architecture.

    Implements a sequence-to-sequence LSTM model for multi-step time series
    forecasting with teacher forcing during training.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension of LSTM layers
        num_layers: Number of LSTM layers
        forecast_horizon: Number of steps to forecast
        dropout: Dropout probability
        teacher_forcing_ratio: Ratio of teacher forcing during training

    Example:
        >>> model = LSTMForecaster(
        ...     input_dim=10,
        ...     hidden_dim=128,
        ...     num_layers=2,
        ...     forecast_horizon=24,
        ... )
        >>> forecaster = LSTMForecaster(model)
        >>> forecaster.fit(train_loader, epochs=100)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 1,
        dropout: float = 0.1,
        teacher_forcing_ratio: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.teacher_forcing_ratio = teacher_forcing_ratio

        model = _LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            dropout=dropout,
        )
        super().__init__(model, device)

    def fit(
        self,
        train_data: Union[DataLoader, np.ndarray, pd.DataFrame],
        val_data: Optional[Union[DataLoader, np.ndarray, pd.DataFrame]] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        **kwargs,
    ) -> "LSTMForecaster":
        """Train the LSTM forecaster."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Convert data to DataLoader if needed
        if isinstance(train_data, (np.ndarray, pd.DataFrame)):
            train_loader = self._create_dataloader(train_data, **kwargs)
        else:
            train_loader = train_data

        if val_data is not None and isinstance(val_data, (np.ndarray, pd.DataFrame)):
            val_loader = self._create_dataloader(val_data, **kwargs)
        else:
            val_loader = val_data

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                if len(batch) == 2:
                    x, y = batch
                else:
                    x = batch
                    y = None

                x = x.to(self.device)
                if y is not None:
                    y = y.to(self.device)

                optimizer.zero_grad()

                # Use teacher forcing
                use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
                output = self.model(
                    x, y if use_teacher_forcing else None, self.teacher_forcing_ratio
                )

                if y is not None:
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        forecast_horizon: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate forecasts."""
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            predictions = self.model(X, None, 0.0)

        return predictions.cpu().numpy()

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate model on validation set."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x, None, 0.0)
                    loss = criterion(output, y)
                    val_loss += loss.item()

        return val_loss / len(val_loader)

    def _create_dataloader(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        seq_length: int = 24,
        batch_size: int = 32,
        **kwargs,
    ) -> DataLoader:
        """Create DataLoader from numpy array or DataFrame."""
        dataset = TimeSeriesDataset(
            data=data,
            seq_length=seq_length,
            forecast_horizon=self.forecast_horizon,
            **kwargs,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class _LSTMModel(nn.Module):
    """Internal LSTM model implementation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        forecast_horizon: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, input_dim)
        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: Tensor,
        target: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        batch_size = x.size(0)

        # Encode
        _, (hidden, cell) = self.encoder_lstm(x)

        # Decode
        decoder_input = x[:, -1:, :]  # Last time step
        outputs = []

        for t in range(self.forecast_horizon):
            output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            prediction = self.fc(output)
            outputs.append(prediction)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t : t + 1, :]
            else:
                decoder_input = prediction

        return torch.cat(outputs, dim=1)


class TransformerForecaster(BaseForecaster):
    """Temporal Fusion Transformer for multi-horizon forecasting.

    Implements a transformer-based architecture with temporal attention mechanisms
    for capturing long-range dependencies and multiple forecasting horizons.

    Args:
        input_dim: Number of input features
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Feedforward network dimension
        forecast_horizon: Number of steps to forecast
        dropout: Dropout probability

    Example:
        >>> model = TransformerForecaster(
        ...     input_dim=10,
        ...     d_model=128,
        ...     nhead=8,
        ...     forecast_horizon=24,
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 512,
        forecast_horizon: int = 1,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        model = _TransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            forecast_horizon=forecast_horizon,
            dropout=dropout,
        )
        super().__init__(model, device)

    def fit(
        self,
        train_data: Union[DataLoader, np.ndarray, pd.DataFrame],
        val_data: Optional[Union[DataLoader, np.ndarray, pd.DataFrame]] = None,
        epochs: int = 100,
        lr: float = 1e-4,
        **kwargs,
    ) -> "TransformerForecaster":
        """Train the transformer forecaster."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        if isinstance(train_data, (np.ndarray, pd.DataFrame)):
            train_loader = self._create_dataloader(train_data, **kwargs)
        else:
            train_loader = train_data

        if val_data is not None and isinstance(val_data, (np.ndarray, pd.DataFrame)):
            val_loader = self._create_dataloader(val_data, **kwargs)
        else:
            val_loader = val_data

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    x = batch.to(self.device)
                    y = None

                optimizer.zero_grad()
                output = self.model(x)

                if y is not None:
                    loss = criterion(output, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        forecast_horizon: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate forecasts."""
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            predictions = self.model(X)

        return predictions.cpu().numpy()

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate model."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    loss = criterion(output, y)
                    val_loss += loss.item()

        return val_loss / len(val_loader)

    def _create_dataloader(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        seq_length: int = 24,
        batch_size: int = 32,
        **kwargs,
    ) -> DataLoader:
        """Create DataLoader."""
        dataset = TimeSeriesDataset(
            data=data,
            seq_length=seq_length,
            forecast_horizon=self.forecast_horizon,
            **kwargs,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class _TransformerModel(nn.Module):
    """Internal transformer model implementation."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        forecast_horizon: int,
        dropout: float,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.decoder_projection = nn.Linear(input_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.output_projection = nn.Linear(d_model, input_dim)
        self.forecast_horizon = forecast_horizon

    def forward(self, x: Tensor) -> Tensor:
        # Encode
        x_proj = self.input_projection(x)
        x_proj = self.pos_encoder(x_proj)
        memory = self.encoder(x_proj)

        # Decode - autoregressive
        batch_size = x.size(0)
        decoder_input = x[:, -1:, :]  # Start with last input
        outputs = []

        for _ in range(self.forecast_horizon):
            dec_proj = self.decoder_projection(decoder_input)
            dec_proj = self.pos_encoder(dec_proj)
            dec_out = self.decoder(dec_proj, memory)
            prediction = self.output_projection(dec_out[:, -1:, :])
            outputs.append(prediction)
            decoder_input = torch.cat([decoder_input, prediction], dim=1)

        return torch.cat(outputs, dim=1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class NBeatsForecaster(BaseForecaster):
    """N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.

    A deep neural architecture based on backward and forward residual links
    with a very deep stack of fully-connected layers.

    Args:
        input_dim: Number of input features
        forecast_horizon: Number of steps to forecast
        stack_types: Types of stacks ('trend', 'seasonality', 'generic')
        nb_blocks_per_stack: Number of blocks per stack
        hidden_dim: Hidden dimension
        thetas_dim: Dimension of basis functions

    Example:
        >>> model = NBeatsForecaster(
        ...     input_dim=1,
        ...     forecast_horizon=24,
        ...     stack_types=['trend', 'seasonality'],
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        forecast_horizon: int = 1,
        stack_types: List[str] = None,
        nb_blocks_per_stack: int = 3,
        hidden_dim: int = 128,
        thetas_dim: List[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon

        if stack_types is None:
            stack_types = ["trend", "seasonality"]
        if thetas_dim is None:
            thetas_dim = [2, 8]

        model = _NBeatsModel(
            input_dim=input_dim,
            forecast_horizon=forecast_horizon,
            stack_types=stack_types,
            nb_blocks_per_stack=nb_blocks_per_stack,
            hidden_dim=hidden_dim,
            thetas_dim=thetas_dim,
        )
        super().__init__(model, device)

    def fit(
        self,
        train_data: Union[DataLoader, np.ndarray, pd.DataFrame],
        val_data: Optional[Union[DataLoader, np.ndarray, pd.DataFrame]] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        **kwargs,
    ) -> "NBeatsForecaster":
        """Train the N-BEATS forecaster."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        if isinstance(train_data, (np.ndarray, pd.DataFrame)):
            train_loader = self._create_dataloader(train_data, **kwargs)
        else:
            train_loader = train_data

        if val_data is not None and isinstance(val_data, (np.ndarray, pd.DataFrame)):
            val_loader = self._create_dataloader(val_data, **kwargs)
        else:
            val_loader = val_data

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    x = batch.to(self.device)
                    continue

                optimizer.zero_grad()
                backcast, forecast = self.model(x)

                # Loss on forecast
                if y is not None:
                    loss = criterion(forecast, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        forecast_horizon: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate forecasts."""
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            _, forecast = self.model(X)

        return forecast.cpu().numpy()

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate model."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    _, forecast = self.model(x)
                    loss = criterion(forecast, y)
                    val_loss += loss.item()

        return val_loss / len(val_loader)

    def _create_dataloader(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        seq_length: int = 24,
        batch_size: int = 32,
        **kwargs,
    ) -> DataLoader:
        """Create DataLoader."""
        dataset = TimeSeriesDataset(
            data=data,
            seq_length=seq_length,
            forecast_horizon=self.forecast_horizon,
            **kwargs,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class _NBeatsModel(nn.Module):
    """Internal N-BEATS model implementation."""

    def __init__(
        self,
        input_dim: int,
        forecast_horizon: int,
        stack_types: List[str],
        nb_blocks_per_stack: int,
        hidden_dim: int,
        thetas_dim: List[int],
    ):
        super().__init__()

        self.stacks = nn.ModuleList()
        for i, stack_type in enumerate(stack_types):
            blocks = nn.ModuleList()
            for _ in range(nb_blocks_per_stack):
                blocks.append(
                    NBeatsBlock(
                        input_dim=input_dim,
                        forecast_horizon=forecast_horizon,
                        hidden_dim=hidden_dim,
                        thetas_dim=thetas_dim[i],
                        block_type=stack_type,
                    )
                )
            self.stacks.append(blocks)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        backcast = x
        forecast = torch.zeros(
            x.size(0), self.stacks[0][0].forecast_horizon, x.size(2), device=x.device
        )

        for stack in self.stacks:
            for block in stack:
                b, f = block(backcast)
                backcast = backcast - b
                forecast = forecast + f

        return backcast, forecast


class NBeatsBlock(nn.Module):
    """N-BEATS block with basis expansion."""

    def __init__(
        self,
        input_dim: int,
        forecast_horizon: int,
        hidden_dim: int,
        thetas_dim: int,
        block_type: str = "generic",
    ):
        super().__init__()

        self.block_type = block_type
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        # Basis expansion layers
        self.theta_b_fc = nn.Linear(hidden_dim, thetas_dim)
        self.theta_f_fc = nn.Linear(hidden_dim, thetas_dim)

        self.backcast_basis = nn.Linear(thetas_dim, input_dim)
        self.forecast_basis = nn.Linear(thetas_dim, forecast_horizon * input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Take mean across time dimension for simplicity
        x_mean = x.mean(dim=1)  # [batch, features]

        h = F.relu(self.fc1(x_mean))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))

        theta_b = self.theta_b_fc(h)
        theta_f = self.theta_f_fc(h)

        backcast = self.backcast_basis(theta_b).unsqueeze(1)
        forecast = self.forecast_basis(theta_f)
        forecast = forecast.view(-1, self.forecast_horizon, self.input_dim)

        return backcast, forecast


class DeepARForecaster(BaseForecaster):
    """DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.

    A probabilistic forecasting model that produces full predictive distributions
    rather than just point forecasts.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension of RNN
        num_layers: Number of RNN layers
        forecast_horizon: Number of steps to forecast
        dropout: Dropout probability

    Example:
        >>> model = DeepARForecaster(
        ...     input_dim=10,
        ...     hidden_dim=128,
        ...     forecast_horizon=24,
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 1,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        model = _DeepARModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            dropout=dropout,
        )
        super().__init__(model, device)

    def fit(
        self,
        train_data: Union[DataLoader, np.ndarray, pd.DataFrame],
        val_data: Optional[Union[DataLoader, np.ndarray, pd.DataFrame]] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        **kwargs,
    ) -> "DeepARForecaster":
        """Train the DeepAR forecaster."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if isinstance(train_data, (np.ndarray, pd.DataFrame)):
            train_loader = self._create_dataloader(train_data, **kwargs)
        else:
            train_loader = train_data

        if val_data is not None and isinstance(val_data, (np.ndarray, pd.DataFrame)):
            val_loader = self._create_dataloader(val_data, **kwargs)
        else:
            val_loader = val_data

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    continue

                optimizer.zero_grad()

                # Compute negative log likelihood
                loss = self.model.loss(x, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        forecast_horizon: Optional[int] = None,
        num_samples: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """Generate point forecasts (mean of distribution)."""
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            # Sample multiple times and take mean
            samples = []
            for _ in range(num_samples):
                sample = self.model.sample(X, forecast_horizon or self.forecast_horizon)
                samples.append(sample)
            predictions = torch.stack(samples).mean(dim=0)

        return predictions.cpu().numpy()

    def predict_interval(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        confidence: float = 0.95,
        num_samples: int = 100,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals via sampling."""
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            samples = []
            for _ in range(num_samples):
                sample = self.model.sample(X, self.forecast_horizon)
                samples.append(sample)

            samples_array = torch.stack(samples).cpu().numpy()
            mean_pred = samples_array.mean(axis=0)

            alpha = (1 - confidence) / 2
            lower_bounds = np.percentile(samples_array, alpha * 100, axis=0)
            upper_bounds = np.percentile(samples_array, (1 - alpha) * 100, axis=0)

        return mean_pred, lower_bounds, upper_bounds

    def _validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    loss = self.model.loss(x, y)
                    val_loss += loss.item()

        return val_loss / len(val_loader)

    def _create_dataloader(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        seq_length: int = 24,
        batch_size: int = 32,
        **kwargs,
    ) -> DataLoader:
        """Create DataLoader."""
        dataset = TimeSeriesDataset(
            data=data,
            seq_length=seq_length,
            forecast_horizon=self.forecast_horizon,
            **kwargs,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class _DeepARModel(nn.Module):
    """Internal DeepAR model implementation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        forecast_horizon: int,
        dropout: float,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Parameters for Gaussian distribution
        self.mu_fc = nn.Linear(hidden_dim, input_dim)
        self.sigma_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output, _ = self.rnn(x)
        hidden = output[:, -1, :]

        mu = self.mu_fc(hidden)
        sigma = F.softplus(self.sigma_fc(hidden)) + 1e-6

        return mu, sigma

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Negative log-likelihood loss."""
        mu, sigma = self.forward(x)

        # Expand mu and sigma for all forecast steps
        mu = mu.unsqueeze(1).expand(-1, y.size(1), -1)
        sigma = sigma.unsqueeze(1).expand(-1, y.size(1), -1)

        # Gaussian negative log-likelihood
        loss = 0.5 * torch.log(2 * torch.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2

        return loss.mean()

    def sample(self, x: Tensor, forecast_horizon: int) -> Tensor:
        """Sample from the predictive distribution."""
        batch_size = x.size(0)
        samples = []

        hidden = None
        input_seq = x

        for _ in range(forecast_horizon):
            output, hidden = self.rnn(input_seq, hidden)
            h = output[:, -1, :]

            mu = self.mu_fc(h)
            sigma = F.softplus(self.sigma_fc(h)) + 1e-6

            # Sample from Gaussian
            sample = mu + sigma * torch.randn_like(mu)
            samples.append(sample.unsqueeze(1))

            # Use sample as next input
            input_seq = torch.cat([input_seq[:, 1:, :], sample.unsqueeze(1)], dim=1)

        return torch.cat(samples, dim=1)


# =============================================================================
# Feature Engineering
# =============================================================================


class FeatureEngineer:
    """Feature engineering utilities for time series.

    Provides methods for creating various time-based, lag, rolling, and
    spectral features for time series forecasting.

    Example:
        >>> fe = FeatureEngineer()
        >>> df_features = fe.transform(
        ...     df,
        ...     datetime_col='timestamp',
        ...     lags=[1, 2, 3],
        ...     rolling_windows=[7, 14],
        ... )
    """

    def __init__(self):
        self.fitted_params = {}

    def transform(
        self,
        df: pd.DataFrame,
        datetime_col: Optional[str] = None,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        rolling_stats: List[str] = None,
        fourier_periods: Optional[List[int]] = None,
        fourier_order: int = 3,
        add_holidays: bool = False,
        country: str = "US",
        drop_original_datetime: bool = True,
    ) -> pd.DataFrame:
        """Transform DataFrame with comprehensive feature engineering.

        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column (if None, uses index)
            lags: List of lag periods to create
            rolling_windows: List of window sizes for rolling stats
            rolling_stats: Rolling statistics to compute (mean, std, min, max, etc.)
            fourier_periods: List of periods for Fourier features
            fourier_order: Order of Fourier series
            add_holidays: Whether to add holiday indicators
            country: Country code for holidays
            drop_original_datetime: Whether to drop original datetime column

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Get datetime series
        if datetime_col is not None:
            dt_series = pd.to_datetime(df[datetime_col])
            if drop_original_datetime:
                df = df.drop(columns=[datetime_col])
        else:
            dt_series = pd.to_datetime(df.index)

        # Time-based features
        df = self._add_time_features(df, dt_series)

        # Lag features
        if lags:
            df = self._add_lag_features(df, lags)

        # Rolling window statistics
        if rolling_windows and rolling_stats:
            df = self._add_rolling_features(df, rolling_windows, rolling_stats)

        # Fourier features for seasonality
        if fourier_periods:
            df = self._add_fourier_features(
                df, dt_series, fourier_periods, fourier_order
            )

        # Holiday indicators
        if add_holidays:
            df = self._add_holiday_features(df, dt_series, country)

        return df

    def _add_time_features(
        self,
        df: pd.DataFrame,
        dt_series: pd.Series,
    ) -> pd.DataFrame:
        """Add time-based features."""
        df["year"] = dt_series.dt.year
        df["month"] = dt_series.dt.month
        df["day"] = dt_series.dt.day
        df["hour"] = dt_series.dt.hour
        df["dayofweek"] = dt_series.dt.dayofweek
        df["dayofyear"] = dt_series.dt.dayofyear
        df["weekofyear"] = dt_series.dt.isocalendar().week.values
        df["quarter"] = dt_series.dt.quarter

        # Cyclical encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
        df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        return df

    def _add_lag_features(
        self,
        df: pd.DataFrame,
        lags: List[int],
    ) -> pd.DataFrame:
        """Add lag features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int],
        stats: List[str],
    ) -> pd.DataFrame:
        """Add rolling window statistics."""
        if stats is None:
            stats = ["mean", "std", "min", "max"]

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            for window in windows:
                for stat in stats:
                    if stat == "mean":
                        df[f"{col}_rolling_{window}_mean"] = (
                            df[col].rolling(window=window).mean()
                        )
                    elif stat == "std":
                        df[f"{col}_rolling_{window}_std"] = (
                            df[col].rolling(window=window).std()
                        )
                    elif stat == "min":
                        df[f"{col}_rolling_{window}_min"] = (
                            df[col].rolling(window=window).min()
                        )
                    elif stat == "max":
                        df[f"{col}_rolling_{window}_max"] = (
                            df[col].rolling(window=window).max()
                        )
                    elif stat == "median":
                        df[f"{col}_rolling_{window}_median"] = (
                            df[col].rolling(window=window).median()
                        )
                    elif stat == "skew":
                        df[f"{col}_rolling_{window}_skew"] = (
                            df[col].rolling(window=window).skew()
                        )

        return df

    def _add_fourier_features(
        self,
        df: pd.DataFrame,
        dt_series: pd.Series,
        periods: List[int],
        order: int,
    ) -> pd.DataFrame:
        """Add Fourier features for seasonality."""
        # Convert datetime to numerical value (seconds since epoch)
        t = (dt_series.astype(np.int64) // 10**9).values

        for period in periods:
            for i in range(1, order + 1):
                df[f"fourier_sin_{period}_{i}"] = np.sin(2 * np.pi * i * t / period)
                df[f"fourier_cos_{period}_{i}"] = np.cos(2 * np.pi * i * t / period)

        return df

    def _add_holiday_features(
        self,
        df: pd.DataFrame,
        dt_series: pd.Series,
        country: str = "US",
    ) -> pd.DataFrame:
        """Add holiday indicator features."""
        try:
            from pandas.tseries.holiday import USFederalHolidayCalendar

            if country == "US":
                cal = USFederalHolidayCalendar()
                holidays = cal.holidays(start=dt_series.min(), end=dt_series.max())
                df["is_holiday"] = dt_series.isin(holidays).astype(int)
            else:
                warnings.warn(f"Holidays for country {country} not implemented")
                df["is_holiday"] = 0
        except ImportError:
            warnings.warn("pandas holiday calendar not available")
            df["is_holiday"] = 0

        df["is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)

        return df


# =============================================================================
# Time Series Metrics
# =============================================================================


class TimeSeriesMetrics:
    """Comprehensive metrics for time series forecasting evaluation.

    Provides standard forecasting metrics including scale-dependent,
    percentage errors, and probabilistic metrics.

    Example:
        >>> metrics = TimeSeriesMetrics()
        >>> mae = metrics.mae(y_true, y_pred)
        >>> rmse = metrics.rmse(y_true, y_pred)
        >>> mape = metrics.mape(y_true, y_pred)
    """

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MAPE value (in percentage)
        """
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            SMAPE value
        """
        return (
            np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
            * 100
        )

    @staticmethod
    def mase(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonality: int = 1,
    ) -> float:
        """Mean Absolute Scaled Error.

        Scale-independent metric that compares forecast errors to
        the in-sample naive forecast error.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            y_train: Training data (for calculating naive error)
            seasonality: Seasonal period for naive forecast

        Returns:
            MASE value (MASE < 1 means better than naive)
        """
        # Calculate naive forecast error on training data
        naive_forecast = y_train[:-seasonality]
        naive_errors = np.abs(y_train[seasonality:] - naive_forecast)
        mae_naive = np.mean(naive_errors)

        if mae_naive == 0:
            return np.inf

        mae_forecast = np.mean(np.abs(y_true - y_pred))
        return mae_forecast / mae_naive

    @staticmethod
    def coverage(
        y_true: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> float:
        """Prediction interval coverage.

        Measures the percentage of true values that fall within
        the prediction intervals.

        Args:
            y_true: Ground truth values
            lower_bounds: Lower prediction interval bounds
            upper_bounds: Upper prediction interval bounds

        Returns:
            Coverage proportion (0 to 1)
        """
        within_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        return np.mean(within_interval)

    @staticmethod
    def crps(
        y_true: np.ndarray,
        y_samples: np.ndarray,
    ) -> float:
        """Continuous Ranked Probability Score.

        Probabilistic scoring rule that measures the accuracy of
        probabilistic forecasts.

        Args:
            y_true: Ground truth values (shape: [batch, horizon, features])
            y_samples: Samples from predictive distribution (shape: [samples, batch, horizon, features])

        Returns:
            CRPS value (lower is better)
        """
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        # Approximate using samples

        # First term: expected absolute error
        term1 = np.mean(np.abs(y_samples - y_true[np.newaxis, ...]), axis=0)

        # Second term: expected absolute difference between samples
        n_samples = y_samples.shape[0]
        term2 = 0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                term2 += np.abs(y_samples[i] - y_samples[j])
        term2 = term2 / (n_samples * (n_samples - 1) / 2)

        crps_values = term1 - 0.5 * term2
        return np.mean(crps_values)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared coefficient of determination.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            R² value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def mpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Percentage Error.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MPE value (positive means under-forecasting)
        """
        mask = y_true != 0
        return np.mean((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

    @staticmethod
    def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Forecast bias.

        Measures systematic over- or under-forecasting.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Bias value
        """
        return np.mean(y_pred - y_true)


# =============================================================================
# Data Utilities
# =============================================================================


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series forecasting.

    Creates sequences from time series data for supervised learning.

    Args:
        data: Time series data (numpy array or DataFrame)
        seq_length: Length of input sequences
        forecast_horizon: Number of steps to forecast
        target_cols: Column indices or names for targets
        feature_cols: Column indices or names for features (None = all)

    Example:
        >>> dataset = TimeSeriesDataset(
        ...     data=df.values,
        ...     seq_length=24,
        ...     forecast_horizon=12,
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        seq_length: int = 24,
        forecast_horizon: int = 1,
        target_cols: Optional[Union[List[int], List[str]]] = None,
        feature_cols: Optional[Union[List[int], List[str]]] = None,
    ):
        if isinstance(data, pd.DataFrame):
            self.data = data.values.astype(np.float32)
            self.column_names = data.columns.tolist()
        else:
            self.data = data.astype(np.float32)
            self.column_names = None

        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

        # Determine feature and target columns
        if target_cols is None:
            # Use all columns as targets
            self.target_indices = list(range(self.data.shape[1]))
        elif isinstance(target_cols[0], str) and self.column_names:
            self.target_indices = [self.column_names.index(c) for c in target_cols]
        else:
            self.target_indices = target_cols

        if feature_cols is None:
            # Use all columns as features
            self.feature_indices = list(range(self.data.shape[1]))
        elif isinstance(feature_cols[0], str) and self.column_names:
            self.feature_indices = [self.column_names.index(c) for c in feature_cols]
        else:
            self.feature_indices = feature_cols

        # Calculate number of sequences
        self.n_samples = len(self.data) - seq_length - forecast_horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Data length ({len(self.data)}) must be greater than "
                f"seq_length ({seq_length}) + forecast_horizon ({forecast_horizon})"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Input sequence
        x = self.data[idx : idx + self.seq_length, self.feature_indices]

        # Target sequence
        y_start = idx + self.seq_length
        y_end = y_start + self.forecast_horizon
        y = self.data[y_start:y_end, self.target_indices]

        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_sequences(
    data: np.ndarray,
    seq_length: int,
    forecast_horizon: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform time series data into sequences for supervised learning.

    Args:
        data: Time series data (2D array: [timesteps, features])
        seq_length: Length of input sequences
        forecast_horizon: Number of steps to forecast
        stride: Step size between consecutive sequences

    Returns:
        X: Input sequences (shape: [n_samples, seq_length, n_features])
        y: Target sequences (shape: [n_samples, forecast_horizon, n_features])

    Example:
        >>> X, y = create_sequences(data, seq_length=24, forecast_horizon=12)
    """
    X, y = [], []

    for i in range(0, len(data) - seq_length - forecast_horizon + 1, stride):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + forecast_horizon])

    return np.array(X), np.array(y)


def temporal_train_test_split(
    data: Union[np.ndarray, pd.DataFrame],
    test_size: float = 0.2,
    val_size: Optional[float] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split time series data temporally (preserving temporal order).

    Args:
        data: Time series data
        test_size: Proportion of data for testing
        val_size: Optional proportion of training data for validation

    Returns:
        Split data (train, test) or (train, val, test)

    Example:
        >>> train, test = temporal_train_test_split(data, test_size=0.2)
        >>> train, val, test = temporal_train_test_split(data, test_size=0.2, val_size=0.1)
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.values
        is_dataframe = True
    else:
        data_array = data
        is_dataframe = False

    n_samples = len(data_array)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    train = data_array[:n_train]
    test = data_array[n_train:]

    if val_size is not None:
        n_val = int(n_train * val_size)
        n_train_new = n_train - n_val

        val = train[n_train_new:]
        train = train[:n_train_new]

        if is_dataframe:
            train = pd.DataFrame(train, columns=data.columns)
            val = pd.DataFrame(val, columns=data.columns)
            test = pd.DataFrame(test, columns=data.columns)

        return train, val, test

    if is_dataframe:
        train = pd.DataFrame(train, columns=data.columns)
        test = pd.DataFrame(test, columns=data.columns)

    return train, test


class ScalerType(Enum):
    """Enum for scaler types."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class TimeSeriesScaler:
    """Scaling utilities for time series data.

    Wraps sklearn scalers with additional functionality for time series.

    Args:
        scaler_type: Type of scaler to use
        feature_range: Range for MinMaxScaler (default: (0, 1))

    Example:
        >>> scaler = TimeSeriesScaler(scaler_type="standard")
        >>> train_scaled = scaler.fit_transform(train_data)
        >>> test_scaled = scaler.transform(test_data)
    """

    def __init__(
        self,
        scaler_type: Union[ScalerType, str] = ScalerType.STANDARD,
        feature_range: Tuple[float, float] = (0, 1),
    ):
        if isinstance(scaler_type, str):
            scaler_type = ScalerType(scaler_type)

        self.scaler_type = scaler_type
        self.feature_range = feature_range
        self.scaler = None
        self._is_fitted = False

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "TimeSeriesScaler":
        """Fit scaler to data."""
        if isinstance(data, pd.DataFrame):
            data = data.values

        if self.scaler_type == ScalerType.STANDARD:
            self.scaler = StandardScaler()
        elif self.scaler_type == ScalerType.MINMAX:
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
        elif self.scaler_type == ScalerType.ROBUST:
            self.scaler = RobustScaler()

        self.scaler.fit(data)
        self._is_fitted = True

        return self

    def transform(
        self,
        data: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using fitted scaler."""
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before transform")

        is_dataframe = isinstance(data, pd.DataFrame)
        columns = data.columns if is_dataframe else None

        if is_dataframe:
            data = data.values

        scaled = self.scaler.transform(data)

        if is_dataframe:
            scaled = pd.DataFrame(scaled, columns=columns)

        return scaled

    def fit_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit scaler and transform data."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Inverse transform scaled data."""
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        is_dataframe = isinstance(data, pd.DataFrame)
        columns = data.columns if is_dataframe else None

        if is_dataframe:
            data = data.values

        unscaled = self.scaler.inverse_transform(data)

        if is_dataframe:
            unscaled = pd.DataFrame(unscaled, columns=columns)

        return unscaled


# =============================================================================
# Ensemble Methods
# =============================================================================


class EnsembleForecaster(BaseForecaster):
    """Ensemble forecaster combining multiple models.

    Supports simple averaging, weighted averaging, and stacking ensemble methods.

    Args:
        forecasters: List of BaseForecaster instances
        method: Ensemble method ('average', 'weighted', 'stacking')
        weights: Optional weights for weighted ensemble
        meta_learner: Optional meta-learner for stacking

    Example:
        >>> ensemble = EnsembleForecaster(
        ...     forecasters=[lstm_forecaster, transformer_forecaster],
        ...     method='weighted',
        ... )
        >>> ensemble.fit(train_data)
        >>> predictions = ensemble.predict(test_data)
    """

    def __init__(
        self,
        forecasters: List[BaseForecaster],
        method: str = "average",
        weights: Optional[np.ndarray] = None,
        meta_learner: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.forecasters = forecasters
        self.method = method
        self.weights = weights
        self.meta_learner = meta_learner
        self.device = device

        # Create a dummy model for compatibility
        model = nn.Identity()
        super().__init__(model, device)

    def fit(
        self,
        train_data: Union[DataLoader, np.ndarray, pd.DataFrame],
        val_data: Optional[Union[DataLoader, np.ndarray, pd.DataFrame]] = None,
        **kwargs,
    ) -> "EnsembleForecaster":
        """Fit ensemble (fits individual forecasters)."""
        # Fit individual forecasters
        for i, forecaster in enumerate(self.forecasters):
            print(f"Training forecaster {i + 1}/{len(self.forecasters)}...")
            forecaster.fit(train_data, val_data, **kwargs)

        # Compute weights for weighted ensemble
        if self.method == "weighted" and self.weights is None:
            self._compute_weights(val_data or train_data)

        # Fit meta-learner for stacking
        if self.method == "stacking" and self.meta_learner is not None:
            self._fit_meta_learner(train_data, val_data)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        forecast_horizon: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate ensemble predictions."""
        # Get predictions from all forecasters
        predictions = []
        for forecaster in self.forecasters:
            pred = forecaster.predict(X, forecast_horizon, **kwargs)
            predictions.append(pred)

        predictions = np.stack(predictions)

        if self.method == "average":
            return np.mean(predictions, axis=0)

        elif self.method == "weighted":
            if self.weights is None:
                raise ValueError("Weights must be provided for weighted ensemble")
            weights = np.array(self.weights).reshape(-1, 1, 1, 1)
            return np.sum(predictions * weights, axis=0)

        elif self.method == "stacking":
            # Reshape predictions for meta-learner
            n_forecasters = len(self.forecasters)
            batch_size = predictions.shape[1]
            horizon = predictions.shape[2]
            features = predictions.shape[3]

            # Flatten and stack predictions
            stacked = predictions.transpose(1, 0, 2, 3).reshape(
                batch_size, n_forecasters * horizon * features
            )

            if self.meta_learner is not None:
                return self.meta_learner.predict(stacked)
            else:
                # Default to mean if no meta-learner
                return np.mean(predictions, axis=0)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict_interval(
        self,
        X: Union[np.ndarray, pd.DataFrame, Tensor],
        confidence: float = 0.95,
        num_samples: int = 100,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with intervals using ensemble diversity."""
        # Get all predictions from ensemble members
        all_predictions = []
        for forecaster in self.forecasters:
            pred = forecaster.predict(X, **kwargs)
            all_predictions.append(pred)

        all_predictions = np.stack(all_predictions)

        # Ensemble mean as point prediction
        mean_pred = np.mean(all_predictions, axis=0)

        # Use ensemble spread for intervals
        alpha = (1 - confidence) / 2
        lower_bounds = np.percentile(all_predictions, alpha * 100, axis=0)
        upper_bounds = np.percentile(all_predictions, (1 - alpha) * 100, axis=0)

        return mean_pred, lower_bounds, upper_bounds

    def _compute_weights(
        self,
        val_data: Union[DataLoader, np.ndarray, pd.DataFrame],
    ) -> None:
        """Compute weights based on validation performance."""
        # Extract X and y from validation data
        if isinstance(val_data, DataLoader):
            # Assume DataLoader yields (X, y) pairs
            all_y_true = []
            all_predictions = []

            for batch in val_data:
                if len(batch) == 2:
                    X_batch, y_batch = batch
                    X_batch = (
                        X_batch.cpu().numpy()
                        if isinstance(X_batch, torch.Tensor)
                        else X_batch
                    )
                    y_batch = (
                        y_batch.cpu().numpy()
                        if isinstance(y_batch, torch.Tensor)
                        else y_batch
                    )

                    all_y_true.append(y_batch)

                    forecaster_preds = []
                    for forecaster in self.forecasters:
                        pred = forecaster.predict(X_batch)
                        forecaster_preds.append(pred)
                    all_predictions.append(np.stack(forecaster_preds))

            y_true = np.concatenate(all_y_true, axis=0)
            predictions = np.concatenate(all_predictions, axis=1)
        else:
            # Assume numpy array or DataFrame
            if isinstance(val_data, pd.DataFrame):
                val_data = val_data.values

            # Split into X and y (assume y is last forecast_horizon steps)
            forecast_horizon = self.forecasters[0].forecast_horizon
            X = val_data[:-forecast_horizon]
            y_true = val_data[-forecast_horizon:]
            y_true = y_true.reshape(1, -1, val_data.shape[1])

            predictions = []
            for forecaster in self.forecasters:
                pred = forecaster.predict(X.reshape(1, -1, val_data.shape[1]))
                predictions.append(pred)
            predictions = np.stack(predictions)

        # Compute errors
        errors = []
        for i in range(len(self.forecasters)):
            mae = np.mean(np.abs(y_true - predictions[i]))
            errors.append(mae)

        # Convert errors to weights (lower error = higher weight)
        inv_errors = 1 / (np.array(errors) + 1e-8)
        self.weights = inv_errors / inv_errors.sum()

    def _fit_meta_learner(
        self,
        train_data: Union[DataLoader, np.ndarray, pd.DataFrame],
        val_data: Optional[Union[DataLoader, np.ndarray, pd.DataFrame]] = None,
    ) -> None:
        """Fit meta-learner for stacking ensemble."""
        # This is a simplified version - in practice, you'd use cross-validation
        # to generate out-of-sample predictions for training the meta-learner
        warnings.warn(
            "Stacking ensemble requires careful implementation with CV. "
            "Using simple averaging instead."
        )


# =============================================================================
# Integration with fishstick Trainer
# =============================================================================


class ForecastingTrainer:
    """Trainer specifically designed for time series forecasting.

    Integrates with fishstick's Trainer class while providing
    time series specific functionality.

    Args:
        model: PyTorch forecasting model
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device for training

    Example:
        >>> trainer = ForecastingTrainer(model, optimizer, nn.MSELoss())
        >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.current_epoch = 0
        self.global_step = 0
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 2:
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
            else:
                data = batch.to(self.device)
                target = None

            output = self.model(data)

            if target is not None:
                loss = self.criterion(output, target)
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                total_loss += loss.item() * self.gradient_accumulation_steps

        self.current_epoch += 1
        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)
                else:
                    continue

                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        return self.history


# =============================================================================
# Convenience Functions
# =============================================================================


def create_forecaster(
    model_type: str, input_dim: int, forecast_horizon: int = 1, **kwargs
) -> BaseForecaster:
    """Factory function to create forecasters.

    Args:
        model_type: Type of model ('lstm', 'transformer', 'nbeats', 'deepar')
        input_dim: Number of input features
        forecast_horizon: Number of steps to forecast
        **kwargs: Additional model-specific arguments

    Returns:
        BaseForecaster instance

    Example:
        >>> forecaster = create_forecaster('lstm', input_dim=10, forecast_horizon=24)
    """
    if model_type.lower() == "lstm":
        return LSTMForecaster(
            input_dim=input_dim, forecast_horizon=forecast_horizon, **kwargs
        )
    elif model_type.lower() == "transformer":
        return TransformerForecaster(
            input_dim=input_dim, forecast_horizon=forecast_horizon, **kwargs
        )
    elif model_type.lower() == "nbeats":
        return NBeatsForecaster(
            input_dim=input_dim, forecast_horizon=forecast_horizon, **kwargs
        )
    elif model_type.lower() == "deepar":
        return DeepARForecaster(
            input_dim=input_dim, forecast_horizon=forecast_horizon, **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Export symbols
# =============================================================================

__all__ = [
    # Base classes
    "BaseForecaster",
    "ForecastingTrainer",
    # Deep learning forecasters
    "LSTMForecaster",
    "TransformerForecaster",
    "NBeatsForecaster",
    "DeepARForecaster",
    # Feature engineering
    "FeatureEngineer",
    # Metrics
    "TimeSeriesMetrics",
    # Data utilities
    "TimeSeriesDataset",
    "create_sequences",
    "temporal_train_test_split",
    "TimeSeriesScaler",
    "ScalerType",
    # Ensemble
    "EnsembleForecaster",
    # Convenience functions
    "create_forecaster",
]
