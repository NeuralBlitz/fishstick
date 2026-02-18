"""
Time Series Anomaly Detection Module.

This module provides time series anomaly detection methods:
- Statistical methods (Z-score, IQR, etc. for time series)
- LSTM-based sequence modeling
- Transformer-based attention detection
- Seasonal decomposition methods
- Prophet-based forecasting detection
- Change point detection
- Spectral analysis methods

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy import signal, stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class TimeSeriesResult:
    """Container for time series anomaly detection results."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    n_anomalies: int
    anomaly_indices: np.ndarray
    timestamps: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None


class BaseTimeSeriesDetector(ABC):
    """Base class for time series anomaly detectors."""

    def __init__(
        self,
        contamination: float = 0.1,
        window_size: int = 10,
        stride: int = 1,
    ):
        self.contamination = contamination
        self.window_size = window_size
        self.stride = stride
        self.threshold: Optional[float] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseTimeSeriesDetector":
        """Fit the detector on normal time series data."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            self.threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return (scores > self.threshold).astype(int)

    def fit_predict(self, X: np.ndarray) -> TimeSeriesResult:
        """Fit and predict in one call."""
        self.fit(X)
        scores = self.score(X)
        labels = self.predict(X)
        return TimeSeriesResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            n_anomalies=int(np.sum(labels)),
            anomaly_indices=np.where(labels == 1)[0],
        )


class StatisticalTimeSeriesDetector(BaseTimeSeriesDetector):
    """
    Statistical time series anomaly detector.

    Uses rolling statistics (mean, std, z-score) to detect anomalies.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    window_size : int
        Rolling window size.
    method : str
        Method: 'zscore', 'iqr', 'mad', 'gesd'.
    threshold : float
        Z-score threshold for anomaly detection.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        window_size: int = 10,
        method: str = "zscore",
        threshold: float = 3.0,
    ):
        super().__init__(
            contamination=contamination,
            window_size=window_size,
        )
        self.method = method
        self.threshold_val = threshold
        self.rolling_mean: Optional[np.ndarray] = None
        self.rolling_std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StatisticalTimeSeriesDetector":
        """Fit the statistical detector."""
        X = np.asarray(X).flatten()
        self.is_fitted = True
        return self

    def _compute_rolling_stats(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rolling mean and std."""
        n = len(X)
        rolling_mean = np.zeros(n)
        rolling_std = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.window_size + 1)
            window = X[start : i + 1]
            rolling_mean[i] = np.mean(window)
            rolling_std[i] = np.std(window) + 1e-10

        return rolling_mean, rolling_std

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        X = np.asarray(X).flatten()

        if self.method == "zscore":
            rolling_mean, rolling_std = self._compute_rolling_stats(X)
            scores = np.abs((X - rolling_mean) / rolling_std)
        elif self.method == "iqr":
            scores = self._iqr_score(X)
        elif self.method == "mad":
            scores = self._mad_score(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return scores

    def _iqr_score(self, X: np.ndarray) -> np.ndarray:
        """Compute IQR-based anomaly scores."""
        n = len(X)
        scores = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.window_size + 1)
            window = X[start : i + 1]
            q1 = np.percentile(window, 25)
            q3 = np.percentile(window, 75)
            iqr = q3 - q1
            median = np.median(window)
            scores[i] = abs((X[i] - median) / (iqr + 1e-10)) if iqr > 0 else 0

        return scores

    def _mad_score(self, X: np.ndarray) -> np.ndarray:
        """Compute MAD-based anomaly scores."""
        n = len(X)
        scores = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.window_size + 1)
            window = X[start : i + 1]
            median = np.median(window)
            mad = np.median(np.abs(window - median)) + 1e-10
            scores[i] = np.abs((X[i] - median) / (mad * 1.4826))

        return scores


class LSTMAnomalyDetector(BaseTimeSeriesDetector):
    """
    LSTM-based time series anomaly detector.

    Uses LSTM to model normal time series patterns and detects
    anomalies based on reconstruction error.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    window_size : int
        Sequence window size.
    hidden_size : int
        LSTM hidden size.
    num_layers : int
        Number of LSTM layers.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        window_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 50,
        lr: float = 1e-3,
    ):
        super().__init__(
            contamination=contamination,
            window_size=window_size,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()

    def _build_model(self, input_dim: int) -> "torch.nn.Module":
        """Build LSTM model."""
        import torch
        import torch.nn as nn

        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_size, num_layers):
                super().__init__()
                self.encoder = nn.LSTM(
                    input_dim,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                )
                self.decoder = nn.LSTM(
                    hidden_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, input_dim)

            def forward(self, x):
                _, (h, _) = self.encoder(x)
                h = h[-1].unsqueeze(0).repeat(x.size(1), 1).unsqueeze(0)
                out, _ = self.decoder(h)
                out = self.fc(out.squeeze(0))
                return out

        return LSTMAutoencoder(input_dim, self.hidden_size, self.num_layers)

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM training."""
        sequences = []
        for i in range(len(X) - self.window_size + 1):
            sequences.append(X[i : i + self.window_size])
        return np.array(sequences)

    def fit(self, X: np.ndarray) -> "LSTMAnomalyDetector":
        """Fit the LSTM detector."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.fit_transform(X)

        sequences = self._create_sequences(X_scaled)
        if len(sequences) == 0:
            sequences = X_scaled.reshape(1, -1, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(X.shape[1]).to(device)

        dataset = TensorDataset(torch.FloatTensor(sequences))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, x[:, -1, :])
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (reconstruction error)."""
        import torch

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)

        sequences = self._create_sequences(X_scaled)
        if len(sequences) == 0:
            sequences = X_scaled.reshape(1, -1, 1)

        device = next(self.model.parameters()).device
        self.model.eval()

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequences).to(device)
            reconstructions = self.model(seq_tensor)
            errors = (
                torch.mean((reconstructions - seq_tensor[:, -1, :]) ** 2, dim=1)
                .cpu()
                .numpy()
            )

        scores = np.zeros(len(X))
        scores[self.window_size - 1 :] = errors
        scores[: self.window_size - 1] = errors[0] if len(errors) > 0 else 0

        return scores


class TransformerAnomalyDetector(BaseTimeSeriesDetector):
    """
    Transformer-based time series anomaly detector.

    Uses self-attention to capture temporal dependencies and
    detect anomalies based on prediction error.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    window_size : int
        Sequence window size.
    d_model : int
        Model dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    epochs : int
        Training epochs.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        window_size: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        epochs: int = 50,
    ):
        super().__init__(
            contamination=contamination,
            window_size=window_size,
        )
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()

    def _build_model(self, input_dim: int) -> "torch.nn.Module":
        """Build Transformer model."""
        import torch
        import torch.nn as nn

        class TransformerPredictor(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(d_model, input_dim)

            def forward(self, x):
                x = self.input_proj(x)
                x = self.transformer(x)
                x = self.fc(x[:, -1, :])
                return x

        return TransformerPredictor(
            input_dim, self.d_model, self.nhead, self.num_layers
        )

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences."""
        sequences = []
        for i in range(len(X) - self.window_size + 1):
            sequences.append(X[i : i + self.window_size])
        return np.array(sequences)

    def fit(self, X: np.ndarray) -> "TransformerAnomalyDetector":
        """Fit the Transformer detector."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.fit_transform(X)
        sequences = self._create_sequences(X_scaled)

        if len(sequences) == 0:
            sequences = X_scaled.reshape(1, -1, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(X.shape[1]).to(device)

        dataset = TensorDataset(torch.FloatTensor(sequences))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, x[:, -1, :])
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        import torch

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)
        sequences = self._create_sequences(X_scaled)

        if len(sequences) == 0:
            sequences = X_scaled.reshape(1, -1, 1)

        device = next(self.model.parameters()).device
        self.model.eval()

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequences).to(device)
            predictions = self.model(seq_tensor)
            errors = (
                torch.mean((predictions - seq_tensor[:, -1, :]) ** 2, dim=1)
                .cpu()
                .numpy()
            )

        scores = np.zeros(len(X))
        scores[self.window_size - 1 :] = errors
        scores[: self.window_size - 1] = errors[0] if len(errors) > 0 else 0

        return scores


class SeasonalDecompositionDetector(BaseTimeSeriesDetector):
    """
    Seasonal decomposition-based anomaly detector.

    Decomposes time series into trend, seasonal, and residual components.
    Anomalies are detected in the residual component.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    period : int
        Seasonal period.
    model : str
        Decomposition model: 'additive', 'multiplicative'.
    threshold : float
        Residual threshold in standard deviations.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        period: int = 12,
        model: str = "additive",
        threshold: float = 3.0,
    ):
        super().__init__(
            contamination=contamination,
            window_size=period,
        )
        self.period = period
        self.model = model
        self.threshold_val = threshold
        self.trend: Optional[np.ndarray] = None
        self.seasonal: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SeasonalDecompositionDetector":
        """Fit the detector (computes seasonal pattern)."""
        X = np.asarray(X).flatten()
        self._decompose(X)
        self.is_fitted = True
        return self

    def _decompose(self, X: np.ndarray) -> None:
        """Perform seasonal decomposition."""
        n = len(X)
        self.trend = self._rolling_mean(X, self.period)
        detrended = X - self.trend

        self.seasonal = np.zeros(n)
        seasonal_pattern = np.zeros(self.period)
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            if len(indices) > 0:
                seasonal_pattern[i] = np.mean(detrended[indices])

        seasonal_pattern -= np.mean(seasonal_pattern)
        for i in range(n):
            self.seasonal[i] = seasonal_pattern[i % self.period]

    def _rolling_mean(self, X: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean for trend."""
        result = np.zeros_like(X)
        half = window // 2

        for i in range(len(X)):
            start = max(0, i - half)
            end = min(len(X), i + half + 1)
            result[i] = np.mean(X[start:end])

        return result

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores from residual component."""
        X = np.asarray(X).flatten()

        if self.trend is None:
            self._decompose(X)

        if self.model == "additive":
            residual = X - self.trend - self.seasonal
        else:
            seasonal_abs = np.abs(self.seasonal) + 1e-10
            residual = (X - self.trend) / seasonal_abs - np.sign(self.seasonal)

        residual_std = np.std(residual)
        scores = np.abs(residual) / (residual_std + 1e-10)

        return scores


class ChangePointDetector:
    """
    Change point detection for time series.

    Detects abrupt changes in time series distribution.

    Parameters
    ----------
    method : str
        Detection method: 'cusum', 'bayesian', 'pelt'.
    threshold : float
        Detection threshold.
    """

    def __init__(
        self,
        method: str = "cusum",
        threshold: float = 5.0,
    ):
        self.method = method
        self.threshold = threshold
        self.change_points: List[int] = []

    def fit_detect(self, X: np.ndarray) -> List[int]:
        """Detect change points."""
        X = np.asarray(X).flatten()

        if self.method == "cusum":
            self.change_points = self._cusum(X)
        elif self.method == "bayesian":
            self.change_points = self._bayesian(X)
        elif self.method == "pelt":
            self.change_points = self._pelt(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self.change_points

    def _cusum(self, X: np.ndarray) -> List[int]:
        """CUSUM change point detection."""
        mean = np.mean(X)
        std = np.std(X) + 1e-10

        X_norm = (X - mean) / std
        cusum_pos = np.zeros(len(X))
        cusum_neg = np.zeros(len(X))

        for i in range(1, len(X)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + X_norm[i] - 1)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - X_norm[i] - 1)

        combined = np.maximum(cusum_pos, cusum_neg)
        change_points = np.where(combined > self.threshold)[0].tolist()

        return change_points

    def _bayesian(self, X: np.ndarray) -> List[int]:
        """Bayesian online change point detection."""
        change_points = []
        run_length = 0
        hazard = 1 / len(X)
        log_evidence = 0

        for i in range(1, len(X)):
            prior = np.array([hazard, 1 - hazard])
            posterior = prior * np.array(
                [
                    np.exp(log_evidence),
                    1 - np.exp(log_evidence),
                ]
            )
            posterior /= np.sum(posterior)

            if posterior[0] > 0.5:
                if run_length > 10:
                    change_points.append(i)
                run_length = 0
                log_evidence = 0
            else:
                run_length += 1
                log_evidence += np.log(1 + (i - run_length) * hazard)

        return change_points

    def _pelt(self, X: np.ndarray) -> List[int]:
        """PELT (Pruned Exact Linear Time) change point detection."""
        n = len(X)

        def cost(start: int, end: int) -> float:
            segment = X[start:end]
            return len(segment) * np.var(segment) if len(segment) > 1 else 0

        R = [0]
        cps = []

        for t in range(1, n):
            costs = [cost(r, t + 1) + (len(R) > 0 and R[-1] > 0) for r in R + [0]]
            min_cost_idx = np.argmin(costs)
            R.append(costs[min_cost_idx])

            if min_cost_idx == len(R) - 1:
                cps.append(t)

        return cps


class SpectralAnomalyDetector(BaseTimeSeriesDetector):
    """
    Spectral anomaly detector using frequency domain analysis.

    Detects anomalies based on spectral density changes.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    window_size : int
        FFT window size.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        window_size: int = 64,
    ):
        super().__init__(
            contamination=contamination,
            window_size=window_size,
        )
        self.baseline_spectrum: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SpectralAnomalyDetector":
        """Fit baseline spectrum."""
        X = np.asarray(X).flatten()
        padded = self._pad_to_window(X)
        spectrum = np.abs(fft(padded))[: len(padded) // 2]
        self.baseline_spectrum = spectrum / (np.linalg.norm(spectrum) + 1e-10)
        self.is_fitted = True
        return self

    def _pad_to_window(self, X: np.ndarray) -> np.ndarray:
        """Pad or truncate to window size."""
        if len(X) >= self.window_size:
            return X[: self.window_size]
        else:
            return np.pad(X, (0, self.window_size - len(X)), mode="constant")

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute spectral anomaly scores."""
        X = np.asarray(X).flatten()
        n = len(X)
        scores = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.window_size + 1)
            window = X[start : i + 1]
            padded = self._pad_to_window(window)
            spectrum = np.abs(fft(padded))[: len(padded) // 2]
            spectrum_norm = spectrum / (np.linalg.norm(spectrum) + 1e-10)

            if self.baseline_spectrum is not None:
                min_len = min(len(spectrum_norm), len(self.baseline_spectrum))
                scores[i] = np.linalg.norm(
                    spectrum_norm[:min_len] - self.baseline_spectrum[:min_len]
                )

        return scores


class KSigmaDetector(BaseTimeSeriesDetector):
    """
    K-Sigma detector with exponential moving average.

    Uses EMA for adaptive thresholding.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    k : float
        Number of standard deviations.
    alpha : float
        EMA smoothing factor.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        k: float = 3.0,
        alpha: float = 0.3,
    ):
        super().__init__(contamination=contamination, window_size=1)
        self.k = k
        self.alpha = alpha

    def fit(self, X: np.ndarray) -> "KSigmaDetector":
        """Fit the detector."""
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores with EMA."""
        X = np.asarray(X).flatten()
        n = len(X)
        scores = np.zeros(n)

        ema = X[0]
        ema_var = 0

        for i in range(n):
            diff = X[i] - ema
            ema += self.alpha * diff

            ema_var = (1 - self.alpha) * (ema_var + self.alpha * diff**2)
            std = np.sqrt(ema_var) + 1e-10

            scores[i] = abs(diff) / std

        return scores
