"""
Financial Time Series Module

Advanced time series processing and feature engineering for financial data.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor


class FinancialTimeSeries:
    """Base class for financial time series data."""

    def __init__(
        self, prices: Union[ndarray, Tensor], timestamps: Optional[ndarray] = None
    ):
        self.prices = torch.tensor(prices) if isinstance(prices, ndarray) else prices
        self.timestamps = timestamps
        self.returns: Optional[Tensor] = None
        self.log_returns: Optional[Tensor] = None

    def compute_returns(self, period: int = 1) -> Tensor:
        """Compute simple returns."""
        if period == 1:
            self.returns = torch.diff(self.prices) / self.prices[:-1]
        else:
            self.returns = (self.prices[period:] - self.prices[:-period]) / self.prices[
                :-period
            ]
        return self.returns

    def compute_log_returns(self, period: int = 1) -> Tensor:
        """Compute log returns."""
        if period == 1:
            self.log_returns = torch.log(self.prices[1:] / self.prices[:-1])
        else:
            self.log_returns = torch.log(self.prices[period:] / self.prices[:-period])
        return self.log_returns


class TechnicalIndicators:
    """Technical indicators for financial time series."""

    @staticmethod
    def sma(prices: Tensor, window: int) -> Tensor:
        """Simple Moving Average."""
        if len(prices) < window:
            raise ValueError(
                f"Window size {window} exceeds series length {len(prices)}"
            )
        return torch.nn.functional.avg_pool1d(
            prices.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=window,
            stride=1,
        ).squeeze()

    @staticmethod
    def ema(prices: Tensor, span: int) -> Tensor:
        """Exponential Moving Average."""
        alpha = 2.0 / (span + 1)
        ema_values = [prices[0]]
        for i in range(1, len(prices)):
            ema_values.append(alpha * prices[i] + (1 - alpha) * ema_values[-1])
        return torch.tensor(ema_values)

    @staticmethod
    def rsi(prices: Tensor, window: int = 14) -> Tensor:
        """Relative Strength Index."""
        deltas = torch.diff(prices)
        gains = torch.clamp(deltas, min=0)
        losses = torch.clamp(-deltas, min=0)

        avg_gains = torch.nn.functional.avg_pool1d(
            gains.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=window,
            stride=1,
        ).squeeze()
        avg_losses = torch.nn.functional.avg_pool1d(
            losses.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=window,
            stride=1,
        ).squeeze()

        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(
        prices: Tensor,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """MACD (Moving Average Convergence Divergence)."""
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        prices: Tensor,
        window: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Bollinger Bands."""
        sma = TechnicalIndicators.sma(prices, window)
        std = torch.std(prices[:window])
        upper = sma + num_std * std
        lower = sma - num_std * std
        return upper, sma, lower

    @staticmethod
    def atr(high: Tensor, low: Tensor, close: Tensor, window: int = 14) -> Tensor:
        """Average True Range."""
        tr1 = high[1:] - low[1:]
        tr2 = torch.abs(high[1:] - close[:-1])
        tr3 = torch.abs(low[1:] - close[:-1])
        tr = torch.maximum(torch.maximum(tr1, tr2), tr3)
        atr = torch.nn.functional.avg_pool1d(
            tr.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=window,
            stride=1,
        ).squeeze()
        return atr


class OHLCVData:
    """OHLCV (Open, High, Low, Close, Volume) data handler."""

    def __init__(
        self,
        open_prices: Tensor,
        high: Tensor,
        low: Tensor,
        close: Tensor,
        volume: Tensor,
    ):
        self.open = open_prices
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def typical_price(self) -> Tensor:
        """Typical price = (High + Low + Close) / 3"""
        return (self.high + self.low + self.close) / 3

    def weighted_price(self) -> Tensor:
        """Weighted price = (Open + High + Low + Close) / 4"""
        return (self.open + self.high + self.low + self.close) / 4

    def true_range(self) -> Tensor:
        """True Range."""
        tr1 = self.high[1:] - self.low[1:]
        tr2 = torch.abs(self.high[1:] - self.close[:-1])
        tr3 = torch.abs(self.low[1:] - self.close[:-1])
        return torch.maximum(torch.maximum(tr1, tr2), tr3)

    def on_balance_volume(self) -> Tensor:
        """On-Balance Volume."""
        obv = [0.0]
        for i in range(1, len(self.close)):
            if self.close[i] > self.close[i - 1]:
                obv.append(obv[-1] + self.volume[i])
            elif self.close[i] < self.close[i - 1]:
                obv.append(obv[-1] - self.volume[i])
            else:
                obv.append(obv[-1])
        return torch.tensor(obv)


class FinancialFeatureEngineer:
    """Feature engineering for financial machine learning."""

    @staticmethod
    def create_lag_features(data: Tensor, lags: list[int]) -> Tensor:
        """Create lagged features."""
        features = [data]
        for lag in lags:
            features.append(torch.roll(data, lag))
        return torch.stack(features, dim=1)

    @staticmethod
    def rolling_statistics(
        data: Tensor,
        windows: list[int],
    ) -> Tensor:
        """Compute rolling statistics (mean, std, skew, kurtosis)."""
        features = []
        for window in windows:
            rolled = FinancialFeatureEngineer._rolling_view(data, window)
            features.append(torch.mean(rolled, dim=1))
            features.append(torch.std(rolled, dim=1))
        return torch.stack(features, dim=1)

    @staticmethod
    def _rolling_view(a: Tensor, window: int) -> Tensor:
        """Create rolling window view of tensor."""
        shape = (a.size(0) - window + 1, window)
        strides = (a.stride(0), a.stride(0))
        return torch.as_strided(a, shape, strides)

    @staticmethod
    def fourier_features(data: Tensor, n_components: int = 5) -> Tensor:
        """Fourier transform features."""
        fft = torch.fft.fft(data)
        magnitude = torch.abs(fft[:n_components])
        return magnitude

    @staticmethod
    def volatility_features(
        returns: Tensor,
        windows: list[int] = [5, 10, 20, 60],
    ) -> Tensor:
        """Compute volatility at different horizons."""
        features = []
        for window in windows:
            if len(returns) >= window:
                rolled = FinancialFeatureEngineer._rolling_view(returns, window)
                vol = torch.std(rolled, dim=1)
                features.append(vol)
        return torch.stack(features, dim=1) if features else returns


class FinancialScaler:
    """Scaling for financial time series."""

    def __init__(self, method: str = "zscore"):
        self.method = method
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.min: Optional[float] = None
        self.max: Optional[float] = None

    def fit(self, data: Tensor) -> "FinancialScaler":
        """Fit scaler parameters."""
        if self.method == "zscore":
            self.mean = float(torch.mean(data))
            self.std = float(torch.std(data)) + 1e-8
        elif self.method == "minmax":
            self.min = float(torch.min(data))
            self.max = float(torch.max(data)) + 1e-8
        return self

    def transform(self, data: Tensor) -> Tensor:
        """Transform data."""
        if self.method == "zscore":
            return (data - self.mean) / self.std
        elif self.method == "minmax":
            return (data - self.min) / (self.max - self.min)
        return data

    def fit_transform(self, data: Tensor) -> Tensor:
        """Fit and transform."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: Tensor) -> Tensor:
        """Inverse transform."""
        if self.method == "zscore":
            return data * self.std + self.mean
        elif self.method == "minmax":
            return data * (self.max - self.min) + self.min
        return data
