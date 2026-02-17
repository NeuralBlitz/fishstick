"""
Time Series Preprocessing Tools
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, date


class TimeSeriesScaler:
    """Scalers for time series data."""

    @staticmethod
    def standard(series: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """StandardScaler."""
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / std, mean, std

    @staticmethod
    def minmax(
        series: np.ndarray, feature_range: Tuple[float, float] = (0, 1)
    ) -> Tuple[np.ndarray, float, float]:
        """MinMaxScaler."""
        min_val = np.min(series)
        max_val = np.max(series)
        a, b = feature_range
        scaled = a + (series - min_val) * (b - a) / (max_val - min_val)
        return scaled, min_val, max_val

    @staticmethod
    def robust(series: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """RobustScaler using median and IQR."""
        median = np.median(series)
        q75, q25 = np.percentile(series, [75, 25])
        iqr = q75 - q25
        return (series - median) / iqr, median, iqr

    @staticmethod
    def inverse_standard(scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Inverse StandardScaler."""
        return scaled * std + mean

    @staticmethod
    def inverse_minmax(
        scaled: np.ndarray,
        min_val: float,
        max_val: float,
        feature_range: Tuple[float, float] = (0, 1),
    ) -> np.ndarray:
        """Inverse MinMaxScaler."""
        a, b = feature_range
        return min_val + (scaled - a) * (max_val - min_val) / (b - a)


class SlidingWindow:
    """Create sliding windows for time series."""

    def __init__(self, window_size: int, horizon: int = 1, stride: int = 1):
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride

    def transform(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform series to windows."""
        X, y = [], []

        for i in range(
            0, len(series) - self.window_size - self.horizon + 1, self.stride
        ):
            X.append(series[i : i + self.window_size])
            y.append(series[i + self.window_size : i + self.window_size + self.horizon])

        return np.array(X), np.array(y)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform windows to series."""
        if X.ndim == 2:
            return X.flatten()
        return X


class HolidayFeatures:
    """Extract holiday features."""

    def __init__(self, country: str = "US"):
        self.country = country
        self.holidays = self._get_holidays()

    def _get_holidays(self) -> set:
        try:
            import holidays

            return set(holidays.US(years=range(2010, 2030)).keys())
        except ImportError:
            return set()

    def transform(self, dates: list) -> np.ndarray:
        """Generate holiday features."""
        features = []
        for d in dates:
            if isinstance(d, str):
                d = pd.to_datetime(d)
            features.append(1 if d.date() in self.holidays else 0)
        return np.array(features)

    def is_holiday(self, date: date) -> bool:
        """Check if date is a holiday."""
        return date in self.holidays


class CalendarFeatures:
    """Extract calendar-based features."""

    @staticmethod
    def transform(dates: list) -> np.ndarray:
        """Generate calendar features."""
        features = []

        for d in dates:
            if isinstance(d, str):
                d = pd.to_datetime(d)

            day_of_week = d.dayofweek
            day_of_month = d.day
            month = d.month
            quarter = d.quarter
            week_of_year = d.isocalendar()[1]
            is_weekend = 1 if day_of_week >= 5 else 0
            is_month_start = 1 if d.is_month_start else 0
            is_month_end = 1 if d.is_month_end else 0

            features.append(
                [
                    day_of_week,
                    day_of_month,
                    month,
                    quarter,
                    week_of_year,
                    is_weekend,
                    is_month_start,
                    is_month_end,
                ]
            )

        return np.array(features)

    @staticmethod
    def fourier_features(dates: list, n_order: int = 3) -> np.ndarray:
        """Generate Fourier features for seasonality."""
        features = []

        for d in dates:
            if isinstance(d, str):
                d = pd.to_datetime(d)

            day_of_year = d.timetuple().tm_yday

            fourier = []
            for k in range(1, n_order + 1):
                fourier.append(np.sin(2 * np.pi * k * day_of_year / 365.25))
                fourier.append(np.cos(2 * np.pi * k * day_of_year / 365.25))

            features.append(fourier)

        return np.array(features)


class LagFeatures:
    """Create lag features."""

    def __init__(self, lags: list):
        self.lags = lags

    def transform(self, series: np.ndarray) -> np.ndarray:
        """Create lag features."""
        n = len(series)
        max_lag = max(self.lags)
        features = []

        for lag in self.lags:
            lag_feature = np.full(n, np.nan)
            lag_feature[lag:] = series[:-lag]
            features.append(lag_feature)

        return np.column_stack(features)


class DifferenceTransformer:
    """Apply differencing to make series stationary."""

    def __init__(self, order: int = 1):
        self.order = order

    def transform(self, series: np.ndarray) -> np.ndarray:
        """Apply differencing."""
        diff = series.copy()
        for _ in range(self.order):
            diff = np.diff(diff)
        return diff

    def inverse_transform(self, series: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Inverse differencing."""
        result = series.copy()
        for _ in range(self.order):
            result = np.cumsum(result)
            result += original[: len(result)]
        return result


class RollingStatistics:
    """Compute rolling statistics."""

    @staticmethod
    def mean(series: np.ndarray, window: int) -> np.ndarray:
        """Rolling mean."""
        return np.convolve(series, np.ones(window) / window, mode="same")

    @staticmethod
    def std(series: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation."""
        result = np.zeros_like(series)
        for i in range(len(series)):
            start = max(0, i - window // 2)
            end = min(len(series), i + window // 2)
            result[i] = np.std(series[start:end])
        return result

    @staticmethod
    def min(series: np.ndarray, window: int) -> np.ndarray:
        """Rolling min."""
        result = np.zeros_like(series)
        for i in range(len(series)):
            start = max(0, i - window // 2)
            end = min(len(series), i + window // 2)
            result[i] = np.min(series[start:end])
        return result

    @staticmethod
    def max(series: np.ndarray, window: int) -> np.ndarray:
        """Rolling max."""
        result = np.zeros_like(series)
        for i in range(len(series)):
            start = max(0, i - window // 2)
            end = min(len(series), i + window // 2)
            result[i] = np.max(series[start:end])
        return result


class OutlierDetector:
    """Detect outliers in time series."""

    @staticmethod
    def zscore(series: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Z-score based outlier detection."""
        z = np.abs((series - np.mean(series)) / np.std(series))
        return z > threshold

    @staticmethod
    def iqr(series: np.ndarray, k: float = 1.5) -> np.ndarray:
        """IQR based outlier detection."""
        q75, q25 = np.percentile(series, [75, 25])
        iqr = q75 - q25
        lower = q25 - k * iqr
        upper = q75 + k * iqr
        return (series < lower) | (series > upper)

    @staticmethod
    def mad(series: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """Median Absolute Deviation based outlier detection."""
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        return np.abs(series - median) > threshold * mad
