"""
Fishstick Time Series Module

A comprehensive time series forecasting and analysis library.
Includes models, preprocessing, feature engineering, evaluation, and utilities.
"""

import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from scipy.stats import norm, multivariate_normal
from scipy.fft import fft, ifft
from scipy.signal import periodogram
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")


# ============================================================================
# Type Definitions
# ============================================================================

T = TypeVar("T")
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]
TimeSeriesData = Union[pd.Series, pd.DataFrame]


class ModelProtocol(Protocol):
    """Protocol for time series models."""

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "ModelProtocol": ...

    def predict(self, X: ArrayLike) -> np.ndarray: ...


# ============================================================================
# Preprocessing
# ============================================================================


class TimeSeriesScaler(BaseEstimator, TransformerMixin):
    """
    Time series aware scaling with multiple strategies.

    Supports: standard, minmax, robust, log, box-cox
    """

    def __init__(
        self,
        method: Literal["standard", "minmax", "robust", "log", "box-cox"] = "standard",
        feature_range: Tuple[float, float] = (0, 1),
    ):
        self.method = method
        self.feature_range = feature_range
        self.scaler_: Optional[Any] = None
        self.lambda_: Optional[float] = None

    def fit(self, X: ArrayLike, y=None):
        X = self._to_numpy(X)

        if self.method == "standard":
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X.reshape(-1, 1) if X.ndim == 1 else X)
        elif self.method == "minmax":
            self.scaler_ = MinMaxScaler(feature_range=self.feature_range)
            self.scaler_.fit(X.reshape(-1, 1) if X.ndim == 1 else X)
        elif self.method == "robust":
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X.reshape(-1, 1) if X.ndim == 1 else X)
        elif self.method == "log":
            if np.any(X <= 0):
                raise ValueError("Log transform requires positive values")
        elif self.method == "box-cox":
            from scipy.stats import boxcox

            if np.any(X <= 0):
                raise ValueError("Box-Cox transform requires positive values")
            _, self.lambda_ = boxcox(X.flatten())

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        X = self._to_numpy(X)

        if self.method in ["standard", "minmax", "robust"]:
            return (
                self.scaler_.transform(X.reshape(-1, 1)).flatten()
                if X.ndim == 1
                else self.scaler_.transform(X)
            )
        elif self.method == "log":
            return np.log(X)
        elif self.method == "box-cox":
            from scipy.stats import boxcox

            if self.lambda_ is None:
                raise ValueError("Scaler not fitted")
            return boxcox(X.flatten(), lmbda=self.lambda_)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        X = self._to_numpy(X)

        if self.method in ["standard", "minmax", "robust"]:
            return (
                self.scaler_.inverse_transform(X.reshape(-1, 1)).flatten()
                if X.ndim == 1
                else self.scaler_.inverse_transform(X)
            )
        elif self.method == "log":
            return np.exp(X)
        elif self.method == "box-cox":
            if self.lambda_ is None:
                raise ValueError("Scaler not fitted")
            if self.lambda_ == 0:
                return np.exp(X)
            else:
                return np.power(self.lambda_ * X + 1, 1 / self.lambda_)

    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, pd.Series):
            return X.values
        elif isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """
    Time series specific imputation methods.

    Supports: forward_fill, backward_fill, linear, spline, seasonal,
              moving_average, kalman, knn
    """

    def __init__(
        self,
        method: Literal[
            "forward_fill",
            "backward_fill",
            "linear",
            "spline",
            "seasonal",
            "moving_average",
            "kalman",
            "knn",
        ] = "linear",
        seasonal_period: int = 7,
        window: int = 3,
        n_neighbors: int = 5,
    ):
        self.method = method
        self.seasonal_period = seasonal_period
        self.window = window
        self.n_neighbors = n_neighbors

    def fit(self, X: ArrayLike, y=None):
        X = self._to_series(X)
        self.mean_ = X.mean()
        self.std_ = X.std()
        return self

    def transform(self, X: ArrayLike) -> pd.Series:
        X = self._to_series(X).copy()
        mask = X.isna()

        if not mask.any():
            return X

        if self.method == "forward_fill":
            X = X.fillna(method="ffill")
            X = X.fillna(method="bfill")  # Fill leading NaNs
        elif self.method == "backward_fill":
            X = X.fillna(method="bfill")
            X = X.fillna(method="ffill")  # Fill trailing NaNs
        elif self.method == "linear":
            X = X.interpolate(method="linear")
            X = X.fillna(method="bfill").fillna(method="ffill")
        elif self.method == "spline":
            X = X.interpolate(method="spline", order=3)
            X = X.fillna(method="bfill").fillna(method="ffill")
        elif self.method == "seasonal":
            X = self._seasonal_interpolate(X)
        elif self.method == "moving_average":
            X = self._moving_average_interpolate(X)
        elif self.method == "kalman":
            X = self._kalman_interpolate(X)
        elif self.method == "knn":
            X = self._knn_interpolate(X)

        return X

    def _seasonal_interpolate(self, X: pd.Series) -> pd.Series:
        """Use seasonal decomposition for interpolation."""
        from statsmodels.tsa.seasonal import seasonal_decompose

        mask = X.isna()
        if mask.sum() == 0:
            return X

        # Simple forward fill for initial decomposition
        temp = X.fillna(method="ffill").fillna(method="bfill")

        try:
            decomposition = seasonal_decompose(
                temp, period=self.seasonal_period, model="additive"
            )
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Interpolate trend
            trend = trend.interpolate(method="linear")
            trend = trend.fillna(method="bfill").fillna(method="ffill")

            # Reconstruct
            X_filled = trend + seasonal
            X[mask] = X_filled[mask]
        except:
            # Fall back to linear interpolation
            X = X.interpolate(method="linear")

        return X.fillna(method="bfill").fillna(method="ffill")

    def _moving_average_interpolate(self, X: pd.Series) -> pd.Series:
        """Use centered moving average for interpolation."""
        X = X.copy()
        mask = X.isna()

        for idx in X[mask].index:
            left_idx = max(0, X.index.get_loc(idx) - self.window)
            right_idx = min(len(X), X.index.get_loc(idx) + self.window + 1)
            window_values = X.iloc[left_idx:right_idx].dropna()

            if len(window_values) > 0:
                X.loc[idx] = window_values.mean()

        return X.fillna(method="bfill").fillna(method="ffill")

    def _kalman_interpolate(self, X: pd.Series) -> pd.Series:
        """Use Kalman filter for imputation."""
        try:
            from pykalman import KalmanFilter

            X_array = X.values
            mask = ~np.isnan(X_array)

            if mask.sum() < 2:
                return X.interpolate(method="linear")

            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            kf = kf.em(X_array[mask].reshape(-1, 1))
            state_means, _ = kf.filter(X_array.reshape(-1, 1))

            X_filled = X.copy()
            X_filled.loc[X.isna()] = state_means[X.isna()].flatten()
            return X_filled
        except ImportError:
            return X.interpolate(method="linear")

    def _knn_interpolate(self, X: pd.Series) -> pd.Series:
        """Use KNN for imputation."""
        X_array = X.values.reshape(-1, 1)
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        X_imputed = imputer.fit_transform(X_array)
        X_filled = pd.Series(X_imputed.flatten(), index=X.index)
        return X_filled

    def _to_series(self, X: ArrayLike) -> pd.Series:
        if isinstance(X, pd.Series):
            return X
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        return pd.Series(X)


class Differencing(BaseEstimator, TransformerMixin):
    """
    Apply differencing to make time series stationary.

    Supports: simple, seasonal, log, and percentage differencing.
    """

    def __init__(
        self,
        order: int = 1,
        seasonal_period: Optional[int] = None,
        method: Literal["simple", "log", "percentage"] = "simple",
    ):
        self.order = order
        self.seasonal_period = seasonal_period
        self.method = method
        self.first_values_: List[float] = []

    def fit(self, X: ArrayLike, y=None):
        X = self._to_series(X)
        self.first_values_ = []

        temp = X.copy()
        for _ in range(self.order):
            self.first_values_.append(temp.iloc[0])
            temp = temp.diff()
            temp = temp.dropna()

        return self

    def transform(self, X: ArrayLike) -> pd.Series:
        X = self._to_series(X)

        if self.method == "simple":
            result = X
            for _ in range(self.order):
                result = result.diff()
            return result.dropna()
        elif self.method == "log":
            result = np.log(X)
            for _ in range(self.order):
                result = result.diff()
            return result.dropna()
        elif self.method == "percentage":
            result = X.pct_change()
            for i in range(1, self.order):
                result = result.diff()
            return result.dropna()

    def inverse_transform(
        self, X: ArrayLike, initial_values: Optional[List] = None
    ) -> pd.Series:
        X = self._to_series(X)
        values = initial_values if initial_values else self.first_values_

        result = X.copy()
        for i, first_val in enumerate(reversed(values)):
            result = result.cumsum() + first_val

        return result

    def _to_series(self, X: ArrayLike) -> pd.Series:
        if isinstance(X, pd.Series):
            return X
        return pd.Series(X)


class SeasonalDecompose(BaseEstimator, TransformerMixin):
    """
    Seasonal decomposition using moving averages.

    Supports: additive, multiplicative decomposition.
    """

    def __init__(
        self,
        period: int,
        model: Literal["additive", "multiplicative"] = "additive",
        extrapolate_trend: Union[int, str] = "freq",
    ):
        self.period = period
        self.model = model
        self.extrapolate_trend = extrapolate_trend
        self.decomposition_ = None

    def fit(self, X: ArrayLike, y=None):
        X = self._to_series(X)

        from statsmodels.tsa.seasonal import seasonal_decompose

        self.decomposition_ = seasonal_decompose(
            X,
            model=self.model,
            period=self.period,
            extrapolate_trend=self.extrapolate_trend,
        )

        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Return detrended and deseasonalized series."""
        X = self._to_series(X)

        if self.decomposition_ is None:
            raise ValueError("Decompose not fitted")

        if self.model == "additive":
            residual = X - self.decomposition_.trend - self.decomposition_.seasonal
        else:
            residual = X / (self.decomposition_.trend * self.decomposition_.seasonal)

        return pd.DataFrame(
            {
                "observed": X,
                "trend": self.decomposition_.trend,
                "seasonal": self.decomposition_.seasonal,
                "residual": residual,
            }
        )

    def get_trend(self) -> pd.Series:
        """Extract trend component."""
        if self.decomposition_ is None:
            raise ValueError("Decompose not fitted")
        return self.decomposition_.trend

    def get_seasonal(self) -> pd.Series:
        """Extract seasonal component."""
        if self.decomposition_ is None:
            raise ValueError("Decompose not fitted")
        return self.decomposition_.seasonal

    def get_residual(self) -> pd.Series:
        """Extract residual component."""
        if self.decomposition_ is None:
            raise ValueError("Decompose not fitted")
        return self.decomposition_.resid

    def _to_series(self, X: ArrayLike) -> pd.Series:
        if isinstance(X, pd.Series):
            return X
        return pd.Series(X)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detect and handle outliers in time series.

    Methods: zscore, iqr, isolation_forest, local_outlier_factor
    Handling: remove, cap, interpolate, replace
    """

    def __init__(
        self,
        method: Literal[
            "zscore", "iqr", "isolation_forest", "local_outlier_factor"
        ] = "zscore",
        threshold: float = 3.0,
        handling: Literal["remove", "cap", "interpolate", "replace"] = "interpolate",
        replace_value: Optional[float] = None,
    ):
        self.method = method
        self.threshold = threshold
        self.handling = handling
        self.replace_value = replace_value
        self.outlier_mask_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y=None):
        X = self._to_series(X)
        self.outlier_mask_ = self._detect_outliers(X)
        self.mean_ = X.mean()
        self.std_ = X.std()
        return self

    def transform(self, X: ArrayLike) -> pd.Series:
        X = self._to_series(X).copy()
        mask = self._detect_outliers(X)

        if self.handling == "remove":
            X = X[~mask]
        elif self.handling == "cap":
            lower = X[~mask].quantile(0.01)
            upper = X[~mask].quantile(0.99)
            X = X.clip(lower, upper)
        elif self.handling == "interpolate":
            X[mask] = np.nan
            X = X.interpolate(method="linear")
            X = X.fillna(method="bfill").fillna(method="ffill")
        elif self.handling == "replace":
            replacement = (
                self.replace_value if self.replace_value else X[~mask].median()
            )
            X[mask] = replacement

        return X

    def _detect_outliers(self, X: pd.Series) -> np.ndarray:
        if self.method == "zscore":
            z_scores = np.abs(stats.zscore(X.dropna()))
            mask = np.abs(stats.zscore(X)) > self.threshold
        elif self.method == "iqr":
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.threshold * IQR
            upper = Q3 + self.threshold * IQR
            mask = (X < lower) | (X > upper)
        elif self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            mask = iso_forest.fit_predict(X.values.reshape(-1, 1)) == -1
        elif self.method == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor

            lof = LocalOutlierFactor(contamination=0.1)
            mask = lof.fit_predict(X.values.reshape(-1, 1)) == -1

        return mask

    def _to_series(self, X: ArrayLike) -> pd.Series:
        if isinstance(X, pd.Series):
            return X
        return pd.Series(X)


# ============================================================================
# Feature Engineering
# ============================================================================


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract datetime features from time series index.
    """

    def __init__(
        self,
        features: List[str] = None,
        cyclical: bool = True,
    ):
        default_features = [
            "year",
            "month",
            "day",
            "hour",
            "dayofweek",
            "dayofyear",
            "weekofyear",
            "quarter",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
        ]
        self.features = features or default_features
        self.cyclical = cyclical

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        df = X.copy()

        for feature in self.features:
            if feature == "year":
                df["year"] = df.index.year
            elif feature == "month":
                df["month"] = df.index.month
                if self.cyclical:
                    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
                    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            elif feature == "day":
                df["day"] = df.index.day
            elif feature == "hour":
                df["hour"] = df.index.hour
                if self.cyclical:
                    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
                    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            elif feature == "dayofweek":
                df["dayofweek"] = df.index.dayofweek
                if self.cyclical:
                    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
                    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
            elif feature == "dayofyear":
                df["dayofyear"] = df.index.dayofyear
            elif feature == "weekofyear":
                df["weekofyear"] = df.index.isocalendar().week
            elif feature == "quarter":
                df["quarter"] = df.index.quarter
            elif feature == "is_month_start":
                df["is_month_start"] = df.index.is_month_start.astype(int)
            elif feature == "is_month_end":
                df["is_month_end"] = df.index.is_month_end.astype(int)
            elif feature == "is_quarter_start":
                df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
            elif feature == "is_quarter_end":
                df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

        return df


class LagFeatures(BaseEstimator, TransformerMixin):
    """
    Create lag features for time series.
    """

    def __init__(
        self,
        lags: Union[int, List[int]] = 12,
        drop_na: bool = True,
    ):
        self.lags = [lags] if isinstance(lags, int) else lags
        self.drop_na = drop_na

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in df.columns:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        if self.drop_na:
            df = df.dropna()

        return df


class RollingFeatures(BaseEstimator, TransformerMixin):
    """
    Create rolling window statistics as features.
    """

    def __init__(
        self,
        windows: Union[int, List[int]] = [7, 14, 30],
        aggregations: List[str] = ["mean", "std", "min", "max", "median"],
        min_periods: int = 1,
        drop_na: bool = True,
    ):
        self.windows = [windows] if isinstance(windows, int) else windows
        self.aggregations = aggregations
        self.min_periods = min_periods
        self.drop_na = drop_na

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in df.columns:
            for window in self.windows:
                for agg in self.aggregations:
                    feature_name = f"{col}_roll_{window}_{agg}"

                    if agg == "mean":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .mean()
                        )
                    elif agg == "std":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .std()
                        )
                    elif agg == "min":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .min()
                        )
                    elif agg == "max":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .max()
                        )
                    elif agg == "median":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .median()
                        )
                    elif agg == "sum":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .sum()
                        )
                    elif agg == "var":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .var()
                        )
                    elif agg == "skew":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .skew()
                        )
                    elif agg == "kurt":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .kurt()
                        )
                    elif agg == "quantile_25":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .quantile(0.25)
                        )
                    elif agg == "quantile_75":
                        df[feature_name] = (
                            df[col]
                            .rolling(window=window, min_periods=self.min_periods)
                            .quantile(0.75)
                        )

        if self.drop_na:
            df = df.dropna()

        return df


class FourierFeatures(BaseEstimator, TransformerMixin):
    """
    Create Fourier terms for capturing seasonality.
    """

    def __init__(
        self,
        period: Union[int, float],
        order: int = 3,
        index_type: Literal["datetime", "numeric"] = "datetime",
    ):
        self.period = period
        self.order = order
        self.index_type = index_type

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if self.index_type == "datetime":
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have DatetimeIndex")
            t = np.arange(len(df))
        else:
            t = np.arange(len(df))

        for i in range(1, self.order + 1):
            df[f"fourier_sin_{i}"] = np.sin(2 * np.pi * i * t / self.period)
            df[f"fourier_cos_{i}"] = np.cos(2 * np.pi * i * t / self.period)

        return df


class HolidayFeatures(BaseEstimator, TransformerMixin):
    """
    Add holiday and event features.
    """

    def __init__(
        self,
        country: str = "US",
        years: Optional[Tuple[int, int]] = None,
        include_weekends: bool = False,
        custom_holidays: Optional[Dict[str, str]] = None,
    ):
        self.country = country
        self.years = years
        self.include_weekends = include_weekends
        self.custom_holidays = custom_holidays or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            import holidays

            df = X.copy()

            if self.years:
                year_range = range(self.years[0], self.years[1] + 1)
            else:
                year_range = range(df.index.year.min(), df.index.year.max() + 1)

            holiday_calendar = holidays.country_holidays(
                self.country,
                years=list(year_range),
            )

            df["is_holiday"] = df.index.map(lambda x: 1 if x in holiday_calendar else 0)

            # Days to nearest holiday
            holiday_dates = pd.to_datetime(list(holiday_calendar.keys()))

            def days_to_nearest_holiday(date):
                if date in holiday_dates:
                    return 0
                diffs = (holiday_dates - date).days
                before = np.min(np.abs(diffs[diffs < 0])) if np.any(diffs < 0) else 365
                after = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else 365
                return min(before, after)

            df["days_to_holiday"] = df.index.map(days_to_nearest_holiday)

            if self.include_weekends:
                df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

            # Custom holidays
            for name, date_str in self.custom_holidays.items():
                date = pd.to_datetime(date_str)
                df[f"is_{name}"] = (df.index == date).astype(int)

            return df

        except ImportError:
            raise ImportError(
                "holidays package required. Install with: pip install holidays"
            )


# ============================================================================
# Evaluation Metrics
# ============================================================================


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return (
        2.0 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    )


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonal_period: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error.

    Scales the error by the in-sample MAE of the naive seasonal forecast.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_train = np.asarray(y_train)

    # Naive forecast error
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.inf

    return np.mean(np.abs(y_true - y_pred)) / mae_naive


def crps(
    y_true: ArrayLike,
    y_pred_distribution: ArrayLike,
) -> float:
    """
    Continuous Ranked Probability Score.

    For probabilistic forecasts.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred_distribution)

    if y_pred.ndim == 1:
        # Point forecast, compute CRPS for normal distribution
        mean, std = y_pred, np.std(y_pred)
        z = (y_true - mean) / std
        return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    else:
        # Empirical distribution
        n_samples = y_pred.shape[0]
        crps_scores = []

        for i in range(len(y_true)):
            samples = y_pred[i]
            samples_sorted = np.sort(samples)
            empirical_cdf = np.arange(1, len(samples) + 1) / len(samples)

            # CRPS integral
            indicator = (samples_sorted >= y_true[i]).astype(float)
            crps_val = np.mean((empirical_cdf - indicator) ** 2)
            crps_scores.append(crps_val)

        return np.mean(crps_scores)


def coverage(
    y_true: ArrayLike,
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
) -> float:
    """
    Prediction interval coverage.

    Returns the percentage of observations within the prediction interval.
    """
    y_true = np.asarray(y_true)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)

    within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    return np.mean(within_interval) * 100


def interval_score(
    y_true: ArrayLike,
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
    alpha: float = 0.1,
) -> float:
    """
    Interval score for prediction intervals.

    Lower scores are better.
    """
    y_true = np.asarray(y_true)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)

    coverage_score = upper_bound - lower_bound
    penalty = 2 / alpha * (lower_bound - y_true) * (
        y_true < lower_bound
    ) + 2 / alpha * (y_true - upper_bound) * (y_true > upper_bound)

    return np.mean(coverage_score + penalty)


class Metrics:
    """Collection of evaluation metrics."""

    @staticmethod
    def evaluate(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        metrics: List[str] = None,
    ) -> Dict[str, float]:
        """Evaluate multiple metrics at once."""
        metrics = metrics or ["mae", "mse", "rmse", "mape", "smape"]

        results = {}
        for metric_name in metrics:
            if metric_name == "mae":
                results["mae"] = mae(y_true, y_pred)
            elif metric_name == "mse":
                results["mse"] = mse(y_true, y_pred)
            elif metric_name == "rmse":
                results["rmse"] = rmse(y_true, y_pred)
            elif metric_name == "mape":
                results["mape"] = mape(y_true, y_pred)
            elif metric_name == "smape":
                results["smape"] = smape(y_true, y_pred)

        return results


# ============================================================================
# Cross-Validation
# ============================================================================


class TimeSeriesSplit:
    """
    Time series cross-validation splitter.

    Splits data maintaining temporal order.
    """

    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X: ArrayLike) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)

        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = (
                train_end + fold_size
                if self.test_size is None
                else train_end + self.test_size
            )
            test_end = min(test_end, n_samples)

            yield indices[:train_end], indices[test_start:test_end]


class ExpandingWindow:
    """
    Expanding window cross-validation.

    Training set grows with each split.
    """

    def __init__(
        self,
        min_train_size: int,
        step: int = 1,
        test_size: int = 1,
    ):
        self.min_train_size = min_train_size
        self.step = step
        self.test_size = test_size

    def split(self, X: ArrayLike) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)

        train_end = self.min_train_size

        while train_end + self.test_size <= n_samples:
            test_start = train_end
            test_end = min(train_end + self.test_size, n_samples)

            yield indices[:train_end], indices[test_start:test_end]
            train_end += self.step


class SlidingWindow:
    """
    Sliding window cross-validation.

    Fixed-size training window that slides forward.
    """

    def __init__(
        self,
        window_size: int,
        step: int = 1,
        test_size: int = 1,
    ):
        self.window_size = window_size
        self.step = step
        self.test_size = test_size

    def split(self, X: ArrayLike) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)

        start = 0
        end = self.window_size

        while end + self.test_size <= n_samples:
            test_start = end
            test_end = min(end + self.test_size, n_samples)

            yield indices[start:end], indices[test_start:test_end]
            start += self.step
            end += self.step


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.

    Removes data within a gap period around test sets to prevent leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.0,
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(self, X: ArrayLike) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)

        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Apply embargo to test set
            embargo = int(self.embargo_pct * (test_end - test_start))
            test_end -= embargo

            # Create train indices, excluding purge gap
            train_indices = np.concatenate(
                [
                    indices[: max(0, test_start - self.purge_gap)],
                    indices[min(n_samples, test_end + self.purge_gap) :],
                ]
            )

            test_indices = indices[test_start:test_end]

            yield train_indices, test_indices


# ============================================================================
# Deep Learning Models
# ============================================================================


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LSTMForecaster(nn.Module):
    """
    LSTM-based time series forecaster.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * direction_factor, output_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output


class GRUForecaster(nn.Module):
    """
    GRU-based time series forecaster.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * direction_factor, output_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, h_n = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output


class TransformerForecaster(nn.Module):
    """
    Transformer-based time series forecaster.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_horizon: int = 1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer.encoder(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        output = self.fc(x)
        return output


class NBeatsBlock(nn.Module):
    """N-BEATS block."""

    def __init__(
        self,
        input_size: int,
        output_horizon: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        expansion_coefficient_dim: int = 5,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_horizon = output_horizon

        # Stack of fully connected layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        self.fc_stack = nn.Sequential(*layers)

        # Backcast and forecast branches
        self.backcast_fc = nn.Linear(hidden_size, input_size)
        self.forecast_fc = nn.Linear(hidden_size, output_horizon)

        # Basis expansion
        self.basis_expansion = nn.Linear(hidden_size, expansion_coefficient_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        x = self.fc_stack(x)

        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)

        # Residual connection
        backcast = block_input - backcast

        return backcast, forecast


class NBeatsForecaster(nn.Module):
    """
    N-BEATS: Neural basis expansion for interpretable time series forecasting.
    """

    def __init__(
        self,
        input_size: int,
        output_horizon: int,
        hidden_size: int = 512,
        num_blocks: int = 3,
        num_layers: int = 4,
        num_stacks: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_horizon = output_horizon
        self.num_blocks = num_blocks
        self.num_stacks = num_stacks

        self.stacks = nn.ModuleList()

        for _ in range(num_stacks):
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(
                    NBeatsBlock(
                        input_size=input_size,
                        output_horizon=output_horizon,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                    )
                )
            self.stacks.append(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecasts = []

        for stack in self.stacks:
            stack_forecast = torch.zeros(
                x.size(0), self.output_horizon, device=x.device
            )
            for block in stack:
                backcast, forecast = block(x)
                x = backcast
                stack_forecast = stack_forecast + forecast
            forecasts.append(stack_forecast)

        # Sum forecasts from all stacks
        return torch.stack(forecasts, dim=0).sum(dim=0)


class NHiTSBlock(nn.Module):
    """N-HiTS block with pooling and interpolation."""

    def __init__(
        self,
        input_size: int,
        output_horizon: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        pool_size: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_horizon = output_horizon
        self.pool_size = pool_size

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        pooled_size = input_size // pool_size

        # MLP
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(pooled_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.theta_fc = nn.Linear(hidden_size, output_horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pool
        x_pooled = self.pool(x.unsqueeze(1)).squeeze(1)

        # MLP
        h = self.mlp(x_pooled)

        # Forecast
        forecast = self.theta_fc(h)

        # Backcast (interpolation)
        backcast = F.interpolate(
            forecast.unsqueeze(1),
            size=self.input_size,
            mode="linear",
            align_corners=True,
        ).squeeze(1)

        return x - backcast, forecast


class NHITSForecaster(nn.Module):
    """
    N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.
    """

    def __init__(
        self,
        input_size: int,
        output_horizon: int,
        hidden_size: int = 512,
        num_blocks: int = 3,
        num_layers: int = 2,
        num_stacks: int = 2,
        pool_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_horizon = output_horizon
        self.num_stacks = num_stacks

        pool_sizes = pool_sizes or [1, 2, 4]

        self.stacks = nn.ModuleList()

        for i in range(num_stacks):
            blocks = nn.ModuleList()
            pool_size = pool_sizes[i % len(pool_sizes)]
            for _ in range(num_blocks):
                blocks.append(
                    NHiTSBlock(
                        input_size=input_size,
                        output_horizon=output_horizon,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        pool_size=pool_size,
                    )
                )
            self.stacks.append(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecasts = []

        for stack in self.stacks:
            stack_forecast = torch.zeros(
                x.size(0), self.output_horizon, device=x.device
            )
            for block in stack:
                backcast, forecast = block(x)
                x = backcast
                stack_forecast = stack_forecast + forecast
            forecasts.append(stack_forecast)

        return torch.stack(forecasts, dim=0).sum(dim=0)


class TFTAttention(nn.Module):
    """Interpretable Multi-Head Attention for TFT."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = (
            self.query_linear(query)
            .view(batch_size, -1, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key_linear(key)
            .view(batch_size, -1, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value_linear(value)
            .view(batch_size, -1, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.output_linear(context), attention_weights


class GatedResidualNetwork(nn.Module):
    """GRN for TFT."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.output_size = output_size or input_size

        self.input_fc = nn.Linear(input_size, hidden_size)
        self.context_fc = nn.Linear(context_size, hidden_size) if context_size else None
        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, self.output_size)
        self.layer_norm = nn.LayerNorm(self.output_size)

        if input_size != self.output_size:
            self.skip_fc = nn.Linear(input_size, self.output_size)
        else:
            self.skip_fc = None

    def forward(self, x, context=None):
        hidden = self.input_fc(x)

        if context is not None and self.context_fc is not None:
            hidden = hidden + self.context_fc(context)

        hidden = torch.relu(hidden)
        hidden = self.hidden_fc(hidden)
        hidden = self.dropout(hidden)

        output = self.output_fc(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        output = output * gate

        skip = self.skip_fc(x) if self.skip_fc else x
        return self.layer_norm(skip + output)


class VariableSelectionNetwork(nn.Module):
    """Variable selection network for TFT."""

    def __init__(
        self,
        num_inputs: int,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_inputs = num_inputs

        self.grns = nn.ModuleList(
            [
                GatedResidualNetwork(input_size, hidden_size, dropout=dropout)
                for _ in range(num_inputs)
            ]
        )

        self.selection_weights = GatedResidualNetwork(
            num_inputs * input_size,
            hidden_size,
            num_inputs,
            dropout=dropout,
            context_size=context_size,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, context=None):
        # inputs: [batch, num_inputs, input_size]
        batch_size = inputs.size(0)

        # Individual transformations
        v = torch.stack(
            [grn(inputs[:, i, :]) for i, grn in enumerate(self.grns)], dim=1
        )

        # Selection weights
        flat_inputs = inputs.view(batch_size, -1)
        weights = self.selection_weights(flat_inputs, context)
        weights = self.softmax(weights)

        # Weighted combination
        weighted = v * weights.unsqueeze(-1)
        output = weighted.sum(dim=1)

        return output, weights


class TFTForecaster(nn.Module):
    """
    Temporal Fusion Transformer for interpretable multi-horizon forecasting.
    """

    def __init__(
        self,
        num_static_features: int = 0,
        num_temporal_known_features: int = 0,
        num_temporal_observed_features: int = 1,
        hidden_size: int = 160,
        lstm_layers: int = 1,
        attention_heads: int = 4,
        dropout: float = 0.1,
        output_horizon: int = 1,
        quantiles: List[float] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_horizon = output_horizon
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        # Static covariate encoders
        if num_static_features > 0:
            self.static_variable_selection = VariableSelectionNetwork(
                num_static_features, 1, hidden_size, dropout
            )
            self.static_context_grns = nn.ModuleList(
                [
                    GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
                    for _ in range(4)
                ]
            )
        else:
            self.static_variable_selection = None

        # Temporal variable selection
        total_temporal = num_temporal_known_features + num_temporal_observed_features
        self.temporal_variable_selection = VariableSelectionNetwork(
            total_temporal, 1, hidden_size, dropout, context_size=hidden_size
        )

        # LSTM encoder-decoder
        self.lstm_encoder = nn.LSTM(
            hidden_size, hidden_size, lstm_layers, batch_first=True, dropout=dropout
        )
        self.lstm_decoder = nn.LSTM(
            hidden_size, hidden_size, lstm_layers, batch_first=True, dropout=dropout
        )

        # Multi-head attention
        self.attention = TFTAttention(hidden_size, attention_heads, dropout)

        # Position-wise feed-forward
        self.pos_wise_ff = GatedResidualNetwork(
            hidden_size, hidden_size, dropout=dropout
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, len(self.quantiles) * output_horizon)

    def forward(
        self,
        temporal_inputs: torch.Tensor,
        static_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = temporal_inputs.size(0)

        # Static context
        static_context = None
        if self.static_variable_selection is not None and static_inputs is not None:
            static_embedding, _ = self.static_variable_selection(
                static_inputs.unsqueeze(-1)
            )
            static_context = [grn(static_embedding) for grn in self.static_context_grns]

        # Temporal variable selection
        context = static_context[0] if static_context else None
        temporal_features, _ = self.temporal_variable_selection(
            temporal_inputs.unsqueeze(-1)
            if temporal_inputs.dim() == 2
            else temporal_inputs,
            context,
        )

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(temporal_features.unsqueeze(1))

        # Attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Position-wise feed-forward
        output = self.pos_wise_ff(attended)

        # Final output
        output = self.output_layer(output[:, -1, :])
        output = output.view(batch_size, len(self.quantiles), self.output_horizon)

        return output


class DeepARForecaster(nn.Module):
    """
    DeepAR: Probabilistic forecasting with autoregressive recurrent networks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 40,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_horizon: int = 1,
        distribution: Literal[
            "gaussian", "student_t", "negative_binomial"
        ] = "gaussian",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_horizon = output_horizon
        self.distribution = distribution

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Distribution parameters
        if distribution == "gaussian":
            self.mu_fc = nn.Linear(hidden_size, output_horizon)
            self.sigma_fc = nn.Linear(hidden_size, output_horizon)
        elif distribution == "student_t":
            self.mu_fc = nn.Linear(hidden_size, output_horizon)
            self.sigma_fc = nn.Linear(hidden_size, output_horizon)
            self.nu_fc = nn.Linear(hidden_size, output_horizon)
        elif distribution == "negative_binomial":
            self.mu_fc = nn.Linear(hidden_size, output_horizon)
            self.alpha_fc = nn.Linear(hidden_size, output_horizon)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        if self.distribution == "gaussian":
            mu = self.mu_fc(last_hidden)
            sigma = F.softplus(self.sigma_fc(last_hidden)) + 1e-6
            return {"mu": mu, "sigma": sigma}
        elif self.distribution == "student_t":
            mu = self.mu_fc(last_hidden)
            sigma = F.softplus(self.sigma_fc(last_hidden)) + 1e-6
            nu = F.softplus(self.nu_fc(last_hidden)) + 2.0
            return {"mu": mu, "sigma": sigma, "nu": nu}
        elif self.distribution == "negative_binomial":
            mu = F.softplus(self.mu_fc(last_hidden)) + 1e-6
            alpha = F.softplus(self.alpha_fc(last_hidden)) + 1e-6
            return {"mu": mu, "alpha": alpha}

    def sample(
        self, params: Dict[str, torch.Tensor], num_samples: int = 100
    ) -> torch.Tensor:
        """Sample from the predicted distribution."""
        if self.distribution == "gaussian":
            mu, sigma = params["mu"], params["sigma"]
            return torch.normal(
                mu.unsqueeze(-1).expand(-1, -1, num_samples), sigma.unsqueeze(-1)
            )
        elif self.distribution == "student_t":
            # Approximate with Gaussian for simplicity
            mu, sigma = params["mu"], params["sigma"]
            return torch.normal(
                mu.unsqueeze(-1).expand(-1, -1, num_samples), sigma.unsqueeze(-1)
            )
        elif self.distribution == "negative_binomial":
            # Approximate with Gamma-Poisson
            mu, alpha = params["mu"], params["alpha"]
            r = 1.0 / alpha
            p = r / (r + mu)
            # Use normal approximation
            variance = mu * (1 + mu * alpha)
            return torch.normal(
                mu.unsqueeze(-1).expand(-1, -1, num_samples),
                variance.sqrt().unsqueeze(-1),
            )


# ============================================================================
# Classical Models
# ============================================================================


class ARIMAForecaster:
    """
    Classical ARIMA time series forecaster.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model_ = None
        self.fitted_model_ = None

    def fit(self, X: ArrayLike, y=None):
        try:
            from statsmodels.tsa.arima.model import ARIMA

            X = self._to_series(X)
            self.model_ = ARIMA(
                X,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
            )
            self.fitted_model_ = self.model_.fit()
            return self
        except ImportError:
            raise ImportError(
                "statsmodels required for ARIMA. Install with: pip install statsmodels"
            )

    def predict(self, steps: int) -> np.ndarray:
        if self.fitted_model_ is None:
            raise ValueError("Model not fitted")
        return self.fitted_model_.forecast(steps=steps)

    def predict_interval(
        self, steps: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return point forecast and confidence interval."""
        if self.fitted_model_ is None:
            raise ValueError("Model not fitted")
        forecast = self.fitted_model_.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=alpha)
        return mean, conf_int.iloc[:, 0], conf_int.iloc[:, 1]

    def _to_series(self, X: ArrayLike) -> pd.Series:
        if isinstance(X, pd.Series):
            return X
        return pd.Series(X)


class ProphetForecaster:
    """
    Facebook Prophet forecaster wrapper.
    """

    def __init__(
        self,
        growth: str = "linear",
        yearly_seasonality: Union[bool, int] = True,
        weekly_seasonality: Union[bool, int] = True,
        daily_seasonality: Union[bool, int] = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
    ):
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: Optional[str] = None):
        try:
            from prophet import Prophet

            if isinstance(X, pd.Series):
                df = pd.DataFrame({"ds": X.index, "y": X.values})
            else:
                df = X.copy()
                if "ds" not in df.columns:
                    df["ds"] = df.index
                if "y" not in df.columns:
                    df["y"] = df.iloc[:, 0]

            self.model_ = Prophet(
                growth=self.growth,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
            )

            self.model_.fit(df)
            return self

        except ImportError:
            raise ImportError("Prophet required. Install with: pip install prophet")

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        if self.model_ is None:
            raise ValueError("Model not fitted")

        future = self.model_.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model_.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

    def add_regressor(self, name: str, **kwargs):
        """Add additional regressor."""
        if self.model_ is None:
            raise ValueError("Model not initialized")
        self.model_.add_regressor(name, **kwargs)

    def add_seasonality(self, name: str, period: float, fourier_order: int, **kwargs):
        """Add custom seasonality."""
        if self.model_ is None:
            raise ValueError("Model not initialized")
        self.model_.add_seasonality(
            name, period=period, fourier_order=fourier_order, **kwargs
        )


# ============================================================================
# Ensembling
# ============================================================================


class EnsembleForecaster:
    """
    Simple ensemble forecaster combining multiple models.
    """

    def __init__(
        self,
        models: List[Any],
        weights: Optional[np.ndarray] = None,
        method: Literal["mean", "median", "weighted"] = "mean",
    ):
        self.models = models
        self.weights = weights
        self.method = method

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models])

        if self.method == "mean":
            return np.mean(predictions, axis=0)
        elif self.method == "median":
            return np.median(predictions, axis=0)
        elif self.method == "weighted":
            weights = self.weights or np.ones(len(self.models)) / len(self.models)
            return np.average(predictions, axis=0, weights=weights)


class StackingForecaster:
    """
    Stacking ensemble with meta-learner.
    """

    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Any = None,
        cv: Optional[Any] = None,
    ):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge()
        self.cv = cv or TimeSeriesSplit(n_splits=5)
        self.fitted_models_ = {}

    def fit(self, X: ArrayLike, y: ArrayLike):
        X = np.asarray(X)
        y = np.asarray(y)

        # Fit base models
        for name, model in self.base_models.items():
            model.fit(X, y)
            self.fitted_models_[name] = model

        # Create meta-features
        meta_features = self._create_meta_features(X)

        # Fit meta-learner
        self.meta_model.fit(meta_features, y)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X)
        meta_features = self._create_meta_features(X)
        return self.meta_model.predict(meta_features)

    def _create_meta_features(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for name, model in self.fitted_models_.items():
            pred = model.predict(X)
            predictions.append(pred.reshape(-1, 1))
        return np.hstack(predictions)


class BlendingForecaster:
    """
    Blending ensemble using holdout set.
    """

    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Any = None,
        blend_split: float = 0.2,
    ):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge()
        self.blend_split = blend_split
        self.fitted_models_ = {}

    def fit(self, X: ArrayLike, y: ArrayLike):
        X = np.asarray(X)
        y = np.asarray(y)

        # Split for blending
        split_idx = int(len(X) * (1 - self.blend_split))
        X_train, X_blend = X[:split_idx], X[split_idx:]
        y_train, y_blend = y[:split_idx], y[split_idx:]

        # Fit base models
        for name, model in self.base_models.items():
            model.fit(X_train, y_train)
            self.fitted_models_[name] = model

        # Create meta-features from blend set
        meta_features = []
        for name, model in self.fitted_models_.items():
            pred = model.predict(X_blend)
            meta_features.append(pred.reshape(-1, 1))

        meta_features = np.hstack(meta_features)
        self.meta_model.fit(meta_features, y_blend)

        # Refit base models on full data
        for name, model in self.base_models.items():
            model.fit(X, y)
            self.fitted_models_[name] = model

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X)
        meta_features = []

        for name, model in self.fitted_models_.items():
            pred = model.predict(X)
            meta_features.append(pred.reshape(-1, 1))

        meta_features = np.hstack(meta_features)
        return self.meta_model.predict(meta_features)


# ============================================================================
# Utilities
# ============================================================================


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """

    def __init__(
        self,
        data: ArrayLike,
        target: Optional[ArrayLike] = None,
        lookback_window: int = 24,
        forecast_horizon: int = 1,
        transform: Optional[Callable] = None,
    ):
        self.data = torch.FloatTensor(np.asarray(data))
        self.target = (
            torch.FloatTensor(np.asarray(target)) if target is not None else self.data
        )
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data) - self.lookback_window - self.forecast_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.lookback_window]
        y = self.target[
            idx + self.lookback_window : idx
            + self.lookback_window
            + self.forecast_horizon
        ]

        if self.transform:
            x = self.transform(x)

        return x, y


class TimeSeriesDataLoader:
    """
    Utility class for creating time series data loaders.
    """

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def create_loader(
        self,
        dataset: Dataset,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class TimeSeriesTrainer:
    """
    Trainer for deep learning time series models.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "auto",
        early_stopping_patience: int = 10,
        lr_scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.criterion = criterion or nn.MSELoss()
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler = lr_scheduler
        self.history = {"train_loss": [], "val_loss": []}
        self.optimizer = optimizer

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def compile(self, optimizer: str = "adam", lr: float = 1e-3, **kwargs):
        """Compile model with optimizer."""
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, **kwargs)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, **kwargs)
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, **kwargs)
        elif optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=lr, **kwargs
            )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and epoch % 10 == 0:
                    print(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )

                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

            if self.lr_scheduler:
                self.lr_scheduler.step()

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]


# ============================================================================
# Specialized
# ============================================================================


class AnomalyDetection:
    """
    Time series anomaly detection.
    """

    def __init__(
        self,
        method: Literal[
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
            "prophet",
            "lstm_autoencoder",
        ] = "isolation_forest",
        threshold: float = 0.05,
    ):
        self.method = method
        self.threshold = threshold
        self.model_ = None

    def fit(self, X: ArrayLike, y=None):
        X = np.asarray(X).reshape(-1, 1) if np.asarray(X).ndim == 1 else np.asarray(X)

        if self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            self.model_ = IsolationForest(contamination=self.threshold, random_state=42)
            self.model_.fit(X)
        elif self.method == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor

            self.model_ = LocalOutlierFactor(contamination=self.threshold, novelty=True)
            self.model_.fit(X)
        elif self.method == "one_class_svm":
            from sklearn.svm import OneClassSVM

            self.model_ = OneClassSVM(nu=self.threshold)
            self.model_.fit(X)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return -1 for anomaly, 1 for normal."""
        X = np.asarray(X).reshape(-1, 1) if np.asarray(X).ndim == 1 else np.asarray(X)

        if self.method in ["isolation_forest", "local_outlier_factor", "one_class_svm"]:
            return self.model_.predict(X)
        elif self.method == "prophet":
            return self._prophet_anomaly(X)

    def anomaly_scores(self, X: ArrayLike) -> np.ndarray:
        """Return anomaly scores."""
        X = np.asarray(X).reshape(-1, 1) if np.asarray(X).ndim == 1 else np.asarray(X)

        if hasattr(self.model_, "score_samples"):
            return -self.model_.score_samples(X)
        elif hasattr(self.model_, "decision_function"):
            return -self.model_.decision_function(X)
        else:
            return np.zeros(len(X))

    def _prophet_anomaly(self, X: ArrayLike) -> np.ndarray:
        """Use Prophet for anomaly detection."""
        try:
            from prophet import Prophet

            df = pd.DataFrame({"ds": range(len(X)), "y": X.flatten()})
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)

            # Detect anomalies based on confidence intervals
            residuals = np.abs(df["y"] - forecast["yhat"])
            threshold = residuals.quantile(1 - self.threshold)

            return np.where(residuals > threshold, -1, 1)

        except ImportError:
            raise ImportError("Prophet required")


class ChangePointDetection:
    """
    Time series change point detection.
    """

    def __init__(
        self,
        method: Literal[
            "pelt",
            "binseg",
            "window",
            "prophet",
            "bayesian_online",
        ] = "pelt",
        penalty: Optional[float] = None,
        min_size: int = 2,
    ):
        self.method = method
        self.penalty = penalty
        self.min_size = min_size

    def detect(self, X: ArrayLike) -> List[int]:
        X = np.asarray(X)

        if self.method == "pelt":
            return self._pelt(X)
        elif self.method == "binseg":
            return self._binseg(X)
        elif self.method == "window":
            return self._window(X)
        elif self.method == "prophet":
            return self._prophet_changepoints(X)
        elif self.method == "bayesian_online":
            return self._bayesian_online(X)

    def _pelt(self, X: np.ndarray) -> List[int]:
        """PELT algorithm for change point detection."""
        try:
            import ruptures as rpt

            algo = rpt.Pelt(model="rbf", min_size=self.min_size).fit(X)
            penalty = self.penalty or np.log(len(X))
            return algo.predict(pen=penalty)
        except ImportError:
            raise ImportError("ruptures required. Install with: pip install ruptures")

    def _binseg(self, X: np.ndarray) -> List[int]:
        """Binary segmentation for change point detection."""
        try:
            import ruptures as rpt

            algo = rpt.Binseg(model="l2").fit(X)
            n_bkps = max(1, len(X) // 50)
            return algo.predict(n_bkps=n_bkps)
        except ImportError:
            raise ImportError("ruptures required")

    def _window(self, X: np.ndarray) -> List[int]:
        """Window-based change point detection."""
        try:
            import ruptures as rpt

            algo = rpt.Window(width=20, model="l2").fit(X)
            n_bkps = max(1, len(X) // 50)
            return algo.predict(n_bkps=n_bkps)
        except ImportError:
            raise ImportError("ruptures required")

    def _prophet_changepoints(self, X: np.ndarray) -> List[int]:
        """Use Prophet to detect change points."""
        try:
            from prophet import Prophet

            df = pd.DataFrame({"ds": range(len(X)), "y": X})
            model = Prophet()
            model.fit(df)

            # Get changepoints
            changepoints = model.changepoints
            return changepoints.index.tolist()

        except ImportError:
            raise ImportError("Prophet required")

    def _bayesian_online(self, X: np.ndarray) -> List[int]:
        """Bayesian online change point detection."""
        try:
            from bayesian_changepoint_detection import online_changepoint_detection
            from bayesian_changepoint_detection.prior import const_prior
            from functools import partial

            R, maxes = online_changepoint_detection(
                X,
                partial(const_prior, p=1 / len(X)),
                online_changepoint_detection.StudentT(0.1, 0.01, 1, 0),
            )

            changepoints = np.where(np.diff(maxes) != 0)[0] + 1
            return changepoints.tolist()

        except ImportError:
            # Fallback: simple threshold-based
            diff = np.diff(X)
            threshold = np.std(diff) * 3
            return np.where(np.abs(diff) > threshold)[0].tolist()


class PatternDiscovery:
    """
    Discover patterns in time series.
    """

    def __init__(
        self,
        method: Literal[
            "motif",
            "matrix_profile",
            "clustering",
            "shapelets",
        ] = "matrix_profile",
        window_size: int = 10,
        n_patterns: int = 5,
    ):
        self.method = method
        self.window_size = window_size
        self.n_patterns = n_patterns

    def discover(self, X: ArrayLike) -> Dict[str, Any]:
        X = np.asarray(X)

        if self.method == "matrix_profile":
            return self._matrix_profile(X)
        elif self.method == "motif":
            return self._motif_discovery(X)
        elif self.method == "clustering":
            return self._clustering(X)
        elif self.method == "shapelets":
            return self._shapelets(X)

    def _matrix_profile(self, X: np.ndarray) -> Dict[str, Any]:
        """Compute matrix profile for pattern discovery."""
        try:
            import stumpy

            mp = stumpy.stump(X, m=self.window_size)

            # Find motifs (lowest matrix profile values)
            motif_indices = np.argsort(mp[:, 0])[: self.n_patterns]

            return {
                "matrix_profile": mp[:, 0],
                "motif_indices": motif_indices,
                "motif_neighbors": [mp[i, 1] for i in motif_indices],
            }

        except ImportError:
            raise ImportError("stumpy required. Install with: pip install stumpy")

    def _motif_discovery(self, X: np.ndarray) -> Dict[str, Any]:
        """Discover motifs using matrix profile."""
        return self._matrix_profile(X)

    def _clustering(self, X: np.ndarray) -> Dict[str, Any]:
        """Discover patterns through time series clustering."""
        from sklearn.cluster import KMeans

        # Extract subsequences
        subsequences = []
        for i in range(len(X) - self.window_size + 1):
            subseq = X[i : i + self.window_size]
            subseq = (subseq - np.mean(subseq)) / (np.std(subseq) + 1e-8)
            subsequences.append(subseq)

        subsequences = np.array(subsequences)

        # Cluster
        kmeans = KMeans(n_clusters=self.n_patterns, random_state=42)
        labels = kmeans.fit_predict(subsequences)

        return {
            "cluster_centers": kmeans.cluster_centers_,
            "labels": labels,
            "subsequences": subsequences,
            "cluster_sizes": np.bincount(labels),
        }

    def _shapelets(self, X: np.ndarray) -> Dict[str, Any]:
        """Discover shapelets (discriminative subsequences)."""
        try:
            from sktime.transformations.panel.shapelet_transform import (
                RandomShapeletTransform,
            )

            # For univariate, create dummy labels
            labels = np.zeros(len(X))

            # Reshape for sktime
            X_3d = X.reshape(1, -1, 1)
            y = np.array([0])

            st = RandomShapeletTransform(
                n_shapelet_samples=100,
                max_shapelets=self.n_patterns,
                random_state=42,
            )
            st.fit(X_3d, y)

            return {
                "shapelets": st.shapelets_,
                "shapelet_lengths": [len(s) for s in st.shapelets_],
            }

        except ImportError:
            raise ImportError("sktime required. Install with: pip install sktime")


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Preprocessing
    "TimeSeriesScaler",
    "TimeSeriesImputer",
    "Differencing",
    "SeasonalDecompose",
    "OutlierHandler",
    # Feature Engineering
    "DateTimeFeatures",
    "LagFeatures",
    "RollingFeatures",
    "FourierFeatures",
    "HolidayFeatures",
    # Evaluation
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mase",
    "crps",
    "coverage",
    "interval_score",
    "Metrics",
    # Cross-Validation
    "TimeSeriesSplit",
    "ExpandingWindow",
    "SlidingWindow",
    "PurgedKFold",
    # Models
    "ARIMAForecaster",
    "ProphetForecaster",
    "LSTMForecaster",
    "GRUForecaster",
    "TransformerForecaster",
    "NBeatsForecaster",
    "NHITSForecaster",
    "TFTForecaster",
    "DeepARForecaster",
    # Ensembling
    "EnsembleForecaster",
    "StackingForecaster",
    "BlendingForecaster",
    # Utilities
    "TimeSeriesDataset",
    "TimeSeriesDataLoader",
    "TimeSeriesTrainer",
    # Specialized
    "AnomalyDetection",
    "ChangePointDetection",
    "PatternDiscovery",
]
