"""
Fishstick Financial ML Module

Comprehensive financial machine learning tools for time series analysis,
portfolio optimization, risk management, and trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class OHLCVData:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timestamp: Optional[np.ndarray] = None

    def __post_init__(self):
        shapes = [
            arr.shape
            for arr in [self.open, self.high, self.low, self.close, self.volume]
        ]
        if len(set(shapes)) > 1:
            raise ValueError("All OHLCV arrays must have the same shape")

    @property
    def returns(self) -> np.ndarray:
        return np.diff(self.close) / self.close[:-1]

    @property
    def log_returns(self) -> np.ndarray:
        return np.diff(np.log(self.close))

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        df = pd.DataFrame(data)
        if self.timestamp is not None:
            df.index = pd.to_datetime(self.timestamp)
        return df


@dataclass
class Portfolio:
    weights: np.ndarray
    assets: List[str]
    expected_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None

    def __post_init__(self):
        if len(self.weights) != len(self.assets):
            raise ValueError("Weights and assets must have same length")
        if not np.isclose(np.sum(self.weights), 1.0):
            self.weights = self.weights / np.sum(self.weights)


@dataclass
class BacktestResult:
    returns: np.ndarray
    positions: np.ndarray
    trades: List[Dict[str, Any]]
    metrics: Dict[str, float]
    equity_curve: np.ndarray
    drawdowns: np.ndarray

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Backtest Results Summary",
            "=" * 50,
            f"Total Return: {self.metrics.get('total_return', 0):.2%}",
            f"Annualized Return: {self.metrics.get('annualized_return', 0):.2%}",
            f"Volatility: {self.metrics.get('volatility', 0):.2%}",
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}",
            f"Win Rate: {self.metrics.get('win_rate', 0):.2%}",
            f"Number of Trades: {len(self.trades)}",
            "=" * 50,
        ]
        return "\n".join(lines)


class PriceForecaster:
    """Price forecasting using multiple methods."""

    def __init__(self, method: str = "lstm", lookback: int = 60):
        self.method = method
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler()
        self.fitted = False

    def fit(self, prices: np.ndarray, **kwargs) -> "PriceForecaster":
        if self.method == "lstm":
            self._fit_lstm(prices, **kwargs)
        elif self.method == "ml":
            self._fit_ml(prices, **kwargs)
        elif self.method == "arima":
            self._fit_arima(prices, **kwargs)
        elif self.method == "exp_smooth":
            self._fit_exp_smooth(prices, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        self.fitted = True
        return self

    def _fit_lstm(
        self,
        prices: np.ndarray,
        epochs: int = 50,
        hidden_size: int = 50,
        lr: float = 0.001,
    ):
        scaled = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        X, y = self._create_sequences(scaled, self.lookback)
        X = torch.FloatTensor(X).unsqueeze(-1)
        y = torch.FloatTensor(y)

        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=50, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])

        self.model = LSTMModel(hidden_size=hidden_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def _fit_ml(self, prices: np.ndarray, model_type: str = "xgboost"):
        features = self._create_features(prices)
        X = features[:-1]
        y = prices[self.lookback :]

        if model_type == "xgboost":
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1
            )
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
        else:
            self.model = Ridge(alpha=1.0)
        self.model.fit(X, y)

    def _fit_arima(self, prices: np.ndarray, order: Tuple[int, int, int] = (5, 1, 0)):
        try:
            from statsmodels.tsa.arima.model import ARIMA

            self.model = ARIMA(prices, order=order).fit()
        except ImportError:
            logger.warning("statsmodels not available, using simple AR")
            self.model = {"coeffs": self._estimate_ar_coeffs(prices, order[0])}

    def _fit_exp_smooth(self, prices: np.ndarray, alpha: float = 0.3):
        self.model = {"alpha": alpha, "last_value": prices[-1]}

    def _create_sequences(
        self, data: np.ndarray, lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def _create_features(self, prices: np.ndarray) -> np.ndarray:
        features = []
        for i in range(self.lookback, len(prices)):
            window = prices[i - self.lookback : i]
            feat = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                prices[i - 1],
                np.mean(window[-5:]),
                np.mean(window[-20:]) if len(window) >= 20 else np.mean(window),
            ]
            features.append(feat)
        return np.array(features)

    def _estimate_ar_coeffs(self, prices: np.ndarray, order: int) -> np.ndarray:
        X, y = self._create_sequences(prices, order)
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(
        self, steps: int = 1, last_prices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if self.method == "lstm":
            return self._predict_lstm(steps, last_prices)
        elif self.method == "ml":
            return self._predict_ml(steps, last_prices)
        elif self.method == "arima":
            return self._predict_arima(steps, last_prices)
        elif self.method == "exp_smooth":
            return self._predict_exp_smooth(steps)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _predict_lstm(self, steps: int, last_prices: np.ndarray) -> np.ndarray:
        self.model.eval()
        predictions = []
        current_seq = self.scaler.transform(
            last_prices[-self.lookback :].reshape(-1, 1)
        ).flatten()

        with torch.no_grad():
            for _ in range(steps):
                x = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1)
                pred = self.model(x).item()
                predictions.append(pred)
                current_seq = np.roll(current_seq, -1)
                current_seq[-1] = pred

        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()

    def _predict_ml(self, steps: int, last_prices: np.ndarray) -> np.ndarray:
        predictions = []
        current_prices = last_prices.copy()

        for _ in range(steps):
            features = self._create_features(current_prices)[-1:]
            pred = self.model.predict(features)[0]
            predictions.append(pred)
            current_prices = np.append(current_prices, pred)

        return np.array(predictions)

    def _predict_arima(self, steps: int, last_prices: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "forecast"):
            return self.model.forecast(steps=steps)
        else:
            order = len(self.model["coeffs"])
            predictions = []
            current = last_prices[-order:].tolist()

            for _ in range(steps):
                pred = np.dot(self.model["coeffs"], current[-order:])
                predictions.append(pred)
                current.append(pred)

            return np.array(predictions)

    def _predict_exp_smooth(self, steps: int) -> np.ndarray:
        alpha = self.model["alpha"]
        last_val = self.model["last_value"]
        forecasts = [last_val] * steps
        return np.array(forecasts)


class VolatilityModel:
    """Volatility modeling with GARCH, EWMA, and Realized Volatility."""

    def __init__(self, method: str = "garch"):
        self.method = method
        self.model = None
        self.params = {}

    def fit(self, returns: np.ndarray, **kwargs):
        if self.method == "garch":
            self._fit_garch(returns, **kwargs)
        elif self.method == "ewma":
            self._fit_ewma(returns, **kwargs)
        elif self.method == "realized":
            self._fit_realized(returns, **kwargs)
        return self

    def _fit_garch(self, returns: np.ndarray, p: int = 1, q: int = 1):
        try:
            from arch import arch_model

            self.model = arch_model(returns, vol="Garch", p=p, q=q)
            self.model = self.model.fit(disp="off")
        except ImportError:
            logger.warning("arch package not available, using EWMA fallback")
            self._fit_ewma(returns)

    def _fit_ewma(self, returns: np.ndarray, lambda_param: float = 0.94):
        var = np.zeros(len(returns))
        var[0] = np.var(returns)

        for t in range(1, len(returns)):
            var[t] = (
                lambda_param * var[t - 1] + (1 - lambda_param) * returns[t - 1] ** 2
            )

        self.model = {"variance": var, "lambda": lambda_param}

    def _fit_realized(self, returns: np.ndarray, window: int = 20):
        rv = pd.Series(returns).rolling(window=window).std().values * np.sqrt(252)
        self.model = {"realized_vol": rv, "window": window}

    def forecast(self, steps: int = 1) -> np.ndarray:
        if self.method == "garch" and hasattr(self.model, "forecast"):
            return np.sqrt(self.model.forecast(horizon=steps).variance.values[-1])
        elif self.method == "ewma":
            last_var = self.model["variance"][-1]
            return np.full(steps, np.sqrt(last_var * 252))
        elif self.method == "realized":
            return np.full(steps, self.model["realized_vol"][-1])
        else:
            raise ValueError("Model not fitted")

    def get_volatility(self) -> np.ndarray:
        if self.method == "garch" and hasattr(self.model, "conditional_volatility"):
            return self.model.conditional_volatility
        elif self.method == "ewma":
            return np.sqrt(self.model["variance"])
        elif self.method == "realized":
            return self.model["realized_vol"]
        else:
            raise ValueError("Model not fitted")


class TrendDetector:
    """Trend detection using multiple methods."""

    def __init__(self, methods: List[str] = None):
        self.methods = methods or ["ma_crossover", "adx", "linear_reg"]
        self.signals = {}

    def detect(
        self, prices: np.ndarray, highs: np.ndarray = None, lows: np.ndarray = None
    ) -> Dict[str, TrendDirection]:
        results = {}

        if "ma_crossover" in self.methods:
            results["ma_crossover"] = self._ma_crossover(prices)

        if "adx" in self.methods and highs is not None and lows is not None:
            results["adx"] = self._adx(prices, highs, lows)

        if "linear_reg" in self.methods:
            results["linear_reg"] = self._linear_trend(prices)

        if "hurst" in self.methods:
            results["hurst"] = self._hurst_exponent(prices)

        return results

    def _ma_crossover(
        self, prices: np.ndarray, short: int = 20, long: int = 50
    ) -> TrendDirection:
        ma_short = pd.Series(prices).rolling(short).mean().iloc[-1]
        ma_long = pd.Series(prices).rolling(long).mean().iloc[-1]

        if ma_short > ma_long * 1.02:
            return TrendDirection.UP
        elif ma_short < ma_long * 0.98:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _adx(
        self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14
    ) -> TrendDirection:
        df = pd.DataFrame({"high": highs, "low": lows, "close": prices})

        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["close"].shift(1))
        df["tr3"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

        df["plus_dm"] = np.where(
            (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
            df["high"] - df["high"].shift(1),
            0,
        )
        df["minus_dm"] = np.where(
            (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
            df["low"].shift(1) - df["low"],
            0,
        )

        df["atr"] = df["tr"].rolling(period).mean()
        df["plus_di"] = 100 * df["plus_dm"].rolling(period).mean() / df["atr"]
        df["minus_di"] = 100 * df["minus_dm"].rolling(period).mean() / df["atr"]
        df["dx"] = (
            100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])
        )
        df["adx"] = df["dx"].rolling(period).mean()

        adx_val = df["adx"].iloc[-1]
        plus_di = df["plus_di"].iloc[-1]
        minus_di = df["minus_di"].iloc[-1]

        if adx_val > 25:
            if plus_di > minus_di:
                return TrendDirection.UP
            else:
                return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _linear_trend(self, prices: np.ndarray, window: int = 30) -> TrendDirection:
        recent = prices[-window:]
        x = np.arange(len(recent))
        slope, _, _, _, _ = stats.linregress(x, recent)

        price_change = slope * window / recent[0]

        if price_change > 0.05:
            return TrendDirection.UP
        elif price_change < -0.05:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL

    def _hurst_exponent(self, prices: np.ndarray, max_lag: int = 100) -> TrendDirection:
        lags = range(2, min(max_lag, len(prices) // 4))
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]

        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0]

        if hurst > 0.6:
            return TrendDirection.UP
        elif hurst < 0.4:
            return TrendDirection.DOWN
        return TrendDirection.NEUTRAL


class PatternRecognizer:
    """Pattern recognition for technical analysis."""

    def __init__(self):
        self.patterns = {}

    def find_patterns(self, ohlcv: OHLCVData) -> Dict[str, List[int]]:
        patterns = {}
        patterns["support_resistance"] = self._support_resistance(ohlcv)
        patterns["head_shoulder"] = self._head_and_shoulders(ohlcv)
        patterns["double_top_bottom"] = self._double_patterns(ohlcv)
        patterns["triangles"] = self._triangles(ohlcv)
        patterns["candlestick"] = self._candlestick_patterns(ohlcv)
        return patterns

    def _support_resistance(self, ohlcv: OHLCVData, window: int = 10) -> List[int]:
        highs = ohlcv.high
        lows = ohlcv.low

        levels = []
        for i in range(window, len(highs) - window):
            if all(highs[i] > highs[i - j] for j in range(1, window + 1)) and all(
                highs[i] > highs[i + j] for j in range(1, window + 1)
            ):
                levels.append((i, highs[i], "resistance"))

            if all(lows[i] < lows[i - j] for j in range(1, window + 1)) and all(
                lows[i] < lows[i + j] for j in range(1, window + 1)
            ):
                levels.append((i, lows[i], "support"))

        return levels

    def _head_and_shoulders(self, ohlcv: OHLCVData) -> List[Dict]:
        close = ohlcv.close
        patterns = []

        for i in range(20, len(close) - 20):
            window = close[i - 20 : i + 20]
            peaks = self._find_peaks(window, order=5)

            if len(peaks) >= 3:
                for j in range(len(peaks) - 2):
                    left, head, right = peaks[j], peaks[j + 1], peaks[j + 2]

                    if head > left and head > right and abs(left - right) / left < 0.05:
                        patterns.append(
                            {
                                "type": "head_and_shoulders",
                                "index": i - 20 + head,
                                "left_shoulder": i - 20 + left,
                                "head": i - 20 + head,
                                "right_shoulder": i - 20 + right,
                            }
                        )

        return patterns

    def _double_patterns(self, ohlcv: OHLCVData, tolerance: float = 0.03) -> List[Dict]:
        close = ohlcv.close
        patterns = []

        peaks = self._find_peaks(close, order=10)
        troughs = self._find_peaks(-close, order=10)

        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                if abs(close[peaks[i]] - close[peaks[j]]) / close[peaks[i]] < tolerance:
                    patterns.append(
                        {
                            "type": "double_top",
                            "indices": [peaks[i], peaks[j]],
                            "price": close[peaks[i]],
                        }
                    )

        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                if (
                    abs(close[troughs[i]] - close[troughs[j]]) / close[troughs[i]]
                    < tolerance
                ):
                    patterns.append(
                        {
                            "type": "double_bottom",
                            "indices": [troughs[i], troughs[j]],
                            "price": close[troughs[i]],
                        }
                    )

        return patterns

    def _triangles(self, ohlcv: OHLCVData) -> List[Dict]:
        patterns = []
        close = ohlcv.close

        for i in range(30, len(close)):
            window_high = ohlcv.high[i - 30 : i]
            window_low = ohlcv.low[i - 30 : i]

            high_slope, _, _, _, _ = stats.linregress(range(30), window_high)
            low_slope, _, _, _, _ = stats.linregress(range(30), window_low)

            if abs(high_slope) < 0.001 and low_slope > 0.001:
                patterns.append({"type": "ascending_triangle", "index": i})
            elif high_slope < -0.001 and abs(low_slope) < 0.001:
                patterns.append({"type": "descending_triangle", "index": i})
            elif high_slope < -0.001 and low_slope > 0.001:
                patterns.append({"type": "symmetrical_triangle", "index": i})

        return patterns

    def _candlestick_patterns(self, ohlcv: OHLCVData) -> List[Dict]:
        patterns = []
        opens = ohlcv.open
        highs = ohlcv.high
        lows = ohlcv.low
        closes = ohlcv.close

        for i in range(2, len(closes)):
            body = abs(closes[i] - opens[i])
            lower_shadow = (
                opens[i] - lows[i] if closes[i] > opens[i] else closes[i] - lows[i]
            )
            upper_shadow = (
                highs[i] - closes[i] if closes[i] > opens[i] else highs[i] - opens[i]
            )

            if lower_shadow > 2 * body and upper_shadow < body * 0.5:
                patterns.append({"type": "hammer", "index": i, "bullish": True})

            if upper_shadow > 2 * body and lower_shadow < body * 0.5:
                patterns.append({"type": "shooting_star", "index": i, "bullish": False})

            if (
                closes[i] > opens[i]
                and closes[i - 1] < opens[i - 1]
                and opens[i] < closes[i - 1]
                and closes[i] > opens[i - 1]
            ):
                patterns.append({"type": "bullish_engulfing", "index": i})

            if (
                closes[i] < opens[i]
                and closes[i - 1] > opens[i - 1]
                and opens[i] > closes[i - 1]
                and closes[i] < opens[i - 1]
            ):
                patterns.append({"type": "bearish_engulfing", "index": i})

        return patterns

    def _find_peaks(self, data: np.ndarray, order: int = 5) -> np.ndarray:
        from scipy.signal import argrelextrema

        return argrelextrema(data, np.greater, order=order)[0]
