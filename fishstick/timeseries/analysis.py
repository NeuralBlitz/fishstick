"""
Time Series Analysis Tools
"""

from typing import Tuple, Optional
import torch
import numpy as np
from scipy import signal, stats


class StationarityTest:
    """Stationarity tests for time series."""

    @staticmethod
    def adfuller(series: np.ndarray) -> dict:
        """Augmented Dickey-Fuller test."""
        result = stats.trendend_adf_table(series)
        return {"statistic": result, "critical_values": result}

    @staticmethod
    def kpss(series: np.ndarray) -> dict:
        """KPSS test."""
        statistic, p_value, lags, critical_values = stats.kpss(series, "c")
        return {
            "statistic": statistic,
            "p_value": p_value,
            "lags": lags,
            "critical_values": critical_values,
        }


class SeasonalDecompose:
    """Seasonal decomposition of time series."""

    def __init__(self, period: int = 12, model: str = "additive"):
        self.period = period
        self.model = model
        self.trend: Optional[np.ndarray] = None
        self.seasonal: Optional[np.ndarray] = None
        self.residual: Optional[np.ndarray] = None

    def fit(self, series: np.ndarray) -> "SeasonalDecompose":
        n = len(series)
        trend = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.period // 2)
            end = min(n, i + self.period // 2 + 1)
            trend[i] = np.mean(series[start:end])

        detrended = series - trend if self.model == "additive" else series / trend

        seasonal = np.zeros(self.period)
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            seasonal[i] = np.mean(detrended[indices])

        seasonal = np.tile(seasonal, n // self.period + 1)[:n]

        residual = (
            series - trend - seasonal
            if self.model == "additive"
            else series / (trend * seasonal)
        )

        self.trend = trend
        self.seasonal = seasonal
        self.residual = residual

        return self

    def transform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.trend, self.seasonal, self.residual


class AutocorrelationAnalysis:
    """Autocorrelation analysis tools."""

    @staticmethod
    def acf(series: np.ndarray, nlags: int = 40) -> np.ndarray:
        """Compute autocorrelation function."""
        n = len(series)
        mean = np.mean(series)
        var = np.var(series)

        acf = np.correlate(series - mean, series - mean, mode="full")
        acf = acf[n - 1 : n + nlags]
        acf /= var * n

        return acf[:nlags]

    @staticmethod
    def pacf(series: np.ndarray, nlags: int = 40) -> np.ndarray:
        """Compute partial autocorrelation function."""
        from statsmodels.tsa.stattools import pacf

        return pacf(series, nlags=nlags)

    @staticmethod
    def plot_acf(series: np.ndarray, nlags: int = 40) -> dict:
        """Compute ACF with confidence intervals."""
        acf_values = AutocorrelationAnalysis.acf(series, nlags)
        n = len(series)

        confidence = 1.96 / np.sqrt(n)

        return {
            "acf": acf_values,
            "confidence": confidence,
            "significant": np.abs(acf_values) > confidence,
        }


class FourierTransform:
    """Fourier transform for frequency analysis."""

    @staticmethod
    def fft(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT."""
        n = len(series)
        fft_result = np.fft.fft(series)
        freq = np.fft.fftfreq(n)

        magnitude = np.abs(fft_result)
        phase = np.fft.fftshift(np.angle(fft_result))
        freq = np.fft.fftshift(freq)

        return magnitude, freq

    @staticmethod
    def power_spectrum(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum."""
        n = len(series)
        fft_result = np.fft.fft(series)
        power = np.abs(fft_result) ** 2 / n

        freq = np.fft.fftfreq(n)

        positive_mask = freq > 0
        return freq[positive_mask], power[positive_mask]


class WaveletTransform:
    """Wavelet transform for time-frequency analysis."""

    @staticmethod
    def cwt(series: np.ndarray, widths: np.ndarray = None) -> np.ndarray:
        """Continuous wavelet transform."""
        if widths is None:
            widths = np.arange(1, 31)

        cwt_matrix = signal.cwt(series, signal.ricker, widths)
        return cwt_matrix

    @staticmethod
    def dwt(series: np.ndarray, wavelet: str = "db4", level: int = None) -> dict:
        """Discrete wavelet transform."""
        import pywt

        if level is None:
            level = pywt.dwt_max_level(len(series), wavelet)

        coeffs = pywt.wavedec(series, wavelet, level=level)

        return {
            "approximation": coeffs[0],
            "details": coeffs[1:],
            "wavelet": wavelet,
            "level": level,
        }


class SpectralAnalysis:
    """Spectral analysis tools."""

    @staticmethod
    def periodogram(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute periodogram."""
        freqs, psd = signal.periodogram(series)
        return freqs, psd

    @staticmethod
    def welch(series: np.ndarray, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Welch's method for PSD estimation."""
        freqs, psd = signal.welch(series, nperseg=nperseg)
        return freqs, psd


class ChangePointDetection:
    """Change point detection algorithms."""

    @staticmethod
    def cusum(series: np.ndarray, threshold: float = 5.0) -> list:
        """CUSUM algorithm for change point detection."""
        mean = np.mean(series)
        std = np.std(series)
        normalized = (series - mean) / std

        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))

        for i in range(1, len(series)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + normalized[i] - 0.5)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - normalized[i] - 0.5)

        change_points = []
        for i in range(len(series)):
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                change_points.append(i)

        return change_points

    @staticmethod
    def binary_segmentation(series: np.ndarray, min_size: int = 30) -> list:
        """Binary segmentation for change point detection."""

        def cost(start, end):
            if end - start < min_size:
                return float("inf")
            segment = series[start:end]
            return len(segment) * np.var(segment)

        def find_best_split(start, end):
            best_cost = float("inf")
            best_split = None

            for t in range(start + min_size, end - min_size):
                c = cost(start, t) + cost(t, end)
                if c < best_cost:
                    best_cost = c
                    best_split = t

            return best_split

        change_points = []
        segments = [(0, len(series))]

        while segments:
            start, end = segments.pop()
            split = find_best_split(start, end)

            if split is not None:
                change_points.append(split)
                segments.append((start, split))
                segments.append((split, end))

        return sorted(change_points)
