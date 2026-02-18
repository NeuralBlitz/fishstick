from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AcquisitionFunction(ABC):
    @abstractmethod
    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        pass


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, xi: float = 0.01, minimize: bool = True):
        self.xi = xi
        self.minimize = minimize

    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        if self.minimize:
            best_value = -best_value
            mu = -mu

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best_value - self.xi) / (std + 1e-10)
            ei = (mu - best_value - self.xi) * self._norm_cdf(z) + std * self._norm_pdf(
                z
            )
            ei[std < 1e-8] = 0.0

        return ei

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + np.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, xi: float = 0.01, minimize: bool = True):
        self.xi = xi
        self.minimize = minimize

    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        if self.minimize:
            best_value = -best_value
            mu = -mu

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best_value - self.xi) / (std + 1e-10)
            pi = self._norm_cdf(z)
            pi[std < 1e-8] = 0.0

        return pi

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + np.erf(x / math.sqrt(2)))


class UpperConfidenceBound(AcquisitionFunction):
    def __init__(self, kappa: float = 2.0, minimize: bool = True):
        self.kappa = kappa
        self.minimize = minimize

    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        if self.minimize:
            return mu - self.kappa * std
        return mu + self.kappa * std


class ThompsonSampling(AcquisitionFunction):
    def __init__(self, minimize: bool = True):
        self.minimize = minimize

    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        rng = kwargs.get("rng", np.random.default_rng())
        samples = rng.normal(mu, std)

        if self.minimize:
            samples = -samples

        return samples


class KnowledgeGradient(AcquisitionFunction):
    def __init__(self, minimize: bool = True):
        self.minimize = minimize

    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        if self.minimize:
            mu = -mu

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best_value) / (std + 1e-10)
            kg = (mu - best_value) * self._norm_cdf(z) + std * self._norm_pdf(z)
            kg[std < 1e-8] = 0.0

        return kg

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + np.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class EntropySearch(AcquisitionFunction):
    def __init__(self, minimize: bool = True):
        self.minimize = minimize

    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            h = 0.5 * np.log(2 * math.pi * std**2) + 0.5
            h[std < 1e-8] = 0.0

        return -h


class PredictiveEntropySearch(AcquisitionFunction):
    def __init__(self, minimize: bool = True):
        self.minimize = minimize

    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        if self.minimize:
            mu = -mu

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best_value) / (std + 1e-10)
            pes = self._norm_cdf(z) * np.log(self._norm_cdf(z) + 1e-10) + (
                1 - self._norm_cdf(z)
            ) * np.log(1 - self._norm_cdf(z) + 1e-10)
            pes = -pes
            pes[std < 1e-8] = 0.0

        return pes

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + np.erf(x / math.sqrt(2)))


class MaxValueEntropySearch(AcquisitionFunction):
    def __init__(self, minimize: bool = True, num_samples: int = 100):
        self.minimize = minimize
        self.num_samples = num_samples

    def __call__(
        self, mu: np.ndarray, std: np.ndarray, best_value: float, **kwargs: Any
    ) -> np.ndarray:
        rng = kwargs.get("rng", np.random.default_rng())

        if self.minimize:
            mu = -mu

        samples = rng.normal(mu, std, size=(self.num_samples, len(mu)))
        max_samples = np.max(samples, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            p_max = self._norm_cdf((max_samples - mu) / (std + 1e-10))
            mes = np.log(p_max + 1e-10)

        return mes

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + np.erf(x / math.sqrt(2)))


def get_acquisition_function(name: str, **kwargs: Any) -> AcquisitionFunction:
    name = name.lower()

    if name == "ei" or name == "expected_improvement":
        return ExpectedImprovement(**kwargs)
    elif name == "pi" or name == "probability_of_improvement":
        return ProbabilityOfImprovement(**kwargs)
    elif name == "ucb" or name == "upper_confidence_bound":
        return UpperConfidenceBound(**kwargs)
    elif name == "ts" or name == "thompson_sampling":
        return ThompsonSampling(**kwargs)
    elif name == "kg" or name == "knowledge_gradient":
        return KnowledgeGradient(**kwargs)
    elif name == "es" or name == "entropy_search":
        return EntropySearch(**kwargs)
    elif name == "pes" or name == "predictive_entropy_search":
        return PredictiveEntropySearch(**kwargs)
    elif name == "mes" or name == "max_value_entropy_search":
        return MaxValueEntropySearch(**kwargs)
    else:
        raise ValueError(f"Unknown acquisition function: {name}")
