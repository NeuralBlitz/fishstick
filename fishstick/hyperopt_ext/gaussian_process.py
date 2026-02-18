from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GaussianProcess:
    length_scale: float = 1.0
    variance: float = 1.0
    noise_variance: float = 1e-6
    length_scale_bounds: tuple[float, float] = (1e-3, 1e3)
    variance_bounds: tuple[float, float] = (1e-3, 1e3)
    X_train: np.ndarray | None = field(default=None, repr=False)
    y_train: np.ndarray | None = field(default=None, repr=False)
    _lml: float = -float("inf")
    _alpha: np.ndarray | None = field(default=None, repr=False)
    _L: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(42)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        if X1.shape[1] != X2.shape[1]:
            raise ValueError(f"Dimension mismatch: {X1.shape[1]} vs {X2.shape[1]}")
        dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        return self.variance * np.exp(-0.5 / self.length_scale**2 * dists)

    def _matern_kernel(
        self, X1: np.ndarray, X2: np.ndarray, nu: float = 1.5
    ) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sqrt(
            np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2) + 1e-10
        )
        if nu == 0.5:
            return self.variance * np.exp(-dists / self.length_scale)
        elif nu == 1.5:
            sqrt3 = math.sqrt(3) * dists / self.length_scale
            return self.variance * (1 + sqrt3) * np.exp(-sqrt3)
        elif nu == 2.5:
            sqrt5 = math.sqrt(5) * dists / self.length_scale
            return self.variance * (1 + sqrt5 + sqrt5**2 / 3) * np.exp(-sqrt5)
        return self._rbf_kernel(X1, X2)

    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return self.variance * (X1 @ X2.T)

    def kernel(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        return self._rbf_kernel(X1, X2)

    def fit(
        self, X: np.ndarray, y: np.ndarray, optimize: bool = True
    ) -> "GaussianProcess":
        self.X_train = np.atleast_2d(np.array(X))
        self.y_train = np.atleast_2d(np.array(y)).reshape(-1, 1)

        if len(self.X_train) < 2:
            self._alpha = np.zeros((len(self.X_train), 1))
            return self

        K = self.kernel(self.X_train) + self.noise_variance * np.eye(len(self.X_train))

        try:
            self._L = np.linalg.cholesky(K)
            self._alpha = np.linalg.solve(
                self._L.T, np.linalg.solve(self._L, self.y_train)
            )
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(len(K))
            self._L = np.linalg.cholesky(K)
            self._alpha = np.linalg.solve(
                self._L.T, np.linalg.solve(self._L, self.y_train)
            )

        if optimize:
            self._optimize_hyperparameters()

        return self

    def _optimize_hyperparameters(self) -> None:
        def neg_log_marginal_likelihood(params: np.ndarray) -> float:
            self.length_scale = float(params[0])
            self.variance = float(params[1])

            K = self.kernel(self.X_train) + self.noise_variance * np.eye(
                len(self.X_train)
            )

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                lml = (
                    -0.5 * self.y_train.T @ alpha
                    - 0.5 * log_det
                    - 0.5 * len(self.X_train) * math.log(2 * math.pi)
                )
                return -float(lml)
            except np.linalg.LinAlgError:
                return 1e10

        best_lml = float("inf")
        best_params = np.array([self.length_scale, self.variance])

        for _ in range(20):
            x0 = np.array(
                [
                    self.rng.uniform(*self.length_scale_bounds),
                    self.rng.uniform(*self.variance_bounds),
                ]
            )

            try:
                from scipy.optimize import minimize

                result = minimize(neg_log_marginal_likelihood, x0, method="L-BFGS-B")

                if result.fun < best_lml:
                    best_lml = result.fun
                    best_params = result.x
            except ImportError:
                pass

        self.length_scale = max(
            self.length_scale_bounds[0],
            min(self.length_scale_bounds[1], best_params[0]),
        )
        self.variance = max(
            self.variance_bounds[0], min(self.variance_bounds[1], best_params[1])
        )

    def predict(
        self, X: np.ndarray, return_std: bool = False, return_cov: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)

        if self.X_train is None or self.y_train is None or len(self.X_train) < 2:
            mu = np.zeros(len(X))
            if return_std:
                return mu, np.ones(len(X)) * math.sqrt(self.variance)
            return mu, np.zeros(len(X))

        K_star = self.kernel(self.X_train, X)
        K_star_star = self.kernel(X)

        try:
            mu = K_star.T @ self._alpha
        except Exception:
            K = self.kernel(self.X_train) + self.noise_variance * np.eye(
                len(self.X_train)
            )
            K_inv = np.linalg.pinv(K)
            mu = K_star.T @ K_inv @ self.y_train

        if return_cov:
            cov = K_star_star - K_star.T @ K_inv @ K_star
            var = np.diag(cov)
            var = np.maximum(var, 0)
            return mu.flatten(), np.sqrt(var), cov

        if return_std:
            try:
                v = np.linalg.solve(self._L, K_star)
                var = np.diag(K_star_star) - np.sum(v**2, axis=0)
            except Exception:
                K = self.kernel(self.X_train) + self.noise_variance * np.eye(
                    len(self.X_train)
                )
                K_inv = np.linalg.pinv(K)
                cov = K_star_star - K_star.T @ K_inv @ K_star
                var = np.diag(cov)

            var = np.maximum(var, 0)
            std = np.sqrt(var)
            return mu.flatten(), std

        return mu.flatten(), np.zeros(len(X))

    def sample_y(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        mu, std = self.predict(X, return_std=True)
        samples = self.rng.normal(mu, std, size=(n_samples, len(X)))
        return samples

    def gradient(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            return np.zeros((len(X), 0))

        X = np.atleast_2d(X)
        grads = np.zeros((len(X), self.X_train.shape[1]))

        K_star = self.kernel(self.X_train, X)
        K = self.kernel(self.X_train) + self.noise_variance * np.eye(len(self.X_train))

        try:
            K_inv = np.linalg.pinv(K)
        except Exception:
            return grads

        for i in range(self.X_train.shape[1]):
            dK_star_dxi = (
                (self.X_train[:, i][:, np.newaxis] - X[:, i][np.newaxis, :])
                / (self.length_scale**2)
                * K_star
            )
            grads[:, i] = (K_inv @ dK_star_dxi @ self._alpha).flatten()

        return grads
