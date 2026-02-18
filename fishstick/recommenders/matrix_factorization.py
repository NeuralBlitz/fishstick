"""
Matrix Factorization Methods.

Implements various matrix factorization techniques for collaborative filtering
including SVD, NMF, ALS, and Bayesian approaches.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings

from .base import InteractionMatrix, RecommenderBase


class SVDRecommender:
    """SVD-based Matrix Factorization.

    Implements truncated SVD for matrix factorization, treating missing
    values as zeros (for implicit feedback) or using imputation strategies.

    Attributes:
        n_factors: Number of latent factors
        n_iterations: Number of iterations for iterative SVD
        regularization: L2 regularization strength
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 20,
        regularization: float = 0.01,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        self.interactions: Optional[InteractionMatrix] = None
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(
        self,
        interactions: InteractionMatrix,
        random_state: int = 42,
    ) -> SVDRecommender:
        """Fit SVD model using alternating least squares.

        Args:
            interactions: User-item interaction matrix
            random_state: Random seed

        Returns:
            Self
        """
        np.random.seed(random_state)

        self.interactions = interactions
        self.n_users = interactions.n_users
        self.n_items = interactions.n_items

        self.global_mean = (
            interactions.global_mean if interactions.ratings.nnz > 0 else 0.0
        )

        R = interactions.ratings.toarray()

        mask = R > 0

        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        for u in range(self.n_users):
            rated = R[u] > 0
            if rated.sum() > 0:
                self.user_bias[u] = (
                    R[u, rated].sum() - self.global_mean * rated.sum()
                ) / rated.sum()

        for i in range(self.n_items):
            rated = R[:, i] > 0
            if rated.sum() > 0:
                self.item_bias[i] = (
                    R[rated, i].sum() - self.global_mean * rated.sum()
                ) / rated.sum()

        R_centered = (
            R
            - self.global_mean
            - self.user_bias[:, np.newaxis]
            - self.item_bias[np.newaxis, :]
        )
        R_centered = np.where(mask, R_centered, 0)

        self.user_factors = np.random.randn(self.n_users, self.n_factors) * 0.1
        self.item_factors = np.random.randn(self.n_items, self.n_factors) * 0.1

        for iteration in range(self.n_iterations):
            for u in range(self.n_users):
                rated_items = np.where(mask[u])[0]
                if len(rated_items) == 0:
                    continue

                V = self.item_factors[rated_items]
                residual = R_centered[u, rated_items]

                A = V.T @ V + self.regularization * np.eye(self.n_factors)
                b = V.T @ residual

                self.user_factors[u] = np.linalg.solve(A, b)

            for i in range(self.n_items):
                rated_users = np.where(mask[:, i])[0]
                if len(rated_users) == 0:
                    continue

                U = self.user_factors[rated_users]
                residual = R_centered[rated_users, i]

                A = U.T @ U + self.regularization * np.eye(self.n_factors)
                b = U.T @ residual

                self.item_factors[i] = np.linalg.solve(A, b)

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        pred = self.global_mean

        if user_idx < self.n_users:
            pred += self.user_bias[user_idx]

        if item_idx < self.n_items:
            pred += self.item_bias[item_idx]

        if user_idx < self.n_users and item_idx < self.n_items:
            pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

        return pred

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        predictions = []

        user_rated = set()
        if exclude_known and self.interactions is not None:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        for item_idx in range(self.n_items):
            if item_idx in user_rated:
                continue

            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class NMFRecommender:
    """Non-negative Matrix Factorization.

    Implements NMF which enforces non-negativity, often resulting in
    more interpretable latent factors.

    Attributes:
        n_factors: Number of latent factors
        n_iterations: Maximum number of iterations
        regularization: Regularization strength (L1 and L2)
    """

    def __init__(
        self,
        n_factors: int = 30,
        n_iterations: int = 200,
        regularization: float = 0.01,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization

        self.W: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.interactions: Optional[InteractionMatrix] = None

    def fit(
        self,
        interactions: InteractionMatrix,
        random_state: int = 42,
    ) -> NMFRecommender:
        """Fit NMF model using multiplicative updates.

        Args:
            interactions: User-item interaction matrix
            random_state: Random seed

        Returns:
            Self
        """
        np.random.seed(random_state)

        self.interactions = interactions

        R = interactions.ratings.toarray()
        R = np.maximum(R, 0)

        n_users, n_items = R.shape

        self.W = np.random.rand(n_users, self.n_factors) + 0.1
        self.H = np.random.rand(self.n_factors, n_items) + 0.1

        for iteration in range(self.n_iterations):
            H_update = (self.W.T @ R) / (self.W.T @ self.W @ self.H + 1e-8)
            self.H *= H_update
            self.H = np.maximum(self.H, 1e-8)

            W_update = (R @ self.H.T) / (self.W @ self.H @ self.H.T + 1e-8)
            self.W *= W_update
            self.W = np.maximum(self.W, 1e-8)

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating."""
        if self.W is None:
            raise RuntimeError("Model not fitted.")

        return np.dot(self.W[user_idx], self.H[:, item_idx])

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if self.W is None:
            raise RuntimeError("Model not fitted.")

        predictions = []

        user_rated = set()
        if exclude_known and self.interactions is not None:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        scores = self.W[user_idx] @ self.H

        for item_idx in range(len(scores)):
            if item_idx in user_rated:
                continue
            predictions.append((item_idx, scores[item_idx]))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class ALSRecommender:
    """Alternating Least Squares Matrix Factorization.

    Efficient implementation of ALS for implicit feedback using
    weighted regularization.

    Attributes:
        n_factors: Number of latent factors
        n_iterations: Number of ALS iterations
        regularization: Regularization strength
        alpha: Confidence scaling for implicit feedback
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 15,
        regularization: float = 0.1,
        alpha: float = 40.0,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

        self.interactions: Optional[InteractionMatrix] = None

    def fit(
        self,
        interactions: InteractionMatrix,
        random_state: int = 42,
    ) -> ALSRecommender:
        """Fit ALS model.

        Args:
            interactions: User-item interaction matrix
            random_state: Random seed

        Returns:
            Self
        """
        np.random.seed(random_state)

        self.interactions = interactions

        n_users = interactions.n_users
        n_items = interactions.n_items

        self.user_factors = np.random.rand(n_users, self.n_factors) * 0.1
        self.item_factors = np.random.rand(n_items, self.n_factors) * 0.1

        Cui = interactions.ratings.T.tocsr()

        for iteration in range(self.n_iterations):
            self._update_factors(Cui, self.user_factors, self.item_factors)
            self._update_factors(
                interactions.ratings, self.item_factors, self.user_factors
            )

        return self

    def _update_factors(
        self,
        Cui: csr_matrix,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        """Update factor matrix."""
        YtY = Y.T @ Y
        lambda_I = self.regularization * np.eye(self.n_factors)

        for u in range(X.shape[0]):
            row = Cui.getrow(u)
            indices = row.indices
            ratings = row.data

            if len(indices) == 0:
                X[u] = np.zeros(self.n_factors)
                continue

            confidence = 1 + self.alpha * ratings
            Y_u = Y[indices]

            A = YtY + Y_u.T @ (Y_u * (confidence - 1)[:, np.newaxis]) + lambda_I
            b = Y_u.T @ (confidence * ratings)

            X[u] = np.linalg.solve(A, b)

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict score."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        predictions = []

        user_rated = set()
        if exclude_known and self.interactions is not None:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        scores = self.user_factors[user_idx] @ self.item_factors.T

        for item_idx in range(len(scores)):
            if item_idx in user_rated:
                continue
            predictions.append((item_idx, scores[item_idx]))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class BiasedMF:
    """Biased Matrix Factorization.

    Standard MF with learned user and item biases.

    Attributes:
        n_factors: Number of latent factors
        n_iterations: Number of training iterations
        learning_rate: Learning rate for SGD
        regularization: L2 regularization strength
        bias_regularization: Bias regularization strength
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        bias_regularization: float = 0.0,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.bias_regularization = bias_regularization

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        self.interactions: Optional[InteractionMatrix] = None

    def fit(
        self,
        interactions: InteractionMatrix,
        random_state: int = 42,
    ) -> BiasedMF:
        """Fit biased MF using SGD.

        Args:
            interactions: User-item interaction matrix
            random_state: Random seed

        Returns:
            Self
        """
        np.random.seed(random_state)

        self.interactions = interactions
        self.global_mean = interactions.global_mean

        n_users = interactions.n_users
        n_items = interactions.n_items

        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.1
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.1
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        coo = interactions.ratings.tocoo()
        users = coo.row
        items = coo.col
        ratings = coo.data

        n_samples = len(ratings)

        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)

            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                pred = self.global_mean + self.user_bias[u] + self.item_bias[i]
                pred += np.dot(self.user_factors[u], self.item_factors[i])

                error = r - pred

                self.user_bias[u] += self.learning_rate * (
                    error - self.bias_regularization * self.user_bias[u]
                )
                self.item_bias[i] += self.learning_rate * (
                    error - self.bias_regularization * self.item_bias[i]
                )

                user_f = self.user_factors[u].copy()
                self.user_factors[u] += self.learning_rate * (
                    error * self.item_factors[i] - self.regularization * user_f
                )
                self.item_factors[i] += self.learning_rate * (
                    error * user_f - self.regularization * self.item_factors[i]
                )

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        pred = self.global_mean

        if user_idx < len(self.user_bias):
            pred += self.user_bias[user_idx]

        if item_idx < len(self.item_bias):
            pred += self.item_bias[item_idx]

        if user_idx < len(self.user_factors) and item_idx < len(self.item_factors):
            pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

        return pred

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        predictions = []

        user_rated = set()
        if exclude_known and self.interactions is not None:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        for item_idx in range(len(self.item_factors)):
            if item_idx in user_rated:
                continue

            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class WeightedRegularizedMF:
    """Weighted Regularized Matrix Factorization.

    MF with per-rating confidence weights and biased predictions.

    Attributes:
        n_factors: Number of latent factors
        n_iterations: Number of training iterations
        learning_rate: Learning rate
        regularization: Regularization strength
        weight_scale: Scale for confidence weights
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        weight_scale: float = 1.0,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.weight_scale = weight_scale

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        self.interactions: Optional[InteractionMatrix] = None

    def fit(
        self,
        interactions: InteractionMatrix,
        random_state: int = 42,
    ) -> WeightedRegularizedMF:
        """Fit weighted regularized MF.

        Args:
            interactions: User-item interaction matrix
            random_state: Random seed

        Returns:
            Self
        """
        np.random.seed(random_state)

        self.interactions = interactions
        self.global_mean = interactions.global_mean

        n_users = interactions.n_users
        n_items = interactions.n_items

        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.1
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.1
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        coo = interactions.ratings.tocoo()
        users = coo.row
        items = coo.col
        ratings = coo.data

        n_samples = len(ratings)

        max_rating = ratings.max() if ratings.max() > 0 else 1.0

        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)

            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                confidence = 1 + self.weight_scale * (r / max_rating)

                pred = self.global_mean + self.user_bias[u] + self.item_bias[i]
                pred += np.dot(self.user_factors[u], self.item_factors[i])

                error = r - pred

                self.user_bias[u] += (
                    self.learning_rate
                    * confidence
                    * (error - self.regularization * self.user_bias[u])
                )
                self.item_bias[i] += (
                    self.learning_rate
                    * confidence
                    * (error - self.regularization * self.item_bias[i])
                )

                user_f = self.user_factors[u].copy()
                self.user_factors[u] += (
                    self.learning_rate
                    * confidence
                    * (error * self.item_factors[i] - self.regularization * user_f)
                )
                self.item_factors[i] += (
                    self.learning_rate
                    * confidence
                    * (error * user_f - self.regularization * self.item_factors[i])
                )

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        pred = self.global_mean

        if user_idx < len(self.user_bias):
            pred += self.user_bias[user_idx]

        if item_idx < len(self.item_bias):
            pred += self.item_bias[item_idx]

        if user_idx < len(self.user_factors) and item_idx < len(self.item_factors):
            pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

        return pred

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        predictions = []

        user_rated = set()
        if exclude_known and self.interactions is not None:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        for item_idx in range(len(self.item_factors)):
            if item_idx in user_rated:
                continue

            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class SVDPlusPlus:
    """SVD++ - Matrix Factorization with implicit feedback.

    Extends SVD by incorporating implicit feedback (which items
    a user has interacted with).

    Attributes:
        n_factors: Number of latent factors
        n_iterations: Number of training iterations
        learning_rate: Learning rate
        regularization: Regularization strength
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 20,
        learning_rate: float = 0.007,
        regularization: float = 0.02,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_implicit: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        self.interactions: Optional[InteractionMatrix] = None

    def fit(
        self,
        interactions: InteractionMatrix,
        random_state: int = 42,
    ) -> SVDPlusPlus:
        """Fit SVD++ model.

        Args:
            interactions: User-item interaction matrix
            random_state: Random seed

        Returns:
            Self
        """
        np.random.seed(random_state)

        self.interactions = interactions
        self.global_mean = interactions.global_mean

        n_users = interactions.n_users
        n_items = interactions.n_items

        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.1
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.1
        self.user_implicit = np.random.randn(n_users, self.n_factors) * 0.1
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        user_items = []
        for u in range(n_users):
            items = interactions.get_positive_items(u)
            user_items.append(items)

        coo = interactions.ratings.tocoo()
        users = coo.row
        items = coo.col
        ratings = coo.data

        n_samples = len(ratings)

        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)

            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                N_u = user_items[u]
                sqrt_nu = np.sqrt(len(N_u))

                if sqrt_nu > 0:
                    implicit_sum = self.user_implicit[N_u].sum(axis=0)
                    y_j = implicit_sum / sqrt_nu
                else:
                    y_j = np.zeros(self.n_factors)

                combined_user = self.user_factors[u] + y_j

                pred = self.global_mean + self.user_bias[u] + self.item_bias[i]
                pred += np.dot(combined_user, self.item_factors[i])

                error = r - pred

                self.user_bias[u] += self.learning_rate * (
                    error - self.regularization * self.user_bias[u]
                )
                self.item_bias[i] += self.learning_rate * (
                    error - self.regularization * self.item_bias[i]
                )

                user_f_old = self.user_factors[u].copy()
                item_f_old = self.item_factors[i].copy()

                self.user_factors[u] += self.learning_rate * (
                    error * item_f_old - self.regularization * user_f_old
                )
                self.item_factors[i] += self.learning_rate * (
                    error * combined_user - self.regularization * item_f_old
                )

                if sqrt_nu > 0:
                    for j in N_u:
                        self.user_implicit[j] += self.learning_rate * (
                            error * item_f_old / sqrt_nu
                            - self.regularization * self.user_implicit[j]
                        )

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        N_u = self.interactions.get_positive_items(user_idx)
        sqrt_nu = np.sqrt(len(N_u))

        if sqrt_nu > 0:
            implicit_sum = self.user_implicit[N_u].sum(axis=0)
            y_j = implicit_sum / sqrt_nu
        else:
            y_j = np.zeros(self.n_factors)

        combined_user = self.user_factors[user_idx] + y_j

        pred = self.global_mean
        pred += self.user_bias[user_idx]
        pred += self.item_bias[item_idx]
        pred += np.dot(combined_user, self.item_factors[item_idx])

        return pred

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted.")

        predictions = []

        user_rated = set()
        if exclude_known and self.interactions is not None:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        for item_idx in range(len(self.item_factors)):
            if item_idx in user_rated:
                continue

            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class BayesianMatrixFactorization:
    """Bayesian Matrix Factorization using Variational Inference.

    Implements probabilistic MF with Gaussian priors over latent factors,
    using variational inference for posterior approximation.

    Attributes:
        n_factors: Number of latent factors
        n_iterations: Number of variational iterations
        beta: Precision prior
        mu: Prior mean for factors
    """

    def __init__(
        self,
        n_factors: int = 30,
        n_iterations: int = 50,
        beta: float = 2.0,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.beta = beta

        self.U_mean: Optional[np.ndarray] = None
        self.U_var: Optional[np.ndarray] = None
        self.V_mean: Optional[np.ndarray] = None
        self.V_var: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        self.interactions: Optional[InteractionMatrix] = None

    def fit(
        self,
        interactions: InteractionMatrix,
        random_state: int = 42,
    ) -> BayesianMatrixFactorization:
        """Fit Bayesian MF using variational inference.

        Args:
            interactions: User-item interaction matrix
            random_state: Random seed

        Returns:
            Self
        """
        np.random.seed(random_state)

        self.interactions = interactions
        self.global_mean = interactions.global_mean

        n_users = interactions.n_users
        n_items = interactions.n_items

        self.U_mean = np.random.randn(n_users, self.n_factors) * 0.1
        self.U_var = np.ones((n_users, self.n_factors)) * 0.1
        self.V_mean = np.random.randn(n_items, self.n_factors) * 0.1
        self.V_var = np.ones((n_items, self.n_factors)) * 0.1

        R = interactions.ratings.toarray()
        mask = R > 0

        for iteration in range(self.n_iterations):
            for u in range(n_users):
                rated_items = np.where(mask[u])[0]

                if len(rated_items) == 0:
                    continue

                V_rated = self.V_mean[rated_items]
                V_var_rated = self.V_var[rated_items]

                precision = 1.0 / (V_var_rated.sum(axis=0) + 1.0 / self.beta)

                temp = V_rated * (R[u, rated_items][:, np.newaxis] - self.global_mean)
                temp_var = V_var_rated.sum(axis=0)

                self.U_mean[u] = precision * temp.sum(axis=0)
                self.U_var[u] = precision

            for i in range(n_items):
                rated_users = np.where(mask[:, i])[0]

                if len(rated_users) == 0:
                    continue

                U_rated = self.U_mean[rated_users]
                U_var_rated = self.U_var[rated_users]

                precision = 1.0 / (U_var_rated.sum(axis=0) + 1.0 / self.beta)

                temp = U_rated * (R[rated_users, i][:, np.newaxis] - self.global_mean)
                temp_var = U_var_rated.sum(axis=0)

                self.V_mean[i] = precision * temp.sum(axis=0)
                self.V_var[i] = precision

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating (mean of predictive distribution)."""
        if self.U_mean is None:
            raise RuntimeError("Model not fitted.")

        return self.global_mean + np.dot(self.U_mean[user_idx], self.V_mean[item_idx])

    def predict_uncertainty(self, user_idx: int, item_idx: int) -> Tuple[float, float]:
        """Predict with uncertainty (mean and variance)."""
        if self.U_mean is None:
            raise RuntimeError("Model not fitted.")

        mean = self.global_mean + np.dot(self.U_mean[user_idx], self.V_mean[item_idx])

        var = (
            (self.U_var[user_idx] * self.V_mean[item_idx] ** 2).sum()
            + (self.U_mean[user_idx] ** 2 * self.V_var[item_idx]).sum()
            + (self.U_var[user_idx] * self.V_var[item_idx]).sum()
            + 1.0 / self.beta
        )

        return mean, var

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if self.U_mean is None:
            raise RuntimeError("Model not fitted.")

        predictions = []

        user_rated = set()
        if exclude_known and self.interactions is not None:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        for item_idx in range(len(self.V_mean)):
            if item_idx in user_rated:
                continue

            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]
