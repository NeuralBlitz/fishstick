"""
Collaborative Filtering Methods.

Provides implementation of memory-based and model-based collaborative filtering
including user-based and item-based approaches with various similarity metrics.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Callable, Dict
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import norm
import warnings

from .base import InteractionMatrix, RecommenderBase


class SimilarityMetric:
    """Base class for similarity computation between users/items."""

    def compute(self, matrix: csr_matrix) -> csr_matrix:
        """Compute pairwise similarity matrix.

        Args:
            matrix: User-item or item-user interaction matrix

        Returns:
            Sparse similarity matrix
        """
        raise NotImplementedError

    def pairwise(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two vectors."""
        raise NotImplementedError


class CosineSimilarity(SimilarityMetric):
    """Cosine similarity for collaborative filtering.

    Computes similarity based on the cosine of the angle between
    rating vectors.

    Attributes:
        min_common: Minimum number of common items/users for similarity computation
        shrinkage: Shrinkage factor for regularization
    """

    def __init__(self, min_common: int = 3, shrinkage: int = 100):
        self.min_common = min_common
        self.shrinkage = shrinkage

    def compute(self, matrix: csr_matrix) -> csr_matrix:
        """Compute cosine similarity matrix.

        Args:
            matrix: Interaction matrix (users x items)

        Returns:
            User-user or item-item similarity matrix
        """
        matrix = matrix.astype(np.float32)

        norms = sparse.linalg.norm(matrix, axis=1)
        norms = np.where(norms == 0, 1, norms)

        normalized = matrix.multiply(1.0 / norms[:, np.newaxis])

        sim = normalized @ normalized.T

        data = sim.data
        rows, cols = sim.nonzero()

        coo = matrix.tocoo()
        common_counts = sparse.csr_matrix(
            (np.ones(len(coo.data)), (coo.row, coo.col)), shape=matrix.shape
        )
        common_matrix = (common_counts.T @ common_counts).toarray()

        shrink_factor = common_matrix[rows, cols] / (
            common_matrix[rows, cols] + self.shrinkage
        )

        sim_data = data * shrink_factor

        sim_matrix = csr_matrix((sim_data, (rows, cols)), shape=sim.shape)

        sim_matrix.setdiag(1.0)

        return sim_matrix

    def pairwise(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        mask = (vec1 != 0) & (vec2 != 0)
        if mask.sum() < self.min_common:
            return 0.0

        v1 = vec1[mask]
        v2 = vec2[mask]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(v1, v2) / (norm1 * norm2)


class PearsonCorrelation(SimilarityMetric):
    """Pearson correlation coefficient for collaborative filtering.

    Computes similarity based on correlation between rating patterns,
    accounting for user/item bias.

    Attributes:
        min_common: Minimum number of common items
        shrinkage: Shrinkage factor for regularization
    """

    def __init__(self, min_common: int = 3, shrinkage: int = 100):
        self.min_common = min_common
        self.shrinkage = shrinkage

    def compute(self, matrix: csr_matrix) -> csr_matrix:
        """Compute Pearson correlation matrix.

        Args:
            matrix: Interaction matrix

        Returns:
            Correlation matrix
        """
        matrix = matrix.copy().astype(np.float32)

        row_means = np.array(matrix.sum(axis=1)).flatten() / (
            np.array(matrix != 0).sum(axis=1) + 1e-8
        )

        masked_matrix = matrix.copy()
        masked_matrix.data -= np.repeat(
            row_means[masked_matrix.row], np.diff(masked_matrix.indptr)
        )

        norms = sparse.linalg.norm(masked_matrix, axis=1)
        norms = np.where(norms == 0, 1, norms)

        normalized = masked_matrix.multiply(1.0 / norms[:, np.newaxis])

        sim = normalized @ normalized.T

        data = sim.data
        rows, cols = sim.nonzero()

        coo = matrix.tocoo()
        common_counts = sparse.csr_matrix(
            (np.ones(len(coo.data)), (coo.row, coo.col)), shape=matrix.shape
        )
        common_matrix = (common_counts.T @ common_counts).toarray()

        shrink_factor = common_matrix[rows, cols] / (
            common_matrix[rows, cols] + self.shrinkage
        )

        sim_data = data * shrink_factor

        sim_matrix = csr_matrix((sim_data, (rows, cols)), shape=sim.shape)

        sim_matrix.setdiag(1.0)

        return sim_matrix

    def pairwise(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Pearson correlation between two vectors."""
        mask = (vec1 != 0) & (vec2 != 0)
        if mask.sum() < self.min_common:
            return 0.0

        v1 = vec1[mask]
        v2 = vec2[mask]

        mean1 = v1.mean()
        mean2 = v2.mean()

        v1_centered = v1 - mean1
        v2_centered = v2 - mean2

        numerator = np.dot(v1_centered, v2_centered)
        denominator = np.sqrt(np.sum(v1_centered**2) * np.sum(v2_centered**2))

        if denominator == 0:
            return 0.0

        return numerator / denominator


class JaccardSimilarity(SimilarityMetric):
    """Jaccard similarity for implicit feedback.

    Computes similarity based on set intersection over union,
    suitable for binary/implicit feedback.
    """

    def compute(self, matrix: csr_matrix) -> csr_matrix:
        """Compute Jaccard similarity matrix.

        Args:
            matrix: Binary interaction matrix

        Returns:
            Jaccard similarity matrix
        """
        matrix = matrix.astype(np.float32)
        matrix.data = np.ones_like(matrix.data)

        intersection = matrix @ matrix.T

        row_sums = np.array(matrix.sum(axis=1)).flatten()
        union = (
            row_sums[:, np.newaxis] + row_sums[np.newaxis, :] - intersection.toarray()
        )

        union = np.maximum(union, 1)

        sim = intersection.multiply(1.0 / union)

        sim.setdiag(1.0)

        return sim

    def pairwise(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Jaccard similarity between binary vectors."""
        intersection = np.sum((vec1 > 0) & (vec2 > 0))
        union = np.sum((vec1 > 0) | (vec2 > 0))

        if union == 0:
            return 0.0

        return intersection / union


class UserBasedCF:
    """User-Based Collaborative Filtering.

    Recommends items based on similarity between users.

    Attributes:
        n_neighbors: Number of neighbors to use for prediction
        similarity: Similarity metric to use
        min_similarity: Minimum similarity threshold
        aggregation: Aggregation method ('weighted', 'mean', 'sum')
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        similarity: Optional[SimilarityMetric] = None,
        min_similarity: float = 0.0,
        aggregation: str = "weighted",
    ):
        self.n_neighbors = n_neighbors
        self.similarity = similarity or CosineSimilarity()
        self.min_similarity = min_similarity
        self.aggregation = aggregation

        self.user_similarity: Optional[csr_matrix] = None
        self.interactions: Optional[InteractionMatrix] = None
        self.user_means: Optional[np.ndarray] = None

    def fit(self, interactions: InteractionMatrix) -> UserBasedCF:
        """Fit the user-based CF model.

        Args:
            interactions: User-item interaction matrix

        Returns:
            Self
        """
        self.interactions = interactions

        self.user_similarity = self.similarity.compute(interactions.ratings)

        self.user_means = np.zeros(interactions.n_users)
        for u in range(interactions.n_users):
            row = interactions.ratings.getrow(u)
            if row.nnz > 0:
                self.user_means[u] = row.data.mean()

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair.

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Predicted rating
        """
        if self.interactions is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        item_ratings = self.interactions.ratings.getcol(item_idx).toarray().flatten()

        users_rated = np.where(item_ratings > 0)[0]

        if len(users_rated) == 0:
            return self.user_means[user_idx] if self.user_means[user_idx] > 0 else 0.0

        sims = np.array(
            [self.user_similarity[user_idx, u] for u in users_rated]
        ).flatten()

        valid = (sims >= self.min_similarity) & (users_rated != user_idx)

        if not valid.any():
            return self.user_means[user_idx] if self.user_means[user_idx] > 0 else 0.0

        valid_users = users_rated[valid]
        valid_sims = sims[valid]
        valid_ratings = item_ratings[valid_users]

        if self.aggregation == "weighted":
            sum_sims = np.sum(np.abs(valid_sims))
            if sum_sims > 0:
                return np.sum(valid_sims * valid_ratings) / sum_sims
            return self.user_means[user_idx]
        elif self.aggregation == "mean":
            return valid_ratings.mean()
        elif self.aggregation == "sum":
            return valid_ratings.sum()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations for a user.

        Args:
            user_idx: User index
            n_items: Number of items to recommend
            exclude_known: Whether to exclude items user has interacted with

        Returns:
            List of (item_idx, score) tuples
        """
        if self.interactions is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions = []

        user_rated = set()
        if exclude_known:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        for item_idx in range(self.interactions.n_items):
            if item_idx in user_rated:
                continue

            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class ItemBasedCF:
    """Item-Based Collaborative Filtering.

    Recommends items based on similarity between items.

    Attributes:
        n_neighbors: Number of neighbors to use for prediction
        similarity: Similarity metric to use
        min_similarity: Minimum similarity threshold
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        similarity: Optional[SimilarityMetric] = None,
        min_similarity: float = 0.0,
    ):
        self.n_neighbors = n_neighbors
        self.similarity = similarity or CosineSimilarity()
        self.min_similarity = min_similarity

        self.item_similarity: Optional[csr_matrix] = None
        self.interactions: Optional[InteractionMatrix] = None
        self.item_means: Optional[np.ndarray] = None

    def fit(self, interactions: InteractionMatrix) -> ItemBasedCF:
        """Fit the item-based CF model.

        Args:
            interactions: User-item interaction matrix

        Returns:
            Self
        """
        self.interactions = interactions

        item_user_matrix = interactions.ratings.T.tocsr()
        self.item_similarity = self.similarity.compute(item_user_matrix)

        self.item_means = np.zeros(interactions.n_items)
        for i in range(interactions.n_items):
            col = interactions.ratings.getcol(i)
            if col.nnz > 0:
                self.item_means[i] = col.data.mean()

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair.

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Predicted rating
        """
        if self.interactions is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_ratings = self.interactions.get_user_ratings(user_idx)

        if len(user_ratings) == 0:
            return self.item_means[item_idx]

        user_items = user_ratings[:, 0].astype(int)
        user_ratings_arr = user_ratings[:, 1]

        sims = np.array([self.item_similarity[item_idx, i] for i in user_items])

        valid = sims >= self.min_similarity

        if not valid.any():
            return self.item_means[item_idx]

        valid_items = user_items[valid]
        valid_sims = sims[valid]
        valid_ratings = user_ratings_arr[valid]

        sum_sims = np.sum(np.abs(valid_sims))
        if sum_sims > 0:
            return np.sum(valid_sims * valid_ratings) / sum_sims
        return self.item_means[item_idx]

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations for a user.

        Args:
            user_idx: User index
            n_items: Number of items to recommend
            exclude_known: Whether to exclude items user has interacted with

        Returns:
            List of (item_idx, score) tuples
        """
        if self.interactions is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions = []

        user_rated = set()
        if exclude_known:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        for item_idx in range(self.interactions.n_items):
            if item_idx in user_rated:
                continue

            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class kNNCollaborativeFiltering:
    """K-Nearest Neighbors Collaborative Filtering wrapper.

    Provides unified interface for both user-based and item-based CF.

    Attributes:
        k: Number of neighbors
        similarity: Similarity metric
        mode: 'user' for user-based, 'item' for item-based
        min_k: Minimum neighbors required for prediction
    """

    def __init__(
        self,
        k: int = 40,
        similarity: Optional[SimilarityMetric] = None,
        mode: str = "item",
        min_k: int = 1,
    ):
        if mode not in ("user", "item"):
            raise ValueError("mode must be 'user' or 'item'")

        self.k = k
        self.similarity = similarity or CosineSimilarity()
        self.mode = mode
        self.min_k = min_k

        if mode == "user":
            self.model = UserBasedCF(
                n_neighbors=k,
                similarity=similarity,
            )
        else:
            self.model = ItemBasedCF(
                n_neighbors=k,
                similarity=similarity,
            )

    def fit(self, interactions: InteractionMatrix) -> kNNCollaborativeFiltering:
        """Fit the kNN CF model.

        Args:
            interactions: Interaction matrix

        Returns:
            Self
        """
        self.model.fit(interactions)
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair."""
        return self.model.predict(user_idx, item_idx)

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        return self.model.recommend(user_idx, n_items, exclude_known)


class SlopeOneRecommender:
    """Slope One Collaborative Filtering.

    A simple yet effective CF algorithm that uses weighted
    average of deviations between item pairs.

    Attributes:
        min_support: Minimum number of common users for deviation computation
    """

    def __init__(self, min_support: int = 2):
        self.min_support = min_support
        self.deviations: Optional[Dict[Tuple[int, int], float]] = None
        self.counts: Optional[Dict[Tuple[int, int], int]] = None
        self.interactions: Optional[InteractionMatrix] = None

    def fit(self, interactions: InteractionMatrix) -> SlopeOneRecommender:
        """Fit the Slope One model.

        Args:
            interactions: User-item interaction matrix

        Returns:
            Self
        """
        self.interactions = interactions

        self.deviations = {}
        self.counts = {}

        for i in range(interactions.n_items):
            for j in range(i + 1, interactions.n_items):
                common_users = self._get_common_users(i, j)

                if len(common_users) >= self.min_support:
                    deviations = []
                    for u in common_users:
                        r_ui = interactions.ratings[u, i]
                        r_uj = interactions.ratings[u, j]
                        deviations.append(r_ui - r_uj)

                    self.deviations[(i, j)] = np.mean(deviations)
                    self.deviations[(j, i)] = -np.mean(deviations)
                    self.counts[(i, j)] = len(common_users)
                    self.counts[(j, i)] = len(common_users)

        return self

    def _get_common_users(self, item1: int, item2: int) -> List[int]:
        """Get users who rated both items."""
        users1 = set(self.interactions.ratings.getcol(item1).indices)
        users2 = set(self.interactions.ratings.getcol(item2).indices)
        return list(users1 & users2)

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using weighted Slope One."""
        if self.interactions is None:
            raise RuntimeError("Model not fitted.")

        user_items = self.interactions.get_positive_items(user_idx)

        if len(user_items) == 0:
            return 0.0

        numerator = 0.0
        denominator = 0.0

        for rated_item in user_items:
            key = (rated_item, item_idx)
            if key in self.deviations:
                dev = self.deviations[key]
                count = self.counts[key]

                rating = self.interactions.ratings[user_idx, rated_item]
                numerator += (rating + dev) * count
                denominator += count

        if denominator == 0:
            return self.interactions.global_mean

        return numerator / denominator

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if self.interactions is None:
            raise RuntimeError("Model not fitted.")

        predictions = []

        user_rated = set()
        if exclude_known:
            user_rated = set(self.interactions.get_positive_items(user_idx))

        for item_idx in range(self.interactions.n_items):
            if item_idx in user_rated:
                continue

            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]
