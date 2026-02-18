"""
Base Data Structures and Utilities for Recommender Systems.

Provides core data structures for handling user-item interactions,
datasets, and common utilities used across all recommender models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable, Protocol
from pathlib import Path
from enum import Enum
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import warnings


class SplitStrategy(Enum):
    """Strategy for splitting data into train/test sets."""

    RANDOM = "random"
    TEMPORAL = "temporal"
    USER_BASED = "user_based"
    ITEM_BASED = "item_based"
    LEAVE_ONE_OUT = "leave_one_out"


@dataclass
class InteractionMatrix:
    """User-Item Interaction Matrix.

    Stores user-item interactions in a sparse matrix format.
    Supports both explicit ratings and implicit feedback.

    Attributes:
        ratings: Sparse matrix of shape (n_users, n_items)
        n_users: Number of unique users
        n_items: Number of unique items
        user_ids: Array mapping internal user indices to original IDs
        item_ids: Array mapping internal item indices to original IDs
        is_implicit: Whether this represents implicit feedback
    """

    ratings: csr_matrix
    n_users: int
    n_items: int
    user_ids: Optional[np.ndarray] = None
    item_ids: Optional[np.ndarray] = None
    is_implicit: bool = False

    @classmethod
    def from_dict(
        cls,
        interactions: Dict[Tuple[int, int], float],
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        user_ids: Optional[np.ndarray] = None,
        item_ids: Optional[np.ndarray] = None,
    ) -> InteractionMatrix:
        """Create InteractionMatrix from dictionary of interactions.

        Args:
            interactions: Dictionary mapping (user_id, item_id) to rating
            n_users: Number of users (if None, inferred from max user_id + 1)
            n_items: Number of items (if None, inferred from max item_id + 1)
            user_ids: Optional mapping from indices to original user IDs
            item_ids: Optional mapping from indices to original item IDs

        Returns:
            InteractionMatrix instance
        """
        if not interactions:
            raise ValueError("Interactions dictionary cannot be empty")

        users, items, ratings = zip(*interactions.items())
        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings)

        if n_users is None:
            n_users = int(users.max()) + 1
        if n_items is None:
            n_items = int(items.max()) + 1

        mat = csr_matrix(
            (ratings, (users, items)), shape=(n_users, n_items), dtype=np.float32
        )

        return cls(
            ratings=mat,
            n_users=n_users,
            n_items=n_items,
            user_ids=user_ids,
            item_ids=item_ids,
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
    ) -> InteractionMatrix:
        """Create InteractionMatrix from pandas DataFrame.

        Args:
            df: DataFrame with user-item interactions
            user_col: Name of user column
            item_col: Name of item column
            rating_col: Name of rating column
            n_users: Optional number of users
            n_items: Optional number of items

        Returns:
            InteractionMatrix instance
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for from_dataframe")

        users = df[user_col].values
        items = df[item_col].values
        ratings = df[rating_col].values.astype(np.float32)

        if n_users is None:
            n_users = int(users.max()) + 1
        if n_items is None:
            n_items = int(items.max()) + 1

        mat = csr_matrix(
            (ratings, (users, items)), shape=(n_users, n_items), dtype=np.float32
        )

        return cls(
            ratings=mat,
            n_users=n_users,
            n_items=n_items,
        )

    def to_coo(self) -> coo_matrix:
        """Convert to COO format."""
        return coo_matrix(self.ratings)

    def to_csc(self) -> csc_matrix:
        """Convert to CSC format."""
        return csc_matrix(self.ratings)

    def get_user_ratings(self, user_idx: int) -> np.ndarray:
        """Get all ratings for a specific user.

        Args:
            user_idx: User index

        Returns:
            Array of (item_idx, rating) pairs
        """
        row = self.ratings.getrow(user_idx)
        return np.column_stack((row.indices, row.data))

    def get_item_ratings(self, item_idx: int) -> np.ndarray:
        """Get all ratings for a specific item.

        Args:
            item_idx: Item index

        Returns:
            Array of (user_idx, rating) pairs
        """
        col = self.ratings.getcol(item_idx)
        return np.column_stack((col.indices, col.data))

    def get_positive_items(self, user_idx: int) -> np.ndarray:
        """Get items with positive interactions for a user.

        Args:
            user_idx: User index

        Returns:
            Array of item indices with positive interactions
        """
        row = self.ratings.getrow(user_idx)
        if self.is_implicit:
            return row.indices
        else:
            return row.indices[row.data > 0]

    def user_has_ratings(self, user_idx: int) -> bool:
        """Check if user has any ratings."""
        return self.ratings.getrow(user_idx).nnz > 0

    def item_has_ratings(self, item_idx: int) -> bool:
        """Check if item has any ratings."""
        return self.ratings.getcol(item_idx).nnz > 0

    @property
    def density(self) -> float:
        """Calculate matrix density (fraction of non-zero entries)."""
        return self.ratings.nnz / (self.n_users * self.n_items)

    @property
    def global_mean(self) -> float:
        """Calculate global mean rating."""
        return self.ratings.data.mean() if self.ratings.nnz > 0 else 0.0

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Get rating for user-item pair."""
        return self.ratings[key]


@dataclass
class SparseInteractionMatrix:
    """Memory-efficient sparse interaction matrix using coordinate format.

    Stores only non-zero interactions, useful for very large datasets
    with sparse feedback.

    Attributes:
        users: Array of user indices
        items: Array of item indices
        ratings: Array of ratings
        n_users: Total number of users
        n_items: Total number of items
        is_implicit: Whether this represents implicit feedback
    """

    users: np.ndarray
    items: np.ndarray
    ratings: np.ndarray
    n_users: int
    n_items: int
    is_implicit: bool = False

    @classmethod
    def from_dense(
        cls, dense: np.ndarray, threshold: float = 0.0
    ) -> SparseInteractionMatrix:
        """Create sparse matrix from dense array.

        Args:
            dense: Dense numpy array
            threshold: Values above this threshold are kept

        Returns:
            SparseInteractionMatrix instance
        """
        mask = dense > threshold
        users, items = np.where(mask)
        ratings = dense[mask]

        return cls(
            users=users,
            items=items,
            ratings=ratings,
            n_users=dense.shape[0],
            n_items=dense.shape[1],
        )

    def to_csr(self) -> csr_matrix:
        """Convert to CSR format."""
        return csr_matrix(
            (self.ratings, (self.users, self.items)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

    def __len__(self) -> int:
        """Number of interactions."""
        return len(self.ratings)


@dataclass
class TrainTestSplit:
    """Train/Test split for recommender evaluation.

    Attributes:
        train: Training interaction matrix
        test: Test interaction matrix
        test_users: Array of user indices in test set
        test_items: Array of item indices in test set
        test_ratings: Array of test ratings
        strategy: The splitting strategy used
    """

    train: InteractionMatrix
    test: InteractionMatrix
    test_users: np.ndarray
    test_items: np.ndarray
    test_ratings: np.ndarray
    strategy: SplitStrategy = SplitStrategy.RANDOM

    @classmethod
    def random_split(
        cls,
        interactions: InteractionMatrix,
        test_ratio: float = 0.2,
        random_state: int = 42,
        min_ratings: int = 1,
    ) -> TrainTestSplit:
        """Split interactions randomly into train/test.

        Args:
            interactions: Interaction matrix to split
            test_ratio: Fraction of interactions to use for testing
            random_state: Random seed for reproducibility
            min_ratings: Minimum number of ratings per user in test

        Returns:
            TrainTestSplit instance
        """
        np.random.seed(random_state)

        coo = coo_matrix(interactions.ratings)
        users = coo.row
        items = coo.col
        ratings = coo.data.copy()

        n_total = len(ratings)
        n_test = int(n_total * test_ratio)

        test_mask = np.zeros(n_total, dtype=bool)
        test_indices = np.random.choice(n_total, size=n_test, replace=False)
        test_mask[test_indices] = True

        train_ratings = ratings[~test_mask]
        train_users = users[~test_mask]
        train_items = items[~test_mask]

        test_ratings_arr = ratings[test_mask]
        test_users_arr = users[test_mask]
        test_items_arr = items[test_mask]

        train_mat = csr_matrix(
            (train_ratings, (train_users, train_items)),
            shape=(interactions.n_users, interactions.n_items),
            dtype=np.float32,
        )

        test_mat = csr_matrix(
            (test_ratings_arr, (test_users_arr, test_items_arr)),
            shape=(interactions.n_users, interactions.n_items),
            dtype=np.float32,
        )

        return cls(
            train=InteractionMatrix(
                ratings=train_mat,
                n_users=interactions.n_users,
                n_items=interactions.n_items,
                is_implicit=interactions.is_implicit,
            ),
            test=InteractionMatrix(
                ratings=test_mat,
                n_users=interactions.n_users,
                n_items=interactions.n_items,
                is_implicit=interactions.is_implicit,
            ),
            test_users=test_users_arr,
            test_items=test_items_arr,
            test_ratings=test_ratings_arr,
            strategy=SplitStrategy.RANDOM,
        )

    @classmethod
    def leave_one_out_split(
        cls,
        interactions: InteractionMatrix,
        max_test_per_user: int = 1,
        random_state: int = 42,
    ) -> TrainTestSplit:
        """Leave-one-out split (common for ranking evaluation).

        Keeps last interaction per user for testing.

        Args:
            interactions: Interaction matrix
            max_test_per_user: Maximum test items per user
            random_state: Random seed

        Returns:
            TrainTestSplit instance
        """
        np.random.seed(random_state)

        coo = coo_matrix(interactions.ratings)

        train_users = []
        train_items = []
        train_ratings = []

        test_users = []
        test_items = []
        test_ratings = []

        for user_idx in range(interactions.n_users):
            user_items = interactions.get_positive_items(user_idx)

            if len(user_items) == 0:
                continue

            n_test = min(max_test_per_user, len(user_items) - 1)

            if n_test > 0:
                test_idx = np.random.choice(len(user_items), size=n_test, replace=False)
                test_item_idx = user_items[test_idx]

                test_users.extend([user_idx] * len(test_item_idx))
                test_items.extend(test_item_idx)
                test_ratings.extend([1.0] * len(test_item_idx))

                train_item_idx = np.delete(user_items, test_idx)
                train_users.extend([user_idx] * len(train_item_idx))
                train_items.extend(train_item_idx)
                train_ratings.extend([1.0] * len(train_item_idx))

        train_mat = csr_matrix(
            (
                np.array(train_ratings, dtype=np.float32),
                (np.array(train_users), np.array(train_items)),
            ),
            shape=(interactions.n_users, interactions.n_items),
            dtype=np.float32,
        )

        test_mat = csr_matrix(
            (
                np.array(test_ratings, dtype=np.float32),
                (np.array(test_users), np.array(test_items)),
            ),
            shape=(interactions.n_users, interactions.n_items),
            dtype=np.float32,
        )

        return cls(
            train=InteractionMatrix(
                ratings=train_mat,
                n_users=interactions.n_users,
                n_items=interactions.n_items,
                is_implicit=True,
            ),
            test=InteractionMatrix(
                ratings=test_mat,
                n_users=interactions.n_users,
                n_items=interactions.n_items,
                is_implicit=True,
            ),
            test_users=np.array(test_users),
            test_items=np.array(test_items),
            test_ratings=np.array(test_ratings),
            strategy=SplitStrategy.LEAVE_ONE_OUT,
        )


class UserItemDataset(Dataset):
    """PyTorch Dataset for user-item interactions.

    Args:
        interactions: Interaction matrix
        negative_samples: Number of negative samples per positive
        shuffle: Whether to shuffle the dataset
    """

    def __init__(
        self,
        interactions: InteractionMatrix,
        negative_samples: int = 0,
        shuffle: bool = True,
    ):
        self.interactions = interactions
        self.negative_samples = negative_samples

        coo = coo_matrix(interactions.ratings)
        self.users = coo.row
        self.items = coo.col
        self.ratings = coo.data

        if shuffle:
            perm = np.random.permutation(len(self.users))
            self.users = self.users[perm]
            self.items = self.items[perm]
            self.ratings = self.ratings[perm]

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]

        sample = {
            "user": torch.tensor(user, dtype=torch.long),
            "item": torch.tensor(item, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float32),
        }

        if self.negative_samples > 0:
            neg_items = []
            for _ in range(self.negative_samples):
                neg_item = np.random.randint(0, self.interactions.n_items)
                while self.interactions.ratings[user, neg_item] > 0:
                    neg_item = np.random.randint(0, self.interactions.n_items)
                neg_items.append(neg_item)
            sample["negative_items"] = torch.tensor(neg_items, dtype=torch.long)

        return sample


class RecommenderBase(Protocol):
    """Abstract base class for recommender models.

    All recommender models should implement this interface.
    """

    def fit(self, interactions: InteractionMatrix) -> RecommenderBase:
        """Fit the model to interaction data.

        Args:
            interactions: Training interaction matrix

        Returns:
            Self
        """
        ...

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair.

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Predicted rating
        """
        ...

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
        ...

    def save(self, path: Path) -> None:
        """Save model to disk."""
        ...

    @classmethod
    def load(cls, path: Path) -> RecommenderBase:
        """Load model from disk."""
        ...


def create_sparse_matrix(
    users: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
    format: str = "csr",
) -> csr_matrix | coo_matrix | csc_matrix:
    """Create sparse matrix from arrays.

    Args:
        users: User indices
        items: Item indices
        ratings: Ratings
        n_users: Number of users
        n_items: Number of items
        format: Output format ('csr', 'coo', or 'csc')

    Returns:
        Sparse matrix in requested format
    """
    if format == "csr":
        return csr_matrix(
            (ratings, (users, items)), shape=(n_users, n_items), dtype=np.float32
        )
    elif format == "coo":
        return coo_matrix(
            (ratings, (users, items)), shape=(n_users, n_items), dtype=np.float32
        )
    elif format == "csc":
        return csc_matrix(
            (ratings, (users, items)), shape=(n_users, n_items), dtype=np.float32
        )
    else:
        raise ValueError(f"Unknown format: {format}")


def train_val_split(
    interactions: InteractionMatrix,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[InteractionMatrix, InteractionMatrix]:
    """Split interactions into train and validation sets.

    Args:
        interactions: Interaction matrix
        val_ratio: Fraction for validation
        random_state: Random seed

    Returns:
        Tuple of (train, validation) InteractionMatrix
    """
    split = TrainTestSplit.random_split(
        interactions,
        test_ratio=val_ratio,
        random_state=random_state,
    )
    return split.train, split.test
