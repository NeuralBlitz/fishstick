"""
Feature Selection Base Module for fishstick

Provides base classes and utilities for feature selection algorithms.
"""

from typing import Optional, List, Union, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch
from enum import Enum


class SelectionMethod(Enum):
    """Enumeration of feature selection methods."""

    FILTER = "filter"
    WRAPPER = "wrapper"
    EMBEDDED = "embedded"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


@dataclass
class SelectionResult:
    """
    Result of a feature selection operation.

    Attributes:
        selected_features: Indices of selected features
        scores: Scores for each feature (importance/ranking)
        n_features: Original number of features
        method: Method used for selection
        metadata: Additional information about selection
    """

    selected_features: np.ndarray
    scores: Optional[np.ndarray] = None
    n_features: int = 0
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.n_features == 0 and self.selected_features is not None:
            self.n_features = len(self.selected_features)

    def get_mask(self, n_total: int) -> np.ndarray:
        """Get boolean mask of selected features."""
        mask = np.zeros(n_total, dtype=bool)
        mask[self.selected_features] = True
        return mask

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to keep only selected features."""
        return X[:, self.selected_features]


class BaseSelector(ABC):
    """
    Abstract base class for feature selectors.

    All feature selectors should inherit from this class
    and implement the fit and transform methods.
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        threshold: Optional[float] = None,
        scoring: str = "auto",
    ):
        """
        Args:
            n_features_to_select: Number of features to select.
                If int, select that many features.
                If float (0-1), select that fraction of features.
            threshold: Threshold for feature scoring
            scoring: Scoring method to use
        """
        self.n_features_to_select = n_features_to_select
        self.threshold = threshold
        self.scoring = scoring
        self.selected_features_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> "BaseSelector":
        """
        Fit the feature selector.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels (optional for unsupervised methods)

        Returns:
            self
        """
        pass

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data to keep only selected features.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Transformed features with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet.")

        if isinstance(X, torch.Tensor):
            return X[:, self.selected_features_]
        return X[:, self.selected_features_]

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Fit and transform in one step.

        Args:
            X: Input features
            y: Target labels

        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[bool]]:
        """
        Get support of selected features.

        Args:
            indices: If True, return indices; otherwise return boolean mask

        Returns:
            Feature support
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet.")

        if indices:
            return self.selected_features_

        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_features_] = True
        return mask

    def _parse_n_features(
        self,
        n_total: int,
    ) -> int:
        """Parse n_features_to_select to actual number."""
        if self.n_features_to_select is None:
            return n_total

        if isinstance(self.n_features_to_select, float):
            return max(1, int(n_total * self.n_features_to_select))

        return min(self.n_features_to_select, n_total)

    def _to_numpy(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, bool]:
        """Convert input to numpy array."""
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            X = X.cpu().numpy()
        return X, is_torch

    def _to_torch(
        self,
        X: np.ndarray,
        is_torch: bool,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert back to torch if input was torch."""
        if is_torch:
            return torch.from_numpy(X)
        return X


class SupervisedSelector(BaseSelector):
    """
    Base class for supervised feature selectors.

    These selectors require target labels for fitting.
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        threshold: Optional[float] = None,
        scoring: str = "auto",
        cv: int = 5,
    ):
        """
        Args:
            n_features_to_select: Number of features to select
            threshold: Threshold for feature scoring
            scoring: Scoring method
            cv: Number of cross-validation folds
        """
        super().__init__(n_features_to_select, threshold, scoring)
        self.cv = cv


class UnsupervisedSelector(BaseSelector):
    """
    Base class for unsupervised feature selectors.

    These selectors do not require target labels.
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        threshold: Optional[float] = None,
    ):
        super().__init__(n_features_to_select, threshold)


def create_selector(
    method: str,
    **kwargs,
) -> BaseSelector:
    """
    Factory function to create feature selectors.

    Args:
        method: Selector method name
        **kwargs: Arguments for the selector

    Returns:
        Feature selector instance
    """
    from .variance_threshold import VarianceThresholdSelector
    from .correlation_filter import CorrelationFilter
    from .mutual_info import MutualInfoSelector
    from .chi_square import ChiSquareSelector
    from .fisher_score import FisherScoreSelector
    from .recursive_feature_elimination import RecursiveFeatureElimination
    from .sequential_feature_selection import SequentialFeatureSelector
    from .anova_ftest import ANOVAFTestSelector, FRegressionSelector
    from .relieff import ReliefFSelector
    from .laplacian_score import LaplacianScoreSelector
    from .genetic_algorithm import GeneticAlgorithmSelector
    from .greedy_stepwise import GreedyStepwiseSelector
    from .lasso_selection import LassoSelector, LassoCVSelector, LassoStabilitySelector
    from .elastic_net_tree import (
        ElasticNetSelector,
        RandomForestImportanceSelector,
        GBDTImportanceSelector,
        PermutationImportanceSelector,
    )
    from .dimensionality_reduction import (
        PCASelector,
        LDASelector,
        ICASelector,
        KernelPCASelector,
    )
    from .shap_importance import SHAPImportanceSelector

    selectors = {
        "variance": VarianceThresholdSelector,
        "correlation": CorrelationFilter,
        "mutual_info": MutualInfoSelector,
        "chi2": ChiSquareSelector,
        "fisher": FisherScoreSelector,
        "rfe": RecursiveFeatureElimination,
        "sequential": SequentialFeatureSelector,
        "lasso": LassoSelector,
        "lasso_cv": LassoCVSelector,
        "lasso_stability": LassoStabilitySelector,
        "elastic_net": ElasticNetSelector,
        "anova_f": ANOVAFTestSelector,
        "f_regression": FRegressionSelector,
        "relieff": ReliefFSelector,
        "laplacian": LaplacianScoreSelector,
        "genetic": GeneticAlgorithmSelector,
        "greedy_forward": GreedyStepwiseSelector,
        "greedy_backward": GreedyStepwiseSelector,
        "rf_importance": RandomForestImportanceSelector,
        "gbdt_importance": GBDTImportanceSelector,
        "permutation": PermutationImportanceSelector,
        "pca": PCASelector,
        "lda": LDASelector,
        "ica": ICASelector,
        "kernel_pca": KernelPCASelector,
        "shap": SHAPImportanceSelector,
    }

    if method == "greedy_forward":
        kwargs["direction"] = "forward"
    elif method == "greedy_backward":
        kwargs["direction"] = "backward"

    if method not in selectors:
        raise ValueError(
            f"Unknown selector: {method}. Available: {list(selectors.keys())}"
        )

    return selectors[method](**kwargs)
