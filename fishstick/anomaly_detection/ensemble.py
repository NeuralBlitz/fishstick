"""
Ensemble Anomaly Detection Module.

This module provides ensemble methods for combining multiple anomaly detectors:
- Voting ensemble (majority, weighted)
- Stacking ensemble with meta-learner
- Score aggregation methods
- Dynamic ensemble selection
- Model uncertainty estimation

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


@dataclass
class EnsembleResult:
    """Container for ensemble anomaly detection results."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    n_anomalies: int
    anomaly_indices: np.ndarray
    individual_scores: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None


class BaseEnsembleDetector(ABC):
    """Base class for ensemble anomaly detectors."""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.detectors: List[Any] = []
        self.threshold: Optional[float] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseEnsembleDetector":
        """Fit the ensemble."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble anomaly scores."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            self.threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return (scores > self.threshold).astype(int)

    def fit_predict(self, X: np.ndarray) -> EnsembleResult:
        """Fit and predict in one call."""
        self.fit(X)
        scores = self.score(X)
        labels = self.predict(X)
        return EnsembleResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            n_anomalies=int(np.sum(labels)),
            anomaly_indices=np.where(labels == 1)[0],
        )


class VotingAnomalyEnsemble(BaseEnsembleDetector):
    """
    Voting ensemble for anomaly detection.

    Combines multiple detectors using various voting strategies.

    Parameters
    ----------
    detectors : List
        List of anomaly detector objects.
    voting : str
        Voting strategy: 'hard', 'soft', 'weighted'.
    weights : List[float], optional
        Detector weights for weighted voting.
    """

    def __init__(
        self,
        detectors: List[Any] = None,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)
        self.detectors = detectors or []
        self.voting = voting
        self.weights = weights

    def add_detector(self, detector: Any) -> "VotingAnomalyEnsemble":
        """Add a detector to the ensemble."""
        self.detectors.append(detector)
        return self

    def fit(self, X: np.ndarray) -> "VotingAnomalyEnsemble":
        """Fit all detectors in the ensemble."""
        if not self.detectors:
            raise ValueError("No detectors in ensemble.")

        for detector in self.detectors:
            if hasattr(detector, "fit"):
                detector.fit(X)

        if self.weights is None:
            self.weights = [1.0 / len(self.detectors)] * len(self.detectors)

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted.")

        scores_list = []
        for detector in self.detectors:
            if hasattr(detector, "score"):
                scores_list.append(detector.score(X))
            elif hasattr(detector, "decision_function"):
                scores_list.append(detector.decision_function(X))

        scores_array = np.column_stack(scores_list)

        if self.voting == "hard":
            votes = (scores_array > 0).astype(int)
            return np.mean(votes, axis=1)
        elif self.voting == "soft":
            return np.average(scores_array, axis=1, weights=self.weights)
        elif self.voting == "weighted":
            normalized_weights = np.array(self.weights) / sum(self.weights)
            return np.average(scores_array, axis=1, weights=normalized_weights)
        else:
            raise ValueError(f"Unknown voting type: {self.voting}")


class StackingAnomalyEnsemble(BaseEnsembleDetector):
    """
    Stacking ensemble with meta-learner.

    Uses base detector outputs as features for a meta-learner.

    Parameters
    ----------
    detectors : List
        List of base anomaly detectors.
    meta_learner : optional
        Meta-learner (default: LogisticRegression).
    use_proba : bool
        Use probability scores instead of raw scores.
    """

    def __init__(
        self,
        detectors: List[Any] = None,
        meta_learner: Any = None,
        use_proba: bool = False,
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)
        self.detectors = detectors or []
        self.meta_learner = meta_learner or LogisticRegression()
        self.use_proba = use_proba
        self.scaler = StandardScaler()

    def add_detector(self, detector: Any) -> "StackingAnomalyEnsemble":
        """Add a detector to the ensemble."""
        self.detectors.append(detector)
        return self

    def fit(self, X: np.ndarray) -> "StackingAnomalyEnsemble":
        """Fit base detectors and meta-learner."""
        if not self.detectors:
            raise ValueError("No detectors in ensemble.")

        base_scores = []
        for detector in self.detectors:
            if hasattr(detector, "fit"):
                detector.fit(X)

            if hasattr(detector, "score"):
                scores = detector.score(X)
            elif hasattr(detector, "decision_function"):
                scores = detector.decision_function(X)
            else:
                continue
            base_scores.append(scores.reshape(-1, 1))

        base_features = np.hstack(base_scores)
        base_features_scaled = self.scaler.fit_transform(base_features)

        y_labels = np.zeros(len(X))
        y_labels[
            np.argsort(np.mean(base_features_scaled, axis=1))[
                -int(len(X) * self.contamination) :
            ]
        ] = 1

        self.meta_learner.fit(base_features_scaled, y_labels)

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble scores via meta-learner."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted.")

        base_scores = []
        for detector in self.detectors:
            if hasattr(detector, "score"):
                scores = detector.score(X)
            elif hasattr(detector, "decision_function"):
                scores = detector.decision_function(X)
            else:
                continue
            base_scores.append(scores.reshape(-1, 1))

        base_features = np.hstack(base_scores)
        base_features_scaled = self.scaler.transform(base_features)

        if self.use_proba and hasattr(self.meta_learner, "predict_proba"):
            return self.meta_learner.predict_proba(base_features_scaled)[:, 1]
        elif hasattr(self.meta_learner, "decision_function"):
            return self.meta_learner.decision_function(base_features_scaled)
        else:
            return self.meta_learner.predict(base_features_scaled)


class ScoreAggregationEnsemble(BaseEnsembleDetector):
    """
    Score aggregation ensemble with various combination methods.

    Methods: mean, max, min, median, geometric mean, power mean.

    Parameters
    ----------
    detectors : List
        List of anomaly detectors.
    aggregation : str
        Aggregation method.
    p : float
        Power for power mean.
    """

    def __init__(
        self,
        detectors: List[Any] = None,
        aggregation: str = "mean",
        p: float = 2.0,
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)
        self.detectors = detectors or []
        self.aggregation = aggregation
        self.p = p

    def add_detector(self, detector: Any) -> "ScoreAggregationEnsemble":
        """Add a detector."""
        self.detectors.append(detector)
        return self

    def fit(self, X: np.ndarray) -> "ScoreAggregationEnsemble":
        """Fit all detectors."""
        for detector in self.detectors:
            if hasattr(detector, "fit"):
                detector.fit(X)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute aggregated scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted.")

        scores_list = []
        for detector in self.detectors:
            if hasattr(detector, "score"):
                scores_list.append(detector.score(X))
            elif hasattr(detector, "decision_function"):
                scores_list.append(detector.decision_function(X))

        scores_array = np.column_stack(scores_list)

        if self.aggregation == "mean":
            return np.mean(scores_array, axis=1)
        elif self.aggregation == "max":
            return np.max(scores_array, axis=1)
        elif self.aggregation == "min":
            return np.min(scores_array, axis=1)
        elif self.aggregation == "median":
            return np.median(scores_array, axis=1)
        elif self.aggregation == "geometric_mean":
            return np.exp(np.mean(np.log(scores_array + 1e-10), axis=1))
        elif self.aggregation == "power_mean":
            return np.power(
                np.mean(np.power(scores_array + 1e-10, self.p), axis=1), 1 / self.p
            )
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class DynamicEnsembleSelector(BaseEnsembleDetector):
    """
    Dynamic ensemble selection based on local data density.

    Selects the most appropriate detector for each sample based on
    local data characteristics.

    Parameters
    ----------
    detectors : List
        List of anomaly detectors.
    n_neighbors : int
        Neighbors for local density estimation.
    selection_method : str
        Selection method: 'density', 'diversity', 'uncertainty'.
    """

    def __init__(
        self,
        detectors: List[Any] = None,
        n_neighbors: int = 10,
        selection_method: str = "density",
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)
        self.detectors = detectors or []
        self.n_neighbors = n_neighbors
        self.selection_method = selection_method
        self.local_densities: Optional[np.ndarray] = None

    def add_detector(self, detector: Any) -> "DynamicEnsembleSelector":
        """Add a detector."""
        self.detectors.append(detector)
        return self

    def fit(self, X: np.ndarray) -> "DynamicEnsembleSelector":
        """Fit all detectors and compute local densities."""
        from sklearn.neighbors import NearestNeighbors

        for detector in self.detectors:
            if hasattr(detector, "fit"):
                detector.fit(X)

        nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, X.shape[0]))
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        self.local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute dynamically selected ensemble scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted.")

        all_scores = []
        for detector in self.detectors:
            if hasattr(detector, "score"):
                all_scores.append(detector.score(X))
            elif hasattr(detector, "decision_function"):
                all_scores.append(detector.decision_function(X))

        all_scores = np.column_stack(all_scores)

        if self.selection_method == "density":
            return np.max(all_scores, axis=1)
        elif self.selection_method == "diversity":
            score_variance = np.var(all_scores, axis=1)
            return np.mean(all_scores, axis=1) * (1 + score_variance)
        elif self.selection_method == "uncertainty":
            sorted_scores = np.sort(all_scores, axis=1)
            uncertainty = (
                sorted_scores[:, -1] - sorted_scores[:, -2]
                if all_scores.shape[1] > 1
                else np.zeros(len(X))
            )
            return np.mean(all_scores, axis=1) + uncertainty
        else:
            return np.mean(all_scores, axis=1)


class BaggingAnomalyDetector(BaseEnsembleDetector):
    """
    Bagging-based anomaly detection with bootstrap sampling.

    Creates multiple detectors on bootstrapped samples and aggregates.

    Parameters
    ----------
    base_detector : optional
        Base detector class to bag.
    n_estimators : int
        Number of bagged detectors.
    max_samples : float or int
        Fraction or number of samples per bootstrap.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        base_detector: Any = None,
        n_estimators: int = 10,
        max_samples: Union[float, int] = 0.8,
        random_state: Optional[int] = None,
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)
        self.base_detector = base_detector
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.detectors: List[Any] = []
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "BaggingAnomalyDetector":
        """Fit bagged detectors."""
        import copy

        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if isinstance(self.max_samples, float):
            n_select = int(n_samples * self.max_samples)
        else:
            n_select = min(self.max_samples, n_samples)

        X_scaled = self.scaler.fit_transform(X)

        for i in range(self.n_estimators):
            indices = rng.choice(n_samples, n_select, replace=True)
            X_boot = X_scaled[indices]

            if self.base_detector is not None:
                detector = copy.deepcopy(self.base_detector)
            else:
                from sklearn.ensemble import IsolationForest

                detector = IsolationForest(random_state=rng.randint(0, 2**31))

            if hasattr(detector, "fit"):
                detector.fit(X_boot)

            self.detectors.append(detector)

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute bagged anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted.")

        X_scaled = self.scaler.transform(X)

        scores_list = []
        for detector in self.detectors:
            if hasattr(detector, "score_samples"):
                scores_list.append(detector.score_samples(X_scaled))
            elif hasattr(detector, "score"):
                scores_list.append(detector.score(X_scaled))

        all_scores = np.column_stack(scores_list)
        return np.mean(all_scores, axis=1)


class AdaptiveWeightedEnsemble(BaseEnsembleDetector):
    """
    Adaptive weighted ensemble with learned weights.

    Learns optimal weights based on detection performance.

    Parameters
    ----------
    detectors : List
        List of anomaly detectors.
    weight_learn_epochs : int
        Epochs for weight optimization.
    lr : float
        Learning rate for weight updates.
    """

    def __init__(
        self,
        detectors: List[Any] = None,
        weight_learn_epochs: int = 50,
        lr: float = 0.1,
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)
        self.detectors = detectors or []
        self.weight_learn_epochs = weight_learn_epochs
        self.lr = lr
        self.weights: Optional[np.ndarray] = None

    def add_detector(self, detector: Any) -> "AdaptiveWeightedEnsemble":
        """Add a detector."""
        self.detectors.append(detector)
        return self

    def fit(self, X: np.ndarray) -> "AdaptiveWeightedEnsemble":
        """Fit detectors and learn weights."""
        if not self.detectors:
            raise ValueError("No detectors in ensemble.")

        self.weights = np.ones(len(self.detectors)) / len(self.detectors)

        for detector in self.detectors:
            if hasattr(detector, "fit"):
                detector.fit(X)

        self._learn_weights(X)

        self.is_fitted = True
        return self

    def _learn_weights(self, X: np.ndarray) -> None:
        """Learn optimal detector weights."""
        scores_list = []
        for detector in self.detectors:
            if hasattr(detector, "score"):
                scores_list.append(detector.score(X))
            elif hasattr(detector, "decision_function"):
                scores_list.append(detector.decision_function(X))

        base_scores = np.column_stack(scores_list)
        labels = np.zeros(len(X))
        labels[
            np.argsort(np.mean(base_scores, axis=1))[
                -int(len(X) * self.contamination) :
            ]
        ] = 1

        weights = np.ones(len(self.detectors)) / len(self.detectors)

        for _ in range(self.weight_learn_epochs):
            combined = np.dot(base_scores, weights)

            grad = np.zeros(len(self.detectors))
            for i in range(len(self.detectors)):
                correlation = np.corrcoef(combined, base_scores[:, i])[0, 1]
                grad[i] = -correlation

            weights = weights - self.lr * grad
            weights = np.maximum(weights, 0)
            weights = weights / (np.sum(weights) + 1e-10)

        self.weights = weights

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted ensemble scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted.")

        scores_list = []
        for detector in self.detectors:
            if hasattr(detector, "score"):
                scores_list.append(detector.score(X))
            elif hasattr(detector, "decision_function"):
                scores_list.append(detector.decision_function(X))

        base_scores = np.column_stack(scores_list)
        return np.dot(base_scores, self.weights)


class UncertaintyAwareEnsemble(BaseEnsembleDetector):
    """
    Ensemble with uncertainty estimation.

    Provides confidence scores along with anomaly predictions.

    Parameters
    ----------
    detectors : List
        List of anomaly detectors.
    uncertainty_method : str
        Method for uncertainty: 'variance', 'entropy', ' disagreement'.
    """

    def __init__(
        self,
        detectors: List[Any] = None,
        uncertainty_method: str = "variance",
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)
        self.detectors = detectors or []
        self.uncertainty_method = uncertainty_method

    def add_detector(self, detector: Any) -> "UncertaintyAwareEnsemble":
        """Add a detector."""
        self.detectors.append(detector)
        return self

    def fit(self, X: np.ndarray) -> "UncertaintyAwareEnsemble":
        """Fit all detectors."""
        for detector in self.detectors:
            if hasattr(detector, "fit"):
                detector.fit(X)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scores and uncertainties.

        Returns
        -------
        scores : np.ndarray
            Anomaly scores.
        uncertainties : np.ndarray
            Uncertainty scores.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted.")

        scores_list = []
        for detector in self.detectors:
            if hasattr(detector, "score"):
                scores_list.append(detector.score(X))
            elif hasattr(detector, "decision_function"):
                scores_list.append(detector.decision_function(X))

        base_scores = np.column_stack(scores_list)

        if self.uncertainty_method == "variance":
            uncertainties = np.var(base_scores, axis=1)
        elif self.uncertainty_method == "entropy":
            probs = (base_scores - base_scores.min(axis=0)) / (
                base_scores.max(axis=0) - base_scores.min(axis=0) + 1e-10
            )
            probs = probs / probs.sum(axis=1, keepdims=True)
            uncertainties = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        elif self.uncertainty_method == "disagreement":
            votes = (base_scores > np.median(base_scores, axis=0)).astype(int)
            disagreements = np.mean(votes, axis=1) * (1 - np.mean(votes, axis=1))
            uncertainties = 1 - np.abs(disagreements - 0.5) * 2
        else:
            uncertainties = np.zeros(len(X))

        scores = np.mean(base_scores, axis=1)
        return scores, uncertainties

    def predict_with_confidence(
        self, X: np.ndarray, confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.

        Returns
        -------
        labels : np.ndarray
            Anomaly labels.
        confidence : np.ndarray
            Confidence scores (1 - normalized uncertainty).
        """
        scores, uncertainties = self.score(X)

        if self.threshold is None:
            self.threshold = np.percentile(scores, (1 - self.contamination) * 100)

        labels = (scores > self.threshold).astype(int)
        max_unc = np.max(uncertainties) + 1e-10
        confidence = 1 - (uncertainties / max_unc)

        return labels, confidence
