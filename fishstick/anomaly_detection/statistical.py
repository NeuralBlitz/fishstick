"""
Statistical Anomaly Detection Module.

This module provides statistical methods for anomaly detection including:
- Z-score based detection
- Interquartile Range (IQR) detection
- Grubbs test for outliers
- Chi-square test for multivariate outliers
- Mahalanobis distance-based detection
- Adjusted boxplot for skewed distributions
- Generalized ESD test

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance, MinCovDet


@dataclass
class DetectionResult:
    """Container for anomaly detection results."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    n_anomalies: int
    anomaly_indices: np.ndarray


class BaseStatisticalDetector(ABC):
    """Base class for statistical anomaly detectors."""

    def __init__(self, contamination: float = 0.1, alpha: float = 0.05):
        self.contamination = contamination
        self.alpha = alpha
        self.threshold: Optional[float] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseStatisticalDetector":
        """Fit the detector on normal data."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            self._compute_threshold(scores)
        return (scores > self.threshold).astype(int)

    def fit_predict(self, X: np.ndarray) -> DetectionResult:
        """Fit and predict in one call."""
        self.fit(X)
        scores = self.score(X)
        labels = self.predict(X)
        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            n_anomalies=int(np.sum(labels)),
            anomaly_indices=np.where(labels == 1)[0],
        )

    def _compute_threshold(self, scores: np.ndarray) -> None:
        """Compute threshold based on contamination rate."""
        self.threshold = np.percentile(scores, (1 - self.contamination) * 100)


class ZScoreDetector(BaseStatisticalDetector):
    """
    Z-score based anomaly detector.

    Detects anomalies based on how many standard deviations each point
    is from the mean. Suitable for unimodal, symmetric distributions.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the data.
    alpha : float
        Significance level for threshold computation.
    threshold : float, optional
        Custom z-score threshold. If None, computed from contamination.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        alpha: float = 0.05,
        threshold: Optional[float] = None,
        robust: bool = False,
    ):
        super().__init__(contamination, alpha)
        self.threshold = threshold
        self.robust = robust
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.median: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        """Fit the detector by computing mean and std."""
        X = np.asarray(X)
        if self.robust:
            self.median = np.median(X, axis=0)
            mad = np.median(np.abs(X - self.median), axis=0)
            self.std = mad * 1.4826  # Scale MAD to match std for normal distribution
            self.std = np.where(self.std == 0, 1e-10, self.std)
        else:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std = np.where(self.std == 0, 1e-10, self.std)

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute z-scores as anomaly scores."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        X = np.asarray(X)
        if self.robust:
            z_scores = np.abs((X - self.median) / self.std)
        else:
            z_scores = np.abs((X - self.mean) / self.std)

        return np.max(z_scores, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            self._compute_threshold(scores)
        return (scores > self.threshold).astype(int)


class IQRDetector(BaseStatisticalDetector):
    """
    Interquartile Range (IQR) based anomaly detector.

    Detects anomalies using the boxplot method. Points beyond
    Q1 - k*IQR or Q3 + k*IQR are flagged as anomalies.
    Robust to extreme values.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    alpha : float
        Not used, kept for API consistency.
    k : float
        Multiplier for IQR. Typically 1.5 for outliers, 3.0 for extreme.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        alpha: float = 0.05,
        k: float = 1.5,
    ):
        super().__init__(contamination, alpha)
        self.k = k
        self.q1: Optional[np.ndarray] = None
        self.q3: Optional[np.ndarray] = None
        self.iqr: Optional[np.ndarray] = None
        self.median: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "IQRDetector":
        """Fit by computing quartiles."""
        X = np.asarray(X)
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        self.iqr = np.where(self.iqr == 0, 1e-10, self.iqr)
        self.median = np.median(X, axis=0)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute distance from IQR bounds as anomaly scores."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        X = np.asarray(X)
        lower_bound = self.q1 - self.k * self.iqr
        upper_bound = self.q3 + self.k * self.iqr

        distance_below = np.maximum(lower_bound - X, 0)
        distance_above = np.maximum(X - upper_bound, 0)

        distances = np.sqrt(distance_below**2 + distance_above**2)
        return np.max(distances, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            self._compute_threshold(scores)
        return (scores > self.threshold).astype(int)


class GrubbsDetector(BaseStatisticalDetector):
    """
    Grubbs' test for univariate outlier detection.

    Tests for outliers assuming data follows a normal distribution.
    Iteratively removes outliers until no significant ones remain.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    alpha : float
        Significance level for the test.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        alpha: float = 0.05,
    ):
        super().__init__(contamination, alpha)
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.n: int = 0

    def fit(self, X: np.ndarray) -> "GrubbsDetector":
        """Fit by computing statistics."""
        X = np.asarray(X).flatten()
        self.mean = np.mean(X)
        self.std = np.std(X, ddof=1)
        self.n = len(X)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Grubbs' scores."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        X = np.asarray(X).flatten()
        if self.std == 0:
            return np.zeros(len(X))

        z_scores = np.abs((X - self.mean) / self.std)
        return z_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels using Grubbs' test threshold."""
        scores = self.score(X)
        n = len(X)

        t_dist = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
        grubbs_threshold = ((n - 1) / np.sqrt(n)) * np.sqrt(
            t_dist**2 / (n - 2 + t_dist**2)
        )

        self.threshold = grubbs_threshold
        return (scores > self.threshold).astype(int)


class ChiSquareDetector(BaseStatisticalDetector):
    """
    Chi-square test for multivariate anomaly detection.

    Detects outliers in multivariate data using chi-square distribution.
    Points with sum of squared z-scores exceeding chi-square threshold
    are flagged as anomalies.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    alpha : float
        Significance level for chi-square test.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        alpha: float = 0.05,
    ):
        super().__init__(contamination, alpha)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.dof: int = 0

    def fit(self, X: np.ndarray) -> "ChiSquareDetector":
        """Fit by computing mean and standard deviation."""
        X = np.asarray(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std == 0, 1e-10, self.std)
        self.dof = X.shape[1]
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute chi-square statistic for each point."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        X = np.asarray(X)
        z_scores = (X - self.mean) / self.std
        chi_square_scores = np.sum(z_scores**2, axis=1)
        return chi_square_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        self.threshold = stats.chi2.ppf(1 - self.alpha, self.dof)
        return (scores > self.threshold).astype(int)


class MahalanobisDetector(BaseStatisticalDetector):
    """
    Mahalanobis distance based anomaly detector.

    Uses Mahalanobis distance which accounts for correlations
    between variables. Suitable for multivariate Gaussian data.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    alpha : float
        Significance level for threshold.
    robust : bool
        Use robust covariance estimation (MinCovDet).
    """

    def __init__(
        self,
        contamination: float = 0.1,
        alpha: float = 0.05,
        robust: bool = False,
    ):
        super().__init__(contamination, alpha)
        self.robust = robust
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.inv_covariance: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        """Fit by computing covariance matrix."""
        X = np.asarray(X)
        self.mean = np.mean(X, axis=0)

        if self.robust:
            mcd = MinCovDet().fit(X)
            self.covariance = mcd.covariance_
            self.inv_covariance = mcd.precision_
        else:
            self.covariance = np.cov(X, rowvar=False)
            try:
                self.inv_covariance = np.linalg.inv(self.covariance)
            except np.linalg.LinAlgError:
                self.covariance = np.diag(np.diag(self.covariance))
                self.inv_covariance = np.linalg.inv(self.covariance)

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        X = np.asarray(X)
        distances = np.array(
            [mahalanobis(x, self.mean, self.inv_covariance) for x in X]
        )
        return distances**2  # Squared distance follows chi-square

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            dof = len(self.mean)
            self.threshold = stats.chi2.ppf(1 - self.alpha, dof)
        return (scores > self.threshold).astype(int)


class AdjustedBoxplotDetector(BaseStatisticalDetector):
    """
    Adjusted boxplot for skewed distributions.

    Uses the medcouple (MC) to adjust the boxplot for skewed data.
    More robust than standard IQR for non-symmetric distributions.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    alpha : float
        Significance level.
    k : float
        IQR multiplier.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        alpha: float = 0.05,
        k: float = 1.5,
    ):
        super().__init__(contamination, alpha)
        self.k = k
        self.q1: Optional[float] = None
        self.q3: Optional[float] = None
        self.iqr: Optional[float] = None
        self.median: Optional[float] = None
        self.mc: Optional[float] = None

    def fit(self, X: np.ndarray) -> "AdjustedBoxplotDetector":
        """Fit by computing adjusted boxplot statistics."""
        X = np.asarray(X).flatten()
        self.q1 = np.percentile(X, 25)
        self.q3 = np.percentile(X, 75)
        self.iqr = self.q3 - self.q1
        self.median = np.median(X)

        if self.iqr > 0:
            X_centered = (X - self.median) / self.iqr
            left = X_centered[X_centered < 0]
            right = X_centered[X_centered > 0]
            if len(left) > 0 and len(right) > 0:
                self.mc = np.median(
                    [(a - b) / (a + b) for a in right for b in np.abs(left)]
                )
            else:
                self.mc = 0
        else:
            self.iqr = 1e-10
            self.mc = 0

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute adjusted distance from boxplot bounds."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        X = np.asarray(X).flatten()

        if self.mc >= 0:
            lower_bound = self.q1 - self.k * self.iqr * np.exp(-3 * self.mc)
            upper_bound = self.q3 + self.k * self.iqr * np.exp(3 * self.mc)
        else:
            lower_bound = self.q1 - self.k * self.iqr * np.exp(-3 * -self.mc)
            upper_bound = self.q3 + self.k * self.iqr * np.exp(-3 * -self.mc)

        distance_below = np.maximum(lower_bound - X, 0)
        distance_above = np.maximum(X - upper_bound, 0)

        return distance_below + distance_above


class GeneralizedESDDetector(BaseStatisticalDetector):
    """
    Generalized Extreme Studentized Deviate (ESD) test.

    Detects multiple outliers in a dataset without specifying
    the exact number. Works by iteratively applying Grubbs' test.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies.
    alpha : float
        Significance level for each test.
    max_outliers : int, optional
        Maximum number of outliers to detect. If None, estimate from contamination.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        alpha: float = 0.05,
        max_outliers: Optional[int] = None,
    ):
        super().__init__(contamination, alpha)
        self.max_outliers = max_outliers
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def fit(self, X: np.ndarray) -> "GeneralizedESDDetector":
        """Fit by computing basic statistics."""
        X = np.asarray(X).flatten()
        self.mean = np.mean(X)
        self.std = np.std(X, ddof=1)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute ESD scores."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        X = np.asarray(X).flatten()
        if self.std == 0:
            return np.zeros(len(X))

        return np.abs((X - self.mean) / self.std)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using generalized ESD test."""
        X = np.asarray(X).flatten()
        n = len(X)

        if self.max_outliers is None:
            max_outliers = int(np.ceil(n * self.contamination))
        else:
            max_outliers = min(self.max_outliers, n - 2)

        if max_outliers < 1:
            return np.zeros(n, dtype=int)

        critical_values = []
        for i in range(1, max_outliers + 1):
            dof = n - i - 1
            if dof < 1:
                break
            t_ppf = stats.t.ppf(1 - self.alpha / (2 * (n - i + 1)), dof)
            critical_value = ((n - i) / np.sqrt(n - i)) * np.sqrt(
                t_ppf**2 / (dof + t_ppf**2)
            )
            critical_values.append(critical_value)

        max_outliers = len(critical_values)
        if max_outliers == 0:
            return np.zeros(n, dtype=int)

        scores = np.abs((X - self.mean) / self.std)
        r = np.zeros(max_outliers)
        outliers_mask = np.zeros(n, dtype=bool)

        for i in range(max_outliers):
            current_data = X[~outliers_mask]
            current_mean = np.mean(current_data)
            current_std = np.std(current_data, ddof=1)

            if current_std == 0:
                break

            current_scores = np.abs((X - current_mean) / current_std)
            current_scores[outliers_mask] = 0

            idx = np.argmax(current_scores)
            r[i] = current_scores[idx]

            if r[i] >= critical_values[i]:
                outliers_mask[idx] = True
            else:
                break

        return outliers_mask.astype(int)


class StatisticalDetectorEnsemble(BaseStatisticalDetector):
    """
       Ensemble of multiple statistical detectors.

       Combines multiple statistical methods for robust anomaly detection.
       Uses weighted.

       Parameters
    voting or score averaging    ----------
       detectors : list
           List of detector instances.
       method : str
           Combination method: 'average', 'max', 'weighted'.
       contamination : float
           Expected proportion of anomalies.
    """

    def __init__(
        self,
        detectors: List[BaseStatisticalDetector],
        method: str = "average",
        contamination: float = 0.1,
    ):
        super().__init__(contamination, alpha=0.05)
        self.detectors = detectors
        self.method = method
        self.weights: Optional[List[float]] = None

    def set_weights(self, weights: List[float]) -> None:
        """Set weights for each detector."""
        if len(weights) != len(self.detectors):
            raise ValueError("Number of weights must match number of detectors")
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def fit(self, X: np.ndarray) -> "StatisticalDetectorEnsemble":
        """Fit all detectors."""
        for detector in self.detectors:
            detector.fit(X)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble scores."""
        scores_list = [detector.score(X) for detector in self.detectors]

        if self.method == "average":
            if self.weights is not None:
                return np.average(scores_list, axis=0, weights=self.weights)
            return np.mean(scores_list, axis=0)
        elif self.method == "max":
            return np.max(scores_list, axis=0)
        elif self.method == "weighted":
            if self.weights is None:
                self.weights = [1.0 / len(self.detectors)] * len(self.detectors)
            return np.average(scores_list, axis=0, weights=self.weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def compute_roc_pr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute ROC and PR curve metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 = normal, 1 = anomaly).
    y_scores : np.ndarray
        Anomaly scores.

    Returns
    -------
    dict
        Dictionary containing ROC-AUC, PR-AUC, and curve data.
    """
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
    }


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


__all__ = [
    "BaseStatisticalDetector",
    "DetectionResult",
    "ZScoreDetector",
    "IQRDetector",
    "GrubbsDetector",
    "ChiSquareDetector",
    "MahalanobisDetector",
    "AdjustedBoxplotDetector",
    "GeneralizedESDDetector",
    "StatisticalDetectorEnsemble",
    "compute_roc_pr",
]
