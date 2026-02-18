"""
Comprehensive Drift Detection Module for Fishstick

This module provides implementations for various types of drift detection:
- Data Drift: Statistical tests for distribution changes
- Concept Drift: Online change detection algorithms
- Feature Drift: Feature-level monitoring
- Prediction Drift: Model output monitoring
- Visualization: Drift visualization tools
- Alerts: Notification systems
- Remediation: Automated responses to drift
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
from collections import deque
import json
import time
from datetime import datetime

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance, entropy, chi2_contingency, ks_2samp
from scipy.spatial.distance import jensenshannon


# =============================================================================
# Enums and Constants
# =============================================================================

class DriftType(Enum):
    """Types of drift that can be detected."""
    DATA = "data"
    CONCEPT = "concept"
    FEATURE = "feature"
    PREDICTION = "prediction"
    CONFIDENCE = "confidence"
    CALIBRATION = "calibration"


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Channels for sending drift alerts."""
    EMAIL = "email"
    SLACK = "slack"
    CONSOLE = "console"
    WEBHOOK = "webhook"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DriftResult:
    """Result of a drift detection test."""
    drift_detected: bool
    statistic: float
    p_value: Optional[float] = None
    threshold: float = 0.05
    drift_type: DriftType = DriftType.DATA
n    severity: DriftSeverity = DriftSeverity.NONE
    feature_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "drift_detected": self.drift_detected,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "threshold": float(self.threshold),
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "feature_name": self.feature_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class DriftReport:
    """Comprehensive drift report containing multiple test results."""
    timestamp: datetime = field(default_factory=datetime.now)
    results: List[DriftResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: DriftResult):
        """Add a drift result to the report."""
        self.results.append(result)
    
    def get_drifted_features(self) -> List[str]:
        """Get list of features with detected drift."""
        return [
            r.feature_name for r in self.results 
            if r.drift_detected and r.feature_name is not None
        ]
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Count results by severity level."""
        counts = {s.value: 0 for s in DriftSeverity}
        for r in self.results:
            counts[r.severity.value] += 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_tests": len(self.results),
                "drift_detected_count": sum(1 for r in self.results if r.drift_detected),
                "drifted_features": self.get_drifted_features(),
                "severity_counts": self.get_severity_counts()
            },
            "user_metadata": self.summary
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# Base Classes
# =============================================================================

class BaseDriftDetector(ABC):
    """Abstract base class for all drift detectors."""
    
    def __init__(self, threshold: float = 0.05, name: Optional[str] = None):
        self.threshold = threshold
        self.name = name or self.__class__.__name__
        self.reference_data: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseDriftDetector':
        """Fit the detector on reference data."""
        self.reference_data = np.array(X)
        self.is_fitted = True
        return self
    
    @abstractmethod
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift in new data."""
        pass
    
    def _validate_input(self, X: np.ndarray):
        """Validate input data."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection. Call fit() first.")
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


# =============================================================================
# 1. DATA DRIFT DETECTORS
# =============================================================================

class KSDrift(BaseDriftDetector):
    """
    Kolmogorov-Smirnov drift detector.
    
    Uses the two-sample KS test to compare distributions.
    Non-parametric test for continuous data.
    """
    
    def __init__(self, threshold: float = 0.05, alternative: str = "two-sided"):
        super().__init__(threshold)
        self.alternative = alternative
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift using KS test."""
        X = self._validate_input(X)
        
        statistics = []
        p_values = []
        
        for col in range(X.shape[1]):
            ref_col = self.reference_data[:, col] if self.reference_data.ndim > 1 else self.reference_data
            test_col = X[:, col] if X.ndim > 1 else X
            
            stat, p_val = ks_2samp(ref_col, test_col, alternative=self.alternative)
            statistics.append(stat)
            p_values.append(p_val)
        
        # Use maximum statistic and minimum p-value
        max_stat = max(statistics)
        min_p = min(p_values)
        drift_detected = min_p < self.threshold
        
        # Determine severity
        if min_p < 0.001:
            severity = DriftSeverity.CRITICAL
        elif min_p < 0.01:
            severity = DriftSeverity.HIGH
        elif min_p < self.threshold:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.NONE
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=max_stat,
            p_value=min_p,
            threshold=self.threshold,
            drift_type=DriftType.DATA,
            severity=severity,
            metadata={"test": "Kolmogorov-Smirnov", "statistics_per_feature": statistics}
        )


class ChiSquareDrift(BaseDriftDetector):
    """
    Chi-square drift detector for categorical data.
    
    Tests if the observed frequencies differ from expected frequencies.
    """
    
    def __init__(self, threshold: float = 0.05, bins: int = 10):
        super().__init__(threshold)
        self.bins = bins
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift using chi-square test."""
        X = self._validate_input(X)
        
        chi2_stats = []
        p_values = []
        
        for col in range(X.shape[1]):
            ref_col = self.reference_data[:, col] if self.reference_data.ndim > 1 else self.reference_data
            test_col = X[:, col] if X.ndim > 1 else X
            
            # Create contingency table
            ref_hist, bin_edges = np.histogram(ref_col, bins=self.bins)
            test_hist, _ = np.histogram(test_col, bins=bin_edges)
            
            # Apply Laplace smoothing to avoid zero counts
            ref_hist = ref_hist + 1
            test_hist = test_hist + 1
            
            # Chi-square test
            contingency = np.array([ref_hist, test_hist])
            chi2, p_val, _, _ = chi2_contingency(contingency)
            
            chi2_stats.append(chi2)
            p_values.append(p_val)
        
        max_chi2 = max(chi2_stats)
        min_p = min(p_values)
        drift_detected = min_p < self.threshold
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=max_chi2,
            p_value=min_p,
            threshold=self.threshold,
            drift_type=DriftType.DATA,
            severity=DriftSeverity.HIGH if drift_detected else DriftSeverity.NONE,
            metadata={"test": "Chi-Square", "chi2_stats": chi2_stats}
        )


class WassersteinDrift(BaseDriftDetector):
    """
    Wasserstein distance drift detector.
    
    Measures the Earth Mover's Distance between distributions.
    More sensitive to location shifts than KS test.
    """
    
    def __init__(self, threshold: float = 0.1, p_norm: int = 1):
        super().__init__(threshold)
        self.p_norm = p_norm
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift using Wasserstein distance."""
        X = self._validate_input(X)
        
        distances = []
        
        for col in range(X.shape[1]):
            ref_col = self.reference_data[:, col] if self.reference_data.ndim > 1 else self.reference_data
            test_col = X[:, col] if X.ndim > 1 else X
            
            dist = wasserstein_distance(ref_col, test_col)
            distances.append(dist)
        
        max_dist = max(distances)
        # Normalize by standard deviation of reference
        ref_std = np.std(self.reference_data)
        normalized_dist = max_dist / (ref_std + 1e-10)
        
        drift_detected = normalized_dist > self.threshold
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=normalized_dist,
            threshold=self.threshold,
            drift_type=DriftType.DATA,
            severity=DriftSeverity.HIGH if normalized_dist > 2 * self.threshold else 
                     (DriftSeverity.MEDIUM if drift_detected else DriftSeverity.NONE),
            metadata={
                "test": "Wasserstein",
                "raw_distances": distances,
                "max_distance": max_dist
            }
        )


class PSI(BaseDriftDetector):
    """
    Population Stability Index (PSI) drift detector.
    
    Commonly used in credit risk modeling.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change
    PSI >= 0.25: Significant change
    """
    
    def __init__(self, threshold: float = 0.25, bins: int = 10):
        super().__init__(threshold)
        self.bins = bins
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift using PSI."""
        X = self._validate_input(X)
        
        psi_values = []
        
        for col in range(X.shape[1]):
            ref_col = self.reference_data[:, col] if self.reference_data.ndim > 1 else self.reference_data
            test_col = X[:, col] if X.ndim > 1 else X
            
            # Create bins based on reference data
            min_val, max_val = np.min(ref_col), np.max(ref_col)
            bin_edges = np.linspace(min_val, max_val, self.bins + 1)
            
            # Calculate proportions
            ref_hist, _ = np.histogram(ref_col, bins=bin_edges)
            test_hist, _ = np.histogram(test_col, bins=bin_edges)
            
            ref_prop = ref_hist / (len(ref_col) + 1e-10)
            test_prop = test_hist / (len(test_col) + 1e-10)
            
            # Add small constant to avoid division by zero
            ref_prop = np.maximum(ref_prop, 0.0001)
            test_prop = np.maximum(test_prop, 0.0001)
            
            # Calculate PSI
            psi = np.sum((test_prop - ref_prop) * np.log(test_prop / ref_prop))
            psi_values.append(psi)
        
        max_psi = max(psi_values)
        drift_detected = max_psi > self.threshold
        
        # Determine severity based on PSI guidelines
        if max_psi >= 0.25:
            severity = DriftSeverity.HIGH
        elif max_psi >= 0.1:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.NONE
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=max_psi,
            threshold=self.threshold,
            drift_type=DriftType.DATA,
            severity=severity,
            metadata={"test": "PSI", "psi_per_feature": psi_values}
        )


class KLDivergence(BaseDriftDetector):
    """
    Kullback-Leibler divergence drift detector.
    
    Measures the information lost when using one distribution
    to approximate another.
    """
    
    def __init__(self, threshold: float = 0.1, bins: int = 20, symmetric: bool = True):
        super().__init__(threshold)
        self.bins = bins
        self.symmetric = symmetric
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift using KL divergence."""
        X = self._validate_input(X)
        
        kl_values = []
        
        for col in range(X.shape[1]):
            ref_col = self.reference_data[:, col] if self.reference_data.ndim > 1 else self.reference_data
            test_col = X[:, col] if X.ndim > 1 else X
            
            # Create histograms
            min_val = min(np.min(ref_col), np.min(test_col))
            max_val = max(np.max(ref_col), np.max(test_col))
            bin_edges = np.linspace(min_val, max_val, self.bins + 1)
            
            ref_hist, _ = np.histogram(ref_col, bins=bin_edges, density=True)
            test_hist, _ = np.histogram(test_col, bins=bin_edges, density=True)
            
            # Add small constant to avoid log(0)
            ref_hist = ref_hist + 1e-10
            test_hist = test_hist + 1e-10
            
            # Normalize
            ref_hist = ref_hist / np.sum(ref_hist)
            test_hist = test_hist / np.sum(test_hist)
            
            # Calculate KL divergence
            kl = entropy(test_hist, ref_hist)
            
            if self.symmetric:
                kl = (kl + entropy(ref_hist, test_hist)) / 2
            
            kl_values.append(kl)
        
        max_kl = max(kl_values)
        drift_detected = max_kl > self.threshold
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=max_kl,
            threshold=self.threshold,
            drift_type=DriftType.DATA,
            severity=DriftSeverity.HIGH if max_kl > 2 * self.threshold else 
                     (DriftSeverity.MEDIUM if drift_detected else DriftSeverity.NONE),
            metadata={"test": "KL-Divergence", "kl_per_feature": kl_values}
        )


# =============================================================================
# 2. CONCEPT DRIFT DETECTORS
# =============================================================================

class ADWIN(BaseDriftDetector):
    """
    Adaptive Windowing (ADWIN) algorithm.
    
    Maintains a sliding window of data and automatically
    shrinks or expands based on detected changes.
    """
    
    def __init__(self, delta: float = 0.002, min_window_size: int = 30):
        super().__init__(threshold=delta)
        self.delta = delta
        self.min_window_size = min_window_size
        self.window = deque(maxlen=10000)
        self.bucket_row = []
        self.sum = 0.0
        self.variance = 0.0
        self.n = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Initialize with reference data."""
        X = np.array(X)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
        
        for val in X[-self.min_window_size:]:
            self._add_element(val)
        
        self.is_fitted = True
        return self
    
    def _add_element(self, value: float):
        """Add element to the window."""
        self.window.append(value)
        self.n += 1
        self.sum += value
        
        # Maintain variance
        if self.n > 1:
            old_mean = (self.sum - value) / (self.n - 1)
            new_mean = self.sum / self.n
            self.variance = ((self.n - 2) * self.variance + (value - old_mean) * (value - new_mean)) / (self.n - 1)
    
    def _cut_expression(self, n_0: float, n_1: float, abs_diff: float, variance: float) -> bool:
        """Check if drift condition is met."""
        m = 1 / (1 / n_0 + 1 / n_1)
        delta_prime = self.delta / (self.n * np.log(self.n)) if self.n > 1 else self.delta
        epsilon = np.sqrt(2 * variance * np.log(2 / delta_prime) / m) + 2 * np.log(2 / delta_prime) / (3 * m)
        return abs_diff > epsilon
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect concept drift."""
        X = np.array(X)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
        
        drift_detected = False
        max_stat = 0.0
        
        for val in X:
            self._add_element(val)
            
            if len(self.window) >= 2 * self.min_window_size:
                # Check for drift by comparing first and second half
                n = len(self.window)
                half = n // 2
                
                w1 = list(self.window)[:half]
                w2 = list(self.window)[half:]
                
                mean_1 = np.mean(w1)
                mean_2 = np.mean(w2)
                var_1 = np.var(w1)
                var_2 = np.var(w2)
                
                abs_diff = abs(mean_1 - mean_2)
                pooled_var = (var_1 + var_2) / 2
                
                if self._cut_expression(len(w1), len(w2), abs_diff, pooled_var + 1e-10):
                    drift_detected = True
                    max_stat = abs_diff
                    # Shrink window
                    for _ in range(half):
                        if self.window:
                            removed = self.window.popleft()
                            self.n -= 1
                            self.sum -= removed
                    break
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=max_stat,
            threshold=self.delta,
            drift_type=DriftType.CONCEPT,
            severity=DriftSeverity.HIGH if drift_detected else DriftSeverity.NONE,
            metadata={"algorithm": "ADWIN", "window_size": len(self.window)}
        )


class DDM(BaseDriftDetector):
    """
    Drift Detection Method (DDM).
    
    Monitors error rate and standard deviation to detect
    significant increase indicating concept drift.
    """
    
    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0):
        super().__init__(threshold=0)
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_error_rate = float('inf')
        self.min_std = float('inf')
        self.n_samples = 0
        self.n_errors = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit with labeled data (predictions and true labels)."""
        # X should be errors (0 for correct, 1 for incorrect)
        X = np.array(X)
        if X.ndim == 1:
            self.n_samples = len(X)
            self.n_errors = np.sum(X)
            error_rate = self.n_errors / self.n_samples
            self.min_error_rate = error_rate
            self.min_std = np.sqrt(error_rate * (1 - error_rate) / self.n_samples)
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift based on error stream."""
        X = np.array(X)
        if X.ndim == 1:
            self.n_samples += len(X)
            self.n_errors += np.sum(X)
            
            error_rate = self.n_errors / self.n_samples
            std = np.sqrt(error_rate * (1 - error_rate) / self.n_samples)
            
            # Update minimum statistics
            if error_rate + std < self.min_error_rate + self.min_std:
                self.min_error_rate = error_rate
                self.min_std = std
            
            # Check drift condition
            drift_score = (error_rate - self.min_error_rate) / (self.min_std + 1e-10)
            
            if drift_score > self.drift_level:
                severity = DriftSeverity.HIGH
                drift_detected = True
            elif drift_score > self.warning_level:
                severity = DriftSeverity.MEDIUM
                drift_detected = False
            else:
                severity = DriftSeverity.NONE
                drift_detected = False
            
            return DriftResult(
                drift_detected=drift_detected,
                statistic=drift_score,
                threshold=self.drift_level,
                drift_type=DriftType.CONCEPT,
                severity=severity,
                metadata={
                    "algorithm": "DDM",
                    "error_rate": error_rate,
                    "min_error_rate": self.min_error_rate
                }
            )
        
        raise ValueError("DDM expects binary error array")


class EDDM(BaseDriftDetector):
    """
    Early Drift Detection Method (EDDM).
    
    Monitors distance between errors to detect gradual drift
    earlier than DDM.
    """
    
    def __init__(self, warning_level: float = 0.95, drift_level: float = 0.9):
        super().__init__(threshold=drift_level)
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.max_p = 0
        self.max_s = 0
        self.error_positions = []
        self.n_samples = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit with error stream."""
        X = np.array(X)
        if X.ndim == 1:
            self.error_positions = list(np.where(X == 1)[0])
            self.n_samples = len(X)
            if len(self.error_positions) > 1:
                distances = np.diff(self.error_positions)
                self.max_p = np.mean(distances)
                self.max_s = np.std(distances)
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift based on error distance."""
        X = np.array(X)
        if X.ndim == 1:
            start_idx = self.n_samples
            new_errors = np.where(X == 1)[0] + start_idx
            self.error_positions.extend(new_errors)
            self.n_samples += len(X)
            
            if len(self.error_positions) < 2:
                return DriftResult(
                    drift_detected=False,
                    statistic=0,
                    threshold=self.drift_level,
                    drift_type=DriftType.CONCEPT,
                    severity=DriftSeverity.NONE,
                    metadata={"algorithm": "EDDM"}
                )
            
            distances = np.diff(self.error_positions)
            p = np.mean(distances)
            s = np.std(distances)
            
            # Update maximums
            if p + s > self.max_p + self.max_s:
                self.max_p = p
                self.max_s = s
            
            # Calculate ratio
            ratio = (p + s) / (self.max_p + self.max_s + 1e-10)
            
            if ratio < self.drift_level:
                severity = DriftSeverity.HIGH
                drift_detected = True
            elif ratio < self.warning_level:
                severity = DriftSeverity.MEDIUM
                drift_detected = False
            else:
                severity = DriftSeverity.NONE
                drift_detected = False
            
            return DriftResult(
                drift_detected=drift_detected,
                statistic=ratio,
                threshold=self.drift_level,
                drift_type=DriftType.CONCEPT,
                severity=severity,
                metadata={"algorithm": "EDDM", "mean_distance": p, "std_distance": s}
            )
        
        raise ValueError("EDDM expects binary error array")


class PageHinkley(BaseDriftDetector):
    """
    Page-Hinkley test for change detection.
    
    Cumulative sum test that is sensitive to small
    changes in the mean.
    """
    
    def __init__(self, threshold: float = 50, alpha: float = 0.9995, delta: float = 0.005):
        super().__init__(threshold)
        self.alpha = alpha
        self.delta = delta
        self.mean = 0
        self.sum = 0
        self.n = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit with reference data."""
        X = np.array(X)
        if X.ndim == 1:
            self.mean = np.mean(X)
            self.n = len(X)
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift using Page-Hinkley test."""
        X = np.array(X)
        if X.ndim == 1:
            max_ph = 0
            drift_detected = False
            
            for val in X:
                self.n += 1
                self.mean = self.mean * self.alpha + val * (1 - self.alpha)
                self.sum += val - self.mean - self.delta
                
                ph_stat = self.sum - np.min(self.sum if self.sum < 0 else 0)
                max_ph = max(max_ph, ph_stat)
                
                if ph_stat > self.threshold:
                    drift_detected = True
                    break
            
            return DriftResult(
                drift_detected=drift_detected,
                statistic=max_ph,
                threshold=self.threshold,
                drift_type=DriftType.CONCEPT,
                severity=DriftSeverity.HIGH if drift_detected else DriftSeverity.NONE,
                metadata={"algorithm": "Page-Hinkley", "current_mean": self.mean}
            )
        
        raise ValueError("PageHinkley expects 1D array")


class Cusum(BaseDriftDetector):
    """
    CUSUM (Cumulative Sum) control chart.
    
    Sequential analysis technique for monitoring
    change detection.
    """
    
    def __init__(self, threshold: float = 5, drift: float = 0.5):
        super().__init__(threshold)
        self.drift = drift
        self.mean = 0
        self.cusum_pos = 0
        self.cusum_neg = 0
        self.n = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit with reference data."""
        X = np.array(X)
        if X.ndim == 1:
            self.mean = np.mean(X)
            self.n = len(X)
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift using CUSUM."""
        X = np.array(X)
        if X.ndim == 1:
            max_cusum = 0
            drift_detected = False
            
            for val in X:
                self.n += 1
                # Update mean incrementally
                self.mean = (self.mean * (self.n - 1) + val) / self.n
                
                normalized_val = val - self.mean
                
                self.cusum_pos = max(0, self.cusum_pos + normalized_val - self.drift)
                self.cusum_neg = min(0, self.cusum_neg + normalized_val + self.drift)
                
                cusum_stat = max(self.cusum_pos, abs(self.cusum_neg))
                max_cusum = max(max_cusum, cusum_stat)
                
                if cusum_stat > self.threshold:
                    drift_detected = True
                    break
            
            return DriftResult(
                drift_detected=drift_detected,
                statistic=max_cusum,
                threshold=self.threshold,
                drift_type=DriftType.CONCEPT,
                severity=DriftSeverity.HIGH if drift_detected else DriftSeverity.NONE,
                metadata={
                    "algorithm": "CUSUM",
                    "cusum_pos": self.cusum_pos,
                    "cusum_neg": self.cusum_neg
                }
            )
        
        raise ValueError("Cusum expects 1D array")


# =============================================================================
# 3. FEATURE DRIFT DETECTORS
# =============================================================================

class FeatureDriftDetector(BaseDriftDetector):
    """
    Feature-level drift detector.
    
    Monitors individual features for drift using
    configurable statistical tests.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        test_type: str = "ks",
        feature_names: Optional[List[str]] = None
    ):
        super().__init__(threshold)
        self.test_type = test_type
        self.feature_names = feature_names
        self.feature_detectors: Dict[str, BaseDriftDetector] = {}
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit detectors for each feature."""
        X = np.array(X)
        n_features = X.shape[1] if X.ndim > 1 else 1
        
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        for i, name in enumerate(self.feature_names):
            if i < n_features:
                if self.test_type == "ks":
                    detector = KSDrift(self.threshold)
                elif self.test_type == "wasserstein":
                    detector = WassersteinDrift(self.threshold)
                elif self.test_type == "psi":
                    detector = PSI(self.threshold)
                elif self.test_type == "kl":
                    detector = KLDivergence(self.threshold)
                else:
                    detector = KSDrift(self.threshold)
                
                feature_data = X[:, i:i+1] if X.ndim > 1 else X.reshape(-1, 1)
                detector.fit(feature_data)
                self.feature_detectors[name] = detector
        
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift in all features."""
        X = self._validate_input(X)
        
        results = []
        any_drift = False
        max_stat = 0
        
        for i, (name, detector) in enumerate(self.feature_detectors.items()):
            if i < X.shape[1]:
                feature_data = X[:, i:i+1]
                result = detector.detect(feature_data)
                result.feature_name = name
                results.append(result)
                
                if result.drift_detected:
                    any_drift = True
                    max_stat = max(max_stat, result.statistic)
        
        return DriftResult(
            drift_detected=any_drift,
            statistic=max_stat,
            threshold=self.threshold,
            drift_type=DriftType.FEATURE,
            severity=DriftSeverity.HIGH if any_drift else DriftSeverity.NONE,
            metadata={
                "test": self.test_type,
                "n_features": len(self.feature_detectors),
                "feature_results": [r.to_dict() for r in results]
            }
        )


class FeatureImportanceDrift(BaseDriftDetector):
    """
    Monitor changes in feature importance over time.
    
    Uses model coefficients or permutation importance
    to detect shifts in feature relevance.
    """
    
    def __init__(self, threshold: float = 0.2, importance_method: str = "coefficient"):
        super().__init__(threshold)
        self.importance_method = importance_method
        self.reference_importance: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        model: Optional[Any] = None
    ):
        """Fit with reference data and model."""
        if model is None:
            raise ValueError("FeatureImportanceDrift requires a fitted model")
        
        # Extract feature importance
        if hasattr(model, 'feature_importances_'):
            self.reference_importance = np.array(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            self.reference_importance = np.abs(model.coef_)
            if self.reference_importance.ndim > 1:
                self.reference_importance = self.reference_importance.flatten()
        else:
            raise ValueError("Model must have feature_importances_ or coef_ attribute")
        
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray, model: Any = None) -> DriftResult:
        """Detect changes in feature importance."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if model is None:
            raise ValueError("Model required for importance comparison")
        
        # Get current importance
        if hasattr(model, 'feature_importances_'):
            current_importance = np.array(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            current_importance = np.abs(model.coef_)
            if current_importance.ndim > 1:
                current_importance = current_importance.flatten()
        else:
            raise ValueError("Model must have feature_importances_ or coef_ attribute")
        
        # Normalize
        ref_norm = self.reference_importance / (np.sum(self.reference_importance) + 1e-10)
        curr_norm = current_importance / (np.sum(current_importance) + 1e-10)
        
        # Calculate distance
        distance = np.mean(np.abs(ref_norm - curr_norm))
        drift_detected = distance > self.threshold
        
        # Identify most changed features
        diff = np.abs(ref_norm - curr_norm)
        top_changed = np.argsort(diff)[-5:][::-1]
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=distance,
            threshold=self.threshold,
            drift_type=DriftType.FEATURE,
            severity=DriftSeverity.HIGH if distance > 2 * self.threshold else 
                     (DriftSeverity.MEDIUM if drift_detected else DriftSeverity.NONE),
            metadata={
                "test": "FeatureImportance",
                "distance": distance,
                "top_changed_features": top_changed.tolist()
            }
        )


class CovariateShift(BaseDriftDetector):
    """
    Detect covariate shift using domain classification.
    
    Trains a classifier to distinguish between reference
    and test distributions.
    """
    
    def __init__(
        self,
        threshold: float = 0.6,
        classifier: Optional[Any] = None,
        n_permutations: int = 100
    ):
        super().__init__(threshold)
        self.classifier = classifier
        self.n_permutations = n_permutations
        self.fitted_classifier = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit with reference data."""
        self.reference_data = np.array(X)
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray, classifier: Any = None) -> DriftResult:
        """Detect covariate shift using domain classification."""
        X = self._validate_input(X)
        
        # Use provided classifier or default
        clf = classifier or self.classifier
        if clf is None:
            # Default: simple logistic regression-like approach
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        # Create domain labels
        n_ref = len(self.reference_data)
        n_test = len(X)
        
        X_combined = np.vstack([self.reference_data, X])
        y_domain = np.hstack([np.zeros(n_ref), np.ones(n_test)])
        
        # Train domain classifier
        clf.fit(X_combined, y_domain)
        y_pred = clf.predict_proba(X_combined)[:, 1]
        
        # Calculate accuracy on test set
        test_pred = y_pred[n_ref:]
        accuracy = np.mean((test_pred > 0.5).astype(int) == 1)
        
        # Permutation test for significance
        acc_permuted = []
        for _ in range(self.n_permutations):
            y_shuffled = np.random.permutation(y_domain)
            clf.fit(X_combined, y_shuffled)
            y_pred_perm = clf.predict_proba(X_combined)[:, 1]
            test_pred_perm = y_pred_perm[n_ref:]
            acc_perm = np.mean((test_pred_perm > 0.5).astype(int) == 1)
            acc_permuted.append(acc_perm)
        
        p_value = np.mean(np.array(acc_permuted) >= accuracy)
        drift_detected = accuracy > self.threshold
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=accuracy,
            p_value=p_value,
            threshold=self.threshold,
            drift_type=DriftType.FEATURE,
            severity=DriftSeverity.HIGH if accuracy > 0.7 else 
                     (DriftSeverity.MEDIUM if drift_detected else DriftSeverity.NONE),
            metadata={
                "test": "CovariateShift",
                "domain_accuracy": accuracy,
                "p_value": p_value
            }
        )


# =============================================================================
# 4. PREDICTION DRIFT DETECTORS
# =============================================================================

class PredictionDrift(BaseDriftDetector):
    """
    Monitor drift in model predictions.
    
    Tracks changes in output distribution over time.
    """
    
    def __init__(self, threshold: float = 0.05, test_type: str = "ks"):
        super().__init__(threshold)
        self.test_type = test_type
        self.sub_detector = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit with reference predictions."""
        X = np.array(X)
        
        if self.test_type == "ks":
            self.sub_detector = KSDrift(self.threshold)
        elif self.test_type == "psi":
            self.sub_detector = PSI(self.threshold)
        elif self.test_type == "wasserstein":
            self.sub_detector = WassersteinDrift(self.threshold)
        else:
            self.sub_detector = KSDrift(self.threshold)
        
        self.sub_detector.fit(X)
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift in predictions."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        result = self.sub_detector.detect(X)
        result.drift_type = DriftType.PREDICTION
        result.metadata["monitoring_target"] = "predictions"
        
        return result


class ConfidenceDrift(BaseDriftDetector):
    """
    Monitor drift in model confidence scores.
    
    Tracks changes in prediction confidence/probability
    distributions.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        confidence_type: str = "max_prob",
        test_type: str = "ks"
    ):
        super().__init__(threshold)
        self.confidence_type = confidence_type
        self.test_type = test_type
        self.sub_detector = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit with reference confidence scores."""
        X = np.array(X)
        
        # Extract confidence
        if X.ndim > 1 and X.shape[1] > 1:
            # Multi-class probabilities
            if self.confidence_type == "max_prob":
                confidence = np.max(X, axis=1)
            elif self.confidence_type == "entropy":
                confidence = -np.sum(X * np.log(X + 1e-10), axis=1)
            else:
                confidence = np.max(X, axis=1)
        else:
            confidence = X.flatten() if X.ndim > 1 else X
        
        if self.test_type == "ks":
            self.sub_detector = KSDrift(self.threshold)
        elif self.test_type == "psi":
            self.sub_detector = PSI(self.threshold)
        else:
            self.sub_detector = KSDrift(self.threshold)
        
        self.sub_detector.fit(confidence)
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray) -> DriftResult:
        """Detect drift in confidence scores."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        # Extract confidence
        X = np.array(X)
        if X.ndim > 1 and X.shape[1] > 1:
            if self.confidence_type == "max_prob":
                confidence = np.max(X, axis=1)
            elif self.confidence_type == "entropy":
                confidence = -np.sum(X * np.log(X + 1e-10), axis=1)
            else:
                confidence = np.max(X, axis=1)
        else:
            confidence = X.flatten() if X.ndim > 1 else X
        
        result = self.sub_detector.detect(confidence)
        result.drift_type = DriftType.CONFIDENCE
        result.metadata["confidence_type"] = self.confidence_type
        
        return result


class CalibrationDrift(BaseDriftDetector):
    """
    Monitor drift in model calibration.
    
    Tracks how well predicted probabilities match
    actual outcomes over time.
    """
    
    def __init__(self, threshold: float = 0.1, n_bins: int = 10):
        super().__init__(threshold)
        self.n_bins = n_bins
        self.reference_ece = 0
        self.reference_calib_curve = None
    
    def _expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        bin_accs = []
        bin_confs = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                bin_accs.append(accuracy_in_bin)
                bin_confs.append(avg_confidence_in_bin)
            else:
                bin_accs.append(0)
                bin_confs.append(0)
        
        return ece, np.array(bin_accs), np.array(bin_confs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit with reference predictions and labels."""
        y_prob = np.array(X)
        y_true = np.array(y)
        
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else np.max(y_prob, axis=1)
        
        self.reference_ece, accs, confs = self._expected_calibration_error(y_true, y_prob)
        self.reference_calib_curve = (accs, confs)
        self.is_fitted = True
        return self
    
    def detect(self, X: np.ndarray, y: np.ndarray) -> DriftResult:
        """Detect drift in calibration."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        y_prob = np.array(X)
        y_true = np.array(y)
        
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else np.max(y_prob, axis=1)
        
        current_ece, accs, confs = self._expected_calibration_error(y_true, y_prob)
        
        # Calculate relative change
        if self.reference_ece > 0:
            relative_change = abs(current_ece - self.reference_ece) / self.reference_ece
        else:
            relative_change = current_ece
        
        drift_detected = relative_change > self.threshold
        
        return DriftResult(
            drift_detected=drift_detected,
            statistic=relative_change,
            threshold=self.threshold,
            drift_type=DriftType.CALIBRATION,
            severity=DriftSeverity.HIGH if relative_change > 2 * self.threshold else 
                     (DriftSeverity.MEDIUM if drift_detected else DriftSeverity.NONE),
            metadata={
                "reference_ece": self.reference_ece,
                "current_ece": current_ece,
                "calibration_curve": {"accuracies": accs.tolist(), "confidences": confs.tolist()}
            }
        )


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

class DriftVisualizer:
    """
    Visualization tools for drift detection results.
    
    Creates plots for distributions, time series, and
    comparison charts.
    """
    
    def __init__(self, style: str = "seaborn"):
        self.style = style
        self._check_matplotlib()
    
    def _check_matplotlib(self):
        """Check if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            self.has_matplotlib = False
            warnings.warn("Matplotlib not available. Visualization will be limited.")
    
    def plot_distribution_comparison(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        feature_name: str = "Feature",
        bins: int = 30,
        figsize: Tuple[int, int] = (10, 5)
    ):
        """Plot side-by-side distribution comparison."""
        if not self.has_matplotlib:
            raise ImportError("Matplotlib required for visualization")
        
        fig, axes = self.plt.subplots(1, 2, figsize=figsize)
        
        # Histograms
        axes[0].hist(reference, bins=bins, alpha=0.7, label="Reference", density=True)
        axes[0].hist(test, bins=bins, alpha=0.7, label="Test", density=True)
        axes[0].set_xlabel(feature_name)
        axes[0].set_ylabel("Density")
        axes[0].set_title("Distribution Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plots
        axes[1].boxplot([reference, test], labels=["Reference", "Test"])
        axes[1].set_ylabel(feature_name)
        axes[1].set_title("Box Plot Comparison")
        axes[1].grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        return fig
    
    def plot_drift_over_time(
        self,
        timestamps: List[datetime],
        drift_scores: List[float],
        threshold: float = 0.05,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """Plot drift scores over time."""
        if not self.has_matplotlib:
            raise ImportError("Matplotlib required for visualization")
        
        fig, ax = self.plt.subplots(figsize=figsize)
        
        ax.plot(timestamps, drift_scores, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        ax.fill_between(timestamps, 0, threshold, alpha=0.2, color='green')
        ax.fill_between(timestamps, threshold, max(drift_scores + [threshold * 2]), 
                        alpha=0.2, color='red')
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Drift Score")
        ax.set_title("Drift Detection Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.plt.xticks(rotation=45)
        self.plt.tight_layout()
        return fig
    
    def plot_feature_drift_heatmap(
        self,
        feature_names: List[str],
        drift_scores: List[float],
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot heatmap of feature drift scores."""
        if not self.has_matplotlib:
            raise ImportError("Matplotlib required for visualization")
        
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Sort by drift score
        sorted_indices = np.argsort(drift_scores)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_scores = [drift_scores[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        colors = ['red' if s > 0.1 else 'orange' if s > 0.05 else 'green' for s in sorted_scores]
        ax.barh(range(len(sorted_names)), sorted_scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Drift Score")
        ax.set_title("Feature Drift Scores (Sorted)")
        ax.grid(True, alpha=0.3, axis='x')
        
        self.plt.tight_layout()
        return fig
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (8, 8)
    ):
        """Plot reliability/calibration curve."""
        if not self.has_matplotlib:
            raise ImportError("Matplotlib required for visualization")
        
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        fig, ax = self.plt.subplots(figsize=figsize)
        
        ax.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
        ax.plot(prob_pred, prob_true, 's-', label="Model", markersize=8, linewidth=2)
        
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve (Reliability Diagram)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        return fig
    
    def save_plot(self, fig, filepath: str, dpi: int = 150):
        """Save plot to file."""
        if not self.has_matplotlib:
            raise ImportError("Matplotlib required for visualization")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


class DistributionComparison:
    """
    Compare distributions with statistical summaries.
    """
    
    def __init__(self):
        pass
    
    def compare(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        feature_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare two distributions statistically."""
        ref = np.array(reference)
        test = np.array(test)
        
        comparison = {
            "feature": feature_name or "unnamed",
            "reference": {
                "mean": float(np.mean(ref)),
                "std": float(np.std(ref)),
                "median": float(np.median(ref)),
                "min": float(np.min(ref)),
                "max": float(np.max(ref)),
                "skewness": float(stats.skew(ref)),
                "kurtosis": float(stats.kurtosis(ref)),
                "n_samples": len(ref)
            },
            "test": {
                "mean": float(np.mean(test)),
                "std": float(np.std(test)),
                "median": float(np.median(test)),
                "min": float(np.min(test)),
                "max": float(np.max(test)),
                "skewness": float(stats.skew(test)),
                "kurtosis": float(stats.kurtosis(test)),
                "n_samples": len(test)
            },
            "differences": {
                "mean_diff": float(np.mean(test) - np.mean(ref)),
                "std_ratio": float(np.std(test) / (np.std(ref) + 1e-10)),
                "median_diff": float(np.median(test) - np.median(ref))
            },
            "statistical_tests": {}
        }
        
        # KS test
        ks_stat, ks_p = ks_2samp(ref, test)
        comparison["statistical_tests"]["kolmogorov_smirnov"] = {
            "statistic": float(ks_stat),
            "p_value": float(ks_p),
            "significant": ks_p < 0.05
        }
        
        # Wasserstein distance
        w_dist = wasserstein_distance(ref, test)
        comparison["statistical_tests"]["wasserstein"] = {
            "distance": float(w_dist),
            "normalized": float(w_dist / (np.std(ref) + 1e-10))
        }
        
        return comparison
    
    def compare_multiple(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Compare multiple features."""
        ref = np.array(reference)
        test = np.array(test)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(ref.shape[1])]
        
        comparisons = []
        for i, name in enumerate(feature_names):
            if i < ref.shape[1]:
                comp = self.compare(ref[:, i], test[:, i], name)
                comparisons.append(comp)
        
        return comparisons


class TimeSeriesDrift:
    """
    Analyze drift in time series data.
    """
    
    def __init__(self, window_size: int = 100, step_size: int = 10):
        self.window_size = window_size
        self.step_size = step_size
        self.drift_history = []
    
    def analyze(
        self,
        data: np.ndarray,
        detector: BaseDriftDetector,
        timestamps: Optional[List[datetime]] = None
    ) -> List[Dict[str, Any]]:
        """Analyze drift over time with sliding windows."""
        data = np.array(data)
        n_samples = len(data)
        
        results = []
        
        for start in range(0, n_samples - self.window_size, self.step_size):
            end = start + self.window_size
            window_data = data[start:end]
            
            # Use first window as reference
            if start == 0:
                detector.fit(window_data)
                continue
            
            result = detector.detect(window_data)
            
            entry = {
                "window_start": start,
                "window_end": end,
                "timestamp": timestamps[end] if timestamps and end < len(timestamps) else None,
                "drift_detected": result.drift_detected,
                "drift_score": result.statistic,
                "severity": result.severity.value
            }
            
            results.append(entry)
            self.drift_history.append(entry)
        
        return results
    
    def get_drift_points(self) -> List[int]:
        """Get indices where drift was detected."""
        return [h["window_end"] for h in self.drift_history if h["drift_detected"]]
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of time series drift analysis."""
        if not self.drift_history:
            return {"message": "No drift analysis performed"}
        
        total_windows = len(self.drift_history)
        drift_windows = sum(1 for h in self.drift_history if h["drift_detected"])
        
        return {
            "total_windows": total_windows,
            "drift_windows": drift_windows,
            "drift_rate": drift_windows / total_windows if total_windows > 0 else 0,
            "first_drift": next((h["window_end"] for h in self.drift_history if h["drift_detected"]), None),
            "severity_distribution": {
                "high": sum(1 for h in self.drift_history if h["severity"] == "high"),
                "medium": sum(1 for h in self.drift_history if h["severity"] == "medium"),
                "low": sum(1 for h in self.drift_history if h["severity"] == "low")
            }
        }


# =============================================================================
# 6. ALERTS
# =============================================================================

class DriftAlert:
    """
    Base class for drift alerts.
    
    Sends notifications when drift is detected.
    """
    
    def __init__(
        self,
        name: str = "DriftAlert",
        severity_threshold: DriftSeverity = DriftSeverity.MEDIUM
    ):
        self.name = name
        self.severity_threshold = severity_threshold
        self.alert_history: List[Dict[str, Any]] = []
    
    def should_alert(self, result: DriftResult) -> bool:
        """Check if alert should be sent based on severity."""
        severity_order = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4
        }
        return severity_order.get(result.severity, 0) >= severity_order.get(self.severity_threshold, 0)
    
    @abstractmethod
    def send(self, result: DriftResult, context: Optional[Dict[str, Any]] = None):
        """Send the alert."""
        pass
    
    def log_alert(self, result: DriftResult, status: str, details: str = ""):
        """Log alert to history."""
        self.alert_history.append({
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "drift_type": result.drift_type.value,
            "severity": result.severity.value,
            "details": details
        })


class EmailAlert(DriftAlert):
    """
    Email-based drift alerts.
    """
    
    def __init__(
        self,
        recipient_emails: List[str],
        smtp_server: str = "localhost",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sender_email: Optional[str] = None,
        severity_threshold: DriftSeverity = DriftSeverity.MEDIUM
    ):
        super().__init__("EmailAlert", severity_threshold)
        self.recipient_emails = recipient_emails
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender_email = sender_email or username
    
    def _format_email(self, result: DriftResult, context: Optional[Dict] = None) -> str:
        """Format alert as HTML email."""
        html = f"""
        <html>
        <body>
            <h2>🚨 Drift Alert: {result.severity.value.upper()}</h2>
            <p><strong>Time:</strong> {result.timestamp.isoformat()}</p>
            <p><strong>Drift Type:</strong> {result.drift_type.value}</p>
            <p><strong>Statistic:</strong> {result.statistic:.4f}</p>
            <p><strong>Threshold:</strong> {result.threshold}</p>
            """
        
        if result.p_value is not None:
            html += f"<p><strong>P-Value:</strong> {result.p_value:.4f}</p>"
        
        if result.feature_name:
            html += f"<p><strong>Feature:</strong> {result.feature_name}</p>"
        
        if context:
            html += "<h3>Context:</h3><ul>"
            for key, value in context.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        html += """
            <hr>
            <p><em>This is an automated alert from Fishstick Drift Detection.</em></p>
        </body>
        </html>
        """
        return html
    
    def send(self, result: DriftResult, context: Optional[Dict[str, Any]] = None):
        """Send email alert."""
        if not self.should_alert(result):
            self.log_alert(result, "skipped", "Severity below threshold")
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Drift Alert: {result.severity.value.upper()} - {result.drift_type.value}"
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            
            html_content = self._format_email(result, context)
            msg.attach(MIMEText(html_content, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.username and self.password:
                    server.starttls()
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            self.log_alert(result, "sent", f"Email sent to {len(self.recipient_emails)} recipients")
            
        except Exception as e:
            self.log_alert(result, "failed", str(e))
            warnings.warn(f"Failed to send email alert: {e}")


class SlackAlert(DriftAlert):
    """
    Slack-based drift alerts.
    """
    
    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "Fishstick Drift Bot",
        severity_threshold: DriftSeverity = DriftSeverity.MEDIUM
    ):
        super().__init__("SlackAlert", severity_threshold)
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
    
    def _format_slack_message(
        self,
        result: DriftResult,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Format alert as Slack message."""
        color = {
            DriftSeverity.CRITICAL: "#FF0000",
            DriftSeverity.HIGH: "#FF8C00",
            DriftSeverity.MEDIUM: "#FFD700",
            DriftSeverity.LOW: "#1E90FF",
            DriftSeverity.NONE: "#808080"
        }.get(result.severity, "#808080")
        
        fields = [
            {"title": "Drift Type", "value": result.drift_type.value, "short": True},
            {"title": "Statistic", "value": f"{result.statistic:.4f}", "short": True},
            {"title": "Threshold", "value": str(result.threshold), "short": True}
        ]
        
        if result.p_value is not None:
            fields.append({"title": "P-Value", "value": f"{result.p_value:.4f}", "short": True})
        
        if result.feature_name:
            fields.append({"title": "Feature", "value": result.feature_name, "short": True})
        
        message = {
            "username": self.username,
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": color,
                    "title": f"Drift Alert: {result.severity.value.upper()}",
                    "fields": fields,
                    "footer": "Fishstick Drift Detection",
                    "ts": int(time.time())
                }
            ]
        }
        
        if self.channel:
            message["channel"] = self.channel
        
        return message
    
    def send(self, result: DriftResult, context: Optional[Dict[str, Any]] = None):
        """Send Slack alert."""
        if not self.should_alert(result):
            self.log_alert(result, "skipped", "Severity below threshold")
            return
        
        try:
            import urllib.request
            import urllib.parse
            
            message = self._format_slack_message(result, context)
            data = json.dumps(message).encode('utf-8')
            
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    self.log_alert(result, "sent", "Slack message delivered")
                else:
                    self.log_alert(result, "failed", f"HTTP {response.status}")
                    
        except Exception as e:
            self.log_alert(result, "failed", str(e))
            warnings.warn(f"Failed to send Slack alert: {e}")


class ThresholdAlert(DriftAlert):
    """
    Alert based on custom threshold conditions.
    """
    
    def __init__(
        self,
        threshold_condition: Callable[[DriftResult], bool],
        alert_callback: Optional[Callable[[DriftResult, Dict], None]] = None,
        name: str = "ThresholdAlert"
    ):
        super().__init__(name, DriftSeverity.LOW)
        self.threshold_condition = threshold_condition
        self.alert_callback = alert_callback
    
    def should_alert(self, result: DriftResult) -> bool:
        """Check if custom threshold condition is met."""
        try:
            return self.threshold_condition(result)
        except Exception as e:
            warnings.warn(f"Error in threshold condition: {e}")
            return False
    
    def send(self, result: DriftResult, context: Optional[Dict[str, Any]] = None):
        """Trigger alert callback."""
        if not self.should_alert(result):
            self.log_alert(result, "skipped", "Threshold condition not met")
            return
        
        try:
            if self.alert_callback:
                self.alert_callback(result, context or {})
                self.log_alert(result, "sent", "Callback executed")
            else:
                print(f"[THRESHOLD ALERT] {result.drift_type.value}: {result.statistic:.4f}")
                self.log_alert(result, "sent", "Console output")
        except Exception as e:
            self.log_alert(result, "failed", str(e))


# =============================================================================
# 7. REMEDIATION
# =============================================================================

class RetrainingTrigger:
    """
    Automated model retraining triggers.
    
    Initiates retraining based on drift detection results.
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.1,
        min_samples: int = 1000,
        max_delay_minutes: int = 60,
        retraining_callback: Optional[Callable] = None
    ):
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.max_delay_minutes = max_delay_minutes
        self.retraining_callback = retraining_callback
        self.trigger_history = []
        self.pending_retrain = False
    
    def should_trigger(self, result: DriftResult, n_new_samples: int) -> bool:
        """Determine if retraining should be triggered."""
        if result.drift_detected and result.statistic > self.drift_threshold:
            if n_new_samples >= self.min_samples:
                return True
        return False
    
    def trigger(
        self,
        result: DriftResult,
        training_data: Optional[Any] = None,
        metadata: Optional[Dict] = None
    ):
        """Trigger model retraining."""
        trigger_info = {
            "timestamp": datetime.now().isoformat(),
            "drift_result": result.to_dict(),
            "triggered": True,
            "metadata": metadata or {}
        }
        
        if self.retraining_callback:
            try:
                self.retraining_callback(training_data, result)
                trigger_info["status"] = "success"
            except Exception as e:
                trigger_info["status"] = "failed"
                trigger_info["error"] = str(e)
        else:
            trigger_info["status"] = "callback_not_set"
            warnings.warn("Retraining triggered but no callback set")
        
        self.trigger_history.append(trigger_info)
        self.pending_retrain = False
        
        return trigger_info
    
    def get_trigger_history(self) -> List[Dict]:
        """Get history of retraining triggers."""
        return self.trigger_history


class EnsembleUpdate:
    """
    Update ensemble models based on drift.
    
    Manages ensemble weights and model selection
    in response to drift.
    """
    
    def __init__(
        self,
        models: List[Any],
        update_strategy: str = "weighted",
        weight_decay: float = 0.9
    ):
        self.models = models
        self.update_strategy = update_strategy
        self.weight_decay = weight_decay
        self.weights = np.ones(len(models)) / len(models)
        self.performance_history = [[] for _ in models]
    
    def update_weights(self, performances: List[float]):
        """Update ensemble weights based on recent performance."""
        performances = np.array(performances)
        
        if self.update_strategy == "weighted":
            # Weight by performance with decay
            new_weights = performances / (np.sum(performances) + 1e-10)
            self.weights = (self.weight_decay * self.weights + 
                          (1 - self.weight_decay) * new_weights)
        elif self.update_strategy == "best":
            # Use only best performing model
            best_idx = np.argmax(performances)
            self.weights = np.zeros(len(self.models))
            self.weights[best_idx] = 1.0
        elif self.update_strategy == "adaptive":
            # Adaptive weighting based on recent accuracy
            for i, perf in enumerate(performances):
                self.performance_history[i].append(perf)
                if len(self.performance_history[i]) > 10:
                    self.performance_history[i].pop(0)
            
            avg_perfs = [np.mean(h) if h else 0.5 for h in self.performance_history]
            self.weights = np.array(avg_perfs) / (np.sum(avg_perfs) + 1e-10)
        
        # Normalize
        self.weights = self.weights / (np.sum(self.weights) + 1e-10)
    
    def get_weighted_prediction(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Get ensemble prediction using current weights."""
        weighted_sum = np.zeros_like(predictions[0])
        for weight, pred in zip(self.weights, predictions):
            weighted_sum += weight * pred
        return weighted_sum
    
    def get_weights(self) -> np.ndarray:
        """Get current ensemble weights."""
        return self.weights.copy()


class CalibrationUpdate:
    """
    Update model calibration based on drift.
    
    Recalibrates models using temperature scaling
    or isotonic regression.
    """
    
    def __init__(
        self,
        method: str = "temperature",
        trigger_threshold: float = 0.05
    ):
        self.method = method
        self.trigger_threshold = trigger_threshold
        self.calibration_params = None
        self.calibration_history = []
    
    def fit_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        method: Optional[str] = None
    ):
        """Fit calibration on new data."""
        method = method or self.method
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        if method == "temperature":
            self.calibration_params = self._fit_temperature_scaling(y_true, y_prob)
        elif method == "isotonic":
            self.calibration_params = self._fit_isotonic(y_true, y_prob)
        elif method == "platt":
            self.calibration_params = self._fit_platt_scaling(y_true, y_prob)
        
        self.calibration_history.append({
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "params": self.calibration_params
        })
        
        return self.calibration_params
    
    def _fit_temperature_scaling(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Fit temperature scaling parameter."""
        from scipy.optimize import minimize_scalar
        
        def nll(T):
            scaled = self._apply_temperature(y_prob, T)
            # Binary cross-entropy
            eps = 1e-15
            scaled = np.clip(scaled, eps, 1 - eps)
            return -np.mean(y_true * np.log(scaled) + (1 - y_true) * np.log(1 - scaled))
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        return result.x
    
    def _apply_temperature(self, y_prob: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling."""
        # Convert to logits, scale, convert back
        logits = np.log(y_prob + 1e-10) - np.log(1 - y_prob + 1e-10)
        scaled_logits = logits / temperature
        return 1 / (1 + np.exp(-scaled_logits))
    
    def _fit_isotonic(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Any:
        """Fit isotonic regression calibration."""
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(y_prob, y_true)
            return iso
        except ImportError:
            warnings.warn("sklearn not available for isotonic calibration")
            return None
    
    def _fit_platt_scaling(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[float, float]:
        """Fit Platt scaling parameters."""
        from scipy.optimize import minimize
        
        def nll(params):
            a, b = params
            scaled = 1 / (1 + np.exp(-(a * y_prob + b)))
            eps = 1e-15
            scaled = np.clip(scaled, eps, 1 - eps)
            return -np.mean(y_true * np.log(scaled) + (1 - y_true) * np.log(1 - scaled))
        
        result = minimize(nll, x0=[1.0, 0.0], method='L-BFGS-B')
        return tuple(result.x)
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        if self.calibration_params is None:
            return y_prob
        
        if self.method == "temperature":
            return self._apply_temperature(y_prob, self.calibration_params)
        elif self.method == "isotonic" and self.calibration_params is not None:
            return self.calibration_params.predict(y_prob)
        elif self.method == "platt":
            a, b = self.calibration_params
            return 1 / (1 + np.exp(-(a * y_prob + b)))
        
        return y_prob


# =============================================================================
# 8. UTILITY FUNCTIONS
# =============================================================================

class DriftDetector:
    """
    Unified drift detection interface.
    
    Combines multiple drift detection methods into
    a single configurable interface.
    """
    
    def __init__(
        self,
        data_drift_detector: Optional[BaseDriftDetector] = None,
        concept_drift_detector: Optional[BaseDriftDetector] = None,
        feature_drift_detector: Optional[BaseDriftDetector] = None,
        prediction_drift_detector: Optional[BaseDriftDetector] = None,
        alerts: Optional[List[DriftAlert]] = None,
        visualizer: Optional[DriftVisualizer] = None
    ):
        self.data_drift_detector = data_drift_detector or KSDrift()
        self.concept_drift_detector = concept_drift_detector
        self.feature_drift_detector = feature_drift_detector
        self.prediction_drift_detector = prediction_drift_detector
        self.alerts = alerts or []
        self.visualizer = visualizer or DriftVisualizer()
        self.history = []
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit all detectors."""
        self.data_drift_detector.fit(X)
        
        if self.feature_drift_detector:
            self.feature_drift_detector.fit(X)
        
        if self.prediction_drift_detector and y is not None:
            self.prediction_drift_detector.fit(y)
        
        return self
    
    def detect(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None
    ) -> DriftReport:
        """Run all drift detection methods."""
        report = DriftReport()
        
        # Data drift
        result = self.data_drift_detector.detect(X)
        report.add_result(result)
        self._send_alerts(result)
        
        # Feature drift
        if self.feature_drift_detector:
            result = self.feature_drift_detector.detect(X)
            report.add_result(result)
            self._send_alerts(result)
        
        # Prediction drift
        if self.prediction_drift_detector and predictions is not None:
            result = self.prediction_drift_detector.detect(predictions)
            report.add_result(result)
            self._send_alerts(result)
        
        self.history.append(report)
        return report
    
    def _send_alerts(self, result: DriftResult):
        """Send alerts for detected drift."""
        for alert in self.alerts:
            alert.send(result)
    
    def add_alert(self, alert: DriftAlert):
        """Add an alert handler."""
        self.alerts.append(alert)
    
    def get_history(self) -> List[DriftReport]:
        """Get detection history."""
        return self.history


def detect_drift(
    reference_data: np.ndarray,
    test_data: np.ndarray,
    method: str = "ks",
    threshold: float = 0.05,
    **kwargs
) -> DriftResult:
    """
    Simple interface for drift detection.
    
    Parameters:
    -----------
    reference_data : np.ndarray
        Reference/baseline data
    test_data : np.ndarray
        Test data to check for drift
    method : str
        Detection method ('ks', 'psi', 'wasserstein', 'kl', 'chisquare')
    threshold : float
        Detection threshold
    **kwargs : dict
        Additional arguments for specific detectors
    
    Returns:
    --------
    DriftResult
        Drift detection result
    """
    # Map method names to detector classes
    detectors = {
        'ks': KSDrift,
        'kolmogorov_smirnov': KSDrift,
        'psi': PSI,
        'wasserstein': WassersteinDrift,
        'wasserstein_distance': WassersteinDrift,
        'kl': KLDivergence,
        'kl_divergence': KLDivergence,
        'chisquare': ChiSquareDrift,
        'chi_square': ChiSquareDrift,
        'adwin': ADWIN,
        'ddm': DDM,
        'eddm': EDDM,
        'pagehinkley': PageHinkley,
        'page_hinkley': PageHinkley,
        'cusum': Cusum
    }
    
    method_lower = method.lower().replace('-', '_')
    
    if method_lower not in detectors:
        raise ValueError(f"Unknown method: {method}. Available: {list(detectors.keys())}")
    
    detector_class = detectors[method_lower]
    detector = detector_class(threshold=threshold, **kwargs)
    
    detector.fit(reference_data)
    result = detector.detect(test_data)
    
    return result


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'DriftType',
    'DriftSeverity', 
    'AlertChannel',
    
    # Data classes
    'DriftResult',
    'DriftReport',
    
    # Base
    'BaseDriftDetector',
    
    # Data Drift
    'KSDrift',
    'ChiSquareDrift',
    'WassersteinDrift',
    'PSI',
    'KLDivergence',
    
    # Concept Drift
    'ADWIN',
    'DDM',
    'EDDM',
    'PageHinkley',
    'Cusum',
    
    # Feature Drift
    'FeatureDriftDetector',
    'FeatureImportanceDrift',
    'CovariateShift',
    
    # Prediction Drift
    'PredictionDrift',
    'ConfidenceDrift',
    'CalibrationDrift',
    
    # Visualization
    'DriftVisualizer',
    'DistributionComparison',
    'TimeSeriesDrift',
    
    # Alerts
    'DriftAlert',
    'EmailAlert',
    'SlackAlert',
    'ThresholdAlert',
    
    # Remediation
    'RetrainingTrigger',
    'EnsembleUpdate',
    'CalibrationUpdate',
    
    # Utilities
    'DriftDetector',
    'detect_drift'
]