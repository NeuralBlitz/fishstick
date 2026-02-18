"""
Statistical Inference for TDA.

Provides statistical methods for persistence diagrams including
hypothesis testing, confidence intervals, and bootstrap methods.
"""

from typing import List, Optional, Dict, Tuple
import torch
from torch import Tensor
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import percentileofscore


@dataclass
class PersistenceStatistic:
    """Statistical summary of persistence diagram."""

    mean_persistence: float
    std_persistence: float
    median_persistence: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None


class BootstrapConfidenceInterval:
    """
    Bootstrap Confidence Intervals for Persistence.

    Computes confidence intervals using bootstrap sampling
    for persistence-based statistics.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize bootstrap CI.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compute_ci(
        self,
        diagrams: List[Tensor],
        statistic_fn: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for persistence statistic.

        Args:
            diagrams: List of persistence diagrams
            statistic_fn: Function to compute statistic

        Returns:
            Tuple of (lower, upper) bounds
        """
        if statistic_fn is None:
            statistic_fn = self._default_statistic

        bootstrap_stats = []

        for _ in range(self.n_bootstrap):
            sample = self._bootstrap_sample(diagrams)
            stat = statistic_fn(sample)
            bootstrap_stats.append(stat)

        lower = np.percentile(bootstrap_stats, 100 * self.alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - self.alpha / 2))

        return float(lower), float(upper)

    def _bootstrap_sample(
        self,
        diagrams: List[Tensor],
    ) -> List[Tensor]:
        """Create bootstrap sample from diagrams."""
        n = len(diagrams)
        indices = np.random.choice(n, size=n, replace=True)

        return [diagrams[i] for i in indices]

    def _default_statistic(
        self,
        diagrams: List[Tensor],
    ) -> float:
        """Default statistic: mean persistence."""
        all_pers = []

        for diag in diagrams:
            if len(diag) > 0:
                persistences = (diag[:, 1] - diag[:, 0]).clamp(min=0)
                all_pers.extend(persistences.tolist())

        if len(all_pers) == 0:
            return 0.0

        return np.mean(all_pers)


class HypothesisTest:
    """
    Hypothesis Testing for Persistence Diagrams.

    Provides statistical tests for comparing
    topological features between groups.
    """

    def __init__(self, test_type: str = "permutation"):
        """
        Initialize hypothesis test.

        Args:
            test_type: Type of test ('permutation', 'bootstrap')
        """
        self.test_type = test_type

    def two_sample_test(
        self,
        group1_diagrams: List[Tensor],
        group2_diagrams: List[Tensor],
        n_permutations: int = 1000,
    ) -> Dict[str, float]:
        """
        Two-sample test between groups.

        Args:
            group1_diagrams: First group diagrams
            group2_diagrams: Second group diagrams
            n_permutations: Number of permutations

        Returns:
            Dictionary with test statistic and p-value
        """
        obs_stat = self._compute_test_statistic(group1_diagrams, group2_diagrams)

        combined = group1_diagrams + group2_diagrams
        n1 = len(group1_diagrams)

        perm_stats = []

        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_group1 = combined[:n1]
            perm_group2 = combined[n1:]

            perm_stat = self._compute_test_statistic(perm_group1, perm_group2)
            perm_stats.append(perm_stat)

        p_value = np.mean([abs(s) >= abs(obs_stat) for s in perm_stats])

        return {
            "statistic": obs_stat,
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    def _compute_test_statistic(
        self,
        group1: List[Tensor],
        group2: List[Tensor],
    ) -> float:
        """Compute test statistic."""
        stats1 = self._extract_statistics(group1)
        stats2 = self._extract_statistics(group2)

        if len(stats1) == 0 or len(stats2) == 0:
            return 0.0

        return abs(np.mean(stats1) - np.mean(stats2))

    def _extract_statistics(
        self,
        diagrams: List[Tensor],
    ) -> List[float]:
        """Extract persistence statistics from diagrams."""
        stats = []

        for diag in diagrams:
            if len(diag) > 0:
                pers = (diag[:, 1] - diag[:, 0]).clamp(min=0)
                stats.extend(pers.tolist())

        return stats


class PersistenceDistributionFitting:
    """
    Fits Statistical Distributions to Persistence Values.

    Provides methods for fitting and evaluating
    distribution fits to persistence data.
    """

    def __init__(self):
        pass

    def fit_exponential(
        self,
        persistences: Tensor,
    ) -> Dict[str, float]:
        """
        Fit exponential distribution to persistences.

        Args:
            persistences: Persistence values

        Returns:
            Dictionary with fitted parameters
        """
        pers_np = persistences.cpu().numpy()
        pers_np = pers_np[pers_np > 0]

        if len(pers_np) < 2:
            return {"rate": 0.0, "scale": 0.0}

        loc, scale = scipy_stats.expon.fit(pers_np, floc=0)

        return {
            "rate": 1.0 / scale if scale > 0 else 0.0,
            "scale": scale,
            "mean": scale,
        }

    def fit_pareto(
        self,
        persistences: Tensor,
    ) -> Dict[str, float]:
        """
        Fit Pareto distribution to persistences.

        Args:
            persistences: Persistence values

        Returns:
            Dictionary with fitted parameters
        """
        pers_np = persistences.cpu().numpy()
        pers_np = pers_np[pers_np > 0]

        if len(pers_np) < 2:
            return {"shape": 0.0, "scale": 0.0}

        shape, loc, scale = scipy_stats.pareto.fit(pers_np, floc=0)

        return {
            "shape": shape,
            "scale": scale,
        }

    def fit_gamma(
        self,
        persistences: Tensor,
    ) -> Dict[str, float]:
        """
        Fit Gamma distribution to persistences.

        Args:
            persistences: Persistence values

        Returns:
            Dictionary with fitted parameters
        """
        pers_np = persistences.cpu().numpy()
        pers_np = pers_np[pers_np > 0]

        if len(pers_np) < 2:
            return {"shape": 0.0, "scale": 0.0}

        shape, loc, scale = scipy_stats.gamma.fit(pers_np, floc=0)

        return {
            "shape": shape,
            "scale": scale,
        }


class StabilityAnalyzer:
    """
    Analyzes Stability of Persistence Features.

    Provides methods to assess stability of
    topological features under perturbations.
    """

    def __init__(self):
        pass

    def compute_stability_bound(
        self,
        diagram: Tensor,
        epsilon: float,
    ) -> Dict[str, float]:
        """
        Compute stability bound for diagram.

        Args:
            diagram: Persistence diagram
            epsilon: Perturbation size

        Returns:
            Stability bounds
        """
        if len(diagram) == 0:
            return {"bottleneck_bound": 0.0, "wasserstein_bound": 0.0}

        bottleneck_bound = epsilon
        wasserstein_bound = epsilon * np.sqrt(2)

        return {
            "bottleneck_bound": bottleneck_bound,
            "wasserstein_bound": wasserstein_bound,
        }

    def noise_sensitivity_analysis(
        self,
        points: Tensor,
        noise_levels: List[float],
    ) -> Dict[float, Dict[str, float]]:
        """
        Analyze sensitivity to noise.

        Args:
            points: Original points
            noise_levels: Levels of noise to test

        Returns:
            Dictionary of noise level to stability metrics
        """
        results = {}

        from .persistence import PersistentHomology

        ph = PersistentHomology()
        original_diagrams = ph.compute_from_distance(points)

        for noise_level in noise_levels:
            noise = torch.randn_like(points) * noise_level
            noisy_points = points + noise

            noisy_diagrams = ph.compute_from_distance(noisy_points)

            distance = self._compute_diagram_distance(original_diagrams, noisy_diagrams)

            results[noise_level] = distance

        return results

    def _compute_diagram_distance(
        self,
        diagrams1: List,
        diagrams2: List,
    ) -> Dict[str, float]:
        """Compute distance between diagram sets."""
        if len(diagrams1) == 0 or len(diagrams2) == 0:
            return {"bottleneck": 0.0, "wasserstein": 0.0}

        from .persistence import bottleneck_distance, wasserstein_distance

        distances = {"bottleneck": [], "wasserstein": []}

        for d1, d2 in zip(diagrams1, diagrams2):
            if d1 is not None and d2 is not None:
                distances["bottleneck"].append(bottleneck_distance(d1, d2))
                distances["wasserstein"].append(wasserstein_distance(d1, d2))

        return {
            "bottleneck": np.mean(distances["bottleneck"])
            if distances["bottleneck"]
            else 0.0,
            "wasserstein": np.mean(distances["wasserstein"])
            if distances["wasserstein"]
            else 0.0,
        }


class PersistenceFeatureExtractor:
    """
    Extracts Statistical Features from Persistence.

    Computes various statistical features from
    persistence diagrams for downstream analysis.
    """

    def __init__(self):
        pass

    def extract_features(
        self,
        diagram: Tensor,
    ) -> Dict[str, float]:
        """
        Extract statistical features.

        Args:
            diagram: Persistence diagram

        Returns:
            Dictionary of features
        """
        if len(diagram) == 0:
            return self._empty_features()

        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistences = (deaths - births).clamp(min=0)

        features = {
            "n_features": len(diagram),
            "mean_birth": births.mean().item(),
            "std_birth": births.std().item(),
            "mean_death": deaths.mean().item(),
            "std_death": deaths.std().item(),
            "mean_persistence": persistences.mean().item(),
            "std_persistence": persistences.std().item(),
            "max_persistence": persistences.max().item(),
            "min_persistence": persistences.min().item(),
            "total_persistence": persistences.sum().item(),
        }

        if len(persistences) > 1:
            features["persistence_entropy"] = self._compute_entropy(persistences)
            features["persistence_skewness"] = self._compute_skewness(persistences)
            features["persistence_kurtosis"] = self._compute_kurtosis(persistences)

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            "n_features": 0,
            "mean_birth": 0.0,
            "std_birth": 0.0,
            "mean_death": 0.0,
            "std_death": 0.0,
            "mean_persistence": 0.0,
            "std_persistence": 0.0,
            "max_persistence": 0.0,
            "min_persistence": 0.0,
            "total_persistence": 0.0,
            "persistence_entropy": 0.0,
            "persistence_skewness": 0.0,
            "persistence_kurtosis": 0.0,
        }

    def _compute_entropy(self, values: Tensor) -> float:
        """Compute entropy of values."""
        vals = values.cpu().numpy()
        vals = vals[vals > 0]

        if len(vals) == 0:
            return 0.0

        hist, _ = np.histogram(vals, bins=50)
        hist = hist / hist.sum()

        return -np.sum(hist * np.log(hist + 1e-10))

    def _compute_skewness(self, values: Tensor) -> float:
        """Compute skewness."""
        vals = values.cpu().numpy()

        if len(vals) < 3:
            return 0.0

        return scipy_stats.skew(vals)

    def _compute_kurtosis(self, values: Tensor) -> float:
        """Compute kurtosis."""
        vals = values.cpu().numpy()

        if len(vals) < 4:
            return 0.0

        return scipy_stats.kurtosis(vals)


def confidence_interval_bootstrap(
    diagrams: List[Tensor],
    statistic: str = "mean_persistence",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute confidence interval for persistence statistic.

    Args:
        diagrams: List of persistence diagrams
        statistic: Statistic to compute
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (lower, upper) bounds
    """
    boot = BootstrapConfidenceInterval(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence,
    )

    return boot.compute_ci(diagrams)


def permutation_test(
    group1: List[Tensor],
    group2: List[Tensor],
    n_permutations: int = 1000,
) -> Dict[str, float]:
    """
    Perform permutation test between two groups.

    Args:
        group1: First group diagrams
        group2: Second group diagrams
        n_permutations: Number of permutations

    Returns:
        Test results
    """
    test = HypothesisTest(test_type="permutation")

    return test.two_sample_test(group1, group2, n_permutations)
