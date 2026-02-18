"""
Comprehensive A/B Testing Module for Fishstick

This module provides a complete A/B testing framework including:
- Experiment design (A/B, multivariate, bandit, split tests)
- Randomization strategies
- Sample size calculation
- Statistical tests
- Metrics calculation
- Analysis tools
- Monitoring and early stopping
- Reporting and visualization
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ttest_ind,
    chi2_contingency,
    mannwhitneyu,
    norm,
    beta,
    binom,
    f as f_dist,
)
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from datetime import datetime, timedelta
import json
import hashlib
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Experiment Design
# ============================================================================


class ExperimentStatus(Enum):
    """Status of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class Variant:
    """Represents a test variant."""

    name: str
    id: str
    traffic_allocation: float = 0.5
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    id: str
    variants: List[Variant]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_sample_size: Optional[int] = None
    primary_metric: Optional[str] = None
    secondary_metrics: List[str] = field(default_factory=list)
    confidence_level: float = 0.95


class BaseExperiment(ABC):
    """Abstract base class for all experiment types."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.status = ExperimentStatus.DRAFT
        self.assignments: Dict[str, str] = {}
        self.results: Dict[str, List[float]] = defaultdict(list)
        self.created_at = datetime.now()

    @abstractmethod
    def assign_variant(self, user_id: str, **kwargs) -> str:
        """Assign a user to a variant."""
        pass

    @abstractmethod
    def record_outcome(self, user_id: str, metric: str, value: float):
        """Record an outcome for a user."""
        pass

    def start(self):
        """Start the experiment."""
        self.status = ExperimentStatus.RUNNING
        self.config.start_date = datetime.now()

    def stop(self):
        """Stop the experiment."""
        self.status = ExperimentStatus.STOPPED
        self.config.end_date = datetime.now()

    def pause(self):
        """Pause the experiment."""
        self.status = ExperimentStatus.PAUSED

    def get_variant_counts(self) -> Dict[str, int]:
        """Get the number of assignments per variant."""
        counts = defaultdict(int)
        for variant_id in self.assignments.values():
            counts[variant_id] += 1
        return dict(counts)


class ABTest(BaseExperiment):
    """
    Standard A/B test with control and treatment groups.

    Example:
        control = Variant(name="Control", id="control", traffic_allocation=0.5)
        treatment = Variant(name="Treatment", id="treatment", traffic_allocation=0.5)
        config = ExperimentConfig(name="Landing Page Test", id="lp_test", variants=[control, treatment])
        test = ABTest(config)
    """

    def __init__(self, config: ExperimentConfig, randomizer=None):
        super().__init__(config)
        if len(config.variants) != 2:
            raise ValueError("A/B test requires exactly 2 variants")
        self.randomizer = randomizer or SimpleRandomization()

    def assign_variant(self, user_id: str, **kwargs) -> str:
        """Assign user to control or treatment using the configured randomizer."""
        if user_id in self.assignments:
            return self.assignments[user_id]

        variant = self.randomizer.assign(
            user_id=user_id, variants=self.config.variants, **kwargs
        )
        self.assignments[user_id] = variant.id
        return variant.id

    def record_outcome(self, user_id: str, metric: str, value: float):
        """Record an outcome metric for a user."""
        if user_id not in self.assignments:
            raise ValueError(f"User {user_id} not assigned to any variant")
        variant_id = self.assignments[user_id]
        key = f"{variant_id}:{metric}"
        self.results[key].append(value)


class MultivariateTest(BaseExperiment):
    """
    Multi-variate test for testing multiple factors simultaneously.

    Creates combinations of factors and tests them all.

    Example:
        factors = {
            'button_color': ['red', 'blue'],
            'headline': ['A', 'B'],
            'image': ['1', '2']
        }
        test = MultivariateTest.from_factors("MVT Test", factors)
    """

    def __init__(self, config: ExperimentConfig, randomizer=None):
        super().__init__(config)
        self.randomizer = randomizer or SimpleRandomization()
        self.factor_combinations = self._extract_factor_combinations()

    @classmethod
    def from_factors(
        cls,
        name: str,
        factors: Dict[str, List[str]],
        traffic_allocation: float = 0.5,
        randomizer=None,
    ) -> "MultivariateTest":
        """Create a multivariate test from factor definitions."""
        from itertools import product

        combinations = list(product(*factors.values()))
        variants = []

        for i, combo in enumerate(combinations):
            variant_name = " | ".join(
                [f"{k}={v}" for k, v in zip(factors.keys(), combo)]
            )
            variant_id = f"variant_{i}"
            config_dict = dict(zip(factors.keys(), combo))
            variants.append(
                Variant(
                    name=variant_name,
                    id=variant_id,
                    traffic_allocation=1.0 / len(combinations),
                    config=config_dict,
                )
            )

        config = ExperimentConfig(
            name=name,
            id=f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}",
            variants=variants,
        )

        return cls(config, randomizer)

    def _extract_factor_combinations(self) -> List[Dict[str, str]]:
        """Extract factor combinations from variant configs."""
        return [v.config for v in self.config.variants]

    def assign_variant(self, user_id: str, **kwargs) -> str:
        """Assign user to a variant combination."""
        if user_id in self.assignments:
            return self.assignments[user_id]

        variant = self.randomizer.assign(
            user_id=user_id, variants=self.config.variants, **kwargs
        )
        self.assignments[user_id] = variant.id
        return variant.id

    def record_outcome(self, user_id: str, metric: str, value: float):
        """Record an outcome metric for a user."""
        if user_id not in self.assignments:
            raise ValueError(f"User {user_id} not assigned to any variant")
        variant_id = self.assignments[user_id]
        key = f"{variant_id}:{metric}"
        self.results[key].append(value)

    def get_factor_effects(self, metric: str) -> Dict[str, Dict[str, float]]:
        """Calculate the effect of each factor level."""
        factor_effects = defaultdict(lambda: defaultdict(list))

        for variant in self.config.variants:
            key = f"{variant.id}:{metric}"
            values = self.results.get(key, [])

            for factor, level in variant.config.items():
                factor_effects[factor][level].extend(values)

        # Calculate means for each factor level
        effects = {}
        for factor, levels in factor_effects.items():
            effects[factor] = {
                level: np.mean(values) if values else 0
                for level, values in levels.items()
            }

        return effects


class BanditTest(BaseExperiment):
    """
    Multi-armed bandit test using Thompson Sampling or epsilon-greedy.

    Automatically adjusts traffic allocation based on performance.

    Example:
        config = ExperimentConfig(name="Bandit Test", id="bandit_1", variants=[...])
        test = BanditTest(config, algorithm='thompson', min_samples=100)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        algorithm: str = "thompson",
        epsilon: float = 0.1,
        min_samples: int = 100,
        randomizer=None,
    ):
        super().__init__(config)
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.randomizer = randomizer or SimpleRandomization()

        # Initialize bandit statistics
        self.successes = {v.id: 0 for v in config.variants}
        self.failures = {v.id: 0 for v in config.variants}
        self.total_samples = {v.id: 0 for v in config.variants}

    def assign_variant(self, user_id: str, **kwargs) -> str:
        """Assign user using bandit algorithm."""
        if user_id in self.assignments:
            return self.assignments[user_id]

        # Exploration phase: random assignment until min samples
        total = sum(self.total_samples.values())
        if total < self.min_samples * len(self.config.variants):
            variant = self.randomizer.assign(
                user_id=user_id, variants=self.config.variants, **kwargs
            )
        else:
            # Exploitation phase: use bandit algorithm
            if self.algorithm == "thompson":
                variant = self._thompson_sampling()
            elif self.algorithm == "epsilon_greedy":
                variant = self._epsilon_greedy(user_id, **kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.assignments[user_id] = variant.id
        self.total_samples[variant.id] += 1
        return variant.id

    def _thompson_sampling(self) -> Variant:
        """Thompson sampling for binary outcomes."""
        samples = []
        for variant in self.config.variants:
            # Sample from Beta distribution
            sample = np.random.beta(
                self.successes[variant.id] + 1, self.failures[variant.id] + 1
            )
            samples.append((sample, variant))

        # Return variant with highest sample
        return max(samples, key=lambda x: x[0])[1]

    def _epsilon_greedy(self, user_id: str, **kwargs) -> Variant:
        """Epsilon-greedy algorithm."""
        if np.random.random() < self.epsilon:
            # Explore: random assignment
            return self.randomizer.assign(
                user_id=user_id, variants=self.config.variants, **kwargs
            )
        else:
            # Exploit: choose best performing variant
            best_variant = None
            best_rate = -1

            for variant in self.config.variants:
                total = self.total_samples[variant.id]
                if total > 0:
                    rate = self.successes[variant.id] / total
                    if rate > best_rate:
                        best_rate = rate
                        best_variant = variant

            return best_variant or self.config.variants[0]

    def record_outcome(self, user_id: str, metric: str, value: float):
        """Record an outcome. For bandits, typically binary (0/1)."""
        if user_id not in self.assignments:
            raise ValueError(f"User {user_id} not assigned to any variant")

        variant_id = self.assignments[user_id]
        key = f"{variant_id}:{metric}"
        self.results[key].append(value)

        # Update bandit statistics (assumes binary outcomes)
        if value > 0:
            self.successes[variant_id] += 1
        else:
            self.failures[variant_id] += 1

    def get_variant_probabilities(self) -> Dict[str, float]:
        """Get the probability of each variant being best."""
        n_samples = 10000
        counts = {v.id: 0 for v in self.config.variants}

        for _ in range(n_samples):
            samples = []
            for variant in self.config.variants:
                sample = np.random.beta(
                    self.successes[variant.id] + 1, self.failures[variant.id] + 1
                )
                samples.append((sample, variant.id))

            best = max(samples, key=lambda x: x[0])[1]
            counts[best] += 1

        return {k: v / n_samples for k, v in counts.items()}


class SplitTest(BaseExperiment):
    """
    Split URL testing for testing completely different page versions.

    Useful for testing major redesigns or different landing pages.

    Example:
        variant1 = Variant(name="Original", id="original", config={'url': '/page-a'})
        variant2 = Variant(name="Redesign", id="redesign", config={'url': '/page-b'})
        test = SplitTest(config)
    """

    def __init__(self, config: ExperimentConfig, randomizer=None):
        super().__init__(config)
        self.randomizer = randomizer or SimpleRandomization()

    def assign_variant(self, user_id: str, **kwargs) -> str:
        """Assign user to a URL variant."""
        if user_id in self.assignments:
            return self.assignments[user_id]

        variant = self.randomizer.assign(
            user_id=user_id, variants=self.config.variants, **kwargs
        )
        self.assignments[user_id] = variant.id
        return variant.id

    def record_outcome(self, user_id: str, metric: str, value: float):
        """Record an outcome metric for a user."""
        if user_id not in self.assignments:
            raise ValueError(f"User {user_id} not assigned to any variant")
        variant_id = self.assignments[user_id]
        key = f"{variant_id}:{metric}"
        self.results[key].append(value)

    def get_redirect_url(self, user_id: str) -> str:
        """Get the redirect URL for a user."""
        variant_id = self.assign_variant(user_id)
        variant = next(v for v in self.config.variants if v.id == variant_id)
        return variant.config.get("url", "")


# ============================================================================
# Randomization
# ============================================================================


class BaseRandomization(ABC):
    """Abstract base class for randomization strategies."""

    @abstractmethod
    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Assign a user to a variant."""
        pass


class SimpleRandomization(BaseRandomization):
    """
    Simple random assignment based on traffic allocation.

    Uses deterministic hashing for reproducibility.
    """

    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Assign user using simple randomization."""
        # Use hash for deterministic assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        rand_val = (hash_val % 1000) / 1000.0

        cumulative = 0
        for variant in variants:
            cumulative += variant.traffic_allocation
            if rand_val <= cumulative:
                return variant

        return variants[-1]


class StratifiedRandomization(BaseRandomization):
    """
    Stratified randomization to ensure balanced groups across segments.

    Example:
        randomizer = StratifiedRandomization(strata_keys=['country', 'device_type'])
    """

    def __init__(self, strata_keys: List[str], buffer_size: int = 100):
        self.strata_keys = strata_keys
        self.buffer_size = buffer_size
        self.strata_assignments: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Assign user using stratified randomization."""
        # Build strata key from user attributes
        strata_values = [str(kwargs.get(k, "unknown")) for k in self.strata_keys]
        strata_key = "|".join(strata_values)

        # Get current counts for this strata
        counts = self.strata_assignments[strata_key]
        total = sum(counts.values())

        # Find variant with lowest proportion
        min_prop = float("inf")
        selected_variant = variants[0]

        for variant in variants:
            current_count = counts[variant.id]
            current_prop = current_count / total if total > 0 else 0
            target_prop = variant.traffic_allocation

            # Calculate deviation from target
            deviation = current_prop - target_prop if total > 0 else -target_prop

            if deviation < min_prop:
                min_prop = deviation
                selected_variant = variant

        self.strata_assignments[strata_key][selected_variant.id] += 1
        return selected_variant


class ClusterRandomization(BaseRandomization):
    """
    Cluster randomization for assigning groups rather than individuals.

    Useful when treatment must be applied at group level (e.g., classrooms, clinics).

    Example:
        randomizer = ClusterRandomization(cluster_key='school_id')
    """

    def __init__(
        self, cluster_key: str, randomizer: Optional[BaseRandomization] = None
    ):
        self.cluster_key = cluster_key
        self.randomizer = randomizer or SimpleRandomization()
        self.cluster_assignments: Dict[str, str] = {}

    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Assign user based on their cluster."""
        cluster_id = str(kwargs.get(self.cluster_key, user_id))

        # Check if cluster already assigned
        if cluster_id in self.cluster_assignments:
            variant_id = self.cluster_assignments[cluster_id]
            return next(v for v in variants if v.id == variant_id)

        # Assign cluster to variant
        variant = self.randomizer.assign(cluster_id, variants, **kwargs)
        self.cluster_assignments[cluster_id] = variant.id
        return variant


class CovariateAdaptiveRandomization(BaseRandomization):
    """
    Covariate-adaptive randomization (Pocock-Simon) to balance covariates.

    Minimizes imbalances across multiple covariates simultaneously.

    Example:
        randomizer = CovariateAdaptiveRandomization(
            covariates=['age_group', 'gender', 'device_type'],
            p=0.8
        )
    """

    def __init__(self, covariates: List[str], p: float = 0.8):
        self.covariates = covariates
        self.p = p
        self.covariate_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Assign user using covariate-adaptive randomization."""
        # Calculate imbalance score for each variant
        scores = {}

        for variant in variants:
            score = 0
            for covariate in self.covariates:
                value = str(kwargs.get(covariate, "unknown"))
                counts = self.covariate_counts[covariate][value]

                # Sum counts of other variants
                other_counts = sum(c for v, c in counts.items() if v != variant.id)
                score += other_counts - counts.get(variant.id, 0)

            scores[variant.id] = score

        # Find variant with minimum score (best balance)
        min_score = min(scores.values())
        best_variants = [v for v in variants if scores[v.id] == min_score]

        # Assign with probability p to best, (1-p)/(n-1) to others
        if len(best_variants) == 1 and np.random.random() < self.p:
            selected = best_variants[0]
        else:
            selected = np.random.choice(variants)

        # Update counts
        for covariate in self.covariates:
            value = str(kwargs.get(covariate, "unknown"))
            self.covariate_counts[covariate][value][selected.id] += 1

        return selected


# ============================================================================
# Sample Size
# ============================================================================


class SampleSizeCalculator:
    """
    Calculate required sample sizes for experiments.

    Supports various test types and parameters.
    """

    @staticmethod
    def two_proportion_test(
        p1: float,
        p2: float,
        alpha: float = 0.05,
        power: float = 0.8,
        ratio: float = 1.0,
    ) -> Tuple[int, int]:
        """
        Calculate sample size for two-proportion z-test.

        Args:
            p1: Baseline proportion
            p2: Expected proportion under treatment
            alpha: Significance level
            power: Desired power
            ratio: Ratio of treatment to control sample size

        Returns:
            Tuple of (control_size, treatment_size)
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p_pooled = (p1 + ratio * p2) / (1 + ratio)
        delta = abs(p2 - p1)

        n1 = (
            (
                z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))
                + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
            )
            / delta
        ) ** 2

        n1 = int(np.ceil(n1))
        n2 = int(np.ceil(n1 * ratio))

        return n1, n2

    @staticmethod
    def two_sample_t_test(
        mu1: float,
        mu2: float,
        sigma: float,
        alpha: float = 0.05,
        power: float = 0.8,
        ratio: float = 1.0,
    ) -> Tuple[int, int]:
        """
        Calculate sample size for two-sample t-test.

        Args:
            mu1: Baseline mean
            mu2: Expected mean under treatment
            sigma: Pooled standard deviation
            alpha: Significance level
            power: Desired power
            ratio: Ratio of treatment to control sample size

        Returns:
            Tuple of (control_size, treatment_size)
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        delta = abs(mu2 - mu1)

        n1 = (z_alpha + z_beta) ** 2 * 2 * sigma**2 / delta**2

        n1 = int(np.ceil(n1))
        n2 = int(np.ceil(n1 * ratio))

        return n1, n2

    @staticmethod
    def anova_test(k: int, f: float, alpha: float = 0.05, power: float = 0.8) -> int:
        """
        Calculate sample size per group for ANOVA.

        Args:
            k: Number of groups
            f: Cohen's f effect size
            alpha: Significance level
            power: Desired power

        Returns:
            Sample size per group
        """
        from scipy.optimize import fsolve

        def power_func(n):
            df_between = k - 1
            df_within = k * (n - 1)
            nc = n * k * f**2

            critical_f = stats.f.ppf(1 - alpha, df_between, df_within)
            actual_power = 1 - stats.ncf.cdf(critical_f, df_between, df_within, nc)
            return actual_power - power

        try:
            n = fsolve(power_func, 50)[0]
            return int(np.ceil(max(n, 2)))
        except:
            return 100  # Default fallback


class PowerAnalysis:
    """
    Power analysis for various statistical tests.
    """

    @staticmethod
    def two_proportion_test_power(
        n1: int, n2: int, p1: float, p2: float, alpha: float = 0.05
    ) -> float:
        """Calculate power for two-proportion test."""
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
        se_null = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
        se_alt = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

        z = (p2 - p1) / se_alt
        critical_val = z_alpha * se_null / se_alt

        power = 1 - stats.norm.cdf(critical_val - z) + stats.norm.cdf(-critical_val - z)
        return power

    @staticmethod
    def two_sample_t_test_power(
        n1: int, n2: int, delta: float, sigma: float, alpha: float = 0.05
    ) -> float:
        """Calculate power for two-sample t-test."""
        df = n1 + n2 - 2
        se = sigma * np.sqrt(1 / n1 + 1 / n2)
        ncp = abs(delta) / se

        critical_t = stats.t.ppf(1 - alpha / 2, df)
        power = (
            1 - stats.nct.cdf(critical_t, df, ncp) + stats.nct.cdf(-critical_t, df, ncp)
        )
        return power


class EffectSize:
    """
    Calculate various effect size measures.
    """

    @staticmethod
    def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Cohen's d for two samples."""
        n1, n2 = len(x1), len(x2)
        s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

        # Pooled standard deviation
        s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

        d = (np.mean(x1) - np.mean(x2)) / s_pooled
        return d

    @staticmethod
    def cohens_h(p1: float, p2: float) -> float:
        """Calculate Cohen's h for proportions."""
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    @staticmethod
    def odds_ratio(a: int, b: int, c: int, d: int) -> float:
        """
        Calculate odds ratio from 2x2 contingency table.

        Table:
            a  b
            c  d
        """
        return (a * d) / (b * c) if b * c != 0 else float("inf")

    @staticmethod
    def relative_risk(a: int, b: int, c: int, d: int) -> float:
        """Calculate relative risk from 2x2 table."""
        p1 = a / (a + b) if (a + b) > 0 else 0
        p2 = c / (c + d) if (c + d) > 0 else 0
        return p1 / p2 if p2 != 0 else float("inf")

    @staticmethod
    def eta_squared(groups: List[np.ndarray]) -> float:
        """Calculate eta-squared (ANOVA effect size)."""
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = sum((x - grand_mean) ** 2 for g in groups for x in g)

        return ss_between / ss_total if ss_total > 0 else 0


class MDE:
    """
    Minimum Detectable Effect calculations.
    """

    @staticmethod
    def continuous_metric(
        n: int, sigma: float, alpha: float = 0.05, power: float = 0.8
    ) -> float:
        """
        Calculate MDE for continuous metric.

        Args:
            n: Sample size per group
            sigma: Standard deviation
            alpha: Significance level
            power: Desired power

        Returns:
            Minimum detectable effect (absolute difference)
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        se = sigma * np.sqrt(2 / n)
        mde = (z_alpha + z_beta) * se

        return mde

    @staticmethod
    def proportion_metric(
        n: int, p: float, alpha: float = 0.05, power: float = 0.8
    ) -> float:
        """
        Calculate MDE for proportion metric.

        Args:
            n: Sample size per group
            p: Baseline proportion
            alpha: Significance level
            power: Desired power

        Returns:
            Minimum detectable effect (absolute difference in proportion)
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        se = np.sqrt(2 * p * (1 - p) / n)
        mde = (z_alpha + z_beta) * se

        return mde

    @staticmethod
    def relative_mde(
        n: int,
        baseline: float,
        metric_type: str = "proportion",
        sigma: Optional[float] = None,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> float:
        """
        Calculate relative MDE as percentage of baseline.

        Returns:
            Relative MDE (e.g., 0.10 for 10% lift)
        """
        if metric_type == "proportion":
            absolute_mde = MDE.proportion_metric(n, baseline, alpha, power)
        elif metric_type == "continuous":
            if sigma is None:
                raise ValueError("sigma required for continuous metrics")
            absolute_mde = MDE.continuous_metric(n, sigma, alpha, power)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        return absolute_mde / baseline if baseline != 0 else float("inf")


# ============================================================================
# Statistical Tests
# ============================================================================


@dataclass
class TestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    control_mean: float
    treatment_mean: float
    control_n: int
    treatment_n: int
    significant: bool
    alpha: float
    additional_info: Dict[str, Any] = field(default_factory=dict)


class TTest:
    """
    Two-sample t-test for comparing means.
    """

    @staticmethod
    def independent(
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05,
        equal_var: bool = False,
    ) -> TestResult:
        """
        Perform independent two-sample t-test.

        Args:
            control: Control group values
            treatment: Treatment group values
            alpha: Significance level
            equal_var: Assume equal variances

        Returns:
            TestResult with statistics
        """
        control = np.array(control)
        treatment = np.array(treatment)

        # Perform t-test
        statistic, p_value = ttest_ind(control, treatment, equal_var=equal_var)

        # Calculate means and effect size
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)

        d = EffectSize.cohens_d(treatment, control)

        # Calculate confidence interval
        diff = treatment_mean - control_mean
        se = np.sqrt(
            np.var(control, ddof=1) / len(control)
            + np.var(treatment, ddof=1) / len(treatment)
        )

        df = len(control) + len(treatment) - 2
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        margin = t_critical * se

        ci = (diff - margin, diff + margin)

        return TestResult(
            test_name="Independent T-Test",
            statistic=statistic,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=d,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_n=len(control),
            treatment_n=len(treatment),
            significant=p_value < alpha,
            alpha=alpha,
        )

    @staticmethod
    def paired(
        before: np.ndarray, after: np.ndarray, alpha: float = 0.05
    ) -> TestResult:
        """Perform paired t-test."""
        before = np.array(before)
        after = np.array(after)

        statistic, p_value = stats.ttest_rel(after, before)

        diffs = after - before
        mean_diff = np.mean(diffs)
        d = mean_diff / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0

        se = stats.sem(diffs)
        df = len(diffs) - 1
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        margin = t_critical * se

        ci = (mean_diff - margin, mean_diff + margin)

        return TestResult(
            test_name="Paired T-Test",
            statistic=statistic,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=d,
            control_mean=np.mean(before),
            treatment_mean=np.mean(after),
            control_n=len(before),
            treatment_n=len(after),
            significant=p_value < alpha,
            alpha=alpha,
        )


class ZTest:
    """
    Z-test for large samples or known variances.
    """

    @staticmethod
    def two_proportion(
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        alpha: float = 0.05,
    ) -> TestResult:
        """
        Z-test for comparing two proportions.

        Args:
            control_conversions: Number of conversions in control
            control_total: Total in control
            treatment_conversions: Number of conversions in treatment
            treatment_total: Total in treatment
            alpha: Significance level

        Returns:
            TestResult with statistics
        """
        p1 = control_conversions / control_total
        p2 = treatment_conversions / treatment_total

        # Pooled proportion
        p_pooled = (control_conversions + treatment_conversions) / (
            control_total + treatment_total
        )

        # Standard error
        se = np.sqrt(
            p_pooled * (1 - p_pooled) * (1 / control_total + 1 / treatment_total)
        )

        # Z-statistic
        z = (p2 - p1) / se if se > 0 else 0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Confidence interval
        se_diff = np.sqrt(
            p1 * (1 - p1) / control_total + p2 * (1 - p2) / treatment_total
        )
        z_critical = stats.norm.ppf(1 - alpha / 2)
        diff = p2 - p1
        margin = z_critical * se_diff
        ci = (diff - margin, diff + margin)

        # Effect size (Cohen's h)
        h = EffectSize.cohens_h(p2, p1)

        return TestResult(
            test_name="Two-Proportion Z-Test",
            statistic=z,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=h,
            control_mean=p1,
            treatment_mean=p2,
            control_n=control_total,
            treatment_n=treatment_total,
            significant=p_value < alpha,
            alpha=alpha,
        )

    @staticmethod
    def two_mean(
        control_mean: float,
        control_sd: float,
        control_n: int,
        treatment_mean: float,
        treatment_sd: float,
        treatment_n: int,
        alpha: float = 0.05,
    ) -> TestResult:
        """Z-test for comparing two means (large samples)."""
        se = np.sqrt(control_sd**2 / control_n + treatment_sd**2 / treatment_n)

        z = (treatment_mean - control_mean) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Confidence interval
        z_critical = stats.norm.ppf(1 - alpha / 2)
        diff = treatment_mean - control_mean
        margin = z_critical * se
        ci = (diff - margin, diff + margin)

        # Cohen's d approximation
        pooled_sd = np.sqrt(
            (control_sd**2 * (control_n - 1) + treatment_sd**2 * (treatment_n - 1))
            / (control_n + treatment_n - 2)
        )
        d = diff / pooled_sd if pooled_sd > 0 else 0

        return TestResult(
            test_name="Two-Mean Z-Test",
            statistic=z,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=d,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_n=control_n,
            treatment_n=treatment_n,
            significant=p_value < alpha,
            alpha=alpha,
        )


class ChiSquare:
    """
    Chi-square test for independence and goodness-of-fit.
    """

    @staticmethod
    def test_independence(
        contingency_table: np.ndarray, alpha: float = 0.05
    ) -> TestResult:
        """
        Chi-square test of independence.

        Args:
            contingency_table: 2D array of observed frequencies
            alpha: Significance level

        Returns:
            TestResult with statistics
        """
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Cramér's V (effect size)
        n = contingency_table.sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        cramers_v = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0

        # Confidence interval for effect size (approximate)
        ci = (0, min(1.0, cramers_v * 1.5))  # Rough approximation

        return TestResult(
            test_name="Chi-Square Test of Independence",
            statistic=chi2,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=cramers_v,
            control_mean=contingency_table[0].mean(),
            treatment_mean=contingency_table[1].mean()
            if len(contingency_table) > 1
            else 0,
            control_n=contingency_table[0].sum(),
            treatment_n=contingency_table[1].sum() if len(contingency_table) > 1 else 0,
            significant=p_value < alpha,
            alpha=alpha,
            additional_info={"degrees_of_freedom": dof, "expected": expected},
        )

    @staticmethod
    def goodness_of_fit(
        observed: np.ndarray, expected: np.ndarray, alpha: float = 0.05
    ) -> TestResult:
        """Chi-square goodness-of-fit test."""
        observed = np.array(observed)
        expected = np.array(expected)

        chi2 = np.sum((observed - expected) ** 2 / expected)
        dof = len(observed) - 1
        p_value = 1 - stats.chi2.cdf(chi2, dof)

        return TestResult(
            test_name="Chi-Square Goodness-of-Fit",
            statistic=chi2,
            p_value=p_value,
            confidence_interval=(0, 1),
            effect_size=chi2 / len(observed),
            control_mean=expected.mean(),
            treatment_mean=observed.mean(),
            control_n=expected.sum(),
            treatment_n=observed.sum(),
            significant=p_value < alpha,
            alpha=alpha,
            additional_info={"degrees_of_freedom": dof},
        )


class MannWhitney:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).
    """

    @staticmethod
    def test(
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> TestResult:
        """
        Perform Mann-Whitney U test.

        Args:
            control: Control group values
            treatment: Treatment group values
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            TestResult with statistics
        """
        control = np.array(control)
        treatment = np.array(treatment)

        statistic, p_value = mannwhitneyu(treatment, control, alternative=alternative)

        # Rank-biserial correlation (effect size)
        n1, n2 = len(treatment), len(control)
        r = 1 - (2 * statistic) / (n1 * n2)

        # Confidence interval (using normal approximation)
        se = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (statistic - n1 * n2 / 2) / se if se > 0 else 0
        z_critical = stats.norm.ppf(1 - alpha / 2)

        # Convert to difference in means approximation
        diff = np.median(treatment) - np.median(control)
        margin = abs(diff) * 0.5  # Rough approximation
        ci = (diff - margin, diff + margin)

        return TestResult(
            test_name="Mann-Whitney U Test",
            statistic=statistic,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=abs(r),
            control_mean=np.median(control),
            treatment_mean=np.median(treatment),
            control_n=len(control),
            treatment_n=len(treatment),
            significant=p_value < alpha,
            alpha=alpha,
        )


class PermutationTest:
    """
    Permutation test for non-parametric comparison.
    """

    @staticmethod
    def test(
        control: np.ndarray,
        treatment: np.ndarray,
        n_permutations: int = 10000,
        alpha: float = 0.05,
        statistic_func: Optional[Callable] = None,
    ) -> TestResult:
        """
        Perform permutation test.

        Args:
            control: Control group values
            treatment: Treatment group values
            n_permutations: Number of permutations
            alpha: Significance level
            statistic_func: Function to compute test statistic (default: difference in means)

        Returns:
            TestResult with statistics
        """
        control = np.array(control)
        treatment = np.array(treatment)

        if statistic_func is None:
            statistic_func = lambda x, y: np.mean(x) - np.mean(y)

        # Observed statistic
        observed_stat = statistic_func(treatment, control)

        # Combine data
        combined = np.concatenate([control, treatment])
        n_control = len(control)

        # Permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_control = combined[:n_control]
            perm_treatment = combined[n_control:]
            perm_stats.append(statistic_func(perm_treatment, perm_control))

        perm_stats = np.array(perm_stats)

        # P-value (two-tailed)
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))

        # Confidence interval from permutation distribution
        ci_lower = np.percentile(perm_stats, 100 * alpha / 2)
        ci_upper = np.percentile(perm_stats, 100 * (1 - alpha / 2))

        # Effect size (Cohen's d)
        d = EffectSize.cohens_d(treatment, control)

        return TestResult(
            test_name="Permutation Test",
            statistic=observed_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=d,
            control_mean=np.mean(control),
            treatment_mean=np.mean(treatment),
            control_n=len(control),
            treatment_n=len(treatment),
            significant=p_value < alpha,
            alpha=alpha,
            additional_info={"n_permutations": n_permutations},
        )


# ============================================================================
# Metrics
# ============================================================================


class ConversionRate:
    """
    Calculate conversion rate and related statistics.
    """

    @staticmethod
    def calculate(conversions: int, visitors: int) -> Dict[str, float]:
        """
        Calculate conversion rate with confidence interval.

        Args:
            conversions: Number of conversions
            visitors: Number of visitors

        Returns:
            Dictionary with rate and statistics
        """
        if visitors == 0:
            return {"rate": 0, "se": 0, "ci_lower": 0, "ci_upper": 0}

        rate = conversions / visitors
        se = np.sqrt(rate * (1 - rate) / visitors)

        # Wilson score interval
        z = 1.96  # 95% CI
        n = visitors
        p = rate

        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

        return {
            "rate": rate,
            "se": se,
            "ci_lower": max(0, centre - margin),
            "ci_upper": min(1, centre + margin),
            "conversions": conversions,
            "visitors": visitors,
        }

    @staticmethod
    def lift(
        control_conversions: int,
        control_visitors: int,
        treatment_conversions: int,
        treatment_visitors: int,
    ) -> Dict[str, float]:
        """Calculate relative and absolute lift."""
        control_rate = (
            control_conversions / control_visitors if control_visitors > 0 else 0
        )
        treatment_rate = (
            treatment_conversions / treatment_visitors if treatment_visitors > 0 else 0
        )

        absolute_lift = treatment_rate - control_rate
        relative_lift = (
            (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        )

        return {
            "absolute_lift": absolute_lift,
            "relative_lift": relative_lift,
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
        }


class ClickThroughRate:
    """
    Calculate CTR and related statistics.
    """

    @staticmethod
    def calculate(clicks: int, impressions: int) -> Dict[str, float]:
        """
        Calculate CTR with confidence interval.

        Args:
            clicks: Number of clicks
            impressions: Number of impressions

        Returns:
            Dictionary with CTR and statistics
        """
        return ConversionRate.calculate(clicks, impressions)

    @staticmethod
    def compare(
        control_clicks: int,
        control_impressions: int,
        treatment_clicks: int,
        treatment_impressions: int,
    ) -> TestResult:
        """Compare CTR between control and treatment."""
        return ZTest.two_proportion(
            control_clicks, control_impressions, treatment_clicks, treatment_impressions
        )


class RevenuePerUser:
    """
    Calculate revenue per user metrics.
    """

    @staticmethod
    def calculate(revenues: np.ndarray) -> Dict[str, float]:
        """
        Calculate RPU statistics.

        Args:
            revenues: Array of revenue per user

        Returns:
            Dictionary with RPU statistics
        """
        revenues = np.array(revenues)

        mean = np.mean(revenues)
        median = np.median(revenues)
        std = np.std(revenues, ddof=1)
        se = std / np.sqrt(len(revenues))

        # Confidence interval
        ci_margin = 1.96 * se

        return {
            "mean": mean,
            "median": median,
            "std": std,
            "se": se,
            "ci_lower": mean - ci_margin,
            "ci_upper": mean + ci_margin,
            "n": len(revenues),
            "purchasers": np.sum(revenues > 0),
            "purchase_rate": np.mean(revenues > 0),
        }

    @staticmethod
    def arpu(revenues: np.ndarray, total_users: int) -> float:
        """Calculate Average Revenue Per User (ARPU)."""
        return np.sum(revenues) / total_users if total_users > 0 else 0

    @staticmethod
    def arppu(revenues: np.ndarray) -> float:
        """Calculate Average Revenue Per Paying User (ARPPU)."""
        paying_users = revenues[revenues > 0]
        return np.mean(paying_users) if len(paying_users) > 0 else 0


class EngagementMetrics:
    """
    Calculate user engagement metrics.
    """

    @staticmethod
    def session_metrics(sessions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate session-based engagement metrics.

        Args:
            sessions: List of session dicts with 'duration', 'pages', 'events'

        Returns:
            Dictionary with engagement statistics
        """
        if not sessions:
            return {}

        durations = [s.get("duration", 0) for s in sessions]
        pages = [s.get("pages", 0) for s in sessions]
        events = [s.get("events", 0) for s in sessions]

        return {
            "avg_session_duration": np.mean(durations),
            "median_session_duration": np.median(durations),
            "avg_pages_per_session": np.mean(pages),
            "avg_events_per_session": np.mean(events),
            "total_sessions": len(sessions),
            "bounce_rate": np.mean([p <= 1 for p in pages]),
        }

    @staticmethod
    def retention(users_active_day0: int, users_active_day_n: int) -> float:
        """Calculate retention rate."""
        return users_active_day_n / users_active_day0 if users_active_day0 > 0 else 0

    @staticmethod
    def churn_rate(users_at_start: int, users_lost: int) -> float:
        """Calculate churn rate."""
        return users_lost / users_at_start if users_at_start > 0 else 0


# ============================================================================
# Analysis
# ============================================================================


class ABAnalyzer:
    """
    Comprehensive A/B test analysis.
    """

    def __init__(self, experiment: BaseExperiment):
        self.experiment = experiment

    def analyze_metric(
        self, metric: str, test_type: str = "auto", alpha: float = 0.05
    ) -> TestResult:
        """
        Analyze a specific metric.

        Args:
            metric: Metric name
            test_type: 'ttest', 'mannwhitney', 'permutation', or 'auto'
            alpha: Significance level

        Returns:
            TestResult with analysis
        """
        # Extract data for each variant
        data = {}
        for variant in self.experiment.config.variants:
            key = f"{variant.id}:{metric}"
            values = self.experiment.results.get(key, [])
            data[variant.id] = np.array(values)

        if len(data) < 2:
            raise ValueError("Need at least 2 variants to compare")

        variant_ids = list(data.keys())
        control_data = data[variant_ids[0]]
        treatment_data = data[variant_ids[1]]

        # Auto-select test
        if test_type == "auto":
            test_type = self._select_test(control_data, treatment_data)

        # Run appropriate test
        if test_type == "ttest":
            return TTest.independent(control_data, treatment_data, alpha)
        elif test_type == "mannwhitney":
            return MannWhitney.test(control_data, treatment_data, alpha)
        elif test_type == "permutation":
            return PermutationTest.test(control_data, treatment_data, alpha=alpha)
        elif test_type == "ztest_proportion":
            # Assume binary data
            control_conv = np.sum(control_data)
            treatment_conv = np.sum(treatment_data)
            return ZTest.two_proportion(
                int(control_conv),
                len(control_data),
                int(treatment_conv),
                len(treatment_data),
                alpha,
            )
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def _select_test(self, control: np.ndarray, treatment: np.ndarray) -> str:
        """Automatically select appropriate statistical test."""
        # Check if binary data (likely conversion)
        if set(np.unique(np.concatenate([control, treatment]))).issubset({0, 1}):
            return "ztest_proportion"

        # Check normality (simplified)
        if len(control) > 30 and len(treatment) > 30:
            return "ttest"

        return "mannwhitney"

    def compare_all_variants(
        self, metric: str, correction: str = "bonferroni"
    ) -> List[TestResult]:
        """
        Compare all variants against control.

        Args:
            metric: Metric name
            correction: Multiple testing correction ('bonferroni' or 'none')

        Returns:
            List of TestResults
        """
        variants = self.experiment.config.variants
        control = variants[0]

        results = []
        n_comparisons = len(variants) - 1

        for treatment in variants[1:]:
            alpha = 0.05 / n_comparisons if correction == "bonferroni" else 0.05
            result = self._compare_two(control.id, treatment.id, metric, alpha)
            results.append(result)

        return results

    def _compare_two(
        self, control_id: str, treatment_id: str, metric: str, alpha: float
    ) -> TestResult:
        """Compare two specific variants."""
        control_key = f"{control_id}:{metric}"
        treatment_key = f"{treatment_id}:{metric}"

        control_data = np.array(self.experiment.results.get(control_key, []))
        treatment_data = np.array(self.experiment.results.get(treatment_key, []))

        test_type = self._select_test(control_data, treatment_data)

        if test_type == "ztest_proportion":
            control_conv = np.sum(control_data)
            treatment_conv = np.sum(treatment_data)
            return ZTest.two_proportion(
                int(control_conv),
                len(control_data),
                int(treatment_conv),
                len(treatment_data),
                alpha,
            )
        else:
            return TTest.independent(control_data, treatment_data, alpha)

    def summary(self) -> pd.DataFrame:
        """Generate summary statistics for all variants."""
        summary_data = []

        for variant in self.experiment.config.variants:
            variant_data = {"variant": variant.name, "variant_id": variant.id}

            # Get all metrics for this variant
            for key, values in self.experiment.results.items():
                if key.startswith(variant.id + ":"):
                    metric_name = key.split(":")[1]
                    if values:
                        variant_data[f"{metric_name}_mean"] = np.mean(values)
                        variant_data[f"{metric_name}_std"] = np.std(values, ddof=1)
                        variant_data[f"{metric_name}_n"] = len(values)

            summary_data.append(variant_data)

        return pd.DataFrame(summary_data)


class ConfidenceInterval:
    """
    Calculate confidence intervals for various statistics.
    """

    @staticmethod
    def proportion(
        successes: int, trials: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Wilson score interval for proportions."""
        if trials == 0:
            return (0, 1)

        p = successes / trials
        z = stats.norm.ppf((1 + confidence) / 2)
        n = trials

        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

        return (max(0, centre - margin), min(1, centre + margin))

    @staticmethod
    def mean(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Confidence interval for mean."""
        data = np.array(data)
        mean = np.mean(data)
        se = stats.sem(data)

        if len(data) < 30:
            # Use t-distribution for small samples
            ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=se)
        else:
            # Use normal distribution
            z = stats.norm.ppf((1 + confidence) / 2)
            margin = z * se
            ci = (mean - margin, mean + margin)

        return ci

    @staticmethod
    def difference_in_proportions(
        x1: int, n1: int, x2: int, n2: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Confidence interval for difference in proportions."""
        p1 = x1 / n1
        p2 = x2 / n2
        diff = p2 - p1

        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * se

        return (diff - margin, diff + margin)

    @staticmethod
    def difference_in_means(
        data1: np.ndarray, data2: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Confidence interval for difference in means."""
        data1, data2 = np.array(data1), np.array(data2)

        mean1, mean2 = np.mean(data1), np.mean(data2)
        diff = mean2 - mean1

        se1 = stats.sem(data1)
        se2 = stats.sem(data2)
        se = np.sqrt(se1**2 + se2**2)

        # Degrees of freedom (Welch-Satterthwaite)
        s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        df = (s1 / n1 + s2 / n2) ** 2 / (
            (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        )

        t = stats.t.ppf((1 + confidence) / 2, df)
        margin = t * se

        return (diff - margin, diff + margin)


class PValue:
    """
    P-value calculations and adjustments.
    """

    @staticmethod
    def bonferroni_correction(p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction."""
        n = len(p_values)
        return [min(1, p * n) for p in p_values]

    @staticmethod
    def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction."""
        p_values = np.array(p_values)
        n = len(p_values)

        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        corrected = np.zeros(n)
        for i, p in enumerate(sorted_p):
            corrected[sorted_indices[i]] = min(1, p * n / (i + 1))

        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            corrected[sorted_indices[i]] = min(
                corrected[sorted_indices[i]], corrected[sorted_indices[i + 1]]
            )

        return corrected.tolist()

    @staticmethod
    def fisher_combined_test(p_values: List[float]) -> Tuple[float, float]:
        """
        Fisher's method for combining p-values.

        Returns:
            Tuple of (chi-square statistic, combined p-value)
        """
        p_values = [p for p in p_values if 0 < p < 1]

        if not p_values:
            return 0, 1

        test_stat = -2 * np.sum(np.log(p_values))
        df = 2 * len(p_values)
        combined_p = 1 - stats.chi2.cdf(test_stat, df)

        return test_stat, combined_p


# ============================================================================
# Monitoring
# ============================================================================


class ExperimentMonitor:
    """
    Monitor experiments in real-time.
    """

    def __init__(self, experiment: BaseExperiment):
        self.experiment = experiment
        self.metrics_history: List[Dict] = []

    def check_health(self) -> Dict[str, Any]:
        """
        Check experiment health.

        Returns:
            Dictionary with health metrics
        """
        health = {
            "status": self.experiment.status.value,
            "duration": datetime.now() - self.experiment.created_at,
            "total_assignments": len(self.experiment.assignments),
            "variant_counts": self.experiment.get_variant_counts(),
            "alerts": [],
        }

        # Check for issues
        if self.experiment.status == ExperimentStatus.RUNNING:
            # Check sample ratio mismatch
            srm_result = SampleRatioMismatch.check(
                list(health["variant_counts"].values())
            )
            if srm_result["srm_detected"]:
                health["alerts"].append(
                    {
                        "type": "SRM",
                        "message": f"Sample Ratio Mismatch detected (p={srm_result['p_value']:.4f})",
                    }
                )

        return health

    def snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current experiment state."""
        snapshot = {
            "timestamp": datetime.now(),
            "experiment_id": self.experiment.config.id,
            "status": self.experiment.status.value,
            "assignments": len(self.experiment.assignments),
            "variant_distribution": self.experiment.get_variant_counts(),
            "metrics": {},
        }

        # Calculate current metrics
        for key, values in self.experiment.results.items():
            if values:
                snapshot["metrics"][key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "n": len(values),
                }

        self.metrics_history.append(snapshot)
        return snapshot


class EarlyStopping:
    """
    Early stopping rules for experiments.
    """

    def __init__(
        self,
        min_samples: int = 100,
        max_samples: Optional[int] = None,
        sequential_testing: bool = False,
        spending_function: Optional[Callable] = None,
    ):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.sequential_testing = sequential_testing
        self.spending_function = spending_function or self._obrien_fleming

    @staticmethod
    def _obrien_fleming(information_fraction: float) -> float:
        """O'Brien-Fleming spending function."""
        from scipy.stats import norm

        z = norm.ppf(1 - 0.025)
        return 2 * (1 - norm.cdf(z / np.sqrt(information_fraction)))

    def check_stopping_rules(
        self, experiment: BaseExperiment, metric: str, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check if experiment should stop early.

        Returns:
            Dictionary with stopping decision and reason
        """
        result = {"should_stop": False, "reason": None, "recommendation": "continue"}

        # Get sample sizes
        variant_counts = experiment.get_variant_counts()
        total_samples = sum(variant_counts.values())

        # Check minimum samples
        if total_samples < self.min_samples:
            result["reason"] = (
                f"Minimum samples not reached ({total_samples}/{self.min_samples})"
            )
            return result

        # Check maximum samples
        if self.max_samples and total_samples >= self.max_samples:
            result["should_stop"] = True
            result["reason"] = "Maximum sample size reached"
            result["recommendation"] = "stop"
            return result

        # Perform statistical test
        try:
            analyzer = ABAnalyzer(experiment)
            test_result = analyzer.analyze_metric(metric, alpha=alpha)

            # Check for significance
            if test_result.significant:
                if self.sequential_testing:
                    # Adjust for sequential testing
                    if experiment.config.target_sample_size:
                        info_fraction = (
                            total_samples / experiment.config.target_sample_size
                        )
                        adjusted_alpha = self.spending_function(info_fraction)

                        if test_result.p_value < adjusted_alpha:
                            result["should_stop"] = True
                            result["reason"] = (
                                f"Statistically significant result (p={test_result.p_value:.4f})"
                            )
                            result["recommendation"] = "stop"
                else:
                    result["should_stop"] = True
                    result["reason"] = (
                        f"Statistically significant result (p={test_result.p_value:.4f})"
                    )

                    # Recommend which variant
                    if test_result.treatment_mean > test_result.control_mean:
                        result["recommendation"] = "treatment_wins"
                    else:
                        result["recommendation"] = "control_wins"

        except Exception as e:
            result["reason"] = f"Error in analysis: {str(e)}"

        return result

    def sequential_bounds(
        self, max_n: int, looks: List[int], alpha: float = 0.05
    ) -> List[float]:
        """
        Calculate sequential testing boundaries.

        Args:
            max_n: Maximum sample size
            looks: List of sample sizes at which to check
            alpha: Overall alpha level

        Returns:
            List of p-value boundaries for each look
        """
        bounds = []
        cumulative_alpha = 0

        for i, n in enumerate(looks):
            info_fraction = n / max_n

            # Spending function gives incremental alpha
            incremental_alpha = self.spending_function(info_fraction)
            if i > 0:
                prev_fraction = looks[i - 1] / max_n
                incremental_alpha -= self.spending_function(prev_fraction)

            cumulative_alpha += incremental_alpha
            bounds.append(cumulative_alpha)

        return bounds


class SampleRatioMismatch:
    """
    Detect and diagnose sample ratio mismatch.
    """

    @staticmethod
    def check(
        observed_counts: List[int], expected_proportions: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Test for sample ratio mismatch using chi-square test.

        Args:
            observed_counts: Observed sample counts per variant
            expected_proportions: Expected proportions (default: equal)

        Returns:
            Dictionary with test results
        """
        observed = np.array(observed_counts)
        total = observed.sum()

        if expected_proportions is None:
            expected_proportions = [1 / len(observed)] * len(observed)

        expected = np.array(expected_proportions) * total

        # Chi-square test
        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1
        p_value = 1 - stats.chi2.cdf(chi2, df)

        return {
            "srm_detected": p_value < 0.001,  # Conservative threshold
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": df,
            "observed_counts": observed.tolist(),
            "expected_counts": expected.tolist(),
            "observed_proportions": (observed / total).tolist(),
            "expected_proportions": expected_proportions,
        }

    @staticmethod
    def diagnose(
        experiment: BaseExperiment, time_windows: List[int] = None
    ) -> Dict[str, Any]:
        """
        Diagnose when SRM started occurring.

        Args:
            experiment: The experiment to diagnose
            time_windows: Time windows (in hours) to check

        Returns:
            Dictionary with diagnosis results
        """
        if time_windows is None:
            time_windows = [1, 6, 12, 24, 48, 72]

        # This would require timestamped assignment data
        # For now, return placeholder
        return {
            "diagnosis": "SRM diagnosis requires timestamped assignment data",
            "time_windows_checked": time_windows,
            "recommendation": "Check assignment logs for patterns",
        }


# ============================================================================
# Reporting
# ============================================================================


class ABReport:
    """
    Generate comprehensive A/B test reports.
    """

    def __init__(self, experiment: BaseExperiment):
        self.experiment = experiment
        self.analyzer = ABAnalyzer(experiment)

    def generate(
        self, metrics: Optional[List[str]] = None, include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete experiment report.

        Args:
            metrics: List of metrics to include (default: all)
            include_plots: Whether to include plot data

        Returns:
            Dictionary with report data
        """
        # Get all metrics if not specified
        if metrics is None:
            metrics = list(
                set(key.split(":")[1] for key in self.experiment.results.keys())
            )

        report = {
            "experiment": {
                "name": self.experiment.config.name,
                "id": self.experiment.config.id,
                "status": self.experiment.status.value,
                "start_date": self.experiment.config.start_date,
                "end_date": self.experiment.config.end_date,
                "duration": (
                    self.experiment.config.end_date - self.experiment.config.start_date
                    if self.experiment.config.end_date
                    and self.experiment.config.start_date
                    else None
                ),
                "total_participants": len(self.experiment.assignments),
            },
            "variants": [
                {
                    "id": v.id,
                    "name": v.name,
                    "traffic_allocation": v.traffic_allocation,
                    "config": v.config,
                }
                for v in self.experiment.config.variants
            ],
            "sample_sizes": self.experiment.get_variant_counts(),
            "metrics_analysis": {},
            "summary_statistics": self.analyzer.summary().to_dict("records"),
            "recommendations": [],
        }

        # Analyze each metric
        for metric in metrics:
            try:
                analysis = self.analyzer.analyze_metric(metric)
                report["metrics_analysis"][metric] = {
                    "test_type": analysis.test_name,
                    "p_value": analysis.p_value,
                    "significant": analysis.significant,
                    "effect_size": analysis.effect_size,
                    "confidence_interval": analysis.confidence_interval,
                    "control_mean": analysis.control_mean,
                    "treatment_mean": analysis.treatment_mean,
                    "lift": (
                        (analysis.treatment_mean - analysis.control_mean)
                        / analysis.control_mean
                        if analysis.control_mean != 0
                        else 0
                    ),
                    "absolute_difference": analysis.treatment_mean
                    - analysis.control_mean,
                }

                # Add recommendation
                if analysis.significant:
                    if analysis.treatment_mean > analysis.control_mean:
                        report["recommendations"].append(
                            f"Treatment shows significant improvement in {metric} "
                            f"(+{report['metrics_analysis'][metric]['lift']:.1%})"
                        )
                    else:
                        report["recommendations"].append(
                            f"Control performs better for {metric}"
                        )
                else:
                    report["recommendations"].append(
                        f"No significant difference detected in {metric}"
                    )

            except Exception as e:
                report["metrics_analysis"][metric] = {"error": str(e)}

        # Check for SRM
        srm_check = SampleRatioMismatch.check(list(report["sample_sizes"].values()))
        report["sample_ratio_check"] = srm_check

        if srm_check["srm_detected"]:
            report["recommendations"].append(
                "WARNING: Sample Ratio Mismatch detected - results may be biased"
            )

        return report

    def to_json(self, filepath: str, **kwargs):
        """Save report to JSON file."""
        report = self.generate(**kwargs)

        # Convert datetime objects
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, timedelta):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=json_serial)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert report to DataFrame."""
        report = self.generate()

        rows = []
        for metric, analysis in report["metrics_analysis"].items():
            if "error" not in analysis:
                rows.append(
                    {
                        "metric": metric,
                        "p_value": analysis["p_value"],
                        "significant": analysis["significant"],
                        "effect_size": analysis["effect_size"],
                        "control_mean": analysis["control_mean"],
                        "treatment_mean": analysis["treatment_mean"],
                        "lift": analysis["lift"],
                        "absolute_difference": analysis["absolute_difference"],
                    }
                )

        return pd.DataFrame(rows)


def visualize_results(
    experiment: BaseExperiment,
    metric: str,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create comprehensive visualization of A/B test results.

    Args:
        experiment: The experiment to visualize
        metric: Metric to visualize
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    # Extract data
    data = {}
    for variant in experiment.config.variants:
        key = f"{variant.id}:{metric}"
        values = experiment.results.get(key, [])
        data[variant.name] = np.array(values)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"A/B Test Results: {metric}", fontsize=16, fontweight="bold")

    # 1. Bar plot of means with confidence intervals
    ax1 = axes[0, 0]
    means = [np.mean(v) if len(v) > 0 else 0 for v in data.values()]
    stds = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in data.values()]
    ns = [len(v) for v in data.values()]
    cis = [1.96 * s / np.sqrt(n) if n > 0 else 0 for s, n in zip(stds, ns)]

    x_pos = np.arange(len(data))
    bars = ax1.bar(
        x_pos, means, yerr=cis, capsize=5, alpha=0.7, color=["#3498db", "#e74c3c"]
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(data.keys())
    ax1.set_ylabel(f"{metric} (mean ± 95% CI)")
    ax1.set_title("Means Comparison")
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + ci,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Distribution plots
    ax2 = axes[0, 1]
    for name, values in data.items():
        if len(values) > 0:
            ax2.hist(values, bins=30, alpha=0.5, label=name, density=True)
    ax2.set_xlabel(metric)
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution Comparison")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Box plots
    ax3 = axes[1, 0]
    box_data = [v for v in data.values() if len(v) > 0]
    box_labels = [k for k, v in data.items() if len(v) > 0]
    if box_data:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax3.set_ylabel(metric)
    ax3.set_title("Distribution Summary")
    ax3.grid(axis="y", alpha=0.3)

    # 4. Cumulative metrics over time (if we had timestamps)
    ax4 = axes[1, 1]
    variant_names = list(data.keys())
    if len(variant_names) >= 2:
        control_data = data[variant_names[0]]
        treatment_data = data[variant_names[1]]

        # Cumulative means
        if len(control_data) > 0 and len(treatment_data) > 0:
            min_len = min(len(control_data), len(treatment_data))
            cumsum_control = np.cumsum(control_data[:min_len]) / np.arange(
                1, min_len + 1
            )
            cumsum_treatment = np.cumsum(treatment_data[:min_len]) / np.arange(
                1, min_len + 1
            )

            ax4.plot(cumsum_control, label=variant_names[0], linewidth=2)
            ax4.plot(cumsum_treatment, label=variant_names[1], linewidth=2)
            ax4.set_xlabel("Sample Size")
            ax4.set_ylabel(f"Cumulative {metric}")
            ax4.set_title("Cumulative Mean Over Time")
            ax4.legend()
            ax4.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ============================================================================
# Utility Functions
# ============================================================================


def create_ab_test(
    name: str,
    control_config: Dict[str, Any],
    treatment_config: Dict[str, Any],
    control_traffic: float = 0.5,
) -> ABTest:
    """
    Convenience function to create a standard A/B test.

    Args:
        name: Test name
        control_config: Control variant configuration
        treatment_config: Treatment variant configuration
        control_traffic: Traffic allocation for control

    Returns:
        Configured ABTest instance
    """
    control = Variant(
        name="Control",
        id="control",
        traffic_allocation=control_traffic,
        config=control_config,
    )

    treatment = Variant(
        name="Treatment",
        id="treatment",
        traffic_allocation=1 - control_traffic,
        config=treatment_config,
    )

    config = ExperimentConfig(
        name=name,
        id=f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        variants=[control, treatment],
    )

    return ABTest(config)


def simulate_ab_test(
    control_mean: float,
    treatment_mean: float,
    std: float,
    n_per_group: int,
    seed: Optional[int] = None,
) -> ABTest:
    """
    Simulate an A/B test with normally distributed outcomes.

    Args:
        control_mean: Mean for control group
        treatment_mean: Mean for treatment group
        std: Standard deviation
        n_per_group: Sample size per group
        seed: Random seed

    Returns:
        Simulated ABTest with results
    """
    if seed is not None:
        np.random.seed(seed)

    # Create experiment
    test = create_ab_test(
        name="Simulated Test",
        control_config={"version": "control"},
        treatment_config={"version": "treatment"},
    )

    # Generate users and assign
    for i in range(n_per_group * 2):
        user_id = f"user_{i}"
        variant = test.assign_variant(user_id)

        # Generate outcome
        if variant == "control":
            value = np.random.normal(control_mean, std)
        else:
            value = np.random.normal(treatment_mean, std)

        test.record_outcome(user_id, "revenue", value)

    return test


# Export all major classes
__all__ = [
    # Experiment Design
    "ABTest",
    "MultivariateTest",
    "BanditTest",
    "SplitTest",
    "ExperimentConfig",
    "Variant",
    "ExperimentStatus",
    "BaseExperiment",
    # Randomization
    "SimpleRandomization",
    "StratifiedRandomization",
    "ClusterRandomization",
    "CovariateAdaptiveRandomization",
    "BaseRandomization",
    # Sample Size
    "SampleSizeCalculator",
    "PowerAnalysis",
    "EffectSize",
    "MDE",
    # Statistical Tests
    "TTest",
    "ZTest",
    "ChiSquare",
    "MannWhitney",
    "PermutationTest",
    "TestResult",
    # Metrics
    "ConversionRate",
    "ClickThroughRate",
    "RevenuePerUser",
    "EngagementMetrics",
    # Analysis
    "ABAnalyzer",
    "ConfidenceInterval",
    "PValue",
    # Monitoring
    "ExperimentMonitor",
    "EarlyStopping",
    "SampleRatioMismatch",
    # Reporting
    "ABReport",
    "visualize_results",
    # Utilities
    "create_ab_test",
    "simulate_ab_test",
]
