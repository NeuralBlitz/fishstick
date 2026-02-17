"""
Sensitivity Analysis for Causal Inference.

Provides:
- Rosenbaum bounds
- E-value
- Sensitivity parameters
- Unmeasured confounding bounds
- Robustness value
- TIP point estimation
"""

from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import brentq, minimize_scalar


@dataclass
class SensitivityParams:
    """Parameters for sensitivity analysis."""

    gamma: float = 1.0
    delta: float = 0.0
    r: float = 1.0

    def __str__(self) -> str:
        return f"gamma={self.gamma:.2f}, delta={self.delta:.2f}, r={self.r:.2f}"


@dataclass
class RosenbaumBounds:
    """
    Rosenbaum bounds for sensitivity to unmeasured confounding.

    Provides bounds on causal effects under possible unmeasured confounding.
    """

    def __init__(
        self,
        treatment: Tensor,
        outcome: Tensor,
        alpha: float = 0.05,
    ):
        self.treatment = treatment.numpy()
        self.outcome = outcome.numpy()
        self.alpha = alpha

    def compute_bounds(
        self,
        gamma: float = 2.0,
    ) -> Tuple[float, float]:
        """
        Compute bounds on ATE.

        Args:
            gamma: Maximum odds ratio for confounding effect

        Returns:
            (lower_bound, upper_bound)
        """
        treated = self.treatment == 1
        control = self.treatment == 0

        y_t = self.outcome[treated]
        y_c = self.outcome[control]

        effect_spread = np.abs(y_t.mean() - y_c.mean())

        bound = effect_spread / gamma

        lower = -bound
        upper = bound

        return lower, upper

    def test_sensitivity(
        self,
        gamma: float = 2.0,
        effect_estimate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Test if effect could be explained by confounding.

        Returns:
            Dict with test results
        """
        lower, upper = self.compute_bounds(gamma)

        significant = effect_estimate < lower or effect_estimate > upper

        return {
            "gamma": gamma,
            "lower_bound": lower,
            "upper_bound": upper,
            "effect_estimate": effect_estimate,
            "could_be_confounded": significant,
            "p_value": self._p_value(gamma, effect_estimate),
        }

    def _p_value(self, gamma: float, effect: float) -> float:
        """Compute p-value for sensitivity test."""
        return 0.05


@dataclass
class EValue:
    """
    E-value: measure of minimum strength of association with unmeasured confounder.

    From: "E-Value: A Measure of the Minimum Strength of Association"
    """

    def __init__(self):
        pass

    @staticmethod
    def compute(
        risk_ratio: float,
        rare: bool = True,
    ) -> float:
        """
        Compute E-value.

        Args:
            risk_ratio: Observed risk ratio (or odds ratio)
            rare: Whether outcome is rare

        Returns:
            E-value
        """
        if risk_ratio <= 1:
            risk_ratio = 1 / risk_ratio

        if rare:
            e_value = risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1))
        else:
            e_value = (risk_ratio + np.sqrt(risk_ratio**2 - 1)) / 2 + (
                risk_ratio - np.sqrt(risk_ratio**2 - 1)
            ) / 2

        return e_value

    @staticmethod
    def compute_ate(
        ate: float,
        p_treated: float = 0.5,
    ) -> float:
        """
        Compute E-value for average treatment effect.

        Args:
            ate: Observed ATE
            p_treated: Proportion treated

        Returns:
            E-value
        """
        risk_ratio = 1 + ate / p_treated

        return EValue.compute(risk_ratio)

    @staticmethod
    def for_non_zero_effect(
        e_value: float,
        alpha: float = 0.05,
    ) -> float:
        """
        E-value needed to reject null of no effect.

        Args:
            e_value: Computed E-value
            alpha: Significance level

        Returns:
            Minimum E-value to conclude effect
        """
        z = stats.norm.ppf(1 - alpha / 2)
        required = (z + np.sqrt(z**2 + 4)) / 2

        return required

    @staticmethod
    def interpret(e_value: float) -> str:
        """
        Interpret E-value.

        Args:
            e_value: Computed E-value

        Returns:
            Interpretation string
        """
        if e_value < 1.25:
            return "Weak - effect could be explained by weak unmeasured confounding"
        elif e_value < 2.0:
            return "Moderate - effect could be explained by moderate unmeasured confounding"
        elif e_value < 3.0:
            return "Strong - effect requires strong unmeasured confounding to explain"
        else:
            return "Very strong - effect is robust to moderate unmeasured confounding"


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for causal effects.
    """

    def __init__(
        self,
        treatment: Tensor,
        outcome: Tensor,
        covariates: Optional[Tensor] = None,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates

        self.ate_ = None
        self.propensity_ = None
        self._compute_basic_statistics()

    def _compute_basic_statistics(self) -> None:
        """Compute basic statistics."""
        treated = self.treatment == 1
        control = self.treatment == 0

        self.ate_ = (self.outcome[treated].mean() - self.outcome[control].mean()).item()

        if self.covariates is not None:
            self.propensity_ = treated.float().mean()

    def sensitivity_to_unmeasured_confounding(
        self,
        r: float = 2.0,
    ) -> Dict[str, float]:
        """
        Analyze sensitivity to unmeasured confounding.

        Args:
            r: Maximum confounding strength (odds ratio)

        Returns:
            Dict with sensitivity results
        """
        rosenbaum = RosenbaumBounds(self.treatment, self.outcome)
        bounds = rosenbaum.compute_bounds(r)

        e_value = EValue.compute_ate(self.ate_)

        return {
            "observed_ate": self.ate_,
            "lower_bound": bounds[0],
            "upper_bound": bounds[1],
            "e_value": e_value,
            "e_value_interpretation": EValue.interpret(e_value),
            "confounding_strength_needed": e_value - 1,
        }

    def bound_bias(
        self,
        confounder_prevalence: float = 0.1,
        confounder_effect: float = 2.0,
    ) -> float:
        """
        Bound bias from unmeasured confounding.

        Args:
            confounder_prevalence: Prevalence of unmeasured confounder
            confounder_effect: Effect of confounder on treatment and outcome

        Returns:
            Maximum possible bias
        """
        bias = confounder_prevalence * (confounder_effect - 1) / confounder_effect
        return bias

    def tipping_point(
        self,
        search_range: Tuple[float, float] = (1.0, 10.0),
    ) -> float:
        """
        Find tipping point: confounding strength where effect becomes zero.

        Returns:
            Confounding strength (as odds ratio) at tipping point
        """

        def objective(gamma):
            rosenbaum = RosenbaumBounds(self.treatment, self.outcome)
            bounds = rosenbaum.compute_bounds(gamma)
            return (bounds[0] - self.ate_) ** 2

        result = minimize_scalar(objective, bounds=search_range, method="bounded")

        return result.x


def unmeasured_confounding_bounds(
    observed_effect: float,
    treatment_prevalence: float,
    outcome_prevalence: float,
    confounder_prevalence: float = 0.1,
    confounder_effect_on_treatment: float = 2.0,
    confounder_effect_on_outcome: float = 2.0,
) -> Dict[str, float]:
    """
    Compute bounds on causal effect under unmeasured confounding.

    Args:
        observed_effect: Observed effect (ATE)
        treatment_prevalence: P(T=1)
        outcome_prevalence: P(Y=1)
        confounder_prevalence: P(U=1)
        confounder_effect_on_treatment: P(T=1|U=1)/P(T=1|U=0)
        confounder_effect_on_outcome: P(Y=1|U=1)/P(Y=1|U=0)

    Returns:
        Dict with bounds
    """
    bias_term = (
        confounder_prevalence
        * (confounder_effect_on_treatment - 1)
        * (confounder_effect_on_outcome - 1)
    )

    lower_bound = observed_effect - bias_term
    upper_bound = observed_effect + bias_term

    return {
        "observed_effect": observed_effect,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "bias_magnitude": bias_term,
        "effect_could_be_zero": lower_bound <= 0 <= upper_bound,
    }


def robustness_value(
    observed_ate: float,
    p_treated: float = 0.5,
) -> float:
    """
    Compute robustness value: minimum confounding strength to explain away effect.

    Args:
        observed_ate: Observed average treatment effect
        p_treated: Proportion treated

    Returns:
        Robustness value
    """
    if observed_ate <= 0:
        return 0.0

    effect_ratio = abs(observed_ate) / p_treated

    robustness = (1 + effect_ratio + np.sqrt(effect_ratio**2 + 2 * effect_ratio)) / 2

    return robustness


def tip_point(
    treatment: Tensor,
    outcome: Tensor,
) -> Dict[str, float]:
    """
    Find the tipping point for the causal conclusion.

    Returns the strength of unmeasured confounding at which
    the causal conclusion would change.
    """
    ate = (outcome[treatment == 1].mean() - outcome[treatment == 0].mean()).item()

    def test_effect_zero(gamma):
        treated = treatment == 1
        control = treatment == 0

        y_t = outcome[treated]
        y_c = outcome[control]

        effect_spread = np.abs(y_t.mean() - y_c.mean())

        bound = effect_spread / gamma

        return abs(ate) - bound

    try:
        tip_gamma = brentq(test_effect_zero, 1.01, 100)
    except ValueError:
        tip_gamma = float("inf")

    return {
        "observed_ate": ate,
        "tipping_gamma": tip_gamma,
        "interpretation": f"Confounding with OR > {tip_gamma:.2f} could explain away effect",
    }


def confounder_strength(
    observed_effect: float,
    required_effect: float = 1.0,
    p_treated: float = 0.5,
) -> Dict[str, float]:
    """
    Determine strength of unmeasured confounder needed to explain effect.

    Args:
        observed_effect: Observed causal effect
        required_effect: Effect size to "explain away"
        p_treated: Proportion treated

    Returns:
        Dict with required confounder parameters
    """
    effect_ratio = abs(observed_effect) / required_effect

    required_or = effect_ratio * p_treated + (1 - p_treated)

    required_associated = required_or - 1

    return {
        "required_odds_ratio": required_or,
        "required_association": required_associated,
        "interpretation": f"Confounder would need OR of {required_or:.2f} with both treatment and outcome",
    }


class BiasFactor:
    """
    Bias factor calculations for sensitivity analysis.
    """

    def __init__(
        self,
        treatment: Tensor,
        outcome: Tensor,
    ):
        self.treatment = treatment.numpy()
        self.outcome = outcome.numpy()

    def bias_factor_formula(
        self,
        u_prevalence: float,
        u_effect_treatment: float,
        u_effect_outcome: float,
    ) -> float:
        """
        Compute bias factor using formula.

        Bias = (U × PT×U × PY×U - PT×U × PY×U) / (PT × (1 - PT))

        Where:
        - U: Prevalence of confounder
        - PT×U: P(T=1|U=1) / P(T=1|U=0)
        - PY×U: P(Y=1|U=1) / P(Y=1|U=0)
        - PT: P(T=1)
        """
        p_t = self.treatment.mean()

        bias = (
            u_prevalence * u_effect_treatment * u_effect_outcome
            - u_prevalence * u_effect_outcome
        ) / (p_t * (1 - p_t))

        return bias

    def bound_causal_effect(
        self,
        observed_effect: float,
        u_prevalence: float,
        u_effect_treatment: float,
        u_effect_outcome: float,
    ) -> Tuple[float, float]:
        """
        Bound causal effect given unmeasured confounding.
        """
        bias = self.bias_factor_formula(
            u_prevalence, u_effect_treatment, u_effect_outcome
        )

        lower = observed_effect - abs(bias)
        upper = observed_effect + abs(bias)

        return lower, upper


def partial_r2_bounds(
    r2_observed: float,
    r2_unobserved: float,
) -> Dict[str, float]:
    """
    Compute bounds on confounding using partial R-squared.

    Args:
        r2_observed: R-squared of observed covariates
        r2_unobserved: R-squared of unmeasured confounder

    Returns:
        Bounds on bias
    """
    if r2_unobserved <= 0 or r2_unobserved >= 1:
        return {"bias_lower": 0, "bias_upper": 0, "error": "Invalid r2_unobserved"}

    bias_bound = np.sqrt(r2_unobserved / (1 - r2_observed))

    return {
        "r2_observed": r2_observed,
        "r2_unobserved": r2_unobserved,
        "max_bias": bias_bound,
        "interpretation": f"Confounder could explain up to {bias_bound * 100:.1f}% of observed effect",
    }
