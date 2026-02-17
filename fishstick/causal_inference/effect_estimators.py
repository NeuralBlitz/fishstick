"""
Advanced Causal Effect Estimators.

Provides:
- Linear regression estimators
- Difference-in-differences
- Regression discontinuity
- Instrumental variables
- Front-door estimator
- Marginal structural models
- G-formula
- IPW and AIPW estimators
"""

from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class EstimatorType(Enum):
    """Types of causal effect estimators."""

    LINEAR_REGRESSION = "linear_regression"
    DID = "did"
    RDD = "rdd"
    IV = "iv"
    FRONT_DOOR = "front_door"
    MSM = "msm"
    G_FORMULA = "g_formula"
    IPW = "ipw"
    AIPW = "aipw"


@dataclass
class LinearRegressionEstimator:
    """
    Linear regression for causal effect estimation.

    Assumes linear relationship: Y = α + βT + γX + ε
    """

    def __init__(self, include_intercept: bool = True):
        self.include_intercept = include_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(
        self,
        treatment: Tensor,
        outcome: Tensor,
        covariates: Optional[Tensor] = None,
    ) -> "LinearRegressionEstimator":
        """Fit linear regression."""
        device = treatment.device

        if covariates is not None:
            X = torch.cat([treatment.unsqueeze(-1), covariates], dim=-1)
        else:
            X = treatment.unsqueeze(-1)

        if self.include_intercept:
            X = torch.cat([torch.ones(X.size(0), 1, device=device), X], dim=-1)

        XtX = X.T @ X + 1e-8 * torch.eye(X.size(1), device=device)
        XtY = X.T @ outcome

        params = torch.linalg.solve(XtX, XtY)

        if self.include_intercept:
            self.intercept_ = params[0].item()
            self.coef_ = params[1:].numpy()
        else:
            self.intercept_ = 0.0
            self.coef_ = params.numpy()

        return self

    def predict(
        self,
        treatment: Tensor,
        covariates: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict outcome."""
        if covariates is not None:
            X = torch.cat([treatment.unsqueeze(-1), covariates], dim=-1)
        else:
            X = treatment.unsqueeze(-1)

        if self.include_intercept:
            X = torch.cat([torch.ones(X.size(0), 1, device=X.device), X], dim=-1)

        params = torch.tensor([self.intercept_] + self.coef_.tolist(), device=X.device)

        return X @ params

    def effect(self) -> float:
        """Get treatment effect coefficient."""
        if self.include_intercept:
            return self.coef_[0] if len(self.coef_) > 0 else 0.0
        return self.coef_[0]


@dataclass
class DifferenceInDifferences:
    """
    Difference-in-Differences estimator.

    Estimates causal effect using pre/post treatment comparisons
    for treatment and control groups.
    """

    def __init__(self):
        self.effect_ = None
        self.se_ = None

    def fit(
        self,
        treatment_pre: Tensor,
        treatment_post: Tensor,
        control_pre: Tensor,
        control_post: Tensor,
    ) -> "DifferenceInDifferences":
        """
        Fit DID estimator.

        Args:
            treatment_pre: Pre-treatment outcomes for treated group
            treatment_post: Post-treatment outcomes for treated group
            control_pre: Pre-treatment outcomes for control group
            control_post: Post-treatment outcomes for control group

        Returns:
            Self
        """
        treat_diff = treatment_post.mean() - treatment_pre.mean()
        control_diff = control_post.mean() - control_pre.mean()

        self.effect_ = (treat_diff - control_diff).item()

        n_treat = treatment_pre.size(0)
        n_control = control_pre.size(0)

        var_treat = treatment_post.var() + treatment_pre.var()
        var_control = control_post.var() + control_pre.var()

        self.se_ = torch.sqrt(var_treat / n_treat + var_control / n_control).item()

        return self

    def confidence_interval(
        self,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """Get confidence interval."""
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)

        lower = self.effect_ - z * self.se_
        upper = self.effect_ + z * self.se_

        return lower, upper


@dataclass
class RegressionDiscontinuity:
    """
    Regression Discontinuity Design estimator.

    Estimates causal effect at the cutoff threshold.
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        kernel: str = "triangular",
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.effect_ = None
        self.se_ = None

    def fit(
        self,
        running_variable: Tensor,
        outcome: Tensor,
        treatment: Tensor,
        cutoff: float = 0.0,
    ) -> "RegressionDiscontinuity":
        """
        Fit RDD estimator.

        Args:
            running_variable: Running variable (assignment)
            outcome: Outcome variable
            treatment: Treatment indicator
            cutoff: Cutoff threshold

        Returns:
            Self
        """
        if self.bandwidth is None:
            self.bandwidth = self._optimal_bandwidth(
                running_variable, treatment, cutoff
            )

        mask = torch.abs(running_variable - cutoff) <= self.bandwidth

        rv = running_variable[mask]
        out = outcome[mask]
        treat = treatment[mask]

        rv_centered = rv - cutoff

        X = torch.stack(
            [
                torch.ones_like(rv_centered),
                rv_centered,
                treat.float(),
                treat.float() * rv_centered,
            ],
            dim=-1,
        )

        XtX = X.T @ X + 1e-8 * torch.eye(4)
        XtY = X.T @ out

        params = torch.linalg.solve(XtX, XtY)

        self.effect_ = params[2].item()

        residuals = out - X @ params
        mse = (residuals**2).mean()
        self.se_ = torch.sqrt(mse * torch.linalg.inv(XtX)[2, 2]).item()

        return self

    def _optimal_bandwidth(
        self,
        running_variable: Tensor,
        treatment: Tensor,
        cutoff: float,
    ) -> float:
        """Compute optimal bandwidth using Imbens-Kalyanaraman method."""
        n = running_variable.size(0)
        return 1.06 * running_variable.std() * (n ** (-1 / 5))


@dataclass
class InstrumentalVariableEstimator:
    """
    Instrumental Variable estimator.

    Uses instruments to estimate causal effects when endogeneity is present.
    """

    def __init__(self, method: str = "2sls"):
        self.method = method
        self.effect_ = None
        self.first_stage_ = None

    def fit(
        self,
        instrument: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        covariates: Optional[Tensor] = None,
    ) -> "InstrumentalVariableEstimator":
        """
        Fit IV estimator.

        Args:
            instrument: Instrumental variable
            treatment: Treatment variable
            outcome: Outcome variable
            covariates: Optional covariates

        Returns:
            Self
        """
        if self.method == "2sls":
            self._fit_2sls(instrument, treatment, outcome, covariates)
        elif self.method == "wald":
            self._fit_wald(instrument, treatment, outcome)

        return self

    def _fit_2sls(
        self,
        instrument: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        covariates: Optional[Tensor],
    ) -> None:
        """Two-stage least squares."""
        device = instrument.device

        if covariates is not None:
            Z = torch.cat([instrument.unsqueeze(-1), covariates], dim=-1)
            X = torch.cat([treatment.unsqueeze(-1), covariates], dim=-1)
        else:
            Z = instrument.unsqueeze(-1)
            X = treatment.unsqueeze(-1)

        ZtZ = Z.T @ Z + 1e-8 * torch.eye(Z.size(1), device=device)
        ZtX = Z.T @ X
        ZtY = Z.T @ outcome

        pi = torch.linalg.solve(ZtZ, ZtX)
        X_hat = Z @ pi

        if covariates is not None:
            X_final = torch.cat([X_hat, covariates], dim=-1)
        else:
            X_final = X_hat

        XtX = X_final.T @ X_final + 1e-8 * torch.eye(X_final.size(1), device=device)
        XtY = X_final.T @ outcome

        beta = torch.linalg.solve(XtX, XtY)

        self.effect_ = beta[0].item()
        self.first_stage_ = pi[0].item() if covariates is None else pi[0, 0].item()

    def _fit_wald(
        self,
        instrument: Tensor,
        treatment: Tensor,
        outcome: Tensor,
    ) -> None:
        """Wald estimator."""
        iv = (instrument * outcome).mean() / (instrument * treatment).mean()
        self.effect_ = iv.item()


@dataclass
class FrontDoorEstimator:
    """
    Front-door adjustment estimator.

    Uses intermediate variable to identify causal effect.
    """

    def __init__(self):
        self.effect_ = None

    def fit(
        self,
        treatment: Tensor,
        mediator: Tensor,
        outcome: Tensor,
    ) -> "FrontDoorEstimator":
        """
        Fit front-door estimator.

        Args:
            treatment: Treatment variable
            mediator: Mediator variable
            outcome: Outcome variable

        Returns:
            Self
        """
        device = treatment.device

        t_vals = torch.unique(treatment)

        p_m_t = []
        for t in t_vals:
            mask = treatment == t
            p_m_t.append(mediator[mask].mean().item())

        m_vals = torch.unique(mediator)

        p_y_m = []
        for m in m_vals:
            mask = mediator == m
            if mask.sum() > 0:
                p_y_m.append(outcome[mask].mean().item())
            else:
                p_y_m.append(0.0)

        p_m = mediator.bincount().float() / mediator.size(0)

        total_effect = 0.0
        for i, m in enumerate(m_vals):
            for j, t in enumerate(t_vals):
                p_m_given_t = (mediator[treatment == t] == m).float().mean()
                p_y_given_m = outcome[mediator == m].mean()
                p_t = (treatment == t).float().mean()

                total_effect += p_m_given_t * p_y_given_m * p_t

        self.effect_ = total_effect
        return self


@dataclass
class MarginalStructuralModel:
    """
    Marginal Structural Model for causal inference.

    Uses inverse probability weighting to estimate causal effects.
    """

    def __init__(self, outcome_model: Optional[nn.Module] = None):
        self.outcome_model = outcome_model
        self.propensity_model = None

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
    ) -> "MarginalStructuralModel":
        """Fit MSM."""
        device = next(covariates.device for _ in [covariates])

        if self.outcome_model is None:
            self.outcome_model = nn.Sequential(
                nn.Linear(covariates.size(1) + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self.propensity_model = nn.Sequential(
            nn.Linear(covariates.size(1), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        opt_outcome = torch.optim.Adam(self.outcome_model.parameters())
        opt_prop = torch.optim.Adam(self.propensity_model.parameters())

        for epoch in range(n_epochs):
            opt_prop.zero_grad()
            e = self.propensity_model(covariates).squeeze()
            loss_prop = F.binary_cross_entropy(e, treatment)
            loss_prop.backward()
            opt_prop.step()

            opt_outcome.zero_grad()
            e = self.propensity_model(covariates).detach()

            X_t = torch.cat([covariates, treatment.unsqueeze(-1)], dim=-1)
            y_pred = self.outcome_model(X_t).squeeze()

            weights = treatment / (e + 1e-10) + (1 - treatment) / (1 - e + 1e-10)
            loss = (weights * (outcome - y_pred) ** 2).mean()
            loss.backward()
            opt_outcome.step()

        return self

    def estimate_effect(
        self,
        covariates: Tensor,
    ) -> Tensor:
        """Estimate treatment effect using MSM."""
        self.outcome_model.eval()

        with torch.no_grad():
            t_0 = torch.zeros(covariates.size(0), 1, device=covariates.device)
            t_1 = torch.ones(covariates.size(0), 1, device=covariates.device)

            X_0 = torch.cat([covariates, t_0], dim=-1)
            X_1 = torch.cat([covariates, t_1], dim=-1)

            y_0 = self.outcome_model(X_0)
            y_1 = self.outcome_model(X_1)

        return (y_1 - y_0).squeeze()


class GFormula:
    """
    G-formula (standardization) estimator.

    Estimates causal effect by standardizing over confounders.
    """

    def __init__(self, outcome_model: Optional[nn.Module] = None):
        self.outcome_model = outcome_model

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
    ) -> "GFormula":
        """Fit g-formula estimator."""
        device = next(covariates.device for _ in [covariates])

        if self.outcome_model is None:
            self.outcome_model = nn.Sequential(
                nn.Linear(covariates.size(1) + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        optimizer = torch.optim.Adam(self.outcome_model.parameters())

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            X_t = torch.cat([covariates, treatment.unsqueeze(-1)], dim=-1)
            y_pred = self.outcome_model(X_t).squeeze()

            loss = F.mse_loss(y_pred, outcome)
            loss.backward()
            optimizer.step()

        return self

    def estimate_effect(
        self,
        covariates: Tensor,
    ) -> Dict[str, float]:
        """Estimate effects using g-formula."""
        self.outcome_model.eval()

        with torch.no_grad():
            t_0 = torch.zeros(covariates.size(0), 1, device=covariates.device)
            t_1 = torch.ones(covariates.size(0), 1, device=covariates.device)

            X_0 = torch.cat([covariates, t_0], dim=-1)
            X_1 = torch.cat([covariates, t_1], dim=-1)

            y_0 = self.outcome_model(X_0).mean().item()
            y_1 = self.outcome_model(X_1).mean().item()

        return {
            "ate": y_1 - y_0,
            "e_y_1": y_1,
            "e_y_0": y_0,
        }


@dataclass
class IPWEstimator:
    """
    Inverse Probability Weighting estimator.

    Estimates ATE using propensity score weighting.
    """

    def __init__(self, propensity_model: Optional[nn.Module] = None):
        self.propensity_model = propensity_model

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
    ) -> "IPWEstimator":
        """Fit propensity model."""
        device = next(covariates.device for _ in [covariates])

        if self.propensity_model is None:
            self.propensity_model = nn.Sequential(
                nn.Linear(covariates.size(1), 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        optimizer = torch.optim.Adam(self.propensity_model.parameters())

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            e = self.propensity_model(covariates).squeeze()
            loss = F.binary_cross_entropy(e, treatment)
            loss.backward()
            optimizer.step()

        return self

    def estimate_ate(
        self,
        treatment: Tensor,
        outcome: Tensor,
    ) -> float:
        """Estimate ATE using IPW."""
        e = self.propensity_model(covariates).squeeze()

        weight_1 = treatment / (e + 1e-10)
        weight_0 = (1 - treatment) / (1 - e + 1e-10)

        ate = (weight_1 * outcome).mean() - (weight_0 * outcome).mean()

        return ate.item()


@dataclass
class AIPWEstimator:
    """
    Augmented Inverse Probability Weighting estimator.

    Doubly robust estimator that combines IPW with outcome modeling.
    """

    def __init__(
        self,
        propensity_model: Optional[nn.Module] = None,
        outcome_model: Optional[nn.Module] = None,
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
    ) -> "AIPWEstimator":
        """Fit both propensity and outcome models."""
        device = next(covariates.device for _ in [covariates])

        if self.propensity_model is None:
            self.propensity_model = nn.Sequential(
                nn.Linear(covariates.size(1), 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        if self.outcome_model is None:
            self.outcome_model = nn.Sequential(
                nn.Linear(covariates.size(1) + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        opt_prop = torch.optim.Adam(self.propensity_model.parameters())
        opt_outcome = torch.optim.Adam(self.outcome_model.parameters())

        for epoch in range(n_epochs):
            opt_prop.zero_grad()
            e = self.propensity_model(covariates).squeeze()
            loss_prop = F.binary_cross_entropy(e, treatment)
            loss_prop.backward()
            opt_prop.step()

            opt_outcome.zero_grad()

            e = self.propensity_model(covariates).detach()

            X_t = torch.cat([covariates, treatment.unsqueeze(-1)], dim=-1)
            mu = self.outcome_model(X_t).squeeze()

            mu_1 = self.outcome_model(
                torch.cat(
                    [covariates, torch.ones_like(treatment.unsqueeze(-1))], dim=-1
                )
            ).squeeze()
            mu_0 = self.outcome_model(
                torch.cat(
                    [covariates, torch.zeros_like(treatment.unsqueeze(-1))], dim=-1
                )
            ).squeeze()

            dr_score = (
                mu_1
                - mu_0
                + treatment * (outcome - mu) / (e + 1e-10)
                - (1 - treatment) * (outcome - mu) / (1 - e + 1e-10)
            )

            loss = dr_score.mean() ** 2
            loss.backward()
            opt_outcome.step()

        return self

    def estimate_ate(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
    ) -> float:
        """Estimate ATE using AIPW."""
        self.propensity_model.eval()
        self.outcome_model.eval()

        with torch.no_grad():
            e = self.propensity_model(covariates).squeeze()

            mu_1 = self.outcome_model(
                torch.cat(
                    [covariates, torch.ones_like(treatment.unsqueeze(-1))], dim=-1
                )
            ).squeeze()
            mu_0 = self.outcome_model(
                torch.cat(
                    [covariates, torch.zeros_like(treatment.unsqueeze(-1))], dim=-1
                )
            ).squeeze()

            ate = (
                mu_1
                - mu_0
                + treatment * (outcome - mu_1) / (e + 1e-10)
                - (1 - treatment) * (outcome - mu_0) / (1 - e + 1e-10)
            ).mean()

        return ate.item()


def causal_calibration(
    treatment: Tensor,
    outcome: Tensor,
    covariates: Tensor,
    method: str = "aipw",
) -> float:
    """
    Main function for causal effect estimation.

    Args:
        treatment: Treatment variable
        outcome: Outcome variable
        covariates: Covariate matrix
        method: Estimation method

    Returns:
        Estimated ATE
    """
    if method == "linear":
        estimator = LinearRegressionEstimator()
        estimator.fit(treatment, outcome, covariates)
        return estimator.effect()

    elif method == "ipw":
        ipw = IPWEstimator()
        ipw.fit(covariates, treatment, outcome)
        return ipw.estimate_ate(treatment, outcome)

    elif method == "aipw":
        aipw = AIPWEstimator()
        aipw.fit(covariates, treatment, outcome)
        return aipw.estimate_ate(covariates, treatment, outcome)

    elif method == "msm":
        msm = MarginalStructuralModel()
        msm.fit(covariates, treatment, outcome)
        return msm.estimate_effect(covariates).mean().item()

    else:
        raise ValueError(f"Unknown method: {method}")
