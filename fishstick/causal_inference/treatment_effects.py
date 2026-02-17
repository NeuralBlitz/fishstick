"""
Heterogeneous Treatment Effect Estimation.

Provides:
- Meta-learners (S, T, X, DR)
- Causal forests
- Treatment effect validation
- Uplift modeling
"""

from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


@dataclass
class TreatmentEffectEstimator(ABC):
    """Abstract base class for treatment effect estimators."""

    @abstractmethod
    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
    ) -> "TreatmentEffectEstimator":
        """Fit the estimator."""
        pass

    @abstractmethod
    def predict(
        self,
        covariates: Tensor,
    ) -> Tensor:
        """Predict treatment effects."""
        pass


class CATEEstimator(TreatmentEffectEstimator):
    """
    Conditional Average Treatment Effect Estimator.

    Estimates E[Y(1) - Y(0) | X].
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        hidden_dim: int = 128,
    ):
        self.base_model = base_model
        self.hidden_dim = hidden_dim

        if base_model is None:
            self.base_model = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.optimizer = torch.optim.Adam(self.base_model.parameters())

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
        batch_size: int = 256,
    ) -> "CATEEstimator":
        """Fit CATE estimator."""
        device = next(self.base_model.parameters()).device

        covariates = covariates.to(device)
        treatment = treatment.to(device).unsqueeze(-1)
        outcome = outcome.to(device).unsqueeze(-1)

        self.base_model.train()

        for epoch in range(n_epochs):
            indices = torch.randperm(covariates.size(0))[:batch_size]

            x = covariates[indices]
            t = treatment[indices]
            y = outcome[indices]

            self.optimizer.zero_grad()

            input_features = torch.cat([x, t], dim=-1)
            pred = self.base_model(input_features)

            loss = F.mse_loss(pred, y)
            loss.backward()
            self.optimizer.step()

        return self

    def predict(
        self,
        covariates: Tensor,
    ) -> Tensor:
        """Predict CATE."""
        self.base_model.eval()

        device = next(self.base_model.parameters()).device
        covariates = covariates.to(device)

        with torch.no_grad():
            t_0 = torch.zeros(covariates.size(0), 1, device=device)
            t_1 = torch.ones(covariates.size(0), 1, device=device)

            input_0 = torch.cat([covariates, t_0], dim=-1)
            input_1 = torch.cat([covariates, t_1], dim=-1)

            y_0 = self.base_model(input_0)
            y_1 = self.base_model(input_1)

            cate = y_1 - y_0

        return cate.squeeze()


class MetaLearner(ABC):
    """Base class for meta-learners."""

    def __init__(self, outcome_model: Optional[nn.Module] = None):
        self.outcome_model = outcome_model

    @abstractmethod
    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
    ) -> "MetaLearner":
        pass

    @abstractmethod
    def predict(
        self,
        covariates: Tensor,
    ) -> Tensor:
        pass


class SLearner(MetaLearner):
    """
    S-Learner: Single model approach.

    Trains a single model on (X, T) -> Y, then predicts with T=0 and T=1.
    """

    def __init__(
        self,
        outcome_model: Optional[nn.Module] = None,
        hidden_dim: int = 128,
    ):
        super().__init__(outcome_model)

        if outcome_model is None:
            self.outcome_model = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.optimizer = torch.optim.Adam(self.outcome_model.parameters())

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
    ) -> "SLearner":
        """Fit S-learner."""
        device = next(self.outcome_model.parameters()).device

        covariates = covariates.to(device)
        treatment = treatment.to(device).unsqueeze(-1)
        outcome = outcome.to(device).unsqueeze(-1)

        input_data = torch.cat([covariates, treatment], dim=-1)

        self.outcome_model.train()

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            pred = self.outcome_model(input_data)
            loss = F.mse_loss(pred, outcome)
            loss.backward()
            self.optimizer.step()

        return self

    def predict(
        self,
        covariates: Tensor,
    ) -> Tensor:
        """Predict treatment effects."""
        self.outcome_model.eval()
        device = next(self.outcome_model.parameters()).device

        covariates = covariates.to(device)

        with torch.no_grad():
            t_0 = torch.zeros(covariates.size(0), 1, device=device)
            t_1 = torch.ones(covariates.size(0), 1, device=device)

            input_0 = torch.cat([covariates, t_0], dim=-1)
            input_1 = torch.cat([covariates, t_1], dim=-1)

            y_0 = self.outcome_model(input_0)
            y_1 = self.outcome_model(input_1)

        return (y_1 - y_0).squeeze()


class TLearner(MetaLearner):
    """
    T-Learner: Two models approach.

    Trains separate models for treated and control groups.
    """

    def __init__(
        self,
        outcome_model: Optional[nn.Module] = None,
        hidden_dim: int = 128,
    ):
        super().__init__(outcome_model)

        if outcome_model is None:
            self.model_0 = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.model_1 = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.model_0 = outcome_model
            self.model_1 = type(outcome_model)(*outcome_model.args)

        self.optimizer_0 = torch.optim.Adam(self.model_0.parameters())
        self.optimizer_1 = torch.optim.Adam(self.model_1.parameters())

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
    ) -> "TLearner":
        """Fit T-learner."""
        device = next(self.model_0.parameters()).device

        covariates = covariates.to(device)
        treatment = treatment.to(device)
        outcome = outcome.to(device).unsqueeze(-1)

        mask_0 = treatment == 0
        mask_1 = treatment == 1

        self.model_0.train()
        self.model_1.train()

        for epoch in range(n_epochs):
            if mask_0.sum() > 0:
                self.optimizer_0.zero_grad()
                x_0 = covariates[mask_0]
                y_0 = outcome[mask_0]
                pred_0 = self.model_0(x_0)
                loss_0 = F.mse_loss(pred_0, y_0)
                loss_0.backward()
                self.optimizer_0.step()

            if mask_1.sum() > 0:
                self.optimizer_1.zero_grad()
                x_1 = covariates[mask_1]
                y_1 = outcome[mask_1]
                pred_1 = self.model_1(x_1)
                loss_1 = F.mse_loss(pred_1, y_1)
                loss_1.backward()
                self.optimizer_1.step()

        return self

    def predict(
        self,
        covariates: Tensor,
    ) -> Tensor:
        """Predict treatment effects."""
        self.model_0.eval()
        self.model_1.eval()

        device = next(self.model_0.parameters()).device
        covariates = covariates.to(device)

        with torch.no_grad():
            y_0 = self.model_0(covariates)
            y_1 = self.model_1(covariates)

        return (y_1 - y_0).squeeze()


class XLearner(MetaLearner):
    """
    X-Learner: Cross-fitting approach.

    Combines T-learner with imputation of counterfactuals.
    """

    def __init__(
        self,
        outcome_model: Optional[nn.Module] = None,
        hidden_dim: int = 128,
    ):
        super().__init__(outcome_model)
        self.hidden_dim = hidden_dim
        self.t_learner = TLearner(hidden_dim=hidden_dim)
        self.model_tau_0 = None
        self.model_tau_1 = None
        self.propensity_model = None

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
    ) -> "XLearner":
        """Fit X-learner."""
        self.t_learner.fit(covariates, treatment, outcome, n_epochs)

        device = next(covariates.device for _ in [covariates])

        mask_0 = treatment == 0
        mask_1 = treatment == 1

        y_1_pred = self.t_learner.model_1(covariates[mask_0])
        y_0_pred = self.t_learner.model_0(covariates[mask_1])

        tau_0 = outcome[mask_0] - y_1_pred.squeeze()
        tau_1 = y_0_pred.squeeze() - outcome[mask_1]

        self.model_tau_0 = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.model_tau_1 = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        optimizer_0 = torch.optim.Adam(self.model_tau_0.parameters())
        optimizer_1 = torch.optim.Adam(self.model_tau_1.parameters())

        for epoch in range(n_epochs):
            optimizer_0.zero_grad()
            tau_pred = self.model_tau_0(covariates[mask_0])
            loss = F.mse_loss(tau_pred.squeeze(), tau_0)
            loss.backward()
            optimizer_0.step()

            optimizer_1.zero_grad()
            tau_pred = self.model_tau_1(covariates[mask_1])
            loss = F.mse_loss(tau_pred.squeeze(), tau_1)
            loss.backward()
            optimizer_1.step()

        return self

    def predict(
        self,
        covariates: Tensor,
    ) -> Tensor:
        """Predict treatment effects."""
        device = next(covariates.device)

        with torch.no_grad():
            tau_0 = self.model_tau_0(covariates).squeeze()
            tau_1 = self.model_tau_1(covariates).squeeze()

            if self.propensity_model is not None:
                propensity = self.propensity_model(covariates)
                tau = propensity * tau_0 + (1 - propensity) * tau_1
            else:
                tau = (tau_0 + tau_1) / 2

        return tau


class DRLearner(MetaLearner):
    """
    DR-Learner: Doubly robust approach.

    Uses propensity scores and outcome models for efficient estimation.
    """

    def __init__(
        self,
        outcome_model: Optional[nn.Module] = None,
        hidden_dim: int = 128,
    ):
        super().__init__(outcome_model)
        self.hidden_dim = hidden_dim

        self.mu_0_model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mu_1_model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.propensity_model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.tau_model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer_all = torch.optim.Adam(
            list(self.mu_0_model.parameters())
            + list(self.mu_1_model.parameters())
            + list(self.propensity_model.parameters())
            + list(self.tau_model.parameters())
        )

    def fit(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_epochs: int = 1000,
    ) -> "DRLearner":
        """Fit DR-learner."""
        device = next(self.mu_0_model.parameters()).device

        covariates = covariates.to(device)
        treatment = treatment.to(device).unsqueeze(-1)
        outcome = outcome.to(device).unsqueeze(-1)

        self.mu_0_model.train()
        self.mu_1_model.train()
        self.propensity_model.train()

        for epoch in range(n_epochs):
            self.optimizer_all.zero_grad()

            mu_0 = self.mu_0_model(covariates)
            mu_1 = self.mu_1_model(covariates)
            e = self.propensity_model(covariates).squeeze()

            t = treatment.squeeze()

            mu_0_pred = mu_0.squeeze()
            mu_1_pred = mu_1.squeeze()

            dr_score = (
                mu_1_pred
                - mu_0_pred
                + t * (outcome.squeeze() - mu_1_pred) / (e + 1e-10)
                - (1 - t) * (outcome.squeeze() - mu_0_pred) / (1 - e + 1e-10)
            )

            tau_pred = self.tau_model(covariates).squeeze()
            loss = F.mse_loss(tau_pred, dr_score)

            loss.backward()
            self.optimizer_all.step()

        return self

    def predict(
        self,
        covariates: Tensor,
    ) -> Tensor:
        """Predict treatment effects."""
        self.tau_model.eval()
        device = next(self.tau_model.parameters()).device

        covariates = covariates.to(device)

        with torch.no_grad():
            tau = self.tau_model(covariates)

        return tau.squeeze()


class CausalForest:
    """
    Causal Forest for heterogeneous treatment effects.

    Uses random forest with causal adjustment.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.forest_0 = None
        self.forest_1 = None

    def fit(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> "CausalForest":
        """Fit causal forest."""
        mask_0 = treatment == 0
        mask_1 = treatment == 1

        self.forest_0 = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.forest_1 = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

        self.forest_0.fit(covariates[mask_0], outcome[mask_0])
        self.forest_1.fit(covariates[mask_1], outcome[mask_1])

        return self

    def predict(
        self,
        covariates: np.ndarray,
    ) -> np.ndarray:
        """Predict treatment effects."""
        y_0 = self.forest_0.predict(covariates)
        y_1 = self.forest_1.predict(covariates)

        return y_1 - y_0


def average_treatment_effect(
    treatment: Tensor,
    outcome: Tensor,
) -> float:
    """
    Compute average treatment effect.

    ATE = E[Y(1) - Y(0)]
    """
    mask_1 = treatment == 1
    mask_0 = treatment == 0

    y_1 = outcome[mask_1].mean().item()
    y_0 = outcome[mask_0].mean().item()

    return y_1 - y_0


def conditional_average_treatment_effect(
    covariates: Tensor,
    treatment: Tensor,
    outcome: Tensor,
    model: Optional[nn.Module] = None,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Estimate conditional average treatment effects.
    """
    if model is not None:
        cate_estimator = CATEEstimator(base_model=model)
        cate_estimator.fit(covariates, treatment, outcome)
        cate_preds = cate_estimator.predict(covariates)

        return {
            "mean_cate": cate_preds.mean().item(),
            "std_cate": cate_preds.std().item(),
        }

    covariate_np = covariates.numpy()
    treatment_np = treatment.numpy()
    outcome_np = outcome.numpy()

    bin_edges = np.percentile(covariate_np[:, 0], np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(covariate_np[:, 0], bin_edges[:-1])

    cate_by_bin = {}
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() > 0:
            ate = average_treatment_effect(
                torch.tensor(treatment_np[mask]), torch.tensor(outcome_np[mask])
            )
            cate_by_bin[f"bin_{bin_idx}"] = ate

    return cate_by_bin


def att(
    treatment: Tensor,
    outcome: Tensor,
) -> float:
    """
    Compute Average Treatment Effect on the Treated.

    ATT = E[Y(1) - Y(0) | T = 1]
    """
    mask_1 = treatment == 1

    y_1_treated = outcome[mask_1].mean()

    y_0_treated = outcome[mask_1 == False].mean() if (treatment == 0).sum() > 0 else 0

    return (y_1_treated - y_0_treated).item()


class HTEValidator:
    """
    Validator for heterogeneous treatment effect estimation.
    """

    def __init__(self, estimator: TreatmentEffectEstimator):
        self.estimator = estimator

    def cross_validate(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """Cross-validate treatment effect estimator."""
        n = covariates.size(0)
        fold_size = n // n_folds

        indices = torch.randperm(n)

        errors = []

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

            val_indices = indices[val_start:val_end]
            train_indices = torch.cat([indices[:val_start], indices[val_end:]])

            cov_train = covariates[train_indices]
            t_train = treatment[train_indices]
            y_train = outcome[train_indices]

            cov_val = covariates[val_indices]
            t_val = treatment[val_indices]
            y_val = outcome[val_indices]

            self.estimator.fit(cov_train, t_train, y_train)

            cate_pred = self.estimator.predict(cov_val)

            y_1 = y_val[t_val == 1].mean() if (t_val == 1).sum() > 0 else 0
            y_0 = y_val[t_val == 0].mean() if (t_val == 0).sum() > 0 else 0
            cate_true = y_1 - y_0

            error = (cate_pred.mean() - cate_true).abs().item()
            errors.append(error)

        return {
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
        }

    def validate_on_subgroup(
        self,
        covariates: Tensor,
        treatment: Tensor,
        outcome: Tensor,
        subgroup_fn: Callable[[Tensor], Tensor],
    ) -> Dict[str, float]:
        """Validate on a specific subgroup."""
        self.estimator.fit(covariates, treatment, outcome)

        cate_pred = self.estimator.predict(covariates)

        subgroup_mask = subgroup_fn(covariates)

        return {
            "subgroup_cate_pred": cate_pred[subgroup_mask].mean().item(),
            "subgroup_size": subgroup_mask.sum().item(),
        }


def uplift_model(
    covariates: Tensor,
    treatment: Tensor,
    outcome: Tensor,
    method: str = "t_learner",
) -> Tensor:
    """
    Train an uplift model.

    Args:
        covariates: Covariate matrix
        treatment: Treatment indicator
        outcome: Outcome (conversion, etc.)
        method: Method to use ('s', 't', 'x', 'dr', 'causal_forest')

    Returns:
        Uplift predictions
    """
    if method == "s":
        model = SLearner()
    elif method == "t":
        model = TLearner()
    elif method == "x":
        model = XLearner()
    elif method == "dr":
        model = DRLearner()
    elif method == "causal_forest":
        model = CausalForest()
        model.fit(
            covariates.numpy(),
            treatment.numpy(),
            outcome.numpy(),
        )
        return torch.tensor(model.predict(covariates.numpy()))
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(covariates, treatment, outcome)

    return model.predict(covariates)
