import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class DoCalculusRule(Enum):
    REMOVEMENT = "removement"
    INSERTION = "insertion"
    INTERVENTION = "intervention"


@dataclass
class DoCalculus:
    graph: Any
    treatment: str = ""
    outcome: str = ""

    def apply_rule1(
        self, P_y_given_x: torch.Tensor, intervention: str, z_set: Set[str]
    ) -> torch.Tensor:
        return P_y_given_x

    def apply_rule2(
        self, P_y_given_xz: torch.Tensor, intervention: str, z_set: Set[str]
    ) -> torch.Tensor:
        return P_y_given_xz

    def apply_rule3(
        self,
        P_y_given_x: torch.Tensor,
        intervention: str,
        z_set: Set[str],
        w_set: Set[str],
    ) -> torch.Tensor:
        return P_y_given_x

    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        if not hasattr(self.graph, "get_ancestors"):
            return False

        ancestors_x = self.graph.get_ancestors(x)
        ancestors_y = self.graph.get_ancestors(y)

        blocked = ancestors_x.intersection(ancestors_y).intersection(z)
        return len(blocked) == 0

    def identify_intervention(
        self, treatment: str, outcome: str, graph: Any
    ) -> Optional[str]:
        self.graph = graph
        self.treatment = treatment
        self.outcome = outcome

        if self._backdoor_criterion(treatment, outcome):
            return self._backdoor_formula(treatment, outcome)

        if self._frontdoor_criterion(treatment, outcome):
            return self._frontdoor_formula(treatment, outcome)

        return None

    def _backdoor_criterion(self, treatment: str, outcome: str) -> bool:
        if not hasattr(self.graph, "get_ancestors"):
            return False

        ancestors_t = self.graph.get_ancestors(treatment)
        ancestors_o = self.graph.get_ancestors(outcome)
        confounders = ancestors_t.intersection(ancestors_o)

        return len(confounders) == 0 or len(self.graph.get_parents(treatment)) == 0

    def _frontdoor_criterion(self, treatment: str, outcome: str) -> bool:
        children_t = self.graph.get_children(treatment)

        for mediator in children_t:
            if self.graph.get_parents(mediator) == [treatment]:
                ancestors_o = self.graph.get_ancestors(outcome)
                if treatment not in ancestors_o:
                    return True
        return False

    def _backdoor_formula(self, treatment: str, outcome: str) -> str:
        return f"P({outcome} | do({treatment})) = Σ_{'confounders'} P({outcome} | {treatment}, confounders) * P(confounders)"

    def _frontdoor_formula(self, treatment: str, outcome: str) -> str:
        children_t = self.graph.get_children(treatment)
        mediator = children_t[0] if children_t else "M"
        return f"P({outcome} | do({treatment})) = Σ_{mediator} P({mediator} | {treatment}) * Σ_{treatment} P({outcome} | {mediator}, {treatment}) * P({treatment})"

    def do(
        self, data: torch.Tensor, treatment_idx: int, treatment_value: float
    ) -> torch.Tensor:
        modified_data = data.clone()
        modified_data[:, treatment_idx] = treatment_value
        return modified_data


@dataclass
class CounterfactualReasoning:
    structural_equations: Dict[str, nn.Module] = field(default_factory=dict)
    graph: Any = None

    def add_structural_equation(self, node: str, model: nn.Module) -> None:
        self.structural_equations[node] = model

    def compute_counterfactual(
        self,
        observed_data: torch.Tensor,
        treatment: str,
        treatment_value: float,
        outcome: str,
        node_to_idx: Dict[str, int],
    ) -> torch.Tensor:
        counterfactual_data = observed_data.clone()

        treat_idx = node_to_idx[treatment]
        counterfactual_data[:, treat_idx] = treatment_value

        return counterfactual_data

    def compute_national_welfare_counterfactual(
        self,
        observed_data: torch.Tensor,
        policy: Dict[str, float],
        outcome: str,
        node_to_idx: Dict[str, int],
    ) -> torch.Tensor:
        cf_data = observed_data.clone()

        for var, value in policy.items():
            if var in node_to_idx:
                cf_data[:, node_to_idx[var]] = value

        return cf_data

    def ab_test(
        self, treatment_data: torch.Tensor, control_data: torch.Tensor, outcome_idx: int
    ) -> Dict[str, float]:
        treatment_outcomes = treatment_data[:, outcome_idx]
        control_outcomes = control_data[:, outcome_idx]

        ate = (treatment_outcomes.mean() - control_outcomes.mean()).item()

        treatment_var = treatment_outcomes.var().item()
        control_var = control_outcomes.var().item()
        pooled_std = np.sqrt((treatment_var + control_var) / 2)

        effect_size = ate / pooled_std if pooled_std > 0 else 0

        return {
            "ate": ate,
            "treatment_mean": treatment_outcomes.mean().item(),
            "control_mean": control_outcomes.mean().item(),
            "effect_size": effect_size,
        }


@dataclass
class CATEEstimator:
    treatment: str
    outcome: str
    data: torch.Tensor
    node_to_idx: Dict[str, int]
    model: Optional[nn.Module] = None

    def __post_init__(self):
        if self.model is None:
            self.model = nn.Sequential(
                nn.Linear(self.data.shape[1] - 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        self.treatment_idx = self.node_to_idx.get(self.treatment)
        self.outcome_idx = self.node_to_idx.get(self.outcome)

    def _prepare_features(
        self, include_treatment: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        treat_idx = self.node_to_idx[self.treatment]
        outcome_idx = self.node_to_idx[self.outcome]

        indices = [
            i
            for i in range(self.data.shape[1])
            if i != outcome_idx and (include_treatment or i != treat_idx)
        ]

        X = self.data[:, indices]
        y = self.data[:, outcome_idx]
        t = self.data[:, treat_idx]

        return X, y, t

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> Dict[str, List[float]]:
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        treatment_mask = t == 1
        control_mask = t == 0

        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            if treatment_mask.sum() > 0 and control_mask.sum() > 0:
                X_treated = X[treatment_mask]
                y_treated = y[treatment_mask]
                X_control = X[control_mask]
                y_control = y[control_mask]

                pred_treated = self.model(X_treated).squeeze()
                pred_control = self.model(X_control).squeeze()

                loss_treated = nn.functional.mse_loss(pred_treated, y_treated)
                loss_control = nn.functional.mse_loss(pred_control, y_control)

                loss = loss_treated + loss_control
            else:
                pred = self.model(X).squeeze()
                loss = nn.functional.mse_loss(pred, y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return {"loss": losses}

    def estimate_cate(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            cate = self.model(X).squeeze()
        return cate

    def estimate_heterogeneous_effects(self, X: torch.Tensor) -> torch.Tensor:
        return self.estimate_cate(X)

    def get_conditional_effect_curve(
        self, X: torch.Tensor, feature_name: str, feature_values: torch.Tensor
    ) -> torch.Tensor:
        self.model.eval()
        effects = []

        with torch.no_grad():
            for val in feature_values:
                X_mod = X.clone()
                X_mod[:, 0] = val
                effect = self.model(X_mod).squeeze()
                effects.append(effect)

        return torch.stack(effects)


class DoublyRobustEstimator:
    def __init__(
        self,
        propensity_model: Optional[nn.Module] = None,
        outcome_model: Optional[nn.Module] = None,
    ):
        self.propensity_model = propensity_model or nn.Linear(10, 1)
        self.outcome_model = outcome_model or nn.Sequential(
            nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor, t: torch.Tensor, epochs: int = 100):
        propensity_opt = torch.optim.Adam(self.propensity_model.parameters(), lr=0.01)
        outcome_opt = torch.optim.Adam(self.outcome_model.parameters(), lr=0.01)

        for _ in range(epochs):
            propensity_opt.zero_grad()
            propensity = torch.sigmoid(self.propensity_model(X).squeeze())
            prop_loss = nn.functional.binary_cross_entropy(propensity, t)
            prop_loss.backward()
            propensity_opt.zero_grad()

            outcome_opt.zero_grad()
            X_with_t = torch.cat([X, t.unsqueeze(1)], dim=1)
            outcome_pred = self.outcome_model(X_with_t).squeeze()
            outcome_loss = nn.functional.mse_loss(outcome_pred, y)
            outcome_loss.backward()
            outcome_opt.step()

    def estimate(
        self, X: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> Dict[str, float]:
        with torch.no_grad():
            propensity = torch.sigmoid(self.propensity_model(X).squeeze())
            propensity = torch.clamp(propensity, 0.01, 0.99)

            X_with_treated = torch.cat([X, torch.ones_like(t).unsqueeze(1)], dim=1)
            X_with_control = torch.cat([X, torch.zeros_like(t).unsqueeze(1)], dim=1)

            mu1 = self.outcome_model(X_with_treated).squeeze()
            mu0 = self.outcome_model(X_with_control).squeeze()

            ipw = (t / propensity + (1 - t) / (1 - propensity)).unsqueeze(1)

            dr = mu1 - mu0 + ipw * (y - mu1) - ipw.shift(1) * (y - mu0)

            cate = dr.mean().item()

            cate_squared = (dr - cate) ** 2
            std_error = torch.sqrt(cate_squared.mean() / len(dr)).item()

        return {"cate": cate, "std_error": std_error}


class MetaLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.tau0_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.tau1_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau0 = self.tau0_net(x).squeeze()
        tau1 = self.tau1_net(x).squeeze()
        return tau0 * (1 - t) + tau1 * t

    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau1_net(x).squeeze() - self.tau0_net(x).squeeze()


class XLearner:
    def __init__(
        self,
        outcome_learner: Optional[nn.Module] = None,
        propensity_learner: Optional[nn.Module] = None,
    ):
        self.outcome_learner = outcome_learner or nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.propensity_learner = propensity_learner or nn.Linear(10, 1)

    def fit(self, X: torch.Tensor, y: torch.Tensor, t: torch.Tensor, epochs: int = 100):
        optimizer = torch.optim.Adam(self.outcome_learner.parameters())

        for _ in range(epochs):
            X_with_t = torch.cat([X, t.unsqueeze(1)], dim=1)
            pred = self.outcome_learner(X_with_t).squeeze()
            loss = nn.functional.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_ite(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu1 = self.outcome_learner(
                torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
            ).squeeze()
            mu0 = self.outcome_learner(
                torch.cat([X, torch.zeros(X.shape[0], 1)], dim=1)
            ).squeeze()

            ite = mu1 - mu0

        return ite
