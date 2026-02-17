"""
Counterfactual Reasoning Engine.

Provides:
- Twin network method
- Structural method for counterfactuals
- Natural direct/indirect effects
- Mediated effects
- Counterfactual uncertainty estimation
"""

from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
import torch
from torch import Tensor, nn
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch.nn.functional as F


@dataclass
class CounterfactualOutcome:
    """Represents a counterfactual outcome."""

    factual: Tensor
    counterfactual: Tensor
    treatment: Tensor
    intervention: Tensor
    potential_outcome_0: Optional[Tensor] = None
    potential_outcome_1: Optional[Tensor] = None

    def individual_treatment_effect(self) -> Tensor:
        """Compute individual treatment effect."""
        return self.counterfactual - self.factual

    def attribution(self, treatment: float = 1.0) -> Tensor:
        """Attribute outcome difference to treatment."""
        if self.potential_outcome_0 is not None:
            return self.counterfactual - self.potential_outcome_0
        return self.individual_treatment_effect()


class CounterfactualEngine(ABC):
    """Abstract base class for counterfactual reasoning."""

    @abstractmethod
    def compute(
        self,
        evidence: Dict[str, Tensor],
        intervention: Dict[str, Tensor],
    ) -> CounterfactualOutcome:
        """Compute counterfactual given evidence and intervention."""
        pass


class TwinNetworkMethod(CounterfactualEngine):
    """
    Twin Network Method for counterfactual inference.

    Uses two network copies: one for factual, one for counterfactual.
    """

    def __init__(
        self,
        outcome_model: nn.Module,
        treatment_model: Optional[nn.Module] = None,
    ):
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model

        self.counterfactual_model = None
        self._clone_model()

    def _clone_model(self) -> None:
        """Create counterfactual copy of model."""
        self.counterfactual_model = type(self.outcome_model)(
            *self.outcome_model.args if hasattr(self.outcome_model, "args") else [],
            **self.outcome_model.kwargs
            if hasattr(self.outcome_model, "kwargs")
            else {},
        )

        self.counterfactual_model.load_state_dict(self.outcome_model.state_dict())

    def compute(
        self,
        evidence: Dict[str, Tensor],
        intervention: Dict[str, Tensor],
    ) -> CounterfactualOutcome:
        """
        Compute counterfactual outcome.

        Args:
            evidence: Observed evidence
            intervention: Intervention to apply

        Returns:
            CounterfactualOutcome
        """
        factual_input = torch.stack(
            [evidence[k] for k in sorted(evidence.keys())], dim=-1
        )

        with torch.no_grad():
            factual = self.outcome_model(factual_input)

        cf_evidence = {**evidence, **intervention}
        cf_input = torch.stack(
            [cf_evidence[k] for k in sorted(cf_evidence.keys())], dim=-1
        )

        with torch.no_grad():
            counterfactual = self.counterfactual_model(cf_input)

        treatment = evidence.get("treatment", torch.zeros_like(factual))
        intervention_value = intervention.get("treatment", torch.ones_like(factual))

        return CounterfactualOutcome(
            factual=factual,
            counterfactual=counterfactual,
            treatment=treatment,
            intervention=intervention_value,
        )

    def compute_potential_outcomes(
        self,
        covariates: Tensor,
        treatment_0: Tensor,
        treatment_1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute potential outcomes Y(0) and Y(1).
        """
        input_0 = torch.cat([covariates, treatment_0], dim=-1)
        input_1 = torch.cat([covariates, treatment_1], dim=-1)

        with torch.no_grad():
            y_0 = self.outcome_model(input_0)
            y_1 = self.outcome_model(input_1)

        return y_0, y_1


class StructuralMethod(CounterfactualEngine):
    """
    Structural Method for counterfactual inference.

    Uses the structural causal model directly to compute counterfactuals.
    """

    def __init__(self, scm: Any):
        self.scm = scm

    def compute(
        self,
        evidence: Dict[str, Tensor],
        intervention: Dict[str, Tensor],
    ) -> CounterfactualOutcome:
        """
        Compute counterfactual using SCM.

        Three steps:
        1. Abduction: Infer noise/parameters from evidence
        2. Action: Apply intervention
        3. Prediction: Compute counterfactual outcome
        """
        factual_samples = self.scm.forward(
            n_samples=evidence[next(iter(evidence))].size(0),
            interventions=None,
        )

        factual_outcome = factual_samples.get(
            "outcome", torch.zeros(evidence[next(iter(evidence))].size(0))
        )

        intervention_indices = {
            self.scm.variable_names.index(k): v for k, v in intervention.items()
        }

        cf_samples = self.scm.forward(
            n_samples=intervention[next(iter(intervention))].size(0),
            interventions=intervention_indices,
        )

        counterfactual = cf_samples.get(
            "outcome", torch.zeros(intervention[next(iter(intervention))].size(0))
        )

        treatment = evidence.get("treatment", torch.zeros_like(factual_outcome))
        intervention_value = intervention.get(
            "treatment", torch.ones_like(counterfactual)
        )

        return CounterfactualOutcome(
            factual=factual_outcome,
            counterfactual=counterfactual,
            treatment=treatment,
            intervention=intervention_value,
        )

    def compute_with_inference(
        self,
        evidence: Dict[str, Tensor],
        intervention: Dict[str, Tensor],
        n_noise_samples: int = 100,
    ) -> Dict[str, Tensor]:
        """
        Compute counterfactual with uncertainty via noise inference.

        Returns:
            Dict with mean and std of counterfactual outcomes
        """
        outcomes = []

        for _ in range(n_noise_samples):
            cf_outcome = self.compute(evidence, intervention)
            outcomes.append(cf_outcome.counterfactual)

        outcomes = torch.stack(outcomes)

        return {
            "mean": outcomes.mean(dim=0),
            "std": outcomes.std(dim=0),
            "outcomes": outcomes,
        }


def compute_counterfactual(
    scm: Any,
    evidence: Dict[str, Tensor],
    intervention: Dict[str, Tensor],
    method: str = "structural",
) -> CounterfactualOutcome:
    """
    Compute counterfactual outcome.

    Args:
        scm: Structural causal model
        evidence: Observed evidence
        intervention: Intervention to apply
        method: Method to use ('structural', 'twin_network')

    Returns:
        CounterfactualOutcome
    """
    if method == "structural":
        engine = StructuralMethod(scm)
    else:
        raise ValueError(f"Unknown method: {method}")

    return engine.compute(evidence, intervention)


def counterfactual_uncertainty(
    scm: Any,
    evidence: Dict[str, Tensor],
    intervention: Dict[str, Tensor],
    n_samples: int = 100,
) -> Dict[str, Tensor]:
    """
    Estimate uncertainty in counterfactual outcomes.

    Uses multiple noise samples to estimate epistemic uncertainty.
    """
    outcomes = []

    for _ in range(n_samples):
        cf = compute_counterfactual(scm, evidence, intervention)
        outcomes.append(cf.counterfactual)

    outcomes = torch.stack(outcomes)

    return {
        "mean": outcomes.mean(dim=0),
        "std": outcomes.std(dim=0),
        "quantile_05": outcomes.quantile(0.05, dim=0),
        "quantile_95": outcomes.quantile(0.95, dim=0),
    }


def natural_direct_effect(
    scm: Any,
    evidence: Dict[str, Tensor],
    treatment: str,
    mediator: str,
    outcome: str,
) -> Tensor:
    """
    Compute natural direct effect.

    NDE = E[Y(1, M(0)) - Y(0, M(0))]

    The effect of treatment on outcome not through the mediator.
    """
    intervention_0 = {mediator: torch.zeros_like(evidence[mediator])}
    intervention_1 = {mediator: torch.ones_like(evidence[mediator])}

    cf_1 = compute_counterfactual(
        scm, evidence, {treatment: torch.ones_like(evidence[treatment])}
    )
    cf_0 = compute_counterfactual(
        scf, evidence, {treatment: torch.zeros_like(evidence[treatment])}
    )

    return cf_1.counterfactual - cf_0.counterfactual


def natural_indirect_effect(
    scm: Any,
    evidence: Dict[str, Tensor],
    treatment: str,
    mediator: str,
    outcome: str,
) -> Tensor:
    """
    Compute natural indirect effect.

    NIE = E[Y(1, M(1)) - Y(1, M(0))]

    The effect of treatment on outcome through the mediator.
    """
    evidence_treated = {**evidence, treatment: torch.ones_like(evidence[treatment])}

    cf_1 = compute_counterfactual(
        scm, evidence_treated, {mediator: torch.ones_like(evidence[mediator])}
    )
    cf_0 = compute_counterfactual(
        scm, evidence_treated, {mediator: torch.zeros_like(evidence[mediator])}
    )

    return cf_1.counterfactual - cf_0.counterfactual


def mediated_effect(
    scm: Any,
    evidence: Dict[str, Tensor],
    treatment: str,
    mediator: str,
    outcome: str,
) -> Dict[str, Tensor]:
    """
    Decompose total effect into direct and indirect effects.

    Returns:
        Dict with total_effect, direct_effect, indirect_effect
    """
    treatment_1 = torch.ones_like(evidence[treatment])
    treatment_0 = torch.zeros_like(evidence[treatment])

    cf_total_1 = compute_counterfactual(scm, evidence, {treatment: treatment_1})
    cf_total_0 = compute_counterfactual(scm, evidence, {treatment: treatment_0})

    total_effect = cf_total_1.counterfactual - cf_total_0.counterfactual

    direct_effect = natural_direct_effect(scm, evidence, treatment, mediator, outcome)
    indirect_effect = natural_indirect_effect(
        scm, evidence, treatment, mediator, outcome
    )

    return {
        "total_effect": total_effect,
        "direct_effect": direct_effect,
        "indirect_effect": indirect_effect,
        "proportion_mediated": indirect_effect / (total_effect + 1e-10),
    }


class CounterfactualVAE(nn.Module):
    """
    VAE-based counterfactual reasoning.

    Learns a latent representation that supports counterfactual generation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.treatment_head = nn.Linear(latent_dim, 1)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode to latent distribution."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decode(z)
        treatment_pred = torch.sigmoid(self.treatment_head(z))

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "treatment_pred": treatment_pred,
        }

    def generate_counterfactual(
        self,
        x: Tensor,
        treatment_intervention: float,
    ) -> Tensor:
        """
        Generate counterfactual outcome.

        Args:
            x: Input features
            treatment_intervention: Treatment value to intervene on (0 or 1)

        Returns:
            Counterfactual outcome
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        z[:, 0] = treatment_intervention

        cf_outcome = self.decode(z)

        return cf_outcome


class IndividualTreatmentEffectEstimator:
    """
    Estimate individual treatment effects using counterfactuals.
    """

    def __init__(
        self, model: Union[TwinNetworkMethod, StructuralMethod, CounterfactualVAE]
    ):
        self.model = model

    def estimate_ite(
        self,
        covariates: Tensor,
    ) -> Tensor:
        """
        Estimate individual treatment effect.

        ITE(x) = E[Y(1) - Y(0) | X = x]
        """
        treatment_0 = torch.zeros_like(covariates[:, :1])
        treatment_1 = torch.ones_like(covariates[:, :1])

        if isinstance(self.model, CounterfactualVAE):
            cf_0 = self.model.generate_counterfactual(covariates, 0.0)
            cf_1 = self.model.generate_counterfactual(covariates, 1.0)
        elif isinstance(self.model, TwinNetworkMethod):
            evidence = {"covariates": covariates}
            cf_0 = self.model.compute(
                evidence, {"treatment": treatment_0}
            ).counterfactual
            cf_1 = self.model.compute(
                evidence, {"treatment": treatment_1}
            ).counterfactual
        else:
            raise ValueError("Unknown model type")

        return cf_1 - cf_0

    def estimate_cate(
        self,
        covariates: Tensor,
        subgroup: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        """
        Estimate conditional average treatment effect.
        """
        ite = self.estimate_ite(covariates)

        if subgroup is not None:
            mask = subgroup(covariates)
            return ite[mask].mean()

        return ite.mean()
