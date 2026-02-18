"""
Epistemic vs Aleatoric Uncertainty Decomposition

Methods for separating epistemic (reducible) and aleatoric (irreducible) uncertainty.
"""

from typing import Optional, Tuple, Dict, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


class EpistemicAleatoricDecomposition:
    """Base class for epistemic/aleatoric uncertainty decomposition.

    Epistemic uncertainty: uncertainty due to lack of knowledge, reducible with more data
    Aleatoric uncertainty: inherent noise in data, irreducible
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute predictions with uncertainty decomposition.

        Returns:
            Tuple of (mean_pred, epistemic_uncertainty, aleatoric_uncertainty)
        """
        raise NotImplementedError


class EnsembleUncertaintyDecomposition(EpistemicAleatoricDecomposition):
    """Decompose uncertainty using deep ensembles.

    Epistemic: variance between ensemble members
    Aleatoric: average variance within each member's predictions

    Args:
        models: List of models in ensemble
        n_classes: Number of classes
    """

    def __init__(self, models: nn.ModuleList, n_classes: int):
        super().__init__(models[0])
        self.models = models
        self.n_classes = n_classes
        for model in models:
            model.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute predictions with uncertainty decomposition.

        Returns:
            Tuple of (mean_probs, epistemic, aleatoric)
        """
        all_probs = []

        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)

        mean_probs = all_probs.mean(dim=0)

        epistemic = all_probs.var(dim=0).mean(dim=-1)

        aleatoric = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return mean_probs, epistemic, aleatoric

    def get_uncertainty_ratio(self, x: Tensor) -> Tensor:
        """Get ratio of epistemic to total uncertainty.

        Returns:
            Ratio of epistemic uncertainty
        """
        _, epistemic, aleatoric = self.forward(x)

        total = epistemic + aleatoric + 1e-10

        return epistemic / total


class MutualInformationUncertainty(EpistemicAleatoricDecomposition):
    """Mutual information based uncertainty decomposition.

    Epistemic = I(y|w, x) - information about label given model parameters
    Aleatoric = H[y|x] - predictive entropy

    Args:
        models: List of models for ensemble
    """

    def __init__(self, models: nn.ModuleList):
        super().__init__(models[0])
        self.models = models
        for model in models:
            model.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute MI-based uncertainty decomposition.

        Returns:
            Tuple of (mean_probs, epistemic, aleatoric)
        """
        all_probs = []

        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)

        mean_probs = all_probs.mean(dim=0)

        aleatoric = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        entropy_per_model = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1)

        expected_entropy = entropy_per_model.mean(dim=0)

        epistemic = aleatoric - expected_entropy

        epistemic = torch.clamp(epistemic, min=0)

        return mean_probs, epistemic, aleatoric


class DropoutUncertaintyDecomposition(EpistemicAleatoricDecomposition):
    """Decompose uncertainty using dropout at inference time (MC Dropout).

    Args:
        model: Model with dropout layers
        n_samples: Number of Monte Carlo samples
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
        dropout_rate: float = 0.5,
    ):
        super().__init__(model)
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self._enable_dropout(model)

    def _enable_dropout(self, model: nn.Module):
        """Enable dropout at inference time."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute predictions with MC dropout uncertainty.

        Returns:
            Tuple of (mean_probs, epistemic, aleatoric)
        """
        self.model.eval()

        samples = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs)

        samples = torch.stack(samples, dim=0)

        mean_probs = samples.mean(dim=0)

        epistemic = samples.var(dim=0).mean(dim=-1)

        aleatoric = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return mean_probs, epistemic, aleatoric


class SWAGUncertaintyDecomposition(EpistemicAleatoricDecomposition):
    """SWAG (SWAG) uncertainty decomposition.

    Uses SWA-Gaussian for efficient Bayesian approximation.

    Args:
        model: Base model
        swag_model: SWAG model
        n_samples: Number of samples
    """

    def __init__(
        self,
        model: nn.Module,
        swag_model: Optional[nn.Module] = None,
        n_samples: int = 20,
    ):
        super().__init__(model)
        self.swag_model = swag_model
        self.n_samples = n_samples

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute SWAG-based uncertainty.

        Returns:
            Tuple of (mean_probs, epistemic, aleatoric)
        """
        self.model.eval()

        if self.swag_model is None:
            raise ValueError("SWAG model required for SWAG uncertainty")

        samples = []

        for _ in range(self.n_samples):
            self.swag_model.sample()
            with torch.no_grad():
                logits = self.swag_model(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs)

        samples = torch.stack(samples, dim=0)

        mean_probs = samples.mean(dim=0)

        epistemic = samples.var(dim=0).mean(dim=-1)

        aleatoric = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return mean_probs, epistemic, aleatoric


class GradientUncertaintyDecomposition(EpistemicAleatoricDecomposition):
    """Gradient-based uncertainty estimation.

    Uses gradient variance as epistemic uncertainty proxy.

    Args:
        model: Base model
        n_samples: Number of gradient samples
    """

    def __init__(self, model: nn.Module, n_samples: int = 10):
        super().__init__(model)
        self.n_samples = n_samples

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute gradient-based uncertainty.

        Returns:
            Tuple of (mean_probs, epistemic, aleatoric)
        """
        self.model.eval()

        predictions = []
        grad_norms = []

        for _ in range(self.n_samples):
            x_temp = x.detach().requires_grad_(True)
            logits = self.model(x_temp)

            probs = F.softmax(logits, dim=-1)
            predictions.append(probs)

            loss = logits.sum()
            loss.backward()

            grad_norm = x_temp.grad.norm(dim=-1)
            grad_norms.append(grad_norm)

        predictions = torch.stack(predictions, dim=0)
        mean_probs = predictions.mean(dim=0)

        grad_norms = torch.stack(grad_norms, dim=0)
        epistemic = grad_norms.var(dim=0)

        aleatoric = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return mean_probs, epistemic, aleatoric


class EvidentialUncertaintyDecomposition(EpistemicAleatoricDecomposition):
    """Evidential deep learning uncertainty decomposition.

    Uses evidence-based uncertainty estimation.

    Args:
        model: Model with evidential output
        n_classes: Number of classes
    """

    def __init__(self, model: nn.Module, n_classes: int):
        super().__init__(model)
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute evidential uncertainty decomposition.

        Returns:
            Tuple of (mean_probs, epistemic, aleatoric)
        """
        logits = self.model(x)

        evidence = F.softplus(logits)

        alpha = evidence + 1

        strength = alpha.sum(dim=-1, keepdim=True)
        mean_probs = alpha / strength

        epistemic = self.n_classes / strength

        dirichlet_entropy = (
            torch.lgamma(alpha.sum(dim=-1))
            - torch.lgamma(alpha).sum(dim=-1)
            + (alpha - 1).sum(dim=-1)
            - (alpha - 1) * (torch.digamma(alpha) - digamma(alpha.sum(dim=-1)))
        )

        aleatoric = dirichlet_entropy

        return mean_probs, epistemic, aleatoric


def digamma(x: Tensor) -> Tensor:
    """Digamma function approximation."""
    return torch.log(x) - 1 / (2 * x) - 1 / (12 * x**2)


class DataUncertaintyEstimator:
    """Estimate aleatoric uncertainty directly from data.

    Uses heteroscedastic regression for data uncertainty.

    Args:
        model: Model that outputs (mean, log_var)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute predictions with data uncertainty.

        Returns:
            Tuple of (mean, log_variance, aleatoric_uncertainty)
        """
        output = self.model(x)

        if isinstance(output, tuple):
            mean, log_var = output
        else:
            mean = output
            log_var = torch.zeros_like(mean)

        aleatoric = torch.exp(log_var)

        return mean, log_var, aleatoric


class SpectralNormalizedDecomposition(EpistemicAleatoricDecomposition):
    """Spectral normalized uncertainty decomposition.

    Uses spectral norm of weights as epistemic uncertainty proxy.

    Args:
        model: Base model
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def compute_spectral_norm(self, x: Tensor) -> Tensor:
        """Compute spectral norm of model Jacobian."""
        self.model.eval()

        x_temp = x.detach().requires_grad_(True)

        logits = self.model(x_temp)

        jacobians = []

        for i in range(logits.size(-1)):
            self.model.zero_grad()
            grad = torch.autograd.grad(
                logits[..., i].sum(),
                x_temp,
                retain_graph=True,
            )[0]
            jacobians.append(grad)

        jacobian = torch.stack(jacobians, dim=-1)

        spectral_norms = torch.linalg.svd(jacobian, compute_uv=False).max(dim=-1)[0]

        return spectral_norms

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute spectral-norm based uncertainty.

        Returns:
            Tuple of (mean_probs, epistemic, aleatoric)
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)
            mean_probs = F.softmax(logits, dim=-1)

        epistemic = self.compute_spectral_norm(x)

        aleatoric = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return mean_probs, epistemic, aleatoric


class NGBoostUncertaintyDecomposition(EpistemicAleatoricDecomposition):
    """NGBoost-style uncertainty decomposition.

    Uses natural gradient boosting for probabilistic predictions.

    Args:
        model: Base model
        n_classes: Number of classes
    """

    def __init__(self, model: nn.Module, n_classes: int):
        super().__init__(model)
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute NGBoost-style uncertainty.

        Returns:
            Tuple of (mean_probs, epistemic, aleatoric)
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

        alpha = F.softplus(logits) + 1

        strength = alpha.sum(dim=-1, keepdim=True)
        mean_probs = alpha / strength

        epistemic = strength.reciprocal()

        dirichlet_param = alpha
        aleatoric = (
            torch.lgamma(dirichlet_param.sum(dim=-1))
            - torch.lgamma(dirichlet_param).sum(dim=-1)
            + (dirichlet_param - 1)
            * (
                torch.digamma(dirichlet_param)
                - torch.digamma(dirichlet_param.sum(dim=-1, keepdim=True))
            )
        ).sum(dim=-1)

        return mean_probs, epistemic.squeeze(-1), aleatoric


class UncertaintyDecompositionAnalyzer:
    """Analyze and compare different uncertainty decomposition methods.

    Args:
        decomposition_methods: List of uncertainty decomposition instances
    """

    def __init__(
        self,
        decomposition_methods: List[EpistemicAleatoricDecomposition],
    ):
        self.methods = decomposition_methods

    def analyze(self, x: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """Analyze uncertainty decomposition across methods.

        Returns:
            Dictionary with results from each method
        """
        results = {}

        for i, method in enumerate(self.methods):
            mean, epistemic, aleatoric = method.forward(x)

            results[f"method_{i}"] = {
                "mean": mean,
                "epistemic": epistemic,
                "aleatoric": aleatoric,
                "total": epistemic + aleatoric,
                "epistemic_ratio": epistemic / (epistemic + aleatoric + 1e-10),
            }

        return results

    def compare_rankings(self, x: Tensor) -> Dict[str, Tensor]:
        """Compare uncertainty rankings across methods.

        Returns:
            Dictionary with rankings
        """
        results = self.analyze(x)

        epistemic_ranking = torch.stack(
            [results[k]["epistemic"] for k in results]
        ).mean(dim=0)

        aleatoric_ranking = torch.stack(
            [results[k]["aleatoric"] for k in results]
        ).mean(dim=0)

        return {
            "epistemic_ranking": epistemic_ranking,
            "aleatoric_ranking": aleatoric_ranking,
            "total_ranking": epistemic_ranking + aleatoric_ranking,
        }
