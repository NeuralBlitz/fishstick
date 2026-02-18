"""
Bayesian Model Averaging Module.

Implements Bayesian Model Averaging (BMA) for combining predictions
from multiple models with learned or computed model weights.
"""

from typing import Optional, Tuple, List, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np


class BayesianModelAveraging(nn.Module):
    """Bayesian Model Averaging for combining multiple models.

    Combines predictions from multiple models weighted by their
    posterior model probabilities.

    Args:
        models: List of models to average
        weights: Optional model weights (log probabilities)
    """

    def __init__(
        self,
        models: Optional[nn.ModuleList] = None,
        weights: Optional[Tensor] = None,
    ):
        super().__init__()
        self.models = models if models is not None else nn.ModuleList()

        if weights is not None:
            self.log_weights = nn.Parameter(weights)
        else:
            self.log_weights = nn.Parameter(
                torch.zeros(len(self.models)) if models else torch.zeros(0)
            )

    @property
    def weights(self) -> Tensor:
        """Get normalized model weights."""
        return F.softmax(self.log_weights, dim=0)

    def add_model(self, model: nn.Module):
        """Add a model to the ensemble."""
        self.models.append(model)
        new_weights = torch.zeros(len(self.models))
        new_weights[: len(self.log_weights)] = self.log_weights.data
        self.log_weights = nn.Parameter(new_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with weighted averaging."""
        if len(self.models) == 0:
            raise RuntimeError("No models in ensemble")

        outputs = []
        for model in self.models:
            outputs.append(model(x))

        outputs = torch.stack(outputs)
        weights = self.weights.unsqueeze(1).unsqueeze(2)

        return (outputs * weights).sum(dim=0)

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: int = 100,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict with uncertainty decomposition.

        Args:
            x: Input tensor
            n_samples: Number of samples for uncertainty estimation

        Returns:
            mean: Weighted mean prediction
            model_uncertainty: Epistemic uncertainty (model disagreement)
            data_uncertainty: Aleatoric uncertainty (within-model variance)
        """
        if len(self.models) == 0:
            raise RuntimeError("No models in ensemble")

        all_outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = []
                for _ in range(n_samples):
                    outputs.append(model(x))
                all_outputs.append(torch.stack(outputs))

        all_outputs = torch.stack(all_outputs)

        weights = self.weights.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mean = (all_outputs * weights).sum(dim=0)

        model_uncertainty = (all_outputs.var(dim=0) * weights).sum(dim=0)
        data_uncertainty = all_outputs.mean(dim=1).var(dim=0)

        return mean, model_uncertainty, data_uncertainty


class ModelWeightOptimizer:
    """Optimizer for learning Bayesian Model Averaging weights.

    Learns optimal model weights using evidence lower bound (ELBO)
    or held-out validation performance.
    """

    def __init__(
        self,
        models: List[nn.Module],
        n_classes: int,
    ):
        self.models = models
        self.n_classes = n_classes
        self.n_models = len(models)

        self.log_weights = nn.Parameter(torch.zeros(self.n_models))

    @property
    def weights(self) -> Tensor:
        return F.softmax(self.log_weights, dim=0)

    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 50,
        lr: float = 0.01,
    ):
        """Fit model weights using validation performance.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam([self.log_weights], lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_weights = None
        best_val_acc = 0.0

        for epoch in range(n_epochs):
            self.train()
            train_loss = 0.0

            for x, y in train_loader:
                optimizer.zero_grad()

                outputs = []
                for model in self.models:
                    outputs.append(model(x))

                outputs = torch.stack(outputs)
                weights = F.softmax(self.log_weights, dim=0).unsqueeze(1).unsqueeze(2)

                weighted_output = (outputs * weights).sum(dim=0)
                loss = criterion(weighted_output, y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            val_acc = self.evaluate(val_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.log_weights.data.clone()

        if best_weights is not None:
            self.log_weights.data = best_weights

    def evaluate(self, data_loader) -> float:
        """Evaluate on data loader."""
        correct = 0
        total = 0

        for x, y in data_loader:
            outputs = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    outputs.append(model(x))

            outputs = torch.stack(outputs)
            weights = self.softmax(self.log_weights, dim=0).unsqueeze(1).unsqueeze(2)
            weighted_output = (outputs * weights).sum(dim=0)

            _, predicted = weighted_output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return correct / total


class HyperpriorBMA(nn.Module):
    """Bayesian Model Averaging with hyperpriors.

    Uses hierarchical Bayesian modeling to learn model weights
    with uncertainty over the weights themselves.

    Args:
        models: List of models to average
        prior_alpha: Prior concentration parameter
    """

    def __init__(
        self,
        models: List[nn.Module],
        prior_alpha: float = 1.0,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        self.prior_alpha = prior_alpha

        self.alpha = nn.Parameter(torch.ones(self.n_models) * prior_alpha)

    def sample_weights(self, n_samples: int = 10) -> Tensor:
        """Sample weights from Dirichlet posterior.

        Args:
            n_samples: Number of weight samples

        Returns:
            Weight samples [n_samples, n_models]
        """
        alpha = F.softplus(self.alpha)
        return torch.distributions.Dirichlet(alpha).sample((n_samples,))

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """Predict with uncertainty from weight posterior.

        Args:
            x: Input tensor
            n_samples: Number of weight samples

        Returns:
            mean: Mean prediction
            uncertainty: Model weight uncertainty
        """
        weights = self.sample_weights(n_samples)

        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))

        outputs = torch.stack(outputs)

        predictions = []
        for w in weights:
            w = w.unsqueeze(1).unsqueeze(2)
            pred = (outputs * w).sum(dim=0)
            predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)

        return mean, uncertainty


class MixtureOfExperts(nn.Module):
    """Mixture of Experts with gating network.

    Uses a learned gating network to weight expert predictions
    based on input-dependent model selection.

    Args:
        experts: List of expert models
        gating_network: Network that produces input-dependent weights
    """

    def __init__(
        self,
        experts: List[nn.Module],
        gating_network: nn.Module,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating = gating_network
        self.n_experts = len(experts)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with gating weights.

        Args:
            x: Input tensor

        Returns:
            output: Gated expert output
            weights: Gating weights
        """
        gate_logits = self.gating(x)
        weights = F.softmax(gate_logits, dim=1)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        expert_outputs = torch.stack(expert_outputs, dim=1)

        weights = weights.unsqueeze(-1)
        output = (expert_outputs * weights).sum(dim=1)

        return output, weights

    def predict_with_uncertainty(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict with uncertainty from gating network.

        Args:
            x: Input tensor

        Returns:
            mean: Mean prediction
            uncertainty: Gating-based uncertainty
            weights: Gating weights
        """
        output, weights = self.forward(x)

        expert_outputs = []
        for expert in self.experts:
            expert.eval()
            with torch.no_grad():
                expert_outputs.append(expert(x))

        expert_outputs = torch.stack(expert_outputs, dim=1)

        variance = (expert_outputs.var(dim=1) * weights).sum(dim=1)

        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=1)

        return output, variance + entropy, weights


class ModelSelectionBMA(nn.Module):
    """Bayesian Model Selection with Automatic Relevance Determination.

    Uses ARD-like priors to automatically determine model relevance
    and perform model selection within the averaging framework.

    Args:
        models: List of candidate models
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        self.log_alpha = nn.Parameter(torch.zeros(self.n_models))

    def get_model_weights(self) -> Tensor:
        """Compute model weights based on relevance.

        Returns:
            Model weights (higher = more relevant)
        """
        alpha = F.softplus(self.log_alpha)
        return 1.0 / (alpha + 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with relevance-weighted averaging."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        outputs = torch.stack(outputs)
        weights = self.get_model_weights()
        weights = weights.unsqueeze(1).unsqueeze(2)

        return (outputs * weights).sum(dim=0)


class StackingBMA(nn.Module):
    """Stacked Bayesian Model Averaging.

    Uses a meta-learner to combine base model predictions
    with learned model weights.

    Args:
        base_models: List of base models
        meta_learner: Meta-learner network
    """

    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: nn.Module,
    ):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_learner = meta_learner

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through stacking."""
        base_outputs = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                base_outputs.append(model(x))

        stacked = torch.stack(base_outputs, dim=1)

        return self.meta_learner(stacked)


class OnlineBMA:
    """Online Bayesian Model Averaging for streaming data.

    Updates model weights incrementally as new data arrives.

    Args:
        models: List of candidate models
        prior_weights: Initial model weights
    """

    def __init__(
        self,
        models: List[nn.Module],
        prior_weights: Optional[Tensor] = None,
    ):
        self.models = models
        self.n_models = len(models)

        if prior_weights is None:
            self.weights = torch.ones(self.n_models) / self.n_models
        else:
            self.weights = prior_weights

    def update(self, x: Tensor, y: Tensor):
        """Update model weights based on new data.

        Args:
            x: Input data
            y: Target data
        """
        likelihoods = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)

                if output.dim() > 1 and output.size(-1) > 1:
                    probs = F.softmax(output, dim=-1)
                    pred = output.argmax(dim=-1)
                    correct = pred.eq(y).float()
                    likelihood = correct.mean()
                else:
                    mse = (output.squeeze() - y).pow(2).mean()
                    likelihood = torch.exp(-mse)

                likelihoods.append(likelihood.item())

        likelihoods = torch.tensor(likelihoods)

        self.weights = self.weights * likelihoods
        self.weights = self.weights / self.weights.sum()

    def predict(self, x: Tensor) -> Tensor:
        """Predict using weighted average."""
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))

        outputs = torch.stack(outputs)

        return (outputs * self.weights.view(-1, 1, 1)).sum(dim=0)


class BootstrapBMA(nn.Module):
    """Bayesian Model Averaging with bootstrap ensembles.

    Creates bootstrap ensembles of each model and combines
    them using Bayesian averaging.

    Args:
        model_fn: Function to create models
        n_bootstrap: Number of bootstrap samples per model
        n_models: Number of models
    """

    def __init__(
        self,
        model_fn,
        n_bootstrap: int = 5,
        n_models: int = 3,
    ):
        super().__init__()
        self.model_fn = model_fn
        self.n_bootstrap = n_bootstrap
        self.n_models = n_models

        self.bootstrap_ensembles = nn.ModuleList(
            [
                nn.ModuleList([model_fn() for _ in range(n_bootstrap)])
                for _ in range(n_models)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all bootstrap ensembles."""
        all_outputs = []

        for ensemble in self.bootstrap_ensembles:
            outputs = []
            for model in ensemble:
                model.eval()
                with torch.no_grad():
                    outputs.append(model(x))
            all_outputs.append(torch.stack(outputs).mean(dim=0))

        all_outputs = torch.stack(all_outputs)

        return all_outputs.mean(dim=0)


class BMAContinuous:
    """Continuous Bayesian Model Averaging for infinite model space.

    Models the space of possible models as continuous and
    averages over the model distribution.
    """

    def __init__(
        self,
        model_family: str = "linear",
        prior_variance: float = 1.0,
    ):
        self.model_family = model_family
        self.prior_variance = prior_variance

        self.model_params: List[Tensor] = []
        self.model_evidences: List[float] = []

    def add_model(self, params: Tensor, evidence: float):
        """Add a model to the ensemble.

        Args:
            params: Model parameters
            evidence: Model evidence (marginal likelihood)
        """
        self.model_params.append(params)
        self.model_evidences.append(evidence)

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict using Bayesian model averaging.

        Args:
            x: Input tensor

        Returns:
            Mean prediction and uncertainty
        """
        if not self.model_params:
            raise RuntimeError("No models available")

        evidences = torch.tensor(self.model_evidences)
        weights = F.softmax(evidences, dim=0)

        predictions = []
        for params in self.model_params:
            pred = self._predict_with_params(x, params)
            predictions.append(pred)

        predictions = torch.stack(predictions)

        mean = (predictions * weights.view(-1, 1, 1)).sum(dim=0)
        variance = (predictions.var(dim=0) * weights).sum(dim=0) + (
            (predictions - mean).pow(2) * weights
        ).sum(dim=0)

        return mean, torch.sqrt(variance)

    def _predict_with_params(self, x: Tensor, params: Tensor) -> Tensor:
        """Make prediction with given parameters."""
        raise NotImplementedError
