"""
Deep Ensembles for Uncertainty Quantification.

Implements deep ensemble methods for uncertainty estimation including
weighted ensembles, diversity-promoting ensembles, and ensemble
calibration techniques.
"""

from typing import Optional, Tuple, List, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np


class DeepEnsemble(nn.Module):
    """Deep Ensemble for uncertainty estimation.

    Creates an ensemble of independently trained neural networks
    for robust prediction and uncertainty quantification.

    Args:
        base_model_fn: Function that creates a new model instance
        n_models: Number of models in ensemble
        device: Device to place models on
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        n_models: int = 5,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_models = n_models

        self.models = nn.ModuleList([base_model_fn() for _ in range(n_models)])

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.models = self.models.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all models and average predictions."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs).mean(dim=0)

    def predict(
        self,
        x: Tensor,
        return_individual: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Predict with uncertainty estimates.

        Args:
            x: Input tensor
            return_individual: Whether to return individual model predictions

        Returns:
            mean: Mean predictions
            uncertainty: Standard deviation across models
        """
        x = x.to(self.device)

        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))

        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)

        if return_individual:
            return mean, std, outputs
        return mean, std

    def predict_proba(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Predict class probabilities with uncertainty.

        Args:
            x: Input tensor

        Returns:
            probs: Mean class probabilities
            variance: Variance across models
        """
        x = x.to(self.device)

        probs_list = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)

        probs = torch.stack(probs_list)
        mean_probs = probs.mean(dim=0)
        variance = probs.var(dim=0)

        return mean_probs, variance


class WeightedDeepEnsemble(nn.Module):
    """Weighted Deep Ensemble with learned model weights.

    Learns optimal weights for combining ensemble members based on
    their contribution to predictive uncertainty.

    Args:
        base_model_fn: Function that creates a new model instance
        n_models: Number of models in ensemble
        learn_weights: Whether to learn ensemble weights
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        n_models: int = 5,
        learn_weights: bool = True,
    ):
        super().__init__()
        self.n_models = n_models

        self.models = nn.ModuleList([base_model_fn() for _ in range(n_models)])

        if learn_weights:
            self.log_weights = nn.Parameter(torch.zeros(n_models))
        else:
            self.register_parameter("log_weights", None)

    @property
    def weights(self) -> Tensor:
        """Get normalized ensemble weights."""
        if self.log_weights is None:
            return torch.ones(self.n_models) / self.n_models
        return F.softmax(self.log_weights, dim=0)

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict with weighted ensemble."""
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))

        outputs = torch.stack(outputs)
        weights = self.weights.unsqueeze(1).unsqueeze(2)

        mean = (outputs * weights).sum(dim=0)
        variance = ((outputs - mean.unsqueeze(0)) ** 2 * weights).sum(dim=0)

        return mean, torch.sqrt(variance)


class DiversityPromotingEnsemble(nn.Module):
    """Diversity-promoting Deep Ensemble.

    Uses adversarial training and diversity loss to encourage
    ensemble members to make different predictions.

    Args:
        base_model_fn: Function that creates a new model instance
        n_models: Number of models in ensemble
        diversity_weight: Weight for diversity loss
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        n_models: int = 5,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.n_models = n_models
        self.diversity_weight = diversity_weight

        self.models = nn.ModuleList([base_model_fn() for _ in range(n_models)])

    def diversity_loss(self, x: Tensor) -> Tensor:
        """Compute diversity loss to encourage disagreement."""
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))

        outputs = torch.stack(outputs)

        pairwise_cosine = []
        for i in range(self.n_models):
            for j in range(i + 1, self.n_models):
                o1 = outputs[i].flatten(1)
                o2 = outputs[j].flatten(1)
                cos = F.cosine_similarity(o1, o2, dim=1).mean()
                pairwise_cosine.append(cos)

        return torch.stack(pairwise_cosine).mean()

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs).mean(dim=0)


class SnapshotEnsemble(nn.Module):
    """Snapshot Ensemble using cyclical learning rates.

    Collects model snapshots during training and combines them
    for ensemble prediction.

    Args:
        base_model_fn: Function that creates a new model instance
        n_snapshots: Number of snapshots to collect
        cycle_length: Training iterations per snapshot
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        n_snapshots: int = 5,
        cycle_length: int = 1000,
    ):
        super().__init__()
        self.n_snapshots = n_snapshots
        self.cycle_length = cycle_length

        self.model_fn = base_model_fn
        self.snapshots: List[nn.Module] = []

    def snapshot(self, model: nn.Module):
        """Save a snapshot of the model."""
        snapshot = self.model_fn()
        snapshot.load_state_dict(model.state_dict())
        self.snapshots.append(snapshot)

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict using all saved snapshots."""
        if not self.snapshots:
            raise RuntimeError("No snapshots available. Train the model first.")

        outputs = []
        for snapshot in self.snapshots:
            snapshot.eval()
            with torch.no_grad():
                outputs.append(snapshot(x))

        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)

        return mean, std


class FastGeometricEnsemble(nn.Module):
    """Fast Geometric Ensemble (FGE).

    Uses geometric mean of predictions for better calibration
    and uncertainty estimation.

    Args:
        base_model_fn: Function that creates a new model instance
        n_models: Number of models in ensemble
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        n_models: int = 5,
    ):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([base_model_fn() for _ in range(n_models)])

    def predict_proba(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict using geometric mean of probabilities."""
        probs_list = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)

        probs = torch.stack(probs_list)

        geometric_mean = probs.exp().mean(dim=0).log()
        geometric_mean = geometric_mean / geometric_mean.sum(dim=-1, keepdim=True)

        variance = probs.var(dim=0)

        return geometric_mean, variance

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.predict_proba(x)


class EnsembleWithKnockout(nn.Module):
    """Deep Ensemble with Knockout for uncertainty estimation.

    Uses dropout-based knockout to assess ensemble member reliability.

    Args:
        base_model_fn: Function that creates a new model instance
        n_models: Number of models in ensemble
        knockout_prob: Probability of knockout per model
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        n_models: int = 5,
        knockout_prob: float = 0.25,
    ):
        super().__init__()
        self.n_models = n_models
        self.knockout_prob = knockout_prob

        self.models = nn.ModuleList([base_model_fn() for _ in range(n_models)])

    def predict_with_reliability(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict with reliability scores for each model."""
        outputs = []
        reliabilities = []

        for i, model in enumerate(self.models):
            model.eval()

            knockout_mask = torch.rand(len(x)) > self.knockout_prob

            with torch.no_grad():
                out = model(x)
                outputs.append(out)
                reliabilities.append(knockout_mask.float())

        outputs = torch.stack(outputs)
        reliabilities = torch.stack(reliabilities)

        weights = reliabilities / reliabilities.sum(dim=0, keepdim=True)
        weights = weights.unsqueeze(-1)

        mean = (outputs * weights).sum(dim=0)
        std = outputs.std(dim=0)

        return mean, std, reliabilities


class BatchEnsemble(nn.Module):
    """BatchEnsemble for efficient large-scale ensembles.

    Uses rank-1 updates for memory-efficient ensemble of many models.

    Args:
        backbone: Shared feature extractor
        n_models: Number of ensemble members
        out_features: Output dimension
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_models: int = 5,
        out_features: int = 10,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_models = n_models
        self.out_features = out_features

        self.r = nn.Parameter(torch.randn(n_models, 1, out_features))
        self.s = nn.Parameter(torch.randn(n_models, 1, out_features))

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)

        outputs = []
        for i in range(self.n_models):
            r_i = torch.sigmoid(self.r[i])
            s_i = torch.sigmoid(self.s[i])

            output = features * r_i * s_i
            outputs.append(output)

        return torch.stack(outputs).mean(dim=0)

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = []
        for i in range(self.n_models):
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(x)
                r_i = torch.sigmoid(self.r[i])
                s_i = torch.sigmoid(self.s[i])
                output = features * r_i * s_i
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)

        return mean, std


class EnsembleCalibration:
    """Calibration methods for deep ensembles.

    Provides temperature scaling and Platt scaling
    for improving ensemble confidence estimates.
    """

    @staticmethod
    def temperature_scale(
        logits: Tensor,
        temperature: Tensor,
    ) -> Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Model logits
            temperature: Temperature parameter (learned)

        Returns:
            Scaled logits
        """
        return logits / temperature

    @staticmethod
    def platt_scale(
        logits: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """Apply Platt scaling to logits.

        Args:
            logits: Model logits
            weight: Platt scaling weight
            bias: Platt scaling bias

        Returns:
            Scaled logits
        """
        return weight * logits + bias

    @staticmethod
    def fit_temperature(
        model: nn.Module,
        val_loader,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Fit temperature parameter on validation set.

        Args:
            model: Model to calibrate
            val_loader: Validation data loader
            device: Device

        Returns:
            Fitted temperature
        """
        temperature = nn.Parameter(torch.ones(1).to(device))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        model.eval()
        nll_criterion = nn.CrossEntropyLoss()

        def eval():
            loss = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    logits = model(x)
                scaled_logits = temperature * logits
                loss += nll_criterion(scaled_logits, y)
            return loss

        optimizer.step(eval)
        return temperature


class EnsembleUncertaintyMetrics:
    """Metrics for evaluating ensemble uncertainty.

    Provides methods to compute various uncertainty metrics
    from ensemble predictions.
    """

    @staticmethod
    def epistemic_uncertainty(
        samples: Tensor,
    ) -> Tensor:
        """Compute epistemic uncertainty (variance across models).

        Args:
            samples: Ensemble predictions [n_models, batch, ...]

        Returns:
            Epistemic uncertainty
        """
        return samples.var(dim=0)

    @staticmethod
    def aleatoric_uncertainty(
        samples: Tensor,
    ) -> Tensor:
        """Compute aleatoric uncertainty (average variance within models).

        Args:
            samples: Ensemble predictions [n_models, batch, ...]

        Returns:
            Aleatoric uncertainty
        """
        return samples.mean(dim=0)

    @staticmethod
    def ensemble_disagreement(
        samples: Tensor,
    ) -> Tensor:
        """Compute disagreement between ensemble members.

        Args:
            samples: Ensemble predictions [n_models, batch, n_classes]

        Returns:
            Disagreement score
        """
        predictions = samples.argmax(dim=-1)
        disagreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                disagreements.append((predictions[i] != predictions[j]).float())

        return torch.stack(disagreements).float().mean(dim=0)

    @staticmethod
    def expected_calibration_error(
        probs: Tensor,
        labels: Tensor,
        n_bins: int = 15,
    ) -> float:
        """Compute expected calibration error (ECE).

        Args:
            probs: Predicted probabilities [batch, n_classes]
            labels: True labels [batch]
            n_bins: Number of bins for ECE

        Returns:
            ECE score
        """
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * (accuracy_in_bin - avg_confidence_in_bin).abs()

        return ece.item()
