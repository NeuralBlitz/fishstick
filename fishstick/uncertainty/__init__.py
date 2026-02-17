"""
Uncertainty Quantification & OOD Detection

Methods for measuring uncertainty and detecting out-of-distribution samples.
"""

from typing import Optional, Tuple, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


class MCAlphaDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.

    Enables dropout at inference time for multiple forward passes.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return F.dropout(x, p=self.p, training=True)
        else:
            mask = (torch.rand_like(x) > self.p).float() / (1 - self.p)
            return x * mask


class EnsembleUncertainty(nn.Module):
    """Ensemble-based uncertainty estimation.

    Args:
        models: List of models for ensemble
    """

    def __init__(self, models: nn.ModuleList):
        super().__init__()
        self.models = models

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Get mean prediction and uncertainty.

        Returns:
            mean: Mean predictions
            uncertainty: Variance/entropy as uncertainty measure
        """
        logits_list = []
        for model in self.models:
            with torch.no_grad():
                logits_list.append(model(x))

        logits = torch.stack(logits_list)
        mean = logits.mean(dim=0)
        variance = logits.var(dim=0)

        return mean, variance

    def get_entropy(self, x: Tensor) -> Tensor:
        """Get predictive entropy as uncertainty measure."""
        logits_list = []
        for model in self.models:
            with torch.no_grad():
                logits_list.append(model(x))

        logits = torch.stack(logits_list)
        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)

        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        return entropy


class BayesianNN(nn.Module):
    """Bayesian Neural Network with dropout approximation.

    Args:
        base_model: Base neural network
        n_samples: Number of Monte Carlo samples
    """

    def __init__(self, base_model: nn.Module, n_samples: int = 10):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Get mean and variance predictions."""
        samples = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                samples.append(self.base_model(x))

        samples = torch.stack(samples)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        return mean, variance


class OODDetector:
    """Out-of-Distribution detection base class."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def score(self, x: Tensor) -> Tensor:
        """Compute OOD score. Higher = more likely OOD."""
        raise NotImplementedError


class MaxSoftmaxOODDetector(OODDetector):
    """OOD detection using maximum softmax probability.

    Lower max probability = more likely OOD.
    """

    def score(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]
        return -max_probs


class EnergyOODDetector(OODDetector):
    """OOD detection using energy score.

    Lower energy = more likely ID, Higher energy = more likely OOD.
    """

    def score(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.model(x)
            energy = torch.logsumexp(logits, dim=-1)
        return energy


class MahalanobisOODDetector(OODDetector):
    """OOD detection using Mahalanobis distance.

    Args:
        features: Feature extractor
        num_classes: Number of classes
        mean: Class means (computed during training)
        precision: Precision matrix (computed during training)
    """

    def __init__(
        self,
        model: nn.Module,
        features: nn.Module,
        num_classes: int,
    ):
        super().__init__(model)
        self.features = features
        self.num_classes = num_classes
        self.class_means = None
        self.precision = None

    def compute_statistics(self, id_data: Tensor):
        """Compute class means and precision from ID data."""
        self.model.eval()
        with torch.no_grad():
            embeddings = []
            labels = []
            for x, y in id_data:
                emb = self.features(x)
                embeddings.append(emb)
                labels.append(y)

            embeddings = torch.cat(embeddings)
            labels = torch.cat(labels)

            self.class_means = []
            for c in range(self.num_classes):
                mask = labels == c
                self.class_means.append(embeddings[mask].mean(dim=0))
            self.class_means = torch.stack(self.class_means)

            # Compute covariance
            centered = embeddings - self.class_means[labels]
            cov = centered.T @ centered / len(embeddings)
            self.precision = torch.linalg.pinv(cov + 1e-6 * torch.eye(cov.size(0)))

    def score(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            emb = self.features(x)
            scores = []
            for c in range(self.num_classes):
                diff = emb - self.class_means[c]
                maha = (diff @ self.precision * diff).sum(dim=-1)
                scores.append(maha)
            return torch.stack(scores).min(dim=0)[0]


class EnsembleOODDetector(OODDetector):
    """OOD detection using ensemble disagreement."""

    def score(self, x: Tensor) -> Tensor:
        logits_list = []
        for model in self.models:
            with torch.no_grad():
                logits_list.append(model(x))

        logits = torch.stack(logits_list)
        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)

        variance = probs.var(dim=0).sum(dim=-1)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return variance + entropy


class TemperatureScaledClassifier(nn.Module):
    """Temperature scaling for calibrated predictions.

    Args:
        model: Base classifier
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return logits / self.temperature

    def calibrate(self, logits: Tensor, labels: Tensor):
        """Find optimal temperature using NLL."""
        import scipy.optimize as opt

        def nll(t):
            scaled = logits / t
            return F.cross_entropy(scaled, labels).item()

        result = opt.minimize_scalar(nll, bounds=(0.01, 10), method="bounded")
        self.temperature.data = torch.tensor([result.x])


class DirichletCalibrator(nn.Module):
    """Dirichlet calibration for multi-class predictions.

    Args:
        num_classes: Number of classes
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_classes))

    def forward(self, logits: Tensor) -> Tensor:
        alpha = F.softplus(self.alpha)
        return logits / alpha.unsqueeze(0)


class ConformalPredictor:
    """Conformal prediction for calibrated uncertainty sets.

    Args:
        model: Base model
        alpha: Miscoverage level (1-alpha = confidence)
    """

    def __init__(self, model: nn.Module, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.calibration_scores = []

    def calibrate(self, calib_x: Tensor, calib_y: Tensor):
        """Compute conformity scores on calibration set."""
        with torch.no_grad():
            logits = self.model(calib_x)
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]
            scores = 1 - max_probs
            self.calibration_scores = scores.cpu().numpy()

        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q = np.quantile(self.calibration_scores, q_level)

    def predict(self, x: Tensor) -> Tensor:
        """Get prediction sets."""
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]

            # Get predictions above threshold
            sets = probs > (1 - self.q)
            return sets.long()
