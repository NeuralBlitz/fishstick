"""
Monte Carlo Dropout Module for Uncertainty Estimation.

Implements various dropout-based uncertainty estimation methods
including MC dropout, dropout as Bayesian approximation, and
uncertainty calibration techniques.
"""

from typing import Optional, Tuple, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MCDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.

    Performs multiple forward passes with dropout enabled to approximate
    Bayesian inference in neural networks.

    Args:
        model: Base model to wrap
        n_samples: Number of forward passes for MC estimation
        dropout_layers: Optional list of dropout layers to use
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        dropout_layers: Optional[List[nn.Dropout]] = None,
    ):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.dropout_layers = dropout_layers

    def predict(
        self,
        x: Tensor,
        return_uncertainty: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Make predictions with MC dropout.

        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            mean: Mean predictions
            uncertainty: Standard deviation (if return_uncertainty=True)
        """
        self.model.train()

        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(x)
                samples.append(out)

        samples = torch.stack(samples)
        mean = samples.mean(dim=0)

        if return_uncertainty:
            uncertainty = samples.std(dim=0)
            return mean, uncertainty
        return mean, None

    def predict_proba(
        self,
        x: Tensor,
        return_uncertainty: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Predict class probabilities with uncertainty.

        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            probs: Mean class probabilities
            uncertainty: Variance of predictions (if return_uncertainty=True)
        """
        self.model.train()

        samples = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                out = self.model(x)
                probs = F.softmax(out, dim=-1)
                samples.append(probs)

        samples = torch.stack(samples)
        mean_probs = samples.mean(dim=0)

        if return_uncertainty:
            variance = samples.var(dim=0)
            return mean_probs, variance
        return mean_probs, None


class DropoutAsBayes(nn.Module):
    """Dropout as Bayesian approximation wrapper.

    Wraps a model to enable dropout at inference time for
    Monte Carlo estimation of prediction uncertainty.

    Args:
        base_model: Base neural network
        n_samples: Number of MC samples
    """

    def __init__(self, base_model: nn.Module, n_samples: int = 10):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout layers in the model."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass."""
        return self.base_model(x)

    def forward_with_uncertainty(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass with uncertainty estimation.

        Args:
            x: Input tensor

        Returns:
            mean: Mean predictions
            variance: Variance of predictions
            samples: Individual samples
        """
        self.base_model.train()

        samples = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                samples.append(self.base_model(x))

        samples = torch.stack(samples)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        return mean, variance, samples


class ConcreteDropout(nn.Module):
    """Concrete Dropout for continuous relaxation of dropout.

    Implements the Concrete Dropout relaxation as described in:
    "Concrete Dropout" (Gal et al., 2017)

    Args:
        model: Model to wrap
        input_dim: Input dimension for computing regularization
        weight: Initial dropout probability (logit)
    """

    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        weight: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.input_dim = input_dim

        if weight is None:
            initial_p = 0.5
            weight = torch.log(torch.tensor(initial_p) / (1 - initial_p))

        self.p_logit = nn.Parameter(weight.view(1))

    @property
    def p(self) -> Tensor:
        """Get current dropout probability."""
        return torch.sigmoid(self.p_logit)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            temp = 0.1
            noise = F.gumbel_softmax(
                torch.cat([self.p_logit, torch.zeros_like(self.p_logit)], dim=1),
                tau=temp,
                hard=False,
            )
            mask = noise[:, 0].unsqueeze(1)
            return self.model(x) * mask
        return self.model(x)

    def regularization(self) -> Tensor:
        """Compute dropout regularization term."""
        p = torch.sigmoid(self.p_logit)
        kl = p * (torch.log(p) - torch.log(torch.tensor(1e-5))) + (1 - p) * (
            torch.log(1 - p) - torch.log(torch.tensor(1 - 1e-5))
        )
        return kl.sum() / self.input_dim


class MCDropoutLinear(nn.Module):
    """Linear layer with MC Dropout for uncertainty estimation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        dropout_p: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        return self.linear(x)


class MCDropoutConv2d(nn.Module):
    """Conv2d layer with MC Dropout for uncertainty estimation.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        dropout_p: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.dropout(x)
        return x


class DropoutSchedule:
    """Schedule for varying dropout rate during training.

    Args:
        initial_p: Initial dropout probability
        final_p: Final dropout probability
        total_steps: Total training steps
        schedule_type: Type of schedule ('linear', 'cosine', 'exp')
    """

    def __init__(
        self,
        initial_p: float,
        final_p: float,
        total_steps: int,
        schedule_type: str = "linear",
    ):
        self.initial_p = initial_p
        self.final_p = final_p
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.step = 0

    def step(self):
        """Update the schedule."""
        self.step += 1

    @property
    def p(self) -> float:
        """Get current dropout probability."""
        progress = min(self.step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            return self.initial_p + (self.final_p - self.initial_p) * progress
        elif self.schedule_type == "cosine":
            return self.final_p + (self.initial_p - self.final_p) * 0.5 * (
                1 + torch.cos(torch.tensor(progress * 3.14159))
            )
        elif self.schedule_type == "exp":
            return self.initial_p * (self.final_p / self.initial_p) ** progress
        else:
            return self.initial_p


class MCDropoutClassifier(nn.Module):
    """MC Dropout classifier with uncertainty estimation.

    A complete classifier wrapper with built-in MC dropout
    for generating calibrated uncertainty estimates.

    Args:
        backbone: Feature extraction backbone
        num_classes: Number of output classes
        n_samples: Number of MC samples
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        n_samples: int = 10,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.n_samples = n_samples

        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self._get_feature_dim(), num_classes)

    def _get_feature_dim(self) -> int:
        """Get feature dimension by passing dummy input."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            return features.view(1, -1).size(1)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return self.classifier(features)

    def predict_with_uncertainty(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict with uncertainty estimation.

        Args:
            x: Input tensor

        Returns:
            mean_probs: Mean predicted probabilities
            variance: Prediction variance
            entropy: Predictive entropy
        """
        self.train()

        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs)

        samples = torch.stack(samples)
        mean_probs = samples.mean(dim=0)
        variance = samples.var(dim=0)

        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return mean_probs, variance, entropy


class DropoutUncertaintyMetrics:
    """Metrics for evaluating dropout-based uncertainty estimates.

    Provides methods to compute various uncertainty metrics
    from MC dropout samples.
    """

    @staticmethod
    def mutual_information(
        samples: Tensor,
    ) -> Tensor:
        """Compute mutual information as uncertainty measure.

        Args:
            samples: MC dropout samples [n_samples, batch, classes]

        Returns:
            Mutual information for each sample
        """
        probs = F.softmax(samples, dim=-1)
        mean_probs = probs.mean(dim=0)

        entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        mean_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean(dim=0)

        return entropy_mean - mean_entropy

    @staticmethod
    def expected_pairwise_kl(
        samples: Tensor,
    ) -> Tensor:
        """Compute expected pairwise KL divergence.

        Args:
            samples: MC dropout samples [n_samples, batch, classes]

        Returns:
            Expected KL divergence
        """
        probs = F.softmax(samples, dim=-1)
        mean_probs = probs.mean(dim=0).unsqueeze(0)

        kl = (probs * (torch.log(probs + 1e-10) - torch.log(mean_probs + 1e-10))).sum(
            dim=-1
        )

        return kl.mean(dim=0)

    @staticmethod
    def predictive_variance(
        samples: Tensor,
    ) -> Tensor:
        """Compute predictive variance.

        Args:
            samples: MC dropout samples [n_samples, batch, classes]

        Returns:
            Variance of predictions
        """
        return samples.var(dim=0)

    @staticmethod
    def prediction_disagreement(
        samples: Tensor,
    ) -> Tensor:
        """Compute prediction disagreement.

        Args:
            samples: MC dropout samples [n_samples, batch, classes]

        Returns:
            Fraction of samples that disagree
        """
        predictions = samples.argmax(dim=-1)
        disagreements = (predictions != predictions[0]).float()
        return disagreements.mean(dim=0)
