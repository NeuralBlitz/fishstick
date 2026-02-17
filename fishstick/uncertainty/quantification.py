"""
Comprehensive Uncertainty Quantification Module

This module provides a wide range of uncertainty quantification techniques for neural networks,
including Bayesian methods, ensemble methods, evidential learning, calibration, and OOD detection.
"""

import math
from typing import Optional, Tuple, List, Callable, Dict, Union, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from scipy import optimize, stats
from sklearn.isotonic import IsotonicRegression as SKIsotonicRegression


# ============================================================================
# 1. BAYESIAN NEURAL NETWORKS
# ============================================================================


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with Variational Inference.

    Implements Bayes by Backprop (Blundell et al., 2015) using
    reparameterization trick for gradient-based optimization.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        prior_sigma: Standard deviation of prior distribution

    Example:
        >>> layer = BayesianLinear(784, 128)
        >>> x = torch.randn(32, 784)
        >>> output, kl = layer(x, return_kl=True)
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with suitable priors."""
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.weight_rho, -3)  # Corresponds to sigma ≈ 0.05

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3)

    def forward(
        self, x: Tensor, return_kl: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass with weight sampling.

        Args:
            x: Input tensor of shape (batch_size, in_features)
            return_kl: Whether to return KL divergence

        Returns:
            Output tensor, and optionally KL divergence
        """
        # Sample weights using reparameterization trick
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_epsilon = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_epsilon

        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_epsilon = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * bias_epsilon

        # Linear transformation
        output = F.linear(x, weight, bias)

        if return_kl:
            kl = self._kl_divergence()
            return output, kl
        return output

    def _kl_divergence(self) -> Tensor:
        """Compute KL divergence between variational posterior and prior."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # KL for weights
        kl_weight = torch.sum(
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2 * self.prior_sigma**2)
            - 0.5
        )

        # KL for biases
        kl_bias = torch.sum(
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2 * self.prior_sigma**2)
            - 0.5
        )

        return kl_weight + kl_bias


class BayesianConv2d(nn.Module):
    """
    Bayesian 2D Convolutional Layer with Variational Inference.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Stride of convolution
        padding: Padding added to input
        prior_sigma: Standard deviation of prior distribution

    Example:
        >>> conv = BayesianConv2d(3, 64, 3)
        >>> x = torch.randn(32, 3, 32, 32)
        >>> output, kl = conv(x, return_kl=True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.prior_sigma = prior_sigma

        kernel_prod = self.kernel_size[0] * self.kernel_size[1]

        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.Tensor(
                out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]
            )
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(
                out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]
            )
        )

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.weight_rho, -3)

        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3)

    def forward(
        self, x: Tensor, return_kl: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass with weight sampling."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_epsilon = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_epsilon

        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_epsilon = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * bias_epsilon

        output = F.conv2d(x, weight, bias, self.stride, self.padding)

        if return_kl:
            kl = self._kl_divergence()
            return output, kl
        return output

    def _kl_divergence(self) -> Tensor:
        """Compute KL divergence."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        kl_weight = torch.sum(
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2 * self.prior_sigma**2)
            - 0.5
        )

        kl_bias = torch.sum(
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2 * self.prior_sigma**2)
            - 0.5
        )

        return kl_weight + kl_bias


class BayesByBackprop(nn.Module):
    """
    Bayes by Backprop for Automatic Relevance Determination.

    Wraps a neural network with Bayesian layers and provides
    the evidence lower bound (ELBO) loss for training.

    Args:
        model: Base neural network with Bayesian layers
        num_samples: Number of Monte Carlo samples for ELBO

    Example:
        >>> model = nn.Sequential(BayesianLinear(784, 128), nn.ReLU(), BayesianLinear(128, 10))
        >>> bbb = BayesByBackprop(model)
        >>> loss, metrics = bbb.elbo_loss(x, y, F.cross_entropy)
    """

    def __init__(self, model: nn.Module, num_samples: int = 1):
        super().__init__()
        self.model = model
        self.num_samples = num_samples

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def elbo_loss(
        self,
        x: Tensor,
        target: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        kl_weight: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        Args:
            x: Input data
            target: Target labels/values
            criterion: Loss function (e.g., F.cross_entropy)
            kl_weight: Weight for KL term (useful for minibatch training)

        Returns:
            elbo_loss: Total loss
            metrics: Dictionary with loss components
        """
        total_kl = 0.0
        total_nll = 0.0

        for _ in range(self.num_samples):
            # Collect KL divergences from all Bayesian layers
            kl_sum = 0.0
            activations = x

            for module in self.model.modules():
                if isinstance(module, (BayesianLinear, BayesianConv2d)):
                    activations, kl = module(activations, return_kl=True)
                    kl_sum += kl
                elif isinstance(module, nn.Module) and not isinstance(
                    module, (BayesianLinear, BayesianConv2d)
                ):
                    if hasattr(module, "forward"):
                        try:
                            activations = module(activations)
                        except:
                            pass

            output = (
                activations if not isinstance(activations, tuple) else activations[0]
            )
            nll = criterion(output, target)

            total_kl += kl_sum
            total_nll += nll

        # Average over samples
        avg_kl = total_kl / self.num_samples
        avg_nll = total_nll / self.num_samples

        # ELBO = -E[log p(y|x, w)] + KL[q(w)||p(w)]
        elbo = avg_nll + kl_weight * avg_kl

        metrics = {"elbo": elbo.item(), "nll": avg_nll.item(), "kl": avg_kl.item()}

        return elbo, metrics


class Flipout(nn.Module):
    """
    Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches.

    Flipout provides more efficient sampling for Bayesian neural networks by
    perturbing activations rather than weights, allowing for better gradient estimates.

    Reference: Wen et al., "Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches" (2018)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        prior_sigma: Prior standard deviation

    Example:
        >>> layer = Flipout(784, 128)
        >>> x = torch.randn(32, 784)
        >>> output = layer(x)
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.weight_rho, -3)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with flipout perturbations.

        Args:
            x: Input tensor

        Returns:
            Perturbed output
        """
        batch_size = x.size(0)

        # Compute mean output
        output = F.linear(x, self.weight_mu, self.bias_mu)

        # Sample perturbations
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # Random signs for flipout
        sign_input = torch.randint_like(x, high=2, dtype=torch.float32) * 2 - 1
        sign_output = torch.randint_like(output, high=2, dtype=torch.float32) * 2 - 1

        # Perturbation terms
        delta_weight = weight_sigma * torch.randn_like(self.weight_mu)
        delta_bias = bias_sigma * torch.randn_like(self.bias_mu)

        # Flipout perturbation: sign(r) * (Delta_W * sign(x))
        perturbed = F.linear(x * sign_input, delta_weight, delta_bias)
        output = output + sign_output * perturbed

        return output


# ============================================================================
# 2. ENSEMBLE METHODS
# ============================================================================


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble with Multiple Independent Networks.

    Combines predictions from multiple independently trained networks
    for improved uncertainty estimates.

    Args:
        base_model_fn: Function that returns a new model instance
        num_models: Number of models in the ensemble

    Example:
        >>> def make_model(): return nn.Linear(10, 2)
        >>> ensemble = DeepEnsemble(make_model, num_models=5)
        >>> mean, var = ensemble.predict(x)
    """

    def __init__(self, base_model_fn: Callable[[], nn.Module], num_models: int = 5):
        super().__init__()
        self.models = nn.ModuleList([base_model_fn() for _ in range(num_models)])
        self.num_models = num_models

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning mean prediction."""
        outputs = torch.stack([model(x) for model in self.models])
        return outputs.mean(dim=0)

    def predict(
        self, x: Tensor, return_individual: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Get ensemble predictions with uncertainty.

        Args:
            x: Input data
            return_individual: Whether to return individual predictions

        Returns:
            mean: Mean prediction
            variance: Predictive variance
            individual: Individual predictions (if return_individual=True)
        """
        with torch.no_grad():
            outputs = torch.stack([model(x) for model in self.models])

        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        if return_individual:
            return mean, variance, outputs
        return mean, variance

    def predict_proba(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get probabilistic predictions for classification.

        Returns:
            mean_probs: Average probabilities
            uncertainty: Predictive entropy
        """
        with torch.no_grad():
            probs = torch.stack([F.softmax(model(x), dim=-1) for model in self.models])

        mean_probs = probs.mean(dim=0)
        uncertainty = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return mean_probs, uncertainty


class SnapshotEnsemble:
    """
    Snapshot Ensemble using Cyclic Learning Rates.

    Saves model snapshots at different points during training with
    cyclic learning rates, creating an ensemble without additional training cost.

    Reference: Huang et al., "Snapshot Ensembles: Train 1, get M for free" (2017)

    Args:
        model: Model to ensemble
        num_snapshots: Number of snapshots to save
        cycle_length: Number of epochs per cycle
        lr_min: Minimum learning rate
        lr_max: Maximum learning rate

    Example:
        >>> snapshot_ens = SnapshotEnsemble(model, num_snapshots=5)
        >>> # During training
        >>> snapshot_ens.save_if_needed(epoch, model)
        >>> # During inference
        >>> mean, var = snapshot_ens.predict(x)
    """

    def __init__(
        self,
        model: nn.Module,
        num_snapshots: int = 5,
        cycle_length: int = 50,
        lr_min: float = 0.001,
        lr_max: float = 0.1,
    ):
        self.num_snapshots = num_snapshots
        self.cycle_length = cycle_length
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.snapshots: List[nn.Module] = []

    def get_lr(self, epoch: int) -> float:
        """
        Compute learning rate for cyclic cosine annealing.

        Args:
            epoch: Current epoch

        Returns:
            Learning rate for this epoch
        """
        t = (epoch % self.cycle_length) / self.cycle_length
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * t)
        )

    def save_if_needed(self, epoch: int, model: nn.Module):
        """
        Save snapshot at the end of each cycle.

        Args:
            epoch: Current epoch
            model: Model to snapshot
        """
        if (epoch + 1) % self.cycle_length == 0 and len(
            self.snapshots
        ) < self.num_snapshots:
            self.snapshots.append(
                type(model)(*model.__init_args__, **model.__init_kwargs__)
            )
            self.snapshots[-1].load_state_dict(model.state_dict())
            self.snapshots[-1].eval()

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get ensemble predictions.

        Args:
            x: Input data

        Returns:
            mean: Mean prediction
            variance: Predictive variance
        """
        if len(self.snapshots) == 0:
            raise RuntimeError("No snapshots saved yet")

        with torch.no_grad():
            outputs = torch.stack([model(x) for model in self.snapshots])

        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        return mean, variance


class MCDropoutEnsemble(nn.Module):
    """
    Monte Carlo Dropout as Implicit Ensemble.

    Uses dropout at test time to approximate a Bayesian neural network
    and provide uncertainty estimates.

    Args:
        model: Neural network with dropout layers
        num_samples: Number of forward passes
        dropout_p: Dropout probability

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 50), nn.Dropout(0.5), nn.Linear(50, 2))
        >>> mc_ensemble = MCDropoutEnsemble(model, num_samples=100)
        >>> mean, var = mc_ensemble.predict(x)
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 100,
        dropout_p: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.dropout_p = dropout_p

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass."""
        return self.model(x)

    def _enable_dropout(self, module: nn.Module):
        """Enable dropout during inference."""
        if isinstance(module, nn.Dropout):
            module.train()
        if isinstance(module, nn.Dropout2d):
            module.train()
        if isinstance(module, nn.Dropout3d):
            module.train()

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get MC Dropout predictions.

        Args:
            x: Input data

        Returns:
            mean: Mean prediction
            variance: Epistemic uncertainty
        """
        self.model.eval()
        self.model.apply(self._enable_dropout)

        outputs = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs.append(self.model(x))

        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        return mean, variance

    def predict_proba(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get probabilistic predictions for classification.

        Returns:
            mean_probs: Mean probabilities
            epistemic: Epistemic uncertainty (mutual information)
            aleatoric: Aleatoric uncertainty (expected entropy)
        """
        self.model.eval()
        self.model.apply(self._enable_dropout)

        probs_list = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)

        probs = torch.stack(probs_list)
        mean_probs = probs.mean(dim=0)

        # Epistemic uncertainty: mutual information
        expected_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean(dim=0)
        predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        epistemic = predictive_entropy - expected_entropy

        # Aleatoric uncertainty
        aleatoric = expected_entropy

        return mean_probs, epistemic, aleatoric


class BatchEnsemble(nn.Module):
    """
    Batch Ensemble: Efficient Ensemble with Batch Processing.

    Implements fast ensembling by using fast weight layers that share
    most parameters while having ensemble-specific fast weights.

    Reference: Wen et al., "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning" (2020)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        ensemble_size: Number of ensemble members

    Example:
        >>> layer = BatchEnsemble(784, 128, ensemble_size=4)
        >>> x = torch.randn(32, 784)  # batch_size doesn't need to be divisible by ensemble_size
        >>> output = layer(x)  # Output shape: (32, 128)
    """

    def __init__(self, in_features: int, out_features: int, ensemble_size: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Shared weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Fast weights (rank-1 factors)
        self.r = nn.Parameter(torch.ones(ensemble_size, in_features))
        self.s = nn.Parameter(torch.ones(ensemble_size, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.r)
        nn.init.ones_(self.s)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with batch ensemble.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.size(0)

        # Tile input for ensemble
        x_tiled = x.repeat(self.ensemble_size, 1)

        # Apply fast weights: r_i * W * s_i
        # For each ensemble member
        outputs = []
        for i in range(self.ensemble_size):
            # Apply rank-1 transformation
            x_fast = x_tiled[i * batch_size : (i + 1) * batch_size] * self.r[i]
            out = F.linear(x_fast, self.weight, self.bias)
            out = out * self.s[i]
            outputs.append(out)

        output = torch.stack(outputs).mean(dim=0)
        return output


# ============================================================================
# 3. EVIDENTIAL DEEP LEARNING
# ============================================================================


class EvidentialLayer(nn.Module):
    """
    Evidential Layer for Classification with Uncertainty.

    Uses Prior Networks to model uncertainty through Dirichlet distributions.
    High concentration parameters indicate confident predictions.

    Reference: Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty" (2018)

    Args:
        in_features: Input dimension
        num_classes: Number of classes

    Example:
        >>> layer = EvidentialLayer(128, 10)
        >>> x = torch.randn(32, 128)
        >>> probs, alpha, uncertainty = layer(x)
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass returning probabilities and uncertainty.

        Args:
            x: Input features

        Returns:
            probs: Expected probabilities
            alpha: Dirichlet concentration parameters
            uncertainty: Total uncertainty (vacuity)
        """
        # Get evidence (must be non-negative)
        evidence = F.softplus(self.fc(x)) + 1.0  # +1 for Laplace smoothing

        # Dirichlet parameters: alpha = e + 1
        alpha = evidence + 1.0

        # Expected probabilities
        alpha_sum = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / alpha_sum

        # Uncertainty (vacuity): 1 - S / (S + K) where S = sum of evidence
        # Equivalent to K / (S + K) where K = num_classes
        uncertainty = self.num_classes / alpha_sum.squeeze(-1)

        return probs, alpha, uncertainty


class EvidentialRegression(nn.Module):
    """
    Evidential Regression with NIG (Normal-Inverse-Gamma) Distribution.

    Models predictive uncertainty through the Normal-Inverse-Gamma distribution,
    providing both epistemic (model) and aleatoric (data) uncertainty.

    Reference: Amini et al., "Deep Evidential Regression" (2020)

    Args:
        in_features: Input dimension

    Example:
        >>> layer = EvidentialRegression(128)
        >>> x = torch.randn(32, 128)
        >>> mean, var, v, alpha, beta = layer(x)
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features

        # NIG parameters: gamma (mean), nu, alpha, beta
        self.fc = nn.Linear(in_features, 4)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass returning NIG parameters.

        Args:
            x: Input features

        Returns:
            gamma: Predicted mean
            var: Predictive variance
            v: Degrees of freedom (> 1)
            alpha: Shape parameter (> 1)
            beta: Scale parameter (> 0)
        """
        output = self.fc(x)

        # gamma: mean (unconstrained)
        gamma = output[:, 0]

        # v: degrees of freedom (must be > 1)
        v = F.softplus(output[:, 1]) + 1.0

        # alpha: shape parameter (must be > 1)
        alpha = F.softplus(output[:, 2]) + 1.0

        # beta: scale parameter (must be > 0)
        beta = F.softplus(output[:, 3]) + 1e-8

        # Predictive variance: beta / (v * (alpha - 1))
        var = beta / (v * (alpha - 1) + 1e-8)

        return gamma, var, v, alpha, beta


class NIGLoss(nn.Module):
    """
    Normal-Inverse-Gamma Loss for Evidential Regression.

    Combines negative log-likelihood with regularization on prediction error.

    Reference: Amini et al., "Deep Evidential Regression" (2020)

    Args:
        coeff: Coefficient for regularization term

    Example:
        >>> loss_fn = NIGLoss(coeff=0.01)
        >>> gamma, v, alpha, beta = model(x)
        >>> loss = loss_fn(gamma, v, alpha, beta, target)
    """

    def __init__(self, coeff: float = 0.01):
        super().__init__()
        self.coeff = coeff

    def forward(
        self, gamma: Tensor, v: Tensor, alpha: Tensor, beta: Tensor, target: Tensor
    ) -> Tensor:
        """
        Compute NIG loss.

        Args:
            gamma: Predicted mean
            v: Degrees of freedom
            alpha: Shape parameter
            beta: Scale parameter
            target: Ground truth

        Returns:
            Loss value
        """
        # Ensure numerical stability
        v = torch.clamp(v, min=1.01)
        alpha = torch.clamp(alpha, min=1.01)
        beta = torch.clamp(beta, min=1e-8)

        # Prediction error
        error = target - gamma

        # NIG loss components
        # Log likelihood term
        omega = 2 * beta * (1 + v)
        log_likelihood = (
            0.5 * torch.log(np.pi / v + 1e-8)
            - alpha * torch.log(omega + 1e-8)
            + (alpha + 0.5) * torch.log(v * error**2 + omega + 1e-8)
        )

        # Regularization on prediction error
        regularization = error**2 * (2 * v + alpha)

        loss = log_likelihood.mean() + self.coeff * regularization.mean()

        return loss


# ============================================================================
# 4. CALIBRATION METHODS
# ============================================================================


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for Model Calibration.

    A simple post-hoc calibration method that uses a single scalar
    temperature parameter to soften/sharpen the softmax distribution.

    Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017)

    Example:
        >>> calibrator = TemperatureScaling()
        >>> calibrator.fit(logits, labels)
        >>> calibrated_logits = calibrator(logits)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: Tensor) -> Tensor:
        """
        Apply temperature scaling.

        Args:
            logits: Unnormalized model outputs

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature

    def fit(self, logits: Tensor, labels: Tensor, max_iter: int = 50) -> float:
        """
        Find optimal temperature using NLL optimization.

        Args:
            logits: Validation logits
            labels: Validation labels
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """

        def nll_loss(t: float) -> float:
            self.temperature.data.fill_(t)
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, labels).item()
            return loss

        # Bounded optimization
        result = optimize.minimize_scalar(
            nll_loss,
            bounds=(0.01, 10.0),
            method="bounded",
            options={"maxiter": max_iter},
        )

        self.temperature.data.fill_(result.x)
        return result.x


class PlattScaling:
    """
    Platt Scaling (Logistic Calibration) for Binary Classification.

    Fits a logistic regression model to the classifier outputs.

    Example:
        >>> calibrator = PlattScaling()
        >>> calibrator.fit(scores, labels)
        >>> calibrated_probs = calibrator(scores)
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit Platt scaling parameters.

        Args:
            scores: Raw model outputs (logits or probabilities)
            labels: Binary ground truth labels
        """
        from sklearn.linear_model import LogisticRegression

        # Reshape for sklearn
        scores = scores.reshape(-1, 1)

        # Fit logistic regression
        clf = LogisticRegression()
        clf.fit(scores, labels)

        self.a = clf.coef_[0][0]
        self.b = clf.intercept_[0]

    def __call__(self, scores: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        """
        Apply Platt scaling.

        Args:
            scores: Raw scores

        Returns:
            Calibrated probabilities
        """
        if isinstance(scores, Tensor):
            scores = scores.cpu().numpy()

        # Sigmoid transformation
        probs = 1 / (1 + np.exp(-(self.a * scores + self.b)))

        return probs


class IsotonicCalibration:
    """
    Isotonic Regression for Non-Parametric Calibration.

    Learns a monotonic mapping from uncalibrated scores to calibrated probabilities.
    More flexible than Platt scaling but requires more data.

    Example:
        >>> calibrator = IsotonicCalibration()
        >>> calibrator.fit(scores, labels)
        >>> calibrated_probs = calibrator(scores)
    """

    def __init__(self):
        self.isotonic = SKIsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit isotonic regression.

        Args:
            scores: Raw scores
            labels: Binary ground truth labels
        """
        self.isotonic.fit(scores, labels)

    def __call__(self, scores: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            scores: Raw scores

        Returns:
            Calibrated probabilities
        """
        if isinstance(scores, Tensor):
            scores = scores.cpu().numpy()

        return self.isotonic.predict(scores)


def compute_ece(
    confidences: Union[Tensor, np.ndarray],
    predictions: Union[Tensor, np.ndarray],
    labels: Union[Tensor, np.ndarray],
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between accuracy and confidence
    across confidence bins.

    Args:
        confidences: Model confidence scores (max probability)
        predictions: Predicted class labels
        labels: True class labels
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better)

    Example:
        >>> ece = compute_ece(confidences, predictions, labels, n_bins=15)
    """
    if isinstance(confidences, Tensor):
        confidences = confidences.cpu().numpy()
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (
                (predictions[in_bin] == labels[in_bin]).astype(float).mean()
            )
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def reliability_diagram(
    confidences: Union[Tensor, np.ndarray],
    predictions: Union[Tensor, np.ndarray],
    labels: Union[Tensor, np.ndarray],
    n_bins: int = 15,
    return_data: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Compute data for reliability diagram.

    Args:
        confidences: Model confidence scores
        predictions: Predicted class labels
        labels: True class labels
        n_bins: Number of bins
        return_data: Whether to return additional calibration data

    Returns:
        bin_centers: Center of each bin
        accuracies: Accuracy in each bin
        If return_data=True, also returns:
            - confidences: Average confidence in each bin
            - bin_counts: Number of samples in each bin

    Example:
        >>> centers, accs = reliability_diagram(confidences, predictions, labels)
        >>> plt.bar(centers, accs)
    """
    if isinstance(confidences, Tensor):
        confidences = confidences.cpu().numpy()
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    accuracies = np.zeros(n_bins)
    avg_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i, (bin_lower, bin_upper) in enumerate(
        zip(bin_boundaries[:-1], bin_boundaries[1:])
    ):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_counts[i] = in_bin.sum()

        if bin_counts[i] > 0:
            accuracies[i] = (predictions[in_bin] == labels[in_bin]).astype(float).mean()
            avg_confidences[i] = confidences[in_bin].mean()

    if return_data:
        return bin_centers, accuracies, avg_confidences, bin_counts
    return bin_centers, accuracies


# ============================================================================
# 5. UNCERTAINTY ESTIMATION
# ============================================================================


def predictive_entropy(probs: Tensor) -> Tensor:
    """
    Compute predictive entropy as total uncertainty measure.

    For classification, this is the entropy of the predictive distribution.
    For ensembles or MC samples, compute on the mean probabilities.

    Args:
        probs: Probability distribution (batch_size, num_classes) or
               (num_samples, batch_size, num_classes) for multiple samples

    Returns:
        Entropy values (batch_size,)

    Example:
        >>> probs = torch.softmax(logits, dim=-1)
        >>> entropy = predictive_entropy(probs)
    """
    if probs.dim() == 3:
        # Multiple samples: compute mean first
        probs = probs.mean(dim=0)

    # Avoid log(0)
    probs = torch.clamp(probs, min=1e-10)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)

    return entropy


def mutual_information(probs: Tensor) -> Tensor:
    """
    Compute mutual information as epistemic uncertainty.

    MI = H[E[p(y|x,θ)]] - E[H[p(y|x,θ)]]

    Captures disagreement between ensemble members or MC samples,
    representing model uncertainty.

    Args:
        probs: Probabilities from multiple samples
               Shape: (num_samples, batch_size, num_classes)

    Returns:
        Mutual information values (batch_size,)

    Example:
        >>> probs = torch.stack([model(x) for _ in range(100)])  # MC samples
        >>> epistemic = mutual_information(probs)
    """
    if probs.dim() != 3:
        raise ValueError("probs must have shape (num_samples, batch_size, num_classes)")

    # Mean probabilities across samples
    mean_probs = probs.mean(dim=0)
    mean_probs = torch.clamp(mean_probs, min=1e-10)

    # Entropy of mean (total uncertainty)
    entropy_mean = -(mean_probs * torch.log(mean_probs)).sum(dim=-1)

    # Mean of entropies (aleatoric uncertainty)
    probs = torch.clamp(probs, min=1e-10)
    entropies = -(probs * torch.log(probs)).sum(dim=-1)
    mean_entropy = entropies.mean(dim=0)

    # Epistemic uncertainty
    mi = entropy_mean - mean_entropy

    return mi


def aleatoric_uncertainty(probs: Tensor) -> Tensor:
    """
    Compute aleatoric (data) uncertainty as expected entropy.

    This captures inherent uncertainty in the data that cannot be
    reduced by collecting more data.

    Args:
        probs: Probabilities from multiple samples
               Shape: (num_samples, batch_size, num_classes)

    Returns:
        Aleatoric uncertainty values (batch_size,)

    Example:
        >>> probs = torch.stack([model(x) for _ in range(100)])
        >>> aleatoric = aleatoric_uncertainty(probs)
    """
    if probs.dim() != 3:
        raise ValueError("probs must have shape (num_samples, batch_size, num_classes)")

    probs = torch.clamp(probs, min=1e-10)
    entropies = -(probs * torch.log(probs)).sum(dim=-1)

    return entropies.mean(dim=0)


def confidence_intervals(
    samples: Tensor, confidence: float = 0.95
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute confidence intervals from multiple predictions.

    Args:
        samples: Predictions from multiple models/samples
                 Shape: (num_samples, batch_size, ...) or (num_samples, ...)
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        mean: Mean prediction
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval

    Example:
        >>> samples = torch.stack([model(x) for _ in range(100)])
        >>> mean, lower, upper = confidence_intervals(samples, confidence=0.95)
    """
    mean = samples.mean(dim=0)

    # Compute percentiles for confidence interval
    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    lower = torch.quantile(samples, lower_percentile / 100, dim=0)
    upper = torch.quantile(samples, upper_percentile / 100, dim=0)

    return mean, lower, upper


# ============================================================================
# 6. OUT-OF-DISTRIBUTION DETECTION
# ============================================================================


class MaxSoftmaxProbability:
    """
    Maximum Softmax Probability (MSP) baseline for OOD detection.

    Uses the maximum softmax probability as an OOD score.
    Lower confidence indicates potential OOD samples.

    Reference: Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks" (2017)

    Example:
        >>> detector = MaxSoftmaxProbability(model)
        >>> ood_scores = detector(x)  # Lower = more likely OOD
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute OOD scores.

        Args:
            x: Input data

        Returns:
            OOD scores (negative max probability, higher = more OOD)
        """
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]

        # Return negative for consistency (higher = more OOD)
        return -max_probs


class ODIN:
    """
    ODIN: Out-of-DIstribution detector for Neural networks.

    Uses temperature scaling and input preprocessing to improve
    OOD detection over the baseline MSP method.

    Reference: Liang et al., "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks" (2018)

    Args:
        model: Neural network classifier
        temperature: Temperature for scaling (higher = smoother)
        epsilon: Magnitude of input perturbation

    Example:
        >>> odin = ODIN(model, temperature=1000, epsilon=0.0014)
        >>> odin_scores = odin(x)
    """

    def __init__(
        self, model: nn.Module, temperature: float = 1000.0, epsilon: float = 0.0014
    ):
        self.model = model
        self.model.eval()
        self.temperature = temperature
        self.epsilon = epsilon

    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute ODIN scores.

        Args:
            x: Input data

        Returns:
            OOD scores (higher = more OOD)
        """
        x.requires_grad = True

        # Forward pass with temperature scaling
        logits = self.model(x) / self.temperature

        # Compute gradient for input preprocessing
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)

        # Gradient of max probability w.r.t. input
        loss = max_probs.sum()
        loss.backward()

        # Add perturbation in gradient direction
        gradient = x.grad.data
        x_perturbed = x - self.epsilon * gradient.sign()
        x_perturbed = x_perturbed.detach()

        # Forward pass on perturbed input
        with torch.no_grad():
            logits = self.model(x_perturbed) / self.temperature
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]

        # Return negative for consistency
        return -max_probs


class MahalanobisDistance:
    """
    Mahalanobis Distance-based OOD Detection.

    Uses feature space distance from class centroids to detect OOD samples.

    Reference: Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" (2018)

    Args:
        model: Neural network with feature extractor
        num_classes: Number of output classes

    Example:
        >>> detector = MahalanobisDistance(model, features, num_classes=10)
        >>> detector.fit(id_loader)
        >>> scores = detector(x)
    """

    def __init__(self, model: nn.Module, feature_layer: nn.Module, num_classes: int):
        self.model = model
        self.feature_layer = feature_layer
        self.num_classes = num_classes

        self.class_means: Optional[Tensor] = None
        self.precision: Optional[Tensor] = None

    def fit(self, dataloader: torch.utils.data.DataLoader):
        """
        Compute class means and precision matrix from ID data.

        Args:
            dataloader: DataLoader with in-distribution data
        """
        self.model.eval()

        features_list = []
        labels_list = []

        with torch.no_grad():
            for x, y in dataloader:
                features = self.feature_layer(x)
                features_list.append(features)
                labels_list.append(y)

        features = torch.cat(features_list)
        labels = torch.cat(labels_list)

        # Compute class means
        self.class_means = []
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.class_means.append(features[mask].mean(dim=0))
            else:
                self.class_means.append(torch.zeros(features.size(1)))
        self.class_means = torch.stack(self.class_means)

        # Compute covariance and precision
        centered = features - self.class_means[labels]
        cov = (centered.T @ centered) / len(features)

        # Add small regularization for numerical stability
        self.precision = torch.linalg.pinv(cov + 1e-4 * torch.eye(cov.size(0)))

    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute Mahalanobis distance scores.

        Args:
            x: Input data

        Returns:
            Minimum Mahalanobis distance to any class (higher = more OOD)
        """
        if self.class_means is None or self.precision is None:
            raise RuntimeError("Must call fit() before using detector")

        with torch.no_grad():
            features = self.feature_layer(x)

            # Compute distance to each class mean
            distances = []
            for c in range(self.num_classes):
                diff = features - self.class_means[c]
                distance = torch.sqrt((diff @ self.precision * diff).sum(dim=-1) + 1e-8)
                distances.append(distance)

            distances = torch.stack(distances)
            min_distance = distances.min(dim=0)[0]

        return min_distance


class EnergyBasedOOD:
    """
    Energy-based Out-of-Distribution Detection.

    Uses the energy score (free energy) from the logits to detect OOD samples.
    More principled than softmax-based methods.

    Reference: Liu et al., "Energy-based Out-of-distribution Detection" (2020)

    Args:
        model: Neural network classifier
        temperature: Temperature parameter

    Example:
        >>> detector = EnergyBasedOOD(model, temperature=1.0)
        >>> energy_scores = detector(x)  # Higher = more OOD
    """

    def __init__(self, model: nn.Module, temperature: float = 1.0):
        self.model = model
        self.model.eval()
        self.temperature = temperature

    def __call__(self, x: Tensor) -> Tensor:
        """
        Compute energy scores.

        Args:
            x: Input data

        Returns:
            Energy scores (higher = more OOD)
        """
        with torch.no_grad():
            logits = self.model(x)
            # Energy = -T * log(sum(exp(logits / T)))
            energy = -self.temperature * torch.logsumexp(
                logits / self.temperature, dim=-1
            )

        # Return negative so higher = more OOD (lower energy = more ID)
        return -energy


def evaluate_ood_detection(
    id_scores: Union[Tensor, np.ndarray], ood_scores: Union[Tensor, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate OOD detection performance.

    Computes AUROC, AUPR, FPR@95%TPR metrics.

    Args:
        id_scores: OOD scores for in-distribution samples
        ood_scores: OOD scores for out-of-distribution samples

    Returns:
        Dictionary with evaluation metrics

    Example:
        >>> metrics = evaluate_ood_detection(id_scores, ood_scores)
        >>> print(f"AUROC: {metrics['auroc']:.3f}")
    """
    if isinstance(id_scores, Tensor):
        id_scores = id_scores.cpu().numpy()
    if isinstance(ood_scores, Tensor):
        ood_scores = ood_scores.cpu().numpy()

    # Create labels (0 = ID, 1 = OOD)
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])

    # AUROC
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    auroc = roc_auc_score(y_true, y_score)

    # AUPR
    aupr = average_precision_score(y_true, y_score)

    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr95 = fpr[np.argmax(tpr >= 0.95)]

    # Detection accuracy
    threshold = np.percentile(y_score[len(id_scores) :], 5)  # 5th percentile of OOD
    preds = (y_score > threshold).astype(int)
    detection_acc = (preds == y_true).mean()

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr_at_95_tpr": float(fpr95),
        "detection_accuracy": float(detection_acc),
    }


# ============================================================================
# 7. CONFORMAL PREDICTION
# ============================================================================


def conformal_prediction_sets(
    model: nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    test_x: Tensor,
    alpha: float = 0.1,
    score_function: str = "softmax",
) -> Tensor:
    """
    Build conformal prediction sets with guaranteed coverage.

    Args:
        model: Trained classifier
        calib_loader: DataLoader with calibration data
        test_x: Test inputs
        alpha: Miscoverage level (1-alpha = coverage guarantee)
        score_function: 'softmax' or 'aps' (adaptive prediction sets)

    Returns:
        Prediction sets as binary tensor (batch_size, num_classes)

    Example:
        >>> sets = conformal_prediction_sets(model, calib_loader, test_x, alpha=0.1)
        >>> # sets[i, j] = 1 if class j is in prediction set for sample i
    """
    model.eval()
    scores = []
    labels = []

    # Compute conformity scores on calibration set
    with torch.no_grad():
        for x, y in calib_loader:
            logits = model(x)

            if score_function == "softmax":
                probs = F.softmax(logits, dim=-1)
                # Score = 1 - probability of true class
                batch_scores = 1 - probs[torch.arange(len(y)), y]
            elif score_function == "aps":
                # Adaptive prediction sets
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

                # Cumulative sum until true class is included
                cumsum = sorted_probs.cumsum(dim=-1)
                true_class_rank = (
                    (sorted_indices == y.unsqueeze(-1)).float().argmax(dim=-1)
                )
                batch_scores = cumsum[torch.arange(len(y)), true_class_rank]
            else:
                raise ValueError(f"Unknown score function: {score_function}")

            scores.append(batch_scores)
            labels.append(y)

    scores = torch.cat(scores)
    n = len(scores)

    # Compute quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q = torch.quantile(scores, q_level)

    # Build prediction sets for test data
    with torch.no_grad():
        test_logits = model(test_x)
        test_probs = F.softmax(test_logits, dim=-1)

        if score_function == "softmax":
            # Include all classes with score <= q
            # Score for class j is 1 - prob_j, so include if prob_j >= 1 - q
            prediction_sets = test_probs >= (1 - q)
        elif score_function == "aps":
            sorted_probs, sorted_indices = test_probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            # Find how many classes to include
            include = cumsum <= q
            # Always include at least one class
            include[:, 0] = True

            # Map back to original class order
            prediction_sets = torch.zeros_like(test_probs, dtype=torch.bool)
            for i in range(len(test_x)):
                prediction_sets[i, sorted_indices[i][include[i]]] = True

    return prediction_sets.long()


class SplitConformal:
    """
    Split Conformal Prediction with train/calibration split.

    The standard approach to conformal prediction where data is split
    into training and calibration sets.

    Args:
        model: Base model to train
        alpha: Miscoverage level

    Example:
        >>> cp = SplitConformal(model, alpha=0.1)
        >>> cp.fit(train_loader, calib_loader)
        >>> sets = cp.predict(test_x)
    """

    def __init__(
        self, model: nn.Module, alpha: float = 0.1, score_function: str = "softmax"
    ):
        self.model = model
        self.alpha = alpha
        self.score_function = score_function
        self.q: Optional[float] = None

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        calib_loader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epochs: int = 10,
    ):
        """
        Train model and compute calibration threshold.

        Args:
            train_loader: Training data
            calib_loader: Calibration data
            optimizer: Optimizer (created if None)
            epochs: Number of training epochs
        """
        # Train model
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters())

        self.model.train()
        for epoch in range(epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

        # Compute calibration scores
        self.model.eval()
        scores = []

        with torch.no_grad():
            for x, y in calib_loader:
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)

                if self.score_function == "softmax":
                    batch_scores = 1 - probs[torch.arange(len(y)), y]
                else:  # APS
                    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
                    cumsum = sorted_probs.cumsum(dim=-1)
                    true_class_rank = (
                        (sorted_indices == y.unsqueeze(-1)).float().argmax(dim=-1)
                    )
                    batch_scores = cumsum[torch.arange(len(y)), true_class_rank]

                scores.append(batch_scores)

        scores = torch.cat(scores)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q = torch.quantile(scores, min(q_level, 1.0)).item()

    def predict(self, x: Tensor) -> Tensor:
        """
        Get prediction sets for test data.

        Args:
            x: Test inputs

        Returns:
            Prediction sets
        """
        if self.q is None:
            raise RuntimeError("Must call fit() before predict()")

        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

            if self.score_function == "softmax":
                prediction_sets = probs >= (1 - self.q)
            else:
                sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                include = cumsum <= self.q
                include[:, 0] = True

                prediction_sets = torch.zeros_like(probs, dtype=torch.bool)
                for i in range(len(x)):
                    prediction_sets[i, sorted_indices[i][include[i]]] = True

        return prediction_sets.long()


class CrossConformal:
    """
    Cross-Conformal Prediction using cross-validation.

    Uses K-fold cross-validation to make better use of data,
    providing tighter prediction sets than split conformal.

    Args:
        model_fn: Function that creates a new model instance
        alpha: Miscoverage level
        n_folds: Number of cross-validation folds

    Example:
        >>> def make_model(): return nn.Linear(10, 2)
        >>> cp = CrossConformal(make_model, alpha=0.1, n_folds=5)
        >>> sets = cp.fit_predict(data_loader)
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        alpha: float = 0.1,
        n_folds: int = 5,
        score_function: str = "softmax",
    ):
        self.model_fn = model_fn
        self.alpha = alpha
        self.n_folds = n_folds
        self.score_function = score_function
        self.models: List[nn.Module] = []
        self.quantiles: List[float] = []

    def fit_predict(
        self,
        dataset: torch.utils.data.Dataset,
        test_x: Tensor,
        epochs_per_fold: int = 10,
        batch_size: int = 32,
    ) -> Tensor:
        """
        Train cross-validation models and predict.

        Args:
            dataset: Full dataset
            test_x: Test inputs
            epochs_per_fold: Training epochs per fold
            batch_size: Batch size for training

        Returns:
            Prediction sets
        """
        from sklearn.model_selection import KFold

        # Convert dataset to lists
        all_data = [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))]
        all_x = torch.stack([d[0] for d in all_data])
        all_y = torch.tensor([d[1] for d in all_data])

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        self.models = []
        self.quantiles = []
        all_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_x)):
            # Create fold datasets
            train_x, train_y = all_x[train_idx], all_y[train_idx]
            val_x, val_y = all_x[val_idx], all_y[val_idx]

            # Create and train model
            model = self.model_fn()
            optimizer = torch.optim.Adam(model.parameters())

            model.train()
            for epoch in range(epochs_per_fold):
                # Simple batch training
                for i in range(0, len(train_x), batch_size):
                    batch_x = train_x[i : i + batch_size]
                    batch_y = train_y[i : i + batch_size]

                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = F.cross_entropy(logits, batch_y)
                    loss.backward()
                    optimizer.step()

            # Compute calibration scores on validation fold
            model.eval()
            with torch.no_grad():
                logits = model(val_x)
                probs = F.softmax(logits, dim=-1)

                if self.score_function == "softmax":
                    scores = 1 - probs[torch.arange(len(val_y)), val_y]
                else:
                    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
                    cumsum = sorted_probs.cumsum(dim=-1)
                    true_class_rank = (
                        (sorted_indices == val_y.unsqueeze(-1)).float().argmax(dim=-1)
                    )
                    scores = cumsum[torch.arange(len(val_y)), true_class_rank]

                all_scores.append(scores)

            self.models.append(model)

        # Aggregate scores and compute global quantile
        all_scores = torch.cat(all_scores)
        n = len(all_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        global_q = torch.quantile(all_scores, min(q_level, 1.0)).item()

        # Predict with all models and aggregate
        all_sets = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(test_x)
                probs = F.softmax(logits, dim=-1)

                if self.score_function == "softmax":
                    sets = probs >= (1 - global_q)
                else:
                    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
                    cumsum = sorted_probs.cumsum(dim=-1)
                    include = cumsum <= global_q
                    include[:, 0] = True

                    sets = torch.zeros_like(probs, dtype=torch.bool)
                    for i in range(len(test_x)):
                        sets[i, sorted_indices[i][include[i]]] = True

                all_sets.append(sets)

        # Take intersection or average of all prediction sets
        final_sets = torch.stack(all_sets).float().mean(dim=0) >= 0.5

        return final_sets.long()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def enable_bayesian_layers(model: nn.Module):
    """
    Enable Bayesian behavior (sampling) for all Bayesian layers in a model.

    Args:
        model: Neural network model
    """
    for module in model.modules():
        if isinstance(module, (BayesianLinear, BayesianConv2d, Flipout)):
            module.train()


def disable_bayesian_layers(model: nn.Module):
    """
    Disable Bayesian behavior (use mean weights) for all Bayesian layers.

    Args:
        model: Neural network model
    """
    for module in model.modules():
        if isinstance(module, (BayesianLinear, BayesianConv2d, Flipout)):
            module.eval()


def get_uncertainty_components(
    model: nn.Module, x: Tensor, num_samples: int = 100
) -> Dict[str, Tensor]:
    """
    Compute all uncertainty components for a given input.

    Args:
        model: Neural network (should support multiple forward passes)
        x: Input data
        num_samples: Number of forward passes

    Returns:
        Dictionary with 'total', 'epistemic', and 'aleatoric' uncertainties
    """
    model.eval()

    # Collect predictions
    probs_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

    probs = torch.stack(probs_list)

    # Compute uncertainties
    total = predictive_entropy(probs)
    epistemic = mutual_information(probs)
    aleatoric = aleatoric_uncertainty(probs)

    return {"total": total, "epistemic": epistemic, "aleatoric": aleatoric}


__all__ = [
    # Bayesian Neural Networks
    "BayesianLinear",
    "BayesianConv2d",
    "BayesByBackprop",
    "Flipout",
    # Ensemble Methods
    "DeepEnsemble",
    "SnapshotEnsemble",
    "MCDropoutEnsemble",
    "BatchEnsemble",
    # Evidential Deep Learning
    "EvidentialLayer",
    "EvidentialRegression",
    "NIGLoss",
    # Calibration Methods
    "TemperatureScaling",
    "PlattScaling",
    "IsotonicCalibration",
    "compute_ece",
    "reliability_diagram",
    # Uncertainty Estimation
    "predictive_entropy",
    "mutual_information",
    "aleatoric_uncertainty",
    "confidence_intervals",
    # OOD Detection
    "MaxSoftmaxProbability",
    "ODIN",
    "MahalanobisDistance",
    "EnergyBasedOOD",
    "evaluate_ood_detection",
    # Conformal Prediction
    "conformal_prediction_sets",
    "SplitConformal",
    "CrossConformal",
    # Utilities
    "enable_bayesian_layers",
    "disable_bayesian_layers",
    "get_uncertainty_components",
]
