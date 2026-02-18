"""
Variational Inference Layers for Bayesian Deep Learning.

Provides layers that perform variational inference for Bayesian neural networks,
including concrete dropout, variational linear layers, and variational convolutions.
"""

from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedBernoulli, Bernoulli


class ConcreteDropout(nn.Module):
    """Concrete Dropout for Bayesian neural networks.

    Implements the Concrete Dropout relaxation as described in:
    "Concrete Dropout" (Gal et al., 2017)

    Args:
        layer: The layer to apply dropout to
        weight: Initial dropout probability (logit space)
        eps: Minimum probability value

    Example:
        >>> linear = nn.Linear(256, 128)
        >>> concrete_dropout = ConcreteDropout(linear)
        >>> output = concrete_dropout(input)
    """

    def __init__(
        self,
        layer: nn.Module,
        weight: Optional[float] = None,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.weight_dim = layer.weight.dim()

        if weight is None:
            initial_p = 0.5 if self.weight_dim == 2 else 0.1
            weight = torch.log(torch.tensor(initial_p) / (1 - initial_p))

        self.weight = nn.Parameter(
            weight.unsqueeze(0) if self.weight_dim == 1 else weight
        )
        self.log_noise = nn.Parameter(torch.zeros_like(layer.weight))

    @property
    def p(self) -> Tensor:
        """Get current dropout probability."""
        return torch.sigmoid(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return self.layer(x)

        p = torch.sigmoid(self.weight)

        if self.weight_dim == 2:
            p = p.squeeze(0)

        temp = 0.1
        noise = RelaxedBernoulli(temp, probs=p).rsample()
        mask = torch.clamp(noise, self.eps, 1 - self.eps)

        if self.weight_dim == 2:
            masked_weight = self.layer.weight * mask
            if self.layer.bias is not None:
                masked_bias = self.layer.bias * mask
            else:
                masked_bias = None
            output = F.linear(x, masked_weight, masked_bias)
        else:
            output = self.layer(x) * mask

        return output

    def kl_divergence(self) -> Tensor:
        """Compute KL divergence for regularization."""
        p = torch.sigmoid(self.weight)
        p = torch.clamp(p, self.eps, 1 - self.eps)

        if self.weight_dim == 2:
            p = p.squeeze(0)

        q_dist = RelaxedBernoulli(0.1, probs=p)
        p_dist = Bernoulli(probs=p.new_zeros(p.size()) + self.eps)

        return torch.distributions.kl_divergence(q_dist, p_dist).sum()


class VariationalLinear(nn.Module):
    """Variational Linear layer with Bayesian inference.

    Implements a linear layer where weights are sampled from a learnable
    posterior distribution, with KL divergence regularization.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        prior_sigma: Prior standard deviation (default: 1.0)
        bias: Whether to include bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_sigma", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, a=0, mode="fan_in")
        nn.init.uniform_(self.weight_sigma, -5, -3)

        if self.bias_mu is not None:
            nn.init.zeros_(self.bias_mu)
            nn.init.uniform_(self.bias_sigma, -5, -3)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight = (
                self.weight_mu + torch.randn_like(self.weight_sigma) * self.weight_sigma
            )
            if self.bias_mu is not None:
                bias = (
                    self.bias_mu + torch.randn_like(self.bias_sigma) * self.bias_sigma
                )
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.bias_mu is not None else None

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> Tensor:
        """Compute KL divergence between posterior and prior."""
        sigma_prior = self.prior_sigma

        weight_kl = (
            torch.log(sigma_prior)
            - 0.5 * torch.log(self.weight_sigma**2)
            + 0.5 * (self.weight_sigma**2 + (self.weight_mu**2)) / (sigma_prior**2)
            - 0.5
        ).sum()

        if self.bias_mu is not None:
            bias_kl = (
                torch.log(sigma_prior)
                - 0.5 * torch.log(self.bias_sigma**2)
                + 0.5 * (self.bias_sigma**2 + (self.bias_mu**2)) / (sigma_prior**2)
                - 0.5
            ).sum()
        else:
            bias_kl = 0

        return weight_kl + bias_kl


class VariationalConv2d(nn.Module):
    """Variational Convolutional 2D layer with Bayesian inference.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution
        padding: Padding added to input
        prior_sigma: Prior standard deviation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_channels))
        self.bias_sigma = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, a=0, mode="fan_in")
        nn.init.uniform_(self.weight_sigma, -5, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.uniform_(self.bias_sigma, -5, -3)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight = (
                self.weight_mu + torch.randn_like(self.weight_sigma) * self.weight_sigma
            )
            bias = self.bias_mu + torch.randn_like(self.bias_sigma) * self.bias_sigma
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.conv2d(x, weight, bias, self.stride, self.padding)

    def kl_divergence(self) -> Tensor:
        """Compute KL divergence between posterior and prior."""
        sigma_prior = self.prior_sigma

        weight_kl = (
            torch.log(sigma_prior)
            - 0.5 * torch.log(self.weight_sigma**2)
            + 0.5 * (self.weight_sigma**2 + (self.weight_mu**2)) / (sigma_prior**2)
            - 0.5
        ).sum()

        bias_kl = (
            torch.log(sigma_prior)
            - 0.5 * torch.log(self.bias_sigma**2)
            + 0.5 * (self.bias_sigma**2 + (self.bias_mu**2)) / (sigma_prior**2)
            - 0.5
        ).sum()

        return weight_kl + bias_kl


class VariationalBatchNorm2d(nn.Module):
    """Variational Batch Normalization with Bayesian treatment of running stats.

    Args:
        num_features: Number of features
        eps: Epsilon for numerical stability
        momentum: Momentum for moving average
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight_mu = nn.Parameter(torch.ones(num_features))
        self.weight_sigma = nn.Parameter(torch.zeros(num_features))
        self.bias_mu = nn.Parameter(torch.zeros(num_features))
        self.bias_sigma = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight = (
                self.weight_mu + torch.randn_like(self.weight_sigma) * self.weight_sigma
            )
            bias = self.bias_mu + torch.randn_like(self.bias_sigma) * self.bias_sigma

            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.squeeze()
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * weight.unsqueeze(0).unsqueeze(2) + bias.unsqueeze(0).unsqueeze(2)

    def kl_divergence(self) -> Tensor:
        return Tensor(0.0)


class FlipoutLinear(nn.Module):
    """Flipout linear layer for efficient gradient estimation.

    Implements the Flipout technique for variance reduction in
    Bayesian neural networks as described in:
    "Flipout: Pseudo-GRandom Perturbations for Efficient BNNs" (Wen et al., 2018)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_perturbations: Number of flipout perturbations per forward pass
        bias: Whether to include bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_perturbations: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_perturbations = num_perturbations

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return F.linear(x, self.weight, self.bias)

        batch_size = x.size(0)
        output = torch.zeros_like(F.linear(x, self.weight, self.bias))

        for _ in range(self.num_perturbations):
            signs = torch.randint_like(x, 2) * 2 - 1
            output += F.linear(x * signs, self.weight, self.bias)

        return output / self.num_perturbations


class ReparameterizedDense(nn.Module):
    """Reparameterized dense layer using the reparameterization trick.

    This layer samples from a diagonal Gaussian distribution and uses
    the reparameterization trick for gradient computation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        prior_sigma: Prior standard deviation
        posterior_scale_init: Initial scale for posterior std
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        posterior_scale_init: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(
            torch.empty(out_features, in_features).fill_(
                torch.log(torch.tensor(posterior_scale_init))
            )
        )

        if prior_sigma > 0:
            self.prior_log_sigma = torch.log(torch.tensor(prior_sigma))
        else:
            self.register_parameter("prior_log_sigma", None)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight_sigma = torch.exp(self.weight_log_sigma)
            weight = self.weight_mu + torch.randn_like(weight_sigma) * weight_sigma
        else:
            weight = self.weight_mu

        return F.linear(x, weight)

    def kl_divergence(self) -> Tensor:
        weight_sigma = torch.exp(self.weight_log_sigma)

        if self.prior_log_sigma is not None:
            prior_log_sigma = self.prior_log_sigma
        else:
            prior_log_sigma = torch.zeros_like(self.weight_log_sigma)

        kl = (
            prior_log_sigma
            - self.weight_log_sigma
            + 0.5
            * (weight_sigma**2 + self.weight_mu**2)
            / torch.exp(2 * prior_log_sigma)
            - 0.5
        ).sum()

        return kl


class HeteroscedasticLoss(nn.Module):
    """Heteroscedastic loss for uncertainty estimation.

    Implements the negative log-likelihood with learned noise variance,
    allowing the model to learn different uncertainty levels for different inputs.

    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_mean: Tensor,
        pred_log_var: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute heteroscedastic loss.

        Args:
            pred_mean: Predicted mean
            pred_log_var: Log of predicted variance
            target: Target values

        Returns:
            Negative log-likelihood
        """
        precision = torch.exp(-pred_log_var)
        nll = 0.5 * (pred_log_var + precision * (target - pred_mean) ** 2)

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll
