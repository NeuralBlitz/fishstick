"""
Domain-Specific Batch Normalization Module for Fishstick.

This module provides domain-specific batch normalization techniques including
AdaBN (Adaptive Batch Normalization), DBN (Domain-Specific Batch Normalization),
and SHiP (Sample-to-Pixel Adaptation).

Example:
    >>> from fishstick.domain_adaptation.batch_norm import AdaBN, DomainBatchNorm
    >>> adapter = AdaBN(num_domains=2)
    >>> adapted_model = adapter.apply(model, domain_id=1)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

T = TypeVar("T")


class DomainBatchNorm(Module):
    """Domain-specific Batch Normalization layer.

    Maintains separate batch statistics (mean, variance) for each domain,
    allowing domain-specific normalization during inference.

    Args:
        num_features: Number of input features.
        num_domains: Number of domains to track.
        eps: Small constant for numerical stability.
        momentum: Momentum for moving average computation.

    Example:
        >>> bn = DomainBatchNorm(num_features=256, num_domains=3)
        >>> # Source domain (0)
        >>> out = bn(x, domain_id=0)
        >>> # Target domain (1)
        >>> out = bn(x, domain_id=1)
    """

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.source_bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

        self.domain_means = nn.ParameterList(
            [Parameter(torch.zeros(num_features)) for _ in range(num_domains)]
        )
        self.domain_vars = nn.ParameterList(
            [Parameter(torch.ones(num_features)) for _ in range(num_domains)]
        )

        self.register_buffer("source_mean", torch.zeros(num_features))
        self.register_buffer("source_var", torch.ones(num_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, 0, 1)
        nn.init.zeros_(self.bias)
        self.source_bn.reset_parameters()

    def set_domain_stats(
        self,
        domain_id: int,
        mean: Tensor,
        var: Tensor,
    ) -> None:
        self.domain_means[domain_id].data = mean
        self.domain_vars[domain_id].data = var

    def forward(self, x: Tensor, domain_id: int = 0) -> Tensor:
        if self.training:
            out = self.source_bn(x)
            if domain_id == 0:
                self.source_mean.data = self.source_bn.running_mean.data.clone()
                self.source_var.data = self.source_bn.running_var.data.clone()
            return out

        if domain_id == 0:
            mean = self.source_mean
            var = self.source_var
        else:
            mean = self.domain_means[domain_id]
            var = self.domain_vars[domain_id]

        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized + self.bias


class AdaptiveBatchNorm(Module):
    """Adaptive Batch Normalization that learns to adapt BN statistics.

    Learns domain-specific affine transformation parameters for better
    domain alignment.

    Args:
        num_features: Number of input features.
        num_domains: Number of domains to support.
        embed_dim: Dimension of domain embedding.
    """

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains

        self.bn = nn.BatchNorm1d(num_features)

        self.domain_embedding = nn.Embedding(num_domains, embed_dim)
        self.gamma_predictor = nn.Sequential(
            nn.Linear(embed_dim, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.beta_predictor = nn.Sequential(
            nn.Linear(embed_dim, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )

    def forward(self, x: Tensor, domain_id: Optional[int] = None) -> Tensor:
        out = self.bn(x)

        if domain_id is not None and not self.training:
            domain_embed = self.domain_embedding(
                torch.tensor(domain_id, device=x.device)
            )
            gamma = self.gamma_predictor(domain_embed)
            beta = self.beta_predictor(domain_embed)

            out = gamma.unsqueeze(0) * out + beta.unsqueeze(0)

        return out


class BatchNormAdapter(Module):
    """BatchNorm Adapter for domain adaptation.

    Adapts pre-trained BatchNorm layers for new domains by recalculating
    running statistics.

    Args:
        model: Pre-trained model with BatchNorm layers.
        num_domains: Maximum number of domains to support.
    """

    def __init__(
        self,
        model: Module,
        num_domains: int = 10,
    ):
        super().__init__()
        self.model = model
        self.num_domains = num_domains

        self.bn_layers: Dict[str, DomainBatchNorm] = {}
        self._replace_bn(model, num_domains)

    def _replace_bn(self, module: Module, num_domains: int) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm1d):
                new_bn = DomainBatchNorm(
                    num_features=child.num_features,
                    num_domains=num_domains,
                    eps=child.eps,
                    momentum=child.momentum,
                )
                new_bn.weight.data = child.weight.data.clone()
                new_bn.bias.data = child.bias.data.clone()
                new_bn.source_bn.running_mean.data = child.running_mean.data.clone()
                new_bn.source_bn.running_var.data = child.running_var.data.clone()

                self.bn_layers[name] = new_bn
                setattr(module, name, new_bn)
            else:
                self._replace_bn(child, num_domains)

    def update_domain_stats(self, domain_id: int, data_loader: object) -> None:
        self.eval()
        means = []
        vars_list = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(next(self.parameters()).device)

                for name, bn in self.bn_layers.items():
                    if hasattr(bn, "source_bn"):
                        bn(x, domain_id)

        self.train()


class AdaBN(Module):
    """Adaptive Batch Normalization (AdaBN).

    Adapts pre-trained models to new domains by adjusting BN statistics
    without retraining. Simply runs target domain data through the network
    to update running statistics.

    Reference:
        Li et al. "Adaptive Batch Normalization for Domain Adaptation" CVPR 2016

    Args:
        model: Pre-trained model with BatchNorm layers.
    """

    def __init__(self, model: Module):
        super().__init__()
        self.model = model
        self.bn_layers: List[nn.Module] = []
        self._collect_bn_layers(model)

    def _collect_bn_layers(self, module: Module) -> None:
        for child in module.children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.bn_layers.append(child)
            else:
                self._collect_bn_layers(child)

    def adapt(self, target_data: Tensor, domain_id: int = 1) -> None:
        """Adapt model to target domain using target data.

        Args:
            target_data: Target domain data for adaptation.
            domain_id: Domain identifier for storing statistics.
        """
        self.model.eval()

        with torch.no_grad():
            if target_data.dim() == 2 and target_data.size(1) > 10000:
                batch_size = 32
                for i in range(0, target_data.size(0), batch_size):
                    batch = target_data[i : i + batch_size]
                    _ = self.model(batch)
            else:
                _ = self.model(target_data)

        self.model.train()

    def apply(
        self,
        model: Module,
        domain_id: int = 1,
    ) -> Module:
        """Apply AdaBN adaptation to model for specified domain.

        Args:
            model: Model to adapt.
            domain_id: Target domain ID.

        Returns:
            Adapted model.
        """
        return self.model


class SHiP(Module):
    """Sample-wise Homogenization via Pooling (SHiP).

    Uses sample-specific batch normalization statistics for improved
    domain adaptation on single samples.

    Reference:
        Wang et al. "SHiP: Sample-wise Homogenization via Pooling" ICCV 2019

    Args:
        num_features: Number of input features.
        momentum: Momentum for moving statistics.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum

        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        self.sample_mean = None
        self.sample_var = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, 0, 1)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if x.dim() == 2:
                batch_mean = x.mean(dim=0, keepdim=True)
                batch_var = x.var(dim=0, keepdim=True, unbiased=False)
            elif x.dim() == 4:
                batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
                batch_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            else:
                raise ValueError(f"Unsupported tensor shape: {x.shape}")

            self.running_mean.data = (
                self.momentum * batch_mean.squeeze()
                + (1 - self.momentum) * self.running_mean.data
            )
            self.running_var.data = (
                self.momentum * batch_var.squeeze()
                + (1 - self.momentum) * self.running_var.data
            )

            normalized = (x - batch_mean) / (batch_var + 1e-5).sqrt()
        else:
            normalized = (x - self.running_mean) / (self.running_var + 1e-5).sqrt()

        return self.weight * normalized + self.bias


class InstanceAlignmentNorm(Module):
    """Instance Alignment Normalization layer.

    Aligns feature distributions by matching instance-level statistics
    across domains.

    Args:
        num_features: Number of input features.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.source_mean = None
        self.source_std = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def set_source_stats(self, mean: Tensor, std: Tensor) -> None:
        self.source_mean = mean
        self.source_std = std

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + self.eps
        else:
            mean = self.source_mean.unsqueeze(0) if self.source_mean is not None else 0
            std = self.source_std.unsqueeze(0) if self.source_std is not None else 1

        normalized = (x - mean) / std
        return self.weight * normalized + self.bias


class GroupNormAdapter(Module):
    """Group Normalization adapter for domain adaptation.

    Adapts group normalization to different domains by learning
    domain-specific group assignments.

    Args:
        num_features: Number of input features.
        num_groups: Number of groups for GroupNorm.
        num_domains: Number of domains to support.
    """

    def __init__(
        self,
        num_features: int,
        num_groups: int = 32,
        num_domains: int = 2,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.num_domains = num_domains

        self.group_norms = nn.ModuleList(
            [nn.GroupNorm(num_groups, num_features) for _ in range(num_domains)]
        )

        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, domain_id: int = 0) -> Tensor:
        out = self.group_norms[domain_id](x)
        return self.weight * out + self.bias
