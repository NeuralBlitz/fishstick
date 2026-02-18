"""
Comprehensive Domain Adaptation Module for Fishstick.

This module provides state-of-the-art domain adaptation techniques for transferring
knowledge from source to target domains. Includes adversarial, discrepancy-based,
reconstruction-based, self-training, partial/open-set, source-free, and multi-source
methods with comprehensive evaluation utilities.

Example:
    >>> from fishstick.adaptation.domain import DANN, DATrainer
    >>> dann = DANN(feature_dim=512, num_classes=10)
    >>> trainer = DATrainer(method=dann, source_loader=src_loader, target_loader=tgt_loader)
    >>> trainer.fit(epochs=100)
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# Type aliases
TensorType = torch.Tensor
ModuleType = nn.Module
T = TypeVar("T")


# =============================================================================
# Utilities
# =============================================================================


class GradientReversalLayer(Function):
    """Gradient Reversal Layer for adversarial domain adaptation.

    Forward: identity
    Backward: multiply gradient by -lambda

    Reference:
        Ganin et al. "Domain-Adversarial Training of Neural Networks" JMLR 2016
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversal(Module):
    """Wrapper module for Gradient Reversal Layer.

    Args:
        lambda_init: Initial value for gradient reversal coefficient.
            Can be updated during training via `set_lambda()`.
    """

    def __init__(self, lambda_init: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_init

    def set_lambda(self, lambda_: float) -> None:
        """Update the gradient reversal coefficient."""
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalLayer.apply(x, self.lambda_)


class DomainDiscriminator(Module):
    """Binary domain discriminator for adversarial DA.

    A multi-layer perceptron that classifies features as source or target.

    Args:
        input_dim: Dimension of input features.
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout rate for regularization.
        batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [1024, 1024],
        dropout: float = 0.5,
        batch_norm: bool = True,
    ):
        super().__init__()

        layers: List[Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Returns domain logits (source=1, target=0)."""
        return self.network(x).squeeze(-1)


class GaussianKernel(Module):
    """Gaussian kernel for MMD computation.

    Args:
        sigma: Bandwidth parameter. If None, uses median heuristic.
    """

    def __init__(self, sigma: Optional[float] = None):
        super().__init__()
        self.sigma = sigma

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """Compute Gaussian kernel matrix between X and Y.

        Args:
            X: Source features [n, d]
            Y: Target features [m, d]

        Returns:
            Kernel matrix [n, m]
        """
        XX = X @ X.T
        XY = X @ Y.T
        YY = Y @ Y.T

        X_sqnorms = XX.diag()
        Y_sqnorms = YY.diag()

        r = lambda x: x.unsqueeze(0).expand_as(XY)

        K = -2 * XY + r(X_sqnorms) + r(Y_sqnorms).T

        if self.sigma is None:
            # Median heuristic
            median = torch.median(K[K > 0])
            sigma = torch.sqrt(median) / 2
        else:
            sigma = self.sigma

        return torch.exp(-K / (2 * sigma**2))


class MultiKernelMMD(Module):
    """Multi-kernel Maximum Mean Discrepancy.

    Uses multiple Gaussian kernels with different bandwidths.

    Args:
        sigmas: List of bandwidth parameters. If None, uses log-spaced values.
        num_kernels: Number of kernels to use with median heuristic.
    """

    def __init__(
        self,
        sigmas: Optional[List[float]] = None,
        num_kernels: int = 5,
    ):
        super().__init__()
        self.sigmas = sigmas
        self.num_kernels = num_kernels

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """Compute MK-MMD between source and target features."""
        if self.sigmas is None:
            # Use multiple bandwidths based on median heuristic
            XX = source @ source.T
            XY = source @ target.T
            YY = target @ target.T

            X_sqnorms = XX.diag()
            Y_sqnorms = YY.diag()

            r = lambda x: x.unsqueeze(0).expand_as(XY)
            K = -2 * XY + r(X_sqnorms) + r(Y_sqnorms).T

            median = torch.median(K[K > 0])
            base_sigma = torch.sqrt(median) / 2

            sigmas = [
                base_sigma * (2**i)
                for i in range(-self.num_kernels // 2, self.num_kernels // 2)
            ]
        else:
            sigmas = self.sigmas

        mmd = torch.tensor(0.0, device=source.device)
        for sigma in sigmas:
            kernel = GaussianKernel(sigma=sigma)
            mmd += self._mmd_with_kernel(source, target, kernel)

        return mmd / len(sigmas)

    def _mmd_with_kernel(
        self,
        source: Tensor,
        target: Tensor,
        kernel: GaussianKernel,
    ) -> Tensor:
        """Compute MMD using a specific kernel."""
        K_ss = kernel.forward(source, source)
        K_st = kernel.forward(source, target)
        K_tt = kernel.forward(target, target)

        n = source.size(0)
        m = target.size(0)

        loss = K_ss.sum() / (n * n) - 2 * K_st.sum() / (n * m) + K_tt.sum() / (m * m)
        return loss


# =============================================================================
# Base Classes
# =============================================================================


class DomainAdaptationMethod(Module, ABC):
    """Abstract base class for domain adaptation methods.

    All domain adaptation methods must implement:
        - forward: Forward pass through the network
        - compute_loss: Compute the adaptation loss
    """

    @abstractmethod
    def forward(
        self,
        source_x: Optional[Tensor] = None,
        target_x: Optional[Tensor] = None,
        source_y: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            source_x: Source domain input features.
            target_x: Target domain input features.
            source_y: Source domain labels.

        Returns:
            Dictionary containing outputs and intermediate representations.
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        source_y: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute the domain adaptation loss.

        Args:
            outputs: Dictionary from forward pass.
            source_y: Source domain labels.

        Returns:
            Dictionary of loss components.
        """
        pass


class BaseFeatureExtractor(Module, ABC):
    """Base class for feature extractors used in DA methods."""

    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Extract features from input."""
        pass


class SimpleFeatureExtractor(BaseFeatureExtractor):
    """Simple CNN-based feature extractor.

    Args:
        input_dim: Input dimension (for flattening).
        feature_dim: Output feature dimension.
        hidden_dims: Hidden layer dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dims: List[int] = [1024, 512],
    ):
        super().__init__(input_dim, feature_dim)

        layers: List[Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Extract features."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
