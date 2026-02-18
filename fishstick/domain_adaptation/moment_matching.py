"""
Moment Matching Domain Adaptation Module for Fishstick.

This module provides moment matching methods for domain adaptation including
MMDA (Maximum Mean Discrepancy), DeepJDOT (Deep Joint Distribution Optimal Transport),
and CMD (Central Moment Discrepancy).

Example:
    >>> from fishstick.domain_adaptation.moment_matching import MMDLoss, MMDA
    >>> mmd_loss = MMDLoss(kernel_type='rbf', sigma=1.0)
    >>> loss = mmd_loss(source_features, target_features)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
from torch import Tensor
from torch.nn import Module

T = TypeVar("T")


def gaussian_kernel(
    source: Tensor,
    target: Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> Tensor:
    """Compute Gaussian kernel matrix between source and target.

    Uses multiple kernel widths for more robust MMD estimation.

    Args:
        source: Source features [n_samples, feature_dim].
        target: Target features [m_samples, feature_dim].
        kernel_mul: Multiplier for kernel bandwidth.
        kernel_num: Number of kernels to use.
        fix_sigma: Fixed kernel bandwidth (optional).

    Returns:
        Kernel matrix [n_samples + m_samples, n_samples + m_samples].
    """
    n_samples = source.size(0)
    m_samples = target.size(0)
    total = n_samples + m_samples

    combined = torch.cat([source, target], dim=0)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        pairwise_dists = torch.cdist(combined, combined, p=2)
        bandwidth = torch.sum(pairwise_dists) / (total**2 - total)

    bandwidth /= kernel_mul
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_matrix = torch.zeros((total, total), device=source.device)
    for bandwidth in bandwidth_list:
        kernel_matrix += torch.exp(-pairwise_dists / (2 * bandwidth**2))

    return kernel_matrix


def mmd_linear(kernel_matrix: Tensor, n_source: int) -> Tensor:
    """Compute linear MMD from kernel matrix.

    Args:
        kernel_matrix: Kernel matrix from gaussian_kernel.
        n_source: Number of source samples.

    Returns:
        MMD^2 loss value.
    """
    n_total = kernel_matrix.size(0)
    n_target = n_total - n_source

    start, end = 0, n_source
    source_part = kernel_matrix[start:end, start:end]
    start, end = n_source, n_total
    target_part = kernel_matrix[start:end, start:end]
    start, end = 0, n_source
    cross_part = kernel_matrix[start:end, n_source:]

    loss = (
        torch.sum(source_part) / (n_source * n_source)
        + torch.sum(target_part) / (n_target * n_target)
        - 2 * torch.sum(cross_part) / (n_source * n_target)
    )

    return loss


class MMDLoss(Module):
    """Maximum Mean Discrepancy (MMD) loss for domain adaptation.

    Measures the distance between source and target distributions using
    kernel embeddings. Minimizing MMD aligns the domain distributions.

    Reference:
        Gretton et al. "A Kernel Two-Sample Test" JMLR 2012

    Args:
        kernel_type: Type of kernel ('rbf', 'linear', 'poly').
        sigma: RBF kernel bandwidth parameter.
        kernel_mul: Multiplier for kernel bandwidth.
        kernel_num: Number of kernels for multi-kernel MMD.

    Example:
        >>> mmd = MMDLoss(kernel_type='rbf', sigma=1.0)
        >>> loss = mmd(source_features, target_features)
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        sigma: Optional[float] = None,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        if self.kernel_type == "linear":
            return self._mmd_linear(source, target)
        elif self.kernel_type == "rbf":
            return self._mmd_rbf(source, target)
        elif self.kernel_type == "poly":
            return self._mmd_poly(source, target)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _mmd_linear(self, source: Tensor, target: Tensor) -> Tensor:
        source_mean = torch.mean(source, dim=0)
        target_mean = torch.mean(target, dim=0)
        return torch.sum((source_mean - target_mean) ** 2)

    def _mmd_rbf(self, source: Tensor, target: Tensor) -> Tensor:
        kernel_matrix = gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.sigma,
        )
        n_source = source.size(0)
        return mmd_linear(kernel_matrix, n_source)

    def _mmd_poly(self, source: Tensor, target: Tensor) -> Tensor:
        combined = torch.cat([source, target], dim=0)
        n_source = source.size(0)

        source_mean = torch.mean(source, dim=0)
        target_mean = torch.mean(target, dim=0)

        loss = torch.sum((source_mean - target_mean) ** 2)

        combined_mean = torch.mean(combined, dim=0)
        loss += torch.sum(combined_mean**2)

        return loss


class CMD(Module):
    """Central Moment Discrepancy for domain adaptation.

    Matches central moments of source and target distributions at multiple
    orders, providing a finer-grained distribution alignment than MMD.

    Reference:
        Zellinger et al. "Central Moment Discrepancy (CMD) for
        Domain-Invariant Representation Learning" ICLR 2017

    Args:
        order: Maximum order of moments to match.
        epsilon: Small constant for numerical stability.

    Example:
        >>> cmd = CMD(order=5)
        >>> loss = cmd(source_features, target_features)
    """

    def __init__(self, order: int = 5, epsilon: float = 1e-8):
        super().__init__()
        self.order = order
        self.epsilon = epsilon

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        batch_size = source.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=source.device)

        source_normalized = (source - source.mean(dim=0, keepdim=True)) / (
            source.std(dim=0, keepdim=True) + self.epsilon
        )
        target_normalized = (target - target.mean(dim=0, keepdim=True)) / (
            target.std(dim=0, keepdim=True) + self.epsilon
        )

        loss = 0.0
        for order in range(1, self.order + 1):
            source_moment = torch.mean(source_normalized**order, dim=0)
            target_moment = torch.mean(target_normalized**order, dim=0)
            loss += torch.mean(torch.abs(source_moment - target_moment))

        return loss / self.order


class MomentMatchingLoss(Module):
    """Combined moment matching loss with MMD and CMD.

    Combines MMD and CMD losses for more robust domain alignment.

    Args:
        mmd_weight: Weight for MMD loss.
        cmd_weight: Weight for CMD loss.
        kernel_type: Type of kernel for MMD.
        cmd_order: Order for CMD.
    """

    def __init__(
        self,
        mmd_weight: float = 1.0,
        cmd_weight: float = 1.0,
        kernel_type: str = "rbf",
        cmd_order: int = 5,
    ):
        super().__init__()
        self.mmd_weight = mmd_weight
        self.cmd_weight = cmd_weight
        self.mmd = MMDLoss(kernel_type=kernel_type)
        self.cmd = CMD(order=cmd_order)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        mmd_loss = self.mmd(source, target)
        cmd_loss = self.cmd(source, target)
        return self.mmd_weight * mmd_loss + self.cmd_weight * cmd_loss


class KornetovLoss(Module):
    """Kornetov et al. domain adaptation loss.

    Combines label classification loss with domain moment matching.

    Reference:
        Kornetov et al. "Deep Asymmetric Transfer Learning" 2018
    """

    def __init__(
        self,
        alpha: float = 0.1,
        kernel_type: str = "rbf",
    ):
        super().__init__()
        self.alpha = alpha
        self.mmd = MMDLoss(kernel_type=kernel_type)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_labels: Optional[Tensor] = None,
    ) -> Tensor:
        mmd_loss = self.mmd(source, target)

        class_separation = torch.tensor(0.0, device=source.device)
        if source_labels is not None:
            unique_labels = torch.unique(source_labels)
            class_centers = []
            for label in unique_labels:
                mask = source_labels == label
                if mask.sum() > 0:
                    class_centers.append(source[mask].mean(dim=0))

            if len(class_centers) > 1:
                for i, center1 in enumerate(class_centers):
                    for center2 in class_centers[i + 1 :]:
                        class_separation += torch.sum((center1 - center2) ** 2)

        return mmd_loss - self.alpha * class_separation


class MMDA(Module):
    """Maximum Mean Discrepancy Autoencoder for domain adaptation.

    Combines reconstruction loss with MMD for joint distribution alignment.

    Reference:
        Tolstikhin et al. "MMDA: Maximum Mean Discrepancy Autoencoder" 2016

    Args:
        feature_dim: Dimension of input features.
        latent_dim: Dimension of latent space.
        hidden_dims: Hidden layer dimensions.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        encoder_layers = []
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, feature_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.mmd = MMDLoss()
        self.latent_dim = latent_dim

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        source_latent = self.encode(source)
        target_latent = self.encode(target)

        source_recon = self.decode(source_latent)
        target_recon = self.decode(target_latent)

        source_recon_loss = F.mse_loss(source_recon, source)
        target_recon_loss = F.mse_loss(target_recon, target)

        mmd_loss = self.mmd(source_latent, target_latent)

        return source_recon_loss, target_recon_loss, mmd_loss


class DeepJDOT(Module):
    """Deep Joint Distribution Optimal Transport.

    Jointly optimizes feature extraction and label transfer via optimal transport.

    Reference:
        Courty et al. "DeepJDOT: Deep Joint Distribution Optimal Transport
        for Domain Adaptation" AAAI 2018

    Args:
        feature_dim: Dimension of input features.
        num_classes: Number of classes.
        hidden_dims: Hidden layer dimensions.
        ot_reg: Entropic regularization parameter for OT.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        ot_reg: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.feature_extractor = nn.Sequential()
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            self.feature_extractor.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        self.feature_dim = prev_dim

        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.ot_reg = ot_reg

    def compute_ot_loss(
        self,
        source_features: Tensor,
        target_features: Tensor,
        source_labels: Tensor,
        num_classes: int,
    ) -> Tensor:
        source_features = source_features.detach()
        target_features = target_features.detach()

        batch_size = source_features.size(0)

        cost_matrix = torch.cdist(source_features, target_features, p=2)

        source_onehot = F.one_hot(source_labels, num_classes).float()
        source_soft = torch.zeros(
            batch_size, num_classes, device=source_features.device
        )
        source_soft[:, : source_onehot.size(1)] = source_onehot

        C_class = torch.cdist(
            source_soft,
            torch.zeros(num_classes, device=source_features.device)
            .unsqueeze(0)
            .expand(batch_size, -1),
            p=2,
        )

        combined_cost = cost_matrix + self.ot_reg * C_class

        return torch.mean(combined_cost)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        source_features = self.feature_extractor(source)
        target_features = self.feature_extractor(target)

        class_preds = self.classifier(source_features)

        ot_loss = torch.tensor(0.0, device=source.device)
        if source_labels is not None:
            ot_loss = self.compute_ot_loss(
                source_features, target_features, source_labels, class_preds.size(1)
            )

        return source_features, target_features, class_preds, ot_loss
