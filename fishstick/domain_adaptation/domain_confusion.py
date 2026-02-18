"""
Domain Confusion Losses Module for Fishstick.

This module provides various domain confusion loss functions for domain adaptation
including MMD (Maximum Mean Discrepancy), CMD (Central Moment Discrepancy),
CORAL (Correlation Alignment), and DDC (Domain Discriminative Clustering).

Example:
    >>> from fishstick.domain_adaptation.domain_confusion import MMDLoss, CORAL
    >>> mmd = MMDLoss(kernel_type='rbf', sigma=1.0)
    >>> loss = mmd(source_features, target_features)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class MMDLoss(Module):
    """Maximum Mean Discrepancy loss for domain alignment.

    Measures the distance between source and target distributions using
    kernel embeddings of mean features.

    Reference:
        Gretton et al. "A Kernel Two-Sample Test" JMLR 2012

    Args:
        kernel_type: Type of kernel ('rbf', 'linear', 'poly').
        sigma: Bandwidth parameter for RBF kernel.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        sigma: Optional[float] = None,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma

    def _gaussian_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        if self.sigma is not None:
            bandwidth = self.sigma
        else:
            n = x.size(0)
            m = y.size(0)
            xy = torch.cat([x, y], dim=0)
            pairwise_sq_dists = torch.cdist(xy, xy, p=2) ** 2
            bandwidth = pairwise_sq_dists.mean() / 2

        x_expand = x.unsqueeze(1).expand(x.size(0), y.size(0), -1)
        y_expand = y.unsqueeze(0).expand(x.size(0), y.size(0), -1)
        dist = torch.sum((x_expand - y_expand) ** 2, dim=2)

        return torch.exp(-dist / (2 * bandwidth**2))

    def _linear_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.mm(x, y.t())

    def _poly_kernel(self, x: Tensor, y: Tensor, degree: int = 2) -> Tensor:
        return (torch.mm(x, y.t()) + 1) ** degree

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        if self.kernel_type == "rbf":
            k = self._gaussian_kernel(source, target)
        elif self.kernel_type == "linear":
            k = self._linear_kernel(source, target)
        elif self.kernel_type == "poly":
            k = self._poly_kernel(source, target)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        n = source.size(0)
        m = target.size(0)

        loss = (
            k[:n, :n].sum() / (n * n)
            + k[n:, n:].sum() / (m * m)
            - 2 * k[:n, n:].sum() / (n * m)
        )

        return loss


class CMMDissimilarity(Module):
    """Central Moment-based Metric for domain dissimilarity.

    Matches central moments at multiple orders between source and target
    distributions for better distribution alignment.

    Reference:
        Zellinger et al. "Central Moment Discrepancy for Domain Adaptation" ICLR 2017

    Args:
        order: Maximum order of moments to match.
    """

    def __init__(self, order: int = 5):
        super().__init__()
        self.order = order

    def _compute_moments(self, x: Tensor, order: int) -> Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-8

        normalized = (x - mean) / std
        moments = torch.stack(
            [(normalized**k).mean(dim=0) for k in range(1, order + 1)], dim=0
        )

        return moments

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        source_moments = self._compute_moments(source, self.order)
        target_moments = self._compute_moments(target, self.order)

        discrepancy = torch.mean(torch.abs(source_moments - target_moments))

        return discrepancy


class CORAL(Module):
    """Correlation Alignment (CORAL) loss.

    Aligns second-order statistics (covariances) of source and target features.

    Reference:
        Sun et al. "Return of Frustratingly Easy Domain Adaptation" AAAI 2016

    Args:
        balance: Balance factor for CORAL loss.
    """

    def __init__(self, balance: float = 1.0):
        super().__init__()
        self.balance = balance

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)

        source_centered = source - source_mean
        target_centered = target - target_mean

        source_cov = torch.mm(source_centered.t(), source_centered) / (
            source.size(0) - 1
        )
        target_cov = torch.mm(target_centered.t(), target_centered) / (
            target.size(0) - 1
        )

        loss = torch.mean((source_cov - target_cov) ** 2)

        return self.balance * loss


class DDC(Module):
    """Deep Domain Confusion (DDC) loss.

    Combines MMD with a multi-kernel approach for domain alignment.

    Reference:
        Tzeng et al. "Deep Domain Confusion for Maximizing Domain Invariance" 2014

    Args:
        kernel_num: Number of kernels to use.
        kernel_mul: Multiplier for kernel bandwidth.
        fix_sigma: Fixed kernel bandwidth.
    """

    def __init__(
        self,
        kernel_num: int = 5,
        kernel_mul: float = 2.0,
        fix_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def _multi_kernel_mmd(self, source: Tensor, target: Tensor) -> Tensor:
        batch_size = source.size(0)

        kernels = []
        if self.fix_sigma is not None:
            bandwidths = [self.fix_sigma]
        else:
            total = batch_size * 2
            xxyy = torch.cat([source, target], dim=0)
            dist = torch.cdist(xxyy, xxyy, p=2)
            bandwidth = dist.sum() / (total * total - total)
            bandwidths = [
                bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)
            ]

        for bandwidth in bandwidths:
            x_expand = source.unsqueeze(1).expand(batch_size, target.size(0), -1)
            y_expand = target.unsqueeze(0).expand(batch_size, batch_size, -1)
            dist = torch.sum((x_expand - y_expand) ** 2, dim=2)
            kernels.append(torch.exp(-dist / (2 * bandwidth**2)))

        return sum(kernels) / len(kernels)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        k = self._multi_kernel_mmd(source, target)
        n = source.size(0)

        loss = (
            k[:n, :n].sum() / (n * n)
            + k[n:, n:].sum() / (n * n)
            - 2 * k[:n, n:].sum() / (n * n)
        )

        return loss


class DomainConfusionLoss(Module):
    """General domain confusion loss combining multiple strategies.

    Combines MMD, CORAL, and entropy minimization for robust domain alignment.

    Args:
        mmd_weight: Weight for MMD component.
        coral_weight: Weight for CORAL component.
        entropy_weight: Weight for entropy component.
    """

    def __init__(
        self,
        mmd_weight: float = 1.0,
        coral_weight: float = 1.0,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.mmd_weight = mmd_weight
        self.coral_weight = coral_weight
        self.entropy_weight = entropy_weight

        self.mmd = MMDLoss(kernel_type="rbf")
        self.coral = CORAL()

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        target_predictions: Optional[Tensor] = None,
    ) -> Tensor:
        loss = torch.tensor(0.0, device=source.device)

        if self.mmd_weight > 0:
            loss = loss + self.mmd_weight * self.mmd(source, target)

        if self.coral_weight > 0:
            loss = loss + self.coral_weight * self.coral(source, target)

        if self.entropy_weight > 0 and target_predictions is not None:
            entropy = -torch.sum(
                target_predictions * torch.log(target_predictions + 1e-10), dim=1
            ).mean()
            loss = loss + self.entropy_weight * entropy

        return loss


class JDA(Module):
    """Joint Distribution Adaptation (JDA) loss.

    Jointly aligns marginal and conditional distributions.

    Reference:
        Long et al. "Transfer Feature Learning with Joint Distribution Adaptation" ICCV 2013

    Args:
        kernel_type: Kernel type for MMD.
    """

    def __init__(self, kernel_type: str = "rbf"):
        super().__init__()
        self.kernel_type = kernel_type
        self.mmd = MMDLoss(kernel_type=kernel_type)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_labels: Tensor,
        num_classes: int,
    ) -> Tensor:
        marginal_mmd = self.mmd(source, target)

        conditional_mmds = []
        for c in range(num_classes):
            source_mask = source_labels == c
            if source_mask.sum() > 0:
                source_class = source[source_mask]
                target_class_mean = target.mean(dim=0, keepdim=True)

                class_mmd = torch.norm(source_class.mean(dim=0) - target_class_mean)
                conditional_mmds.append(class_mmd)

        conditional_loss = (
            torch.stack(conditional_mmds).mean()
            if conditional_mmds
            else torch.tensor(0.0)
        )

        return marginal_mmd + conditional_loss


class BDA(Module):
    """Balanced Distribution Adaptation (BDA) loss.

    Adaptively balances source and target distributions using
    domain ratio estimation.

    Reference:
        Wang et al. "Balanced Distribution Adaptation for Transfer Learning" ICDCS 2017

    Args:
        kernel_type: Kernel type for MMD.
        beta: Balance factor between marginal and conditional adaptation.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        beta: float = 0.5,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.beta = beta
        self.mmd = MMDLoss(kernel_type=kernel_type)

    def compute_domain_ratio(self, source: Tensor, target: Tensor) -> float:
        source_var = torch.var(source).item()
        target_var = torch.var(target).item()

        ratio = target_var / (source_var + target_var + 1e-10)
        return min(max(ratio, 0.0), 1.0)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_labels: Tensor,
        num_classes: int,
    ) -> Tensor:
        domain_ratio = self.compute_domain_ratio(source, target)

        marginal_mmd = self.mmd(source, target)

        conditional_mmds = []
        for c in range(num_classes):
            source_mask = source_labels == c
            if source_mask.sum() > 0:
                source_class = source[source_mask]
                class_ratio = source_mask.sum().item() / source.size(0)
                conditional_mmds.append(class_ratio * self.mmd(source_class, target))

        conditional_loss = (
            torch.stack(conditional_mmds).sum()
            if conditional_mmds
            else torch.tensor(0.0)
        )

        loss = (1 - self.beta) * domain_ratio * marginal_mmd + self.beta * (
            1 - domain_ratio
        ) * conditional_loss

        return loss


class JGSA(Module):
    """Joint Geometrical and Statistical Alignment (JGSA).

    Aligns both geometrical and statistical properties of source and target domains.

    Reference:
        Zhang et al. "Joint Geometrical and Statistical Alignment for Visual Domain Adaptation" CVPR 2017

    Args:
        kernel_type: Kernel type for MMD.
    """

    def __init__(self, kernel_type: str = "rbf"):
        super().__init__()
        self.kernel_type = kernel_type
        self.mmd = MMDLoss(kernel_type=kernel_type)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_labels: Tensor,
        num_classes: int,
    ) -> Tensor:
        marginal_mmd = self.mmd(source, target)

        source_mean_per_class = []
        target_mean = target.mean(dim=0)

        for c in range(num_classes):
            mask = source_labels == c
            if mask.sum() > 0:
                source_mean_per_class.append(source[mask].mean(dim=0))

        class_alignment = torch.tensor(0.0, device=source.device)
        if source_mean_per_class:
            source_centers = torch.stack(source_mean_per_class)
            class_alignment = torch.norm(source_centers.mean(dim=0) - target_mean)

        return marginal_mmd + 0.5 * class_alignment


class MEDA(Module):
    """Manifold Embedded Distribution Alignment (MEDA).

    Performs distribution alignment in manifold space using Grassmann manifold.

    Reference:
        Wang et al. "Manifold Embedded Distribution Alignment" ACM MM 2018

    Args:
        kernel_type: Kernel type.
        p: Dimensionality reduction parameter.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        p: int = 30,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.p = p
        self.mmd = MMDLoss(kernel_type=kernel_type)

    def _pca_reduce(self, x: Tensor) -> Tensor:
        if x.size(1) <= self.p:
            return x

        mean = x.mean(dim=0)
        x_centered = x - mean
        cov = torch.mm(x_centered.t(), x_centered) / (x.size(0) - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        top_eigenvectors = eigenvectors[:, -self.p :]

        return torch.mm(x_centered, top_eigenvectors)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        source_reduced = self._pca_reduce(source)
        target_reduced = self._pca_reduce(target)

        return self.mmd(source_reduced, target_reduced)
