"""
Comprehensive Contrastive Learning Module for Fishstick

This module implements a wide variety of contrastive learning methods:

1. **Instance Discrimination**:
   - SimCLR: Simple contrastive learning
   - MoCo: Momentum contrast
   - MoCov2: Improved MoCo
   - MoCov3: ViT-based MoCo
   - SimSiam: Simple siamese
   - BYOL: Bootstrap your own latent
   - SWaV: Swapping assignments
   - NNCLR: Nearest-neighbor CLR
   - BarlowTwins: Barlow twins

2. **Clustering-Based**:
   - DeepCluster: Deep clustering
   - SeLa: Self-labeling
   - SwAV: Online clustering
   - PCL: Prototypical contrastive
   - SCAN: Semantic clustering
   - SPICE: Semantic pseudo-labels

3. **Hard Negative Mining**:
   - HardNegatives: Hard negative mining
   - MixingNegatives: Mix hard negatives
   - DebiasCL: Debias negatives
   - HCL: Hard contrastive loss
   - SupervisedCL: Supervised contrastive

4. **Multimodal Contrastive**:
   - CLIP: Image-text contrastive
   - ALIGN: Large-scale alignment
   - Florence: Unified model
   - Data2Vec: Unified framework
   - ImageBind: Bind modalities

5. **Video Contrastive**:
   - VideoMoCo: Video MoCo
   - CVRL: Contrastive video
   - CoCLR: Cooperative learning
   - VINCE: Video instance
   - Pace: Pace prediction

6. **Contrastive Architectures**:
   - ProjectionHead: MLP projection
   - PredictionHead: BYOL predictor
   - PrototypeLayer: SwAV prototypes
   - Queue: MoCo dictionary
   - EMAUpdater: EMA for BYOL/MoCo

7. **Loss Functions**:
   - NTXentLoss: Normalized temperature
   - InfoNCE: Information NCE
   - DecoupledCL: Decoupled contrastive
   - VICReg: Variance-invariance

8. **Evaluation**:
   - LinearEvaluation: Linear probe
   - KNNEvaluation: KNN classifier
   - FineTuning: Full fine-tuning
   - TransferLearning: Downstream tasks
"""

from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from abc import ABC, abstractmethod
import copy
import math
import warnings
from collections import deque

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributed import all_gather, get_world_size, get_rank
import numpy as np


# =============================================================================
# 6. Contrastive Architectures
# =============================================================================


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Used in SimCLR, MoCo, and other contrastive methods.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output projection dimension
        num_layers: Number of layers in the MLP
        use_bn: Whether to use batch normalization
        bias_last: Whether to use bias in the last layer
        activation: Activation function ('relu', 'gelu', 'swish')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        use_bn: bool = True,
        bias_last: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "swish":
            act_fn = nn.SiLU(inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)

        for i in range(num_layers - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act_fn)

        layers.append(nn.Linear(dims[-2], dims[-1], bias=bias_last))
        self.projection = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Projected tensor of shape (batch_size, output_dim)
        """
        return self.projection(x)


class PredictionHead(nn.Module):
    """Prediction head for BYOL and related methods.

    This is a 2-layer MLP used in BYOL to predict the target projection.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        use_bn: Whether to use batch normalization
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, use_bn: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [
            nn.Linear(input_dim, hidden_dim),
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            ]
        )

        self.predictor = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predicted tensor of shape (batch_size, output_dim)
        """
        return self.predictor(x)


class PrototypeLayer(nn.Module):
    """Prototype layer for SwAV and clustering-based methods.

    Maintains a set of prototypes for online clustering.

    Args:
        dim: Feature dimension
        n_prototypes: Number of prototypes
        epsilon: Small constant for numerical stability
    """

    def __init__(self, dim: int, n_prototypes: int, epsilon: float = 0.05):
        super().__init__()
        self.dim = dim
        self.n_prototypes = n_prototypes
        self.epsilon = epsilon

        # Prototypes as learnable parameters
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim))
        nn.init.uniform_(self.prototypes, -1.0 / dim, 1.0 / dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input features of shape (batch_size, dim)

        Returns:
            Similarity scores of shape (batch_size, n_prototypes)
        """
        # Normalize features and prototypes
        x = F.normalize(x, dim=1)
        w = F.normalize(self.prototypes, dim=1)

        # Compute similarity
        return torch.mm(x, w.t())

    def get_prototypes(self) -> Tensor:
        """Get prototype vectors."""
        return F.normalize(self.prototypes, dim=1)


class Queue(nn.Module):
    """Dictionary queue for MoCo and similar methods.

    Maintains a queue of negative samples.

    Args:
        queue_size: Maximum queue size
        dim: Feature dimension
    """

    def __init__(self, queue_size: int, dim: int):
        super().__init__()
        self.queue_size = queue_size
        self.dim = dim

        # Initialize queue with random features
        self.register_buffer("queue", torch.randn(queue_size, dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Normalize queue
        self.queue = F.normalize(self.queue, dim=1)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: Tensor):
        """Update queue with new keys.

        Args:
            keys: New keys to add (batch_size, dim)
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Replace oldest entries
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr : ptr + batch_size] = keys
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[: batch_size - remaining] = keys[remaining:]

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def get_queue(self) -> Tensor:
        """Get current queue contents."""
        return self.queue.clone()


class EMAUpdater:
    """Exponential Moving Average (EMA) updater for BYOL, MoCo, etc.

    Maintains a shadow model that is updated via EMA from the online model.

    Args:
        online_model: The online model being trained
        target_model: The target model to update
        momentum: EMA momentum coefficient (default: 0.996)
        update_after_steps: Number of steps before starting updates
    """

    def __init__(
        self,
        online_model: nn.Module,
        target_model: nn.Module,
        momentum: float = 0.996,
        update_after_steps: int = 0,
    ):
        self.online_model = online_model
        self.target_model = target_model
        self.momentum = momentum
        self.update_after_steps = update_after_steps
        self.steps = 0

        # Freeze target model
        for param in target_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self):
        """Perform EMA update of target model."""
        self.steps += 1

        if self.steps < self.update_after_steps:
            return

        # Cosine schedule for momentum
        momentum = self.momentum
        if hasattr(self, "cosine_schedule"):
            momentum = (
                1
                - (1 - self.momentum)
                * (math.cos(math.pi * self.steps / self.max_steps) + 1)
                / 2
            )

        for online_param, target_param in zip(
            self.online_model.parameters(), self.target_model.parameters()
        ):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1 - momentum)

    def set_cosine_schedule(self, max_steps: int):
        """Enable cosine schedule for momentum."""
        self.cosine_schedule = True
        self.max_steps = max_steps


# =============================================================================
# 7. Loss Functions
# =============================================================================


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    The standard contrastive loss used in SimCLR.

    Args:
        temperature: Temperature parameter
        batch_size: Batch size
        device: Device to use
    """

    def __init__(
        self, temperature: float = 0.5, batch_size: int = 256, device: str = "cuda"
    ):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """
        Args:
            z_i: First view (batch_size, dim)
            z_j: Second view (batch_size, dim)

        Returns:
            NT-Xent loss
        """
        batch_size = z_i.shape[0]

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate features
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, dim)

        # Compute similarity matrix
        similarity_matrix = (
            torch.mm(representations, representations.t()) / self.temperature
        )

        # Create mask for positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix.masked_fill_(mask, -9e15)

        # Positive pairs are at distance batch_size
        pos_sim = torch.cat(
            [
                torch.diag(similarity_matrix, batch_size),
                torch.diag(similarity_matrix, -batch_size),
            ]
        )

        # Compute loss
        loss = -torch.log(pos_sim / similarity_matrix.sum(dim=1)).mean()

        return loss


class InfoNCELoss(nn.Module):
    """Information Noise Contrastive Estimation (InfoNCE) loss.

    General form of contrastive loss used in various methods.

    Args:
        temperature: Temperature parameter
        reduction: Loss reduction method
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        query: Tensor,
        positive_key: Tensor,
        negative_keys: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            query: Query features (batch_size, dim)
            positive_key: Positive key features (batch_size, dim)
            negative_keys: Negative key features (n_negatives, dim) or (batch_size, n_negatives, dim)

        Returns:
            InfoNCE loss
        """
        # Normalize
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)

        # Positive logits
        positive_logit = (
            torch.sum(query * positive_key, dim=-1, keepdim=True) / self.temperature
        )

        # Negative logits
        if negative_keys is not None:
            negative_keys = F.normalize(negative_keys, dim=-1)

            if negative_keys.dim() == 2:
                # Shared negatives
                negative_logits = torch.mm(query, negative_keys.t()) / self.temperature
            else:
                # Per-query negatives
                negative_logits = (
                    torch.bmm(
                        query.unsqueeze(1), negative_keys.transpose(-2, -1)
                    ).squeeze(1)
                    / self.temperature
                )

            logits = torch.cat([positive_logit, negative_logits], dim=-1)
        else:
            logits = positive_logit

        # Labels: positive is always at index 0
        labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels, reduction=self.reduction)


class DecoupledCLLoss(nn.Module):
    """Decoupled Contrastive Learning Loss.

    Separates positive and negative pairs for better learning.

    Args:
        temperature: Temperature parameter
        pos_weight: Weight for positive pairs
        neg_weight: Weight for negative pairs
    """

    def __init__(
        self, temperature: float = 0.1, pos_weight: float = 1.0, neg_weight: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """
        Args:
            z_i: First view (batch_size, dim)
            z_j: Second view (batch_size, dim)

        Returns:
            Decoupled contrastive loss
        """
        batch_size = z_i.shape[0]

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute all pairwise similarities
        sim_ii = torch.mm(z_i, z_i.t()) / self.temperature
        sim_ij = torch.mm(z_i, z_j.t()) / self.temperature
        sim_ji = torch.mm(z_j, z_i.t()) / self.temperature
        sim_jj = torch.mm(z_j, z_j.t()) / self.temperature

        # Create masks
        mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        sim_ii = sim_ii.masked_fill(mask, -9e15)
        sim_jj = sim_jj.masked_fill(mask, -9e15)

        # Positive pairs (diagonal)
        pos_ij = torch.diag(sim_ij)
        pos_ji = torch.diag(sim_ji)

        # Negative pairs
        neg_i = torch.cat([sim_ij, sim_ii], dim=1)
        neg_j = torch.cat([sim_jj, sim_ji], dim=1)

        # Loss computation
        loss_pos = -torch.mean(pos_ij + pos_ji) * self.pos_weight

        # Log-sum-exp for numerical stability
        loss_neg = (
            torch.mean(torch.logsumexp(neg_i, dim=1) + torch.logsumexp(neg_j, dim=1))
            * self.neg_weight
        )

        return loss_pos + loss_neg


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance Regularization Loss.

    Alternative to contrastive loss without negative samples.

    Args:
        sim_coeff: Weight for invariance loss
        std_coeff: Weight for variance loss
        cov_coeff: Weight for covariance loss
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Args:
            z_a: First view (batch_size, dim)
            z_b: Second view (batch_size, dim)

        Returns:
            Total loss and dictionary of individual losses
        """
        batch_size = z_a.shape[0]
        dim = z_a.shape[1]

        # Invariance loss (MSE)
        sim_loss = F.mse_loss(z_a, z_b)

        # Variance loss (hinge loss on standard deviation)
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.eps)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.eps)
        std_loss = (
            torch.mean(F.relu(1 - std_z_a)) / 2 + torch.mean(F.relu(1 - std_z_b)) / 2
        )

        # Covariance loss (off-diagonal elements of covariance matrix)
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        cov_z_a = (z_a.T @ z_a) / (batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (batch_size - 1)

        cov_loss = (
            self.off_diagonal(cov_z_a).pow_(2).sum() / dim
            + self.off_diagonal(cov_z_b).pow_(2).sum() / dim
        )

        # Total loss
        loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss, {
            "loss": loss.item(),
            "invariance": sim_loss.item(),
            "variance": std_loss.item(),
            "covariance": cov_loss.item(),
        }

    @staticmethod
    def off_diagonal(x: Tensor) -> Tensor:
        """Return off-diagonal elements of a matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# =============================================================================
# 1. Instance Discrimination Methods
# =============================================================================


class SimCLR(nn.Module):
    """SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

    Chen et al., 2020

    Args:
        encoder: Backbone encoder network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        # Get encoder output dimension
        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Projection head
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim, num_layers=2
        )

        # Loss
        self.criterion = NTXentLoss(temperature=temperature)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        """Infer encoder output dimension."""
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "fc"):
            return encoder.fc.in_features
        else:
            # Try forward pass
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                out = encoder(dummy)
                return out.flatten(1).shape[-1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x1: First augmented view
            x2: Second augmented view

        Returns:
            h1, h2: Encoder outputs
            z1, z2: Projected features
        """
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1.flatten(1))
        z2 = self.projector(h2.flatten(1))

        return h1, h2, z1, z2

    def get_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute contrastive loss."""
        return self.criterion(z1, z2)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get normalized embedding for downstream tasks."""
        h = self.encoder(x)
        z = self.projector(h.flatten(1))
        return F.normalize(z, dim=-1)


class MoCo(nn.Module):
    """MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.

    He et al., 2020

    Args:
        encoder: Backbone encoder
        projection_dim: Projection head output dimension
        hidden_dim: Hidden dimension
        queue_size: Queue size for negative samples
        momentum: Momentum for key encoder update
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature

        # Get encoder output dimension
        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Query encoder (online)
        self.encoder_q = encoder
        self.projector_q = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim, num_layers=2
        )

        # Key encoder (target, updated with momentum)
        self.encoder_k = copy.deepcopy(encoder)
        self.projector_k = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim, num_layers=2
        )

        # Freeze key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        # Queue
        self.queue = Queue(queue_size, projection_dim)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        """Infer encoder output dimension."""
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                out = encoder(dummy)
                return out.flatten(1).shape[-1]

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of key encoder."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1 - self.momentum
            )

        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1 - self.momentum
            )

    def forward(
        self, x_q: Tensor, x_k: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x_q: Query view
            x_k: Key view

        Returns:
            q: Query features
            k: Key features
            logits: Similarity logits
            labels: Ground truth labels
        """
        # Query
        q = self.projector_q(self.encoder_q(x_q).flatten(1))
        q = F.normalize(q, dim=1)

        # Key
        with torch.no_grad():
            self._momentum_update()
            k = self.projector_k(self.encoder_k(x_k).flatten(1))
            k = F.normalize(k, dim=1)

        # Compute logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.queue.clone().detach().t()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)

        # Update queue
        self.queue.dequeue_and_enqueue(k)

        return q, k, logits, labels

    def get_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute contrastive loss."""
        return self.criterion(logits, labels)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get normalized embedding."""
        h = self.encoder_q(x)
        z = self.projector_q(h.flatten(1))
        return F.normalize(z, dim=-1)


class MoCov2(MoCo):
    """MoCo v2: Improved Baselines with Momentum Contrastive Learning.

    Chen et al., 2020

    Improvements over MoCo v1:
    - MLP projection head
    - Stronger augmentations
    - Cosine learning rate schedule
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.2,
    ):
        super().__init__(
            encoder=encoder,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            queue_size=queue_size,
            momentum=momentum,
            temperature=temperature,
        )

        # Replace projection heads with MLP + prediction head
        self.predictor_q = PredictionHead(projection_dim, hidden_dim, projection_dim)

        self.predictor_k = PredictionHead(projection_dim, hidden_dim, projection_dim)

        for param in self.predictor_k.parameters():
            param.requires_grad = False

    def forward(
        self, x_q: Tensor, x_k: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward with prediction head."""
        # Query
        f_q = self.projector_q(self.encoder_q(x_q).flatten(1))
        q = F.normalize(self.predictor_q(f_q), dim=1)

        # Key
        with torch.no_grad():
            self._momentum_update()
            f_k = self.projector_k(self.encoder_k(x_k).flatten(1))
            k = F.normalize(self.predictor_k(f_k), dim=1)

        # Compute logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.queue.clone().detach().t()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)

        # Update queue
        self.queue.dequeue_and_enqueue(k)

        return q, k, logits, labels


class MoCov3(nn.Module):
    """MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers.

    Chen et al., 2021

    Specifically designed for Vision Transformers with stability improvements.

    Args:
        encoder: ViT or transformer encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        queue_size: Queue size
        momentum: Momentum coefficient
        temperature: Temperature
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        queue_size: int = 65536,
        momentum: float = 0.99,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Query encoder
        self.encoder_q = encoder
        self.projector_q = nn.Sequential(
            nn.Linear(self.encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Key encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.projector_k = nn.Sequential(
            nn.Linear(self.encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        self.queue = Queue(queue_size, projection_dim)

        self.criterion = nn.CrossEntropyLoss()

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 768  # Default for ViT-Base

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update with cosine schedule option."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    def forward(
        self, x_q: Tensor, x_k: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward with stability improvements."""
        # Query
        h_q = self.encoder_q(x_q)
        if isinstance(h_q, tuple):
            h_q = h_q[0]  # Handle transformers with tuple outputs

        # Global average pooling if needed
        if h_q.dim() == 3:
            h_q = h_q.mean(dim=1)

        q = self.projector_q(h_q)
        q = F.normalize(q, dim=1)

        # Key
        with torch.no_grad():
            self._momentum_update()
            h_k = self.encoder_k(x_k)
            if isinstance(h_k, tuple):
                h_k = h_k[0]
            if h_k.dim() == 3:
                h_k = h_k.mean(dim=1)
            k = self.projector_k(h_k)
            k = F.normalize(k, dim=1)

        # Compute logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.queue.clone().detach().t()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)

        self.queue.dequeue_and_enqueue(k)

        return q, k, logits, labels

    def get_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.criterion(logits, labels)


class SimSiam(nn.Module):
    """SimSiam: Exploring Simple Siamese Representation Learning.

    Chen & He, 2020

    Uses stop-gradient and predictor to prevent collapse without negative samples.

    Args:
        encoder: Backbone encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
    """

    def __init__(
        self, encoder: nn.Module, projection_dim: int = 2048, hidden_dim: int = 2048
    ):
        super().__init__()
        self.encoder = encoder

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Projection head
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim, num_layers=2
        )

        # Predictor
        self.predictor = PredictionHead(projection_dim, hidden_dim, projection_dim)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x1: First view
            x2: Second view

        Returns:
            z1, z2: Projections
            p1, p2: Predictions
        """
        # Compute projections
        z1 = self.projector(self.encoder(x1).flatten(1))
        z2 = self.projector(self.encoder(x2).flatten(1))

        # Compute predictions
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return z1, z2, p1, p2

    def get_loss(self, p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute negative cosine similarity with stop-gradient."""
        # Normalize
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Negative cosine similarity with stop-gradient
        loss = (
            -(p1 * z2.detach()).sum(dim=-1).mean() / 2
            - (p2 * z1.detach()).sum(dim=-1).mean() / 2
        )

        return loss

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get normalized projection."""
        h = self.encoder(x)
        z = self.projector(h.flatten(1))
        return F.normalize(z, dim=-1)


class BYOL(nn.Module):
    """BYOL: Bootstrap Your Own Latent.

    Grill et al., 2020

    Uses online and target networks with momentum update.

    Args:
        encoder: Backbone encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        momentum: Momentum coefficient
        update_after_steps: Steps before starting EMA
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        momentum: float = 0.996,
        update_after_steps: int = 0,
    ):
        super().__init__()

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Online network
        self.online_encoder = encoder
        self.online_projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim, num_layers=2, use_bn=False
        )
        self.online_predictor = PredictionHead(
            projection_dim, hidden_dim, projection_dim
        )

        # Target network
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim, num_layers=2, use_bn=False
        )

        # Freeze target
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        # EMA updater
        self.ema_updater = EMAUpdater(
            nn.ModuleList([self.online_encoder, self.online_projector]),
            nn.ModuleList([self.target_encoder, self.target_projector]),
            momentum=momentum,
            update_after_steps=update_after_steps,
        )

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x1: First view
            x2: Second view

        Returns:
            p1, p2: Online predictions
            z1, z2: Target projections
        """
        # Update target network
        self.ema_updater.update()

        # Online network
        p1 = self.online_predictor(
            self.online_projector(self.online_encoder(x1).flatten(1))
        )
        p2 = self.online_predictor(
            self.online_projector(self.online_encoder(x2).flatten(1))
        )

        # Target network
        with torch.no_grad():
            z1 = self.target_projector(self.target_encoder(x1).flatten(1))
            z2 = self.target_projector(self.target_encoder(x2).flatten(1))

        return p1, p2, z1, z2

    def get_loss(self, p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute MSE loss."""
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        loss = (
            2 - 2 * (p1 * z2).sum(dim=-1).mean() + 2 - 2 * (p2 * z1).sum(dim=-1).mean()
        )

        return loss / 2

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get normalized online projection."""
        h = self.online_encoder(x)
        z = self.online_projector(h.flatten(1))
        return F.normalize(z, dim=-1)


class SwAV(nn.Module):
    """SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.

    Caron et al., 2020

    Uses online clustering with Sinkhorn-Knopp algorithm.

    Args:
        encoder: Backbone encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        n_prototypes: Number of prototypes
        n_crops: Number of crops
        temperature: Temperature for clustering
        sinkhorn_iterations: Sinkhorn-Knopp iterations
        epsilon: Sinkhorn regularization
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        n_prototypes: int = 3000,
        n_crops: int = 2,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        epsilon: float = 0.05,
    ):
        super().__init__()
        self.n_crops = n_crops
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Encoder and projector
        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim, num_layers=2
        )

        # Prototypes
        self.prototypes = PrototypeLayer(projection_dim, n_prototypes, epsilon=epsilon)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def sinkhorn_knopp(self, scores: Tensor, iterations: int = 3) -> Tensor:
        """Sinkhorn-Knopp algorithm for optimal transport.

        Args:
            scores: Similarity scores (batch_size, n_prototypes)
            iterations: Number of iterations

        Returns:
            Soft assignments (batch_size, n_prototypes)
        """
        Q = torch.exp(scores / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]

        # Make doubly stochastic
        for _ in range(iterations):
            Q = Q / Q.sum(dim=1, keepdim=True)
            Q = Q / Q.sum(dim=0, keepdim=True)

        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.t()

    def forward(self, crops: List[Tensor]) -> List[Tensor]:
        """
        Args:
            crops: List of augmented crops

        Returns:
            List of prototype scores for each crop
        """
        batch_size = crops[0].shape[0]

        # Encode all crops
        embeddings = []
        for crop in crops:
            h = self.encoder(crop).flatten(1)
            z = self.projector(h)
            z = F.normalize(z, dim=1)
            embeddings.append(z)

        # Stack embeddings
        embeddings = torch.stack(embeddings)  # (n_crops, batch_size, dim)

        # Compute prototype scores
        scores = self.prototypes(embeddings.reshape(-1, embeddings.shape[-1]))
        scores = scores.reshape(self.n_crops, batch_size, -1)

        return scores

    def get_loss(self, scores: List[Tensor]) -> Tensor:
        """Compute swapped prediction loss."""
        losses = []

        for i in range(self.n_crops):
            for j in range(self.n_crops):
                if i != j:
                    # Compute soft assignments with Sinkhorn-Knopp
                    q_i = self.sinkhorn_knopp(scores[i])
                    q_j = self.sinkhorn_knopp(scores[j])

                    # Swapped prediction
                    loss = -torch.mean(
                        torch.sum(
                            q_i * F.log_softmax(scores[j] / self.temperature, dim=1),
                            dim=1,
                        )
                    )
                    losses.append(loss)

        return sum(losses) / len(losses)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get normalized embedding."""
        h = self.encoder(x)
        z = self.projector(h.flatten(1))
        return F.normalize(z, dim=-1)


class NNCLR(nn.Module):
    """NNCLR: With Nearest Neighbors For Contrastive Learning.

    Dwibedi et al., 2021

    Uses nearest neighbors in embedding space as positives.

    Args:
        encoder: Backbone encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        queue_size: Size of support set
        temperature: Temperature
        top_k: Number of nearest neighbors
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        queue_size: int = 10000,
        temperature: float = 0.1,
        top_k: int = 1,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Encoder and projectors
        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim, num_layers=2
        )
        self.predictor = PredictionHead(projection_dim, hidden_dim, projection_dim)

        # Support set (queue)
        self.register_buffer("queue", torch.randn(queue_size, projection_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: Tensor):
        """Update support set."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr : ptr + batch_size] = keys
        else:
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[: batch_size - remaining] = keys[remaining:]

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def get_nearest_neighbors(self, queries: Tensor, k: int = 1) -> Tensor:
        """Find k nearest neighbors in support set."""
        # Compute similarities
        similarities = torch.mm(queries, self.queue.t())

        # Get top-k
        _, indices = similarities.topk(k, dim=1)

        # Retrieve neighbors
        neighbors = self.queue[indices[:, 0]]  # Take closest for now
        return neighbors

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x1: First view
            x2: Second view

        Returns:
            z1, z2: Embeddings
            nn1, nn2: Nearest neighbors
        """
        # Compute embeddings
        z1 = self.projector(self.encoder(x1).flatten(1))
        z2 = self.projector(self.encoder(x2).flatten(1))

        # Get nearest neighbors
        with torch.no_grad():
            nn1 = self.get_nearest_neighbors(F.normalize(z1, dim=1))
            nn2 = self.get_nearest_neighbors(F.normalize(z2, dim=1))

        # Predictions
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Update queue
        self.dequeue_and_enqueue(F.normalize(z1, dim=1).detach())

        return p1, p2, nn1, nn2

    def get_loss(self, p1: Tensor, p2: Tensor, nn1: Tensor, nn2: Tensor) -> Tensor:
        """Compute contrastive loss with nearest neighbors."""
        # Normalize
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        nn1 = F.normalize(nn1, dim=-1)
        nn2 = F.normalize(nn2, dim=-1)

        # Compute logits
        l_pos1 = torch.sum(p1 * nn2, dim=-1, keepdim=True)
        l_pos2 = torch.sum(p2 * nn1, dim=-1, keepdim=True)

        # Use queue as negatives
        l_neg1 = torch.mm(p1, self.queue.t())
        l_neg2 = torch.mm(p2, self.queue.t())

        logits1 = torch.cat([l_pos1, l_neg1], dim=1) / self.temperature
        logits2 = torch.cat([l_pos2, l_neg2], dim=1) / self.temperature

        labels = torch.zeros(p1.shape[0], dtype=torch.long, device=p1.device)

        loss = F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)
        return loss / 2


class BarlowTwins(nn.Module):
    """Barlow Twins: Self-Supervised Learning via Redundancy Reduction.

    Zbontar et al., 2021

    Uses cross-correlation matrix to enforce similarity while reducing redundancy.

    Args:
        encoder: Backbone encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        lambd: Weight for redundancy reduction loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 8192,
        hidden_dim: int = 8192,
        lambd: float = 5e-3,
    ):
        super().__init__()
        self.lambd = lambd

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Encoder and projector
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_out_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim, bias=False),
        )

        # Batch normalization for the last layer (no affine parameters)
        self.bn = nn.BatchNorm1d(projection_dim, affine=False)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x1: First view
            x2: Second view

        Returns:
            z1, z2: Projected features
        """
        # Encode
        h1 = self.encoder(x1).flatten(1)
        h2 = self.encoder(x2).flatten(1)

        # Project
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return z1, z2

    def get_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute Barlow Twins loss."""
        batch_size = z1.shape[0]

        # Apply batch norm
        z1 = self.bn(z1)
        z2 = self.bn(z2)

        # Cross-correlation matrix
        c = torch.mm(z1.T, z2) / batch_size

        # Loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambd * off_diag
        return loss

    @staticmethod
    def off_diagonal(x: Tensor) -> Tensor:
        """Return off-diagonal elements."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get normalized embedding."""
        h = self.encoder(x)
        z = self.projector(h.flatten(1))
        return F.normalize(z, dim=-1)


# =============================================================================
# 2. Clustering-Based Methods
# =============================================================================


class DeepCluster(nn.Module):
    """DeepCluster: Deep Clustering for Unsupervised Learning of Visual Features.

    Caron et al., 2018

    Alternates between clustering features and training with pseudo-labels.

    Args:
        encoder: Backbone encoder
        n_clusters: Number of clusters
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_clusters: int = 10000,
        projection_dim: int = 128,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.n_clusters = n_clusters

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        # Clustering classifier
        self.classifier = nn.Linear(projection_dim, n_clusters)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        logits = self.classifier(z)
        return logits

    def get_features(self, x: Tensor) -> Tensor:
        """Get features for clustering."""
        with torch.no_grad():
            h = self.encoder(x).flatten(1)
            z = self.projector(h)
        return F.normalize(z, dim=1)

    def cluster(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """Perform k-means clustering on features.

        Returns:
            Pseudo-labels for all samples
        """
        from sklearn.cluster import KMeans

        # Extract features
        features = []
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.cuda() if torch.cuda.is_available() else x
            z = self.get_features(x)
            features.append(z.cpu().numpy())

        features = np.concatenate(features, axis=0)

        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        pseudo_labels = kmeans.fit_predict(features)

        return pseudo_labels


class SeLa(nn.Module):
    """Self-Labeling via Simultaneous Clustering and Representation Learning.

    Asano et al., 2020

    Joint optimization of clustering and representation.

    Args:
        encoder: Backbone encoder
        n_classes: Number of clusters
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int = 1000,
        projection_dim: int = 128,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        # Label head (cluster assignment)
        self.label_head = nn.Linear(projection_dim, n_classes)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        logits = self.label_head(z)
        return z, logits

    def get_loss(self, logits1: Tensor, logits2: Tensor) -> Tensor:
        """Compute self-labeling loss."""
        # Cross-entropy between soft assignments
        p1 = F.softmax(logits1, dim=1)
        p2 = F.log_softmax(logits2, dim=1)

        loss = -torch.mean(torch.sum(p1 * p2, dim=1))
        return loss


class PCL(nn.Module):
    """Prototypical Contrastive Learning.

    Li et al., 2021

    Uses prototypes (cluster centers) for contrastive learning.

    Args:
        encoder: Backbone encoder
        n_prototypes: Number of prototypes
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        temperature: Temperature
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_prototypes: int = 1000,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.temperature = temperature

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        # Prototypes
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, projection_dim))
        nn.init.uniform_(self.prototypes, -1.0 / projection_dim, 1.0 / projection_dim)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)

        # Compute prototype scores
        scores = torch.mm(z, self.prototypes.t())
        return z, scores

    def get_loss(
        self, z1: Tensor, z2: Tensor, scores1: Tensor, scores2: Tensor
    ) -> Tensor:
        """Compute prototypical contrastive loss."""
        # Use prototypes as positive/negative keys
        proto_norm = F.normalize(self.prototypes, dim=1)

        # Positive pairs
        pos_sim = torch.sum(z1 * z2, dim=1) / self.temperature

        # Negative pairs with prototypes
        neg_sim1 = torch.mm(z1, proto_norm.t()) / self.temperature
        neg_sim2 = torch.mm(z2, proto_norm.t()) / self.temperature

        # Contrastive loss
        loss1 = -torch.mean(pos_sim - torch.logsumexp(neg_sim1, dim=1))
        loss2 = -torch.mean(pos_sim - torch.logsumexp(neg_sim2, dim=1))

        return (loss1 + loss2) / 2


class SCAN(nn.Module):
    """SCAN: Learning to Classify Images without Labels.

    Van Gansbeke et al., 2020

    Semantic clustering with nearest neighbors.

    Args:
        encoder: Backbone encoder
        n_classes: Number of clusters
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int = 10,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_classes),
        )

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        logits = self.cluster_head(z)
        return z, logits

    def get_loss(self, logits: Tensor, neighbor_logits: Tensor) -> Tensor:
        """Compute SCAN loss.

        Consistency between sample and its nearest neighbor.
        """
        # Cross-entropy between sample and neighbor predictions
        p = F.softmax(logits, dim=1)
        p_neighbor = F.log_softmax(neighbor_logits, dim=1)

        # Consistency loss
        consistency_loss = -torch.mean(torch.sum(p * p_neighbor, dim=1))

        # Entropy loss (encourage confident predictions)
        p_avg = p.mean(dim=0)
        entropy_loss = torch.sum(p_avg * torch.log(p_avg + 1e-8))

        return consistency_loss - entropy_loss


class SPICE(nn.Module):
    """SPICE: Semantic Pseudo-Labeling for Image Clustering.

    Ni et al., 2021

    Combines contrastive learning with semantic pseudo-labels.

    Args:
        encoder: Backbone encoder
        n_classes: Number of clusters
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        temperature: Temperature
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int = 10,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.temperature = temperature

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        # Cluster head
        self.cluster_head = nn.Linear(projection_dim, n_classes)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[-1]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        logits = self.cluster_head(z)
        return z, logits

    def get_loss(
        self,
        z1: Tensor,
        z2: Tensor,
        logits1: Tensor,
        logits2: Tensor,
        pseudo_labels: Tensor,
    ) -> Tensor:
        """Compute SPICE loss."""
        # Contrastive loss
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        pos_sim = torch.sum(z1 * z2, dim=1) / self.temperature

        # Use pseudo-labels for negative selection
        mask = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)
        neg_sim = torch.mm(z1, z2.t()) / self.temperature
        neg_sim = neg_sim.masked_fill(mask, -9e15)

        contrastive_loss = -torch.mean(pos_sim - torch.logsumexp(neg_sim, dim=1))

        # Classification loss with pseudo-labels
        cls_loss = F.cross_entropy(logits1, pseudo_labels) + F.cross_entropy(
            logits2, pseudo_labels
        )

        return contrastive_loss + cls_loss / 2


# =============================================================================
# 3. Hard Negative Mining Methods
# =============================================================================


class HardNegatives(nn.Module):
    """Hard Negative Mining for Contrastive Learning.

    Selects hardest negatives from the queue.

    Args:
        base_loss: Base contrastive loss module
        mining_ratio: Ratio of hard negatives to sample
    """

    def __init__(self, base_loss: nn.Module, mining_ratio: float = 0.5):
        super().__init__()
        self.base_loss = base_loss
        self.mining_ratio = mining_ratio

    def mine_hard_negatives(self, query: Tensor, negatives: Tensor, k: int) -> Tensor:
        """Select k hardest negatives."""
        # Compute similarities
        sim = torch.mm(query, negatives.t())

        # Select hardest (highest similarity)
        _, indices = torch.topk(sim, k, dim=1, largest=True)

        # Gather hard negatives
        batch_size = query.shape[0]
        hard_negatives = negatives[indices.view(-1)].view(batch_size, k, -1)

        return hard_negatives

    def forward(self, query: Tensor, positive: Tensor, negatives: Tensor) -> Tensor:
        """Compute loss with hard negatives."""
        # Mine hard negatives
        n_hard = int(negatives.shape[0] * self.mining_ratio)
        hard_negatives = self.mine_hard_negatives(query, negatives, n_hard)

        # Compute loss
        return self.base_loss(query, positive, hard_negatives)


class MixingNegatives(nn.Module):
    """Mixing Hard Negatives for Contrastive Learning.

    Mixes hard negatives to create harder negatives.

    Args:
        base_loss: Base loss module
        mix_ratio: Mixing coefficient
    """

    def __init__(self, base_loss: nn.Module, mix_ratio: float = 0.5):
        super().__init__()
        self.base_loss = base_loss
        self.mix_ratio = mix_ratio

    def mix_negatives(self, query: Tensor, negatives: Tensor) -> Tensor:
        """Mix negatives with query."""
        # Mix each negative with query
        mixed = self.mix_ratio * query.unsqueeze(1) + (1 - self.mix_ratio) * negatives

        return mixed

    def forward(self, query: Tensor, positive: Tensor, negatives: Tensor) -> Tensor:
        """Compute loss with mixed negatives."""
        # Mix negatives
        mixed_negatives = self.mix_negatives(query, negatives)

        # Compute loss
        return self.base_loss(query, positive, mixed_negatives)


class DebiasCL(nn.Module):
    """Debiased Contrastive Learning.

    Corrects for sampling bias in negative pairs.

    Args:
        temperature: Temperature
        debias_weight: Weight for debiasing
    """

    def __init__(self, temperature: float = 0.5, debias_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.debias_weight = debias_weight

    def forward(self, z_i: Tensor, z_j: Tensor, tau_plus: float = 0.1) -> Tensor:
        """
        Args:
            z_i, z_j: Positive pairs
            tau_plus: Probability of sampling a true positive
        """
        batch_size = z_i.shape[0]

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = (
            torch.mm(representations, representations.t()) / self.temperature
        )

        # Create mask
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Positive pairs
        pos_sim = torch.cat(
            [
                torch.diag(similarity_matrix, batch_size),
                torch.diag(similarity_matrix, -batch_size),
            ]
        )

        # Debiased loss
        # Estimate negative distribution
        neg_exp = torch.exp(similarity_matrix).sum(dim=1)
        pos_exp = torch.exp(pos_sim)

        # Debiasing correction
        N = 2 * batch_size - 2
        Ng = (-tau_plus * N * pos_exp + neg_exp) / (1 - tau_plus)
        Ng = torch.clamp(Ng, min=N * math.exp(-1 / self.temperature))

        loss = -torch.log(pos_exp / (pos_exp + Ng)).mean()

        return loss


class HCL(nn.Module):
    """Hard Contrastive Loss.

    Explicitly handles hard negatives with importance weighting.

    Args:
        temperature: Temperature
        hard_negative_weight: Weight for hard negatives
        threshold: Hard negative threshold
    """

    def __init__(
        self,
        temperature: float = 0.5,
        hard_negative_weight: float = 10.0,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.threshold = threshold

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute HCL loss."""
        batch_size = z_i.shape[0]

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute all pairwise similarities
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = (
            torch.mm(representations, representations.t()) / self.temperature
        )

        # Create mask
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Positive pairs
        pos_sim = torch.cat(
            [
                torch.diag(similarity_matrix, batch_size),
                torch.diag(similarity_matrix, -batch_size),
            ]
        )

        # Hard negative detection
        neg_sim = torch.cat(
            [
                similarity_matrix[:batch_size, :batch_size],
                similarity_matrix[:batch_size, batch_size:],
                similarity_matrix[batch_size:, :batch_size],
                similarity_matrix[batch_size:, batch_size:],
            ],
            dim=0,
        )

        # Weight hard negatives
        is_hard = neg_sim > self.threshold / self.temperature
        weights = torch.ones_like(neg_sim)
        weights = weights + is_hard.float() * (self.hard_negative_weight - 1)

        # Compute weighted loss
        neg_exp = (torch.exp(neg_sim) * weights).sum(dim=1)
        pos_exp = torch.exp(pos_sim)

        loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()

        return loss


class SupervisedCL(nn.Module):
    """Supervised Contrastive Learning.

    Khosla et al., 2020

    Uses class labels to define positive and negative pairs.

    Args:
        temperature: Temperature
        base_temperature: Base temperature
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            features: Feature representations (batch_size * n_views, dim)
            labels: Labels for each sample (batch_size,)
        """
        device = features.device
        batch_size = labels.shape[0]
        n_views = features.shape[0] // batch_size

        # Normalize
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        anchor_feature = features
        anchor_count = n_views

        # Tile labels
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, anchor_feature.T), self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, anchor_count)

        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# =============================================================================
# 4. Multimodal Contrastive Methods
# =============================================================================


class CLIP(nn.Module):
    """CLIP: Contrastive Language-Image Pre-training.

    Radford et al., 2021

    Learns joint embeddings of images and text.

    Args:
        image_encoder: Vision encoder
        text_encoder: Text encoder
        embed_dim: Joint embedding dimension
        temperature: Temperature (learnable if init is None)
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 512,
        temperature: Optional[float] = 0.07,
    ):
        super().__init__()

        # Encoders
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # Projection layers
        self.image_out_dim = self._get_image_dim(image_encoder)
        self.text_out_dim = self._get_text_dim(text_encoder)

        self.image_projection = nn.Linear(self.image_out_dim, embed_dim)
        self.text_projection = nn.Linear(self.text_out_dim, embed_dim)

        # Learnable temperature
        if temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = temperature

    def _get_image_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 768

    def _get_text_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "hidden_size"):
            return encoder.hidden_size
        else:
            return 512

    def encode_image(self, image: Tensor) -> Tensor:
        """Encode image to joint embedding space."""
        features = self.image_encoder(image)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 3:
            features = features.mean(dim=1)

        return F.normalize(self.image_projection(features), dim=-1)

    def encode_text(self, text: Tensor) -> Tensor:
        """Encode text to joint embedding space."""
        features = self.text_encoder(text)
        if isinstance(features, tuple):
            features = features[0]

        return F.normalize(self.text_projection(features), dim=-1)

    def forward(self, images: Tensor, texts: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            images: Image tensors
            texts: Text token tensors

        Returns:
            image_features, text_features, logits
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # Compute similarity logits
        logit_scale = (
            self.logit_scale.exp()
            if isinstance(self.logit_scale, nn.Parameter)
            else self.logit_scale
        )
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return image_features, text_features, (logits_per_image, logits_per_text)

    def get_loss(self, logits_per_image: Tensor, logits_per_text: Tensor) -> Tensor:
        """Compute symmetric cross-entropy loss."""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        return (loss_i + loss_t) / 2


class ALIGN(nn.Module):
    """ALIGN: Scaling Up Visual and Vision-Language Representation Learning.

    Jia et al., 2021

    Large-scale noisy multimodal learning.

    Args:
        image_encoder: Vision encoder
        text_encoder: Text encoder
        embed_dim: Joint embedding dimension
    """

    def __init__(
        self, image_encoder: nn.Module, text_encoder: nn.Module, embed_dim: int = 640
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # Get output dimensions
        self.image_out_dim = self._get_image_dim(image_encoder)
        self.text_out_dim = self._get_text_dim(text_encoder)

        # Projection layers
        self.image_projection = nn.Linear(self.image_out_dim, embed_dim)
        self.text_projection = nn.Linear(self.text_out_dim, embed_dim)

    def _get_image_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 2048

    def _get_text_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "hidden_size"):
            return encoder.hidden_size
        else:
            return 768

    def encode_image(self, image: Tensor) -> Tensor:
        """Encode image."""
        features = self.image_encoder(image)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 3:
            features = features.mean(dim=1)

        return F.normalize(self.image_projection(features), dim=-1)

    def encode_text(self, text: Tensor) -> Tensor:
        """Encode text."""
        features = self.text_encoder(text)
        if isinstance(features, tuple):
            features = features[0]

        return F.normalize(self.text_projection(features), dim=-1)

    def forward(self, images: Tensor, texts: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        return image_features, text_features

    def get_loss(
        self, image_features: Tensor, text_features: Tensor, temperature: float = 1.0
    ) -> Tensor:
        """Compute contrastive loss with temperature."""
        logits = image_features @ text_features.t() / temperature

        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)

        return (loss_i + loss_t) / 2


class Florence(nn.Module):
    """Florence: A New Foundation Model for Computer Vision.

    Yuan et al., 2021

    Unified vision-language model.

    Args:
        image_encoder: Vision encoder
        text_encoder: Text encoder
        multimodal_encoder: Multimodal fusion encoder
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        multimodal_encoder: Optional[nn.Module] = None,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.multimodal_encoder = multimodal_encoder

        # Output dimensions
        self.image_out_dim = self._get_dim(image_encoder)
        self.text_out_dim = self._get_dim(text_encoder)

        # Projections
        self.image_projection = nn.Linear(self.image_out_dim, embed_dim)
        self.text_projection = nn.Linear(self.text_out_dim, embed_dim)

    def _get_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        elif hasattr(encoder, "hidden_size"):
            return encoder.hidden_size
        else:
            return 768

    def encode_image(self, image: Tensor) -> Tensor:
        """Encode image."""
        features = self.image_encoder(image)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 3:
            features = features[:, 0]  # CLS token

        return self.image_projection(features)

    def encode_text(self, text: Tensor) -> Tensor:
        """Encode text."""
        features = self.text_encoder(text)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 3:
            features = features[:, 0]

        return self.text_projection(features)

    def forward(
        self, images: Optional[Tensor] = None, texts: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Forward pass (unimodal or multimodal)."""
        outputs = {}

        if images is not None:
            outputs["image_features"] = self.encode_image(images)

        if texts is not None:
            outputs["text_features"] = self.encode_text(texts)

        if (
            images is not None
            and texts is not None
            and self.multimodal_encoder is not None
        ):
            # Multimodal fusion
            fused = self.multimodal_encoder(
                outputs["image_features"], outputs["text_features"]
            )
            outputs["fused_features"] = fused

        return outputs


class Data2Vec(nn.Module):
    """Data2Vec: A General Framework for Self-supervised Learning.

    Baevski et al., 2022

    Unified self-supervised learning across modalities.

    Args:
        encoder: Modalities encoder
        embed_dim: Embedding dimension
        ema_decay: EMA decay rate
    """

    def __init__(
        self, encoder: nn.Module, embed_dim: int = 768, ema_decay: float = 0.999
    ):
        super().__init__()

        # Student (online)
        self.encoder = encoder

        # Teacher (target, EMA)
        self.teacher_encoder = copy.deepcopy(encoder)

        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

        self.ema_decay = ema_decay
        self.embed_dim = embed_dim

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher with EMA."""
        for param_s, param_t in zip(
            self.encoder.parameters(), self.teacher_encoder.parameters()
        ):
            param_t.data.mul_(self.ema_decay).add_(
                param_s.data, alpha=1 - self.ema_decay
            )

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input (image, text, speech, etc.)
            mask: Mask for masked prediction (optional)

        Returns:
            student_output, teacher_output
        """
        # Student prediction
        if mask is not None:
            # Apply masking
            x_masked = self.apply_mask(x, mask)
            student_output = self.encoder(x_masked)
        else:
            student_output = self.encoder(x)

        # Teacher prediction (no gradient)
        with torch.no_grad():
            self.update_teacher()
            teacher_output = self.teacher_encoder(x)

        return student_output, teacher_output

    def apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        """Apply mask to input."""
        # Implementation depends on modality
        return x * mask.unsqueeze(-1)

    def get_loss(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        """Compute smooth L1 loss."""
        return F.smooth_l1_loss(student_output, teacher_output)


class ImageBind(nn.Module):
    """ImageBind: One Embedding Space To Bind Them All.

    Girdhar et al., 2023

    Joint embedding space for multiple modalities.

    Args:
        encoders: Dict of modality encoders
        embed_dim: Joint embedding dimension
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        embed_dim: int = 1024,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.encoders = nn.ModuleDict(encoders)
        self.temperature = temperature

        # Projection layers for each modality
        self.projections = nn.ModuleDict()
        for modality, encoder in encoders.items():
            out_dim = self._get_encoder_dim(encoder)
            self.projections[modality] = nn.Sequential(
                nn.Linear(out_dim, embed_dim), nn.LayerNorm(embed_dim)
            )

        self.embed_dim = embed_dim

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 768

    def encode(self, x: Tensor, modality: str) -> Tensor:
        """Encode input from specific modality."""
        features = self.encoders[modality](x)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 3:
            features = features.mean(dim=1)

        return F.normalize(self.projections[modality](features), dim=-1)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            inputs: Dict mapping modality to input tensor

        Returns:
            Dict mapping modality to embeddings
        """
        embeddings = {}
        for modality, x in inputs.items():
            embeddings[modality] = self.encode(x, modality)

        return embeddings

    def get_loss(
        self, embeddings: Dict[str, Tensor], anchor_modality: str = "vision"
    ) -> Tensor:
        """Compute contrastive loss across modalities."""
        losses = []
        anchor = embeddings[anchor_modality]
        batch_size = anchor.shape[0]
        labels = torch.arange(batch_size, device=anchor.device)

        for modality, features in embeddings.items():
            if modality == anchor_modality:
                continue

            logits = anchor @ features.t() / self.temperature
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.t(), labels)
            losses.append((loss_i + loss_t) / 2)

        return sum(losses) / len(losses) if losses else torch.tensor(0.0)


# =============================================================================
# 5. Video Contrastive Methods
# =============================================================================


class VideoMoCo(nn.Module):
    """Video MoCo: Momentum Contrast for Video Representation Learning.

    Extension of MoCo to video.

    Args:
        encoder: Video encoder (3D CNN or transformer)
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        queue_size: Queue size
        momentum: Momentum coefficient
        temperature: Temperature
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        # Query encoder
        self.encoder_q = encoder
        self.projector_q = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        # Key encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.projector_k = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        self.queue = Queue(queue_size, projection_dim)

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 2048

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1 - self.momentum
            )

        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1 - self.momentum
            )

    def forward(
        self, clip_q: Tensor, clip_k: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            clip_q: Query clip (B, C, T, H, W)
            clip_k: Key clip (B, C, T, H, W)

        Returns:
            q, k, logits, labels
        """
        # Query
        f_q = self.encoder_q(clip_q)
        if isinstance(f_q, tuple):
            f_q = f_q[0]
        if f_q.dim() > 2:
            f_q = f_q.mean(dim=[2, 3, 4] if f_q.dim() == 5 else [2, 3])

        q = F.normalize(self.projector_q(f_q), dim=1)

        # Key
        with torch.no_grad():
            self._momentum_update()
            f_k = self.encoder_k(clip_k)
            if isinstance(f_k, tuple):
                f_k = f_k[0]
            if f_k.dim() > 2:
                f_k = f_k.mean(dim=[2, 3, 4] if f_k.dim() == 5 else [2, 3])
            k = F.normalize(self.projector_k(f_k), dim=1)

        # Compute logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.queue.clone().detach().t()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)

        self.queue.dequeue_and_enqueue(k)

        return q, k, logits, labels

    def get_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(logits, labels)


class CVRL(nn.Module):
    """Contrastive Video Representation Learning.

    Uses temporal contrastive learning for videos.

    Args:
        encoder: Video encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        temperature: Temperature
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 2048

    def forward(self, clip: Tensor) -> Tensor:
        """Encode video clip."""
        f = self.encoder(clip)
        if isinstance(f, tuple):
            f = f[0]
        if f.dim() > 2:
            f = f.mean(dim=[2, 3, 4] if f.dim() == 5 else [2, 3])

        z = self.projector(f)
        return F.normalize(z, dim=1)

    def get_loss(
        self, z1: Tensor, z2: Tensor, negatives: Optional[Tensor] = None
    ) -> Tensor:
        """Compute temporal contrastive loss."""
        if negatives is not None:
            l_pos = torch.sum(z1 * z2, dim=1, keepdim=True)
            l_neg = torch.mm(z1, negatives.t())
            logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
            labels = torch.zeros(z1.shape[0], dtype=torch.long, device=z1.device)
            return F.cross_entropy(logits, labels)
        else:
            # NT-Xent loss
            criterion = NTXentLoss(temperature=self.temperature)
            return criterion(z1, z2)


class CoCLR(nn.Module):
    """CoCLR: Cooperative Learning of Video Representations.

    Han et al., 2021

    Uses both RGB and optical flow for cooperative learning.

    Args:
        rgb_encoder: RGB video encoder
        flow_encoder: Optical flow encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        temperature: Temperature
    """

    def __init__(
        self,
        rgb_encoder: nn.Module,
        flow_encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        self.rgb_out_dim = self._get_encoder_dim(rgb_encoder)
        self.flow_out_dim = self._get_encoder_dim(flow_encoder)

        self.rgb_encoder = rgb_encoder
        self.flow_encoder = flow_encoder

        # Projection heads
        self.rgb_projector = ProjectionHead(
            self.rgb_out_dim, hidden_dim, projection_dim
        )
        self.flow_projector = ProjectionHead(
            self.flow_out_dim, hidden_dim, projection_dim
        )

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 2048

    def encode_rgb(self, clip: Tensor) -> Tensor:
        """Encode RGB clip."""
        f = self.rgb_encoder(clip)
        if isinstance(f, tuple):
            f = f[0]
        if f.dim() > 2:
            f = f.mean(dim=[2, 3, 4] if f.dim() == 5 else [2, 3])
        return F.normalize(self.rgb_projector(f), dim=1)

    def encode_flow(self, clip: Tensor) -> Tensor:
        """Encode flow clip."""
        f = self.flow_encoder(clip)
        if isinstance(f, tuple):
            f = f[0]
        if f.dim() > 2:
            f = f.mean(dim=[2, 3, 4] if f.dim() == 5 else [2, 3])
        return F.normalize(self.flow_projector(f), dim=1)

    def forward(self, rgb_clip: Tensor, flow_clip: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward both modalities."""
        rgb_z = self.encode_rgb(rgb_clip)
        flow_z = self.encode_flow(flow_clip)
        return rgb_z, flow_z

    def get_loss(self, rgb_z: Tensor, flow_z: Tensor) -> Tensor:
        """Compute cooperative contrastive loss."""
        # Within-modality contrast
        criterion = NTXentLoss(temperature=self.temperature)

        # RGB-RGB contrast (assuming we have two RGB views)
        # For simplicity, treat as cross-modal contrast
        l_rgb_flow = criterion(rgb_z, flow_z)

        return l_rgb_flow


class VINCE(nn.Module):
    """VINCE: Video Instance Contrastive Learning.

    Uses instance discrimination for video clips.

    Args:
        encoder: Video encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        queue_size: Queue size
        temperature: Temperature
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        queue_size: int = 65536,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        self.encoder_out_dim = self._get_encoder_dim(encoder)

        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        # Queue for negatives
        self.register_buffer("queue", torch.randn(queue_size, projection_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 2048

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: Tensor):
        """Update queue."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr : ptr + batch_size] = keys
        else:
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[: batch_size - remaining] = keys[remaining:]

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, clip: Tensor) -> Tensor:
        """Encode clip."""
        f = self.encoder(clip)
        if isinstance(f, tuple):
            f = f[0]
        if f.dim() > 2:
            f = f.mean(dim=[2, 3, 4] if f.dim() == 5 else [2, 3])

        z = self.projector(f)
        return F.normalize(z, dim=1)

    def get_loss(self, z_q: Tensor, z_k: Tensor) -> Tensor:
        """Compute instance contrastive loss."""
        l_pos = torch.einsum("nc,nc->n", [z_q, z_k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [z_q, self.queue.clone().detach().t()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z_q.device)

        self.dequeue_and_enqueue(z_k)

        return F.cross_entropy(logits, labels)


class Pace(nn.Module):
    """Pace: Pace Prediction for Video Representation Learning.

    Wang et al., 2020

    Predicts playback speed/pace as pretext task.

    Args:
        encoder: Video encoder
        projection_dim: Projection dimension
        hidden_dim: Hidden dimension
        n_paces: Number of pace classes
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        n_paces: int = 4,
    ):
        super().__init__()

        self.encoder_out_dim = self._get_encoder_dim(encoder)
        self.n_paces = n_paces

        self.encoder = encoder
        self.projector = ProjectionHead(
            self.encoder_out_dim, hidden_dim, projection_dim
        )

        # Pace classifier
        self.pace_classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_paces),
        )

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        elif hasattr(encoder, "num_features"):
            return encoder.num_features
        else:
            return 2048

    def forward(self, clip: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            clip: Video clip

        Returns:
            embedding, pace_logits
        """
        f = self.encoder(clip)
        if isinstance(f, tuple):
            f = f[0]
        if f.dim() > 2:
            f = f.mean(dim=[2, 3, 4] if f.dim() == 5 else [2, 3])

        z = self.projector(f)
        pace_logits = self.pace_classifier(z)

        return z, pace_logits

    def get_loss(self, pace_logits: Tensor, pace_labels: Tensor) -> Tensor:
        """Compute pace prediction loss."""
        return F.cross_entropy(pace_logits, pace_labels)


# =============================================================================
# 8. Evaluation Methods
# =============================================================================


class LinearEvaluation(nn.Module):
    """Linear evaluation protocol.

    Train a linear classifier on frozen representations.

    Args:
        encoder: Pre-trained encoder
        n_classes: Number of classes
        feature_dim: Feature dimension
    """

    def __init__(
        self, encoder: nn.Module, n_classes: int, feature_dim: Optional[int] = None
    ):
        super().__init__()
        self.encoder = encoder

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Infer feature dimension if not provided
        if feature_dim is None:
            feature_dim = self._infer_dim(encoder)

        self.classifier = nn.Linear(feature_dim, n_classes)

    def _infer_dim(self, encoder: nn.Module) -> int:
        """Infer encoder output dimension."""
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            out = encoder(dummy)
            if isinstance(out, tuple):
                out = out[0]
            return out.flatten(1).shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        with torch.no_grad():
            features = self.encoder(x)
            if isinstance(features, tuple):
                features = features[0]
            features = features.flatten(1)

        logits = self.classifier(features)
        return logits

    def get_accuracy(
        self, dataloader: torch.utils.data.DataLoader, device: str = "cuda"
    ) -> float:
        """Evaluate accuracy."""
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.forward(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100.0 * correct / total


class KNNEvaluation:
    """k-NN evaluation for representations.

    Uses k-nearest neighbors classifier on frozen features.

    Args:
        encoder: Pre-trained encoder
        k: Number of neighbors
        device: Device to use
    """

    def __init__(self, encoder: nn.Module, k: int = 20, device: str = "cuda"):
        self.encoder = encoder
        self.k = k
        self.device = device
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def extract_features(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[Tensor, Tensor]:
        """Extract features from dataloader."""
        features = []
        labels = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)

                feats = self.encoder(images)
                if isinstance(feats, tuple):
                    feats = feats[0]
                feats = feats.flatten(1)
                feats = F.normalize(feats, dim=1)

                features.append(feats.cpu())
                labels.append(targets)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels

    def evaluate(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> float:
        """Evaluate using k-NN."""
        print("Extracting training features...")
        train_features, train_labels = self.extract_features(train_loader)

        print("Extracting test features...")
        test_features, test_labels = self.extract_features(test_loader)

        # k-NN classification
        correct = 0
        total = test_features.shape[0]

        for i in range(test_features.shape[0]):
            # Compute distances
            distances = torch.sum(
                (train_features - test_features[i].unsqueeze(0)) ** 2, dim=1
            )

            # Get k nearest neighbors
            _, indices = distances.topk(self.k, largest=False)

            # Vote
            neighbor_labels = train_labels[indices]
            pred = torch.bincount(neighbor_labels).argmax()

            if pred == test_labels[i]:
                correct += 1

        accuracy = 100.0 * correct / total
        print(f"k-NN (k={self.k}) Accuracy: {accuracy:.2f}%")

        return accuracy


class FineTuning(nn.Module):
    """Fine-tuning protocol.

    Fine-tune entire pre-trained model on downstream task.

    Args:
        encoder: Pre-trained encoder
        n_classes: Number of classes
        feature_dim: Feature dimension
        freeze_bn: Whether to freeze batch norm
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        feature_dim: Optional[int] = None,
        freeze_bn: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_bn = freeze_bn

        # Infer feature dimension
        if feature_dim is None:
            feature_dim = self._infer_dim(encoder)

        self.classifier = nn.Linear(feature_dim, n_classes)

    def _infer_dim(self, encoder: nn.Module) -> int:
        """Infer encoder output dimension."""
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            out = encoder(dummy)
            if isinstance(out, tuple):
                out = out[0]
            return out.flatten(1).shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        features = self.encoder(x)
        if isinstance(features, tuple):
            features = features[0]
        features = features.flatten(1)

        logits = self.classifier(features)
        return logits

    def train(self, mode: bool = True):
        """Override train to optionally freeze BN."""
        super().train(mode)

        if self.freeze_bn and mode:
            for module in self.encoder.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

        return self


class TransferLearning(nn.Module):
    """Transfer learning to various downstream tasks.

    Args:
        encoder: Pre-trained encoder
        task_type: Type of task ('classification', 'detection', 'segmentation')
        num_classes: Number of classes
    """

    def __init__(
        self,
        encoder: nn.Module,
        task_type: str = "classification",
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        self.encoder = encoder

        if task_type == "classification":
            self.head = self._build_classification_head(num_classes, **kwargs)
        elif task_type == "detection":
            self.head = self._build_detection_head(num_classes, **kwargs)
        elif task_type == "segmentation":
            self.head = self._build_segmentation_head(num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _build_classification_head(
        self, num_classes: int, feature_dim: Optional[int] = None
    ) -> nn.Module:
        """Build classification head."""
        if feature_dim is None:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                out = self.encoder(dummy)
                if isinstance(out, tuple):
                    out = out[0]
                feature_dim = out.flatten(1).shape[-1]

        return nn.Linear(feature_dim, num_classes)

    def _build_detection_head(self, num_classes: int, **kwargs) -> nn.Module:
        """Build detection head (simplified)."""
        # This would typically use a detection framework like Faster R-CNN
        # Simplified version here
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)
        )

    def _build_segmentation_head(self, num_classes: int, **kwargs) -> nn.Module:
        """Build segmentation head."""
        return nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        features = self.encoder(x)

        if self.task_type == "classification":
            if isinstance(features, tuple):
                features = features[0]
            features = features.flatten(1)

        return self.head(features)

    def get_features(self, x: Tensor) -> Tensor:
        """Get intermediate features."""
        return self.encoder(x)


# =============================================================================
# Utilities
# =============================================================================


def off_diagonal(x: Tensor) -> Tensor:
    """Return off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_encoder_output_dim(encoder: nn.Module) -> int:
    """Infer encoder output dimension."""
    if hasattr(encoder, "output_dim"):
        return encoder.output_dim
    elif hasattr(encoder, "embed_dim"):
        return encoder.embed_dim
    elif hasattr(encoder, "num_features"):
        return encoder.num_features
    elif hasattr(encoder, "fc"):
        return encoder.fc.in_features
    else:
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            out = encoder(dummy)
            if isinstance(out, tuple):
                out = out[0]
            return out.flatten(1).shape[-1]


def gather_distributed(x: Tensor) -> Tensor:
    """Gather tensors from all GPUs."""
    if not torch.distributed.is_initialized():
        return x

    world_size = get_world_size()
    if world_size == 1:
        return x

    x_list = [torch.zeros_like(x) for _ in range(world_size)]
    all_gather(x_list, x)
    return torch.cat(x_list, dim=0)


# =============================================================================
# Factory function
# =============================================================================


def create_contrastive_model(method: str, encoder: nn.Module, **kwargs) -> nn.Module:
    """Factory function to create contrastive learning models.

    Args:
        method: Method name
        encoder: Backbone encoder
        **kwargs: Additional arguments for the method

    Returns:
        Contrastive learning model
    """
    method = method.lower()

    # Instance discrimination methods
    if method == "simclr":
        return SimCLR(encoder, **kwargs)
    elif method in ["moco", "moco_v1"]:
        return MoCo(encoder, **kwargs)
    elif method in ["moco_v2", "mocov2"]:
        return MoCov2(encoder, **kwargs)
    elif method in ["moco_v3", "mocov3"]:
        return MoCov3(encoder, **kwargs)
    elif method == "simsiam":
        return SimSiam(encoder, **kwargs)
    elif method == "byol":
        return BYOL(encoder, **kwargs)
    elif method == "swav":
        return SwAV(encoder, **kwargs)
    elif method == "nnclr":
        return NNCLR(encoder, **kwargs)
    elif method == "barlowtwins":
        return BarlowTwins(encoder, **kwargs)

    # Clustering-based methods
    elif method == "deepcluster":
        return DeepCluster(encoder, **kwargs)
    elif method == "sela":
        return SeLa(encoder, **kwargs)
    elif method == "pcl":
        return PCL(encoder, **kwargs)
    elif method == "scan":
        return SCAN(encoder, **kwargs)
    elif method == "spice":
        return SPICE(encoder, **kwargs)

    # Video methods
    elif method == "videomoco":
        return VideoMoCo(encoder, **kwargs)
    elif method == "cvrl":
        return CVRL(encoder, **kwargs)
    elif method == "coclr":
        raise ValueError("CoCLR requires both RGB and flow encoders")
    elif method == "vince":
        return VINCE(encoder, **kwargs)
    elif method == "pace":
        return Pace(encoder, **kwargs)

    else:
        raise ValueError(f"Unknown method: {method}")


__all__ = [
    # Architectures
    "ProjectionHead",
    "PredictionHead",
    "PrototypeLayer",
    "Queue",
    "EMAUpdater",
    # Losses
    "NTXentLoss",
    "InfoNCELoss",
    "DecoupledCLLoss",
    "VICRegLoss",
    # Instance discrimination
    "SimCLR",
    "MoCo",
    "MoCov2",
    "MoCov3",
    "SimSiam",
    "BYOL",
    "SwAV",
    "NNCLR",
    "BarlowTwins",
    # Clustering-based
    "DeepCluster",
    "SeLa",
    "PCL",
    "SCAN",
    "SPICE",
    # Hard negative mining
    "HardNegatives",
    "MixingNegatives",
    "DebiasCL",
    "HCL",
    "SupervisedCL",
    # Multimodal
    "CLIP",
    "ALIGN",
    "Florence",
    "Data2Vec",
    "ImageBind",
    # Video
    "VideoMoCo",
    "CVRL",
    "CoCLR",
    "VINCE",
    "Pace",
    # Evaluation
    "LinearEvaluation",
    "KNNEvaluation",
    "FineTuning",
    "TransferLearning",
    # Utilities
    "create_contrastive_model",
    "get_encoder_output_dim",
    "gather_distributed",
]
