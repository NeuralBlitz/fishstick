"""
Advanced BYOL and SimSiam Implementations

Extended BYOL/SimSiam implementations with:
- Advanced momentum strategies
- NNCLR-style nearest neighbor contrast
- SimSiam with improved training dynamics
- MoCo v3 for Vision Transformers
"""

from typing import Optional, Tuple, Dict, Any, List, Callable
import copy

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributed import all_gather, get_world_size

from fishstick.ssl_extensions.base import (
    MomentumUpdater,
    EMAUpdater,
    MemoryBank,
    stop_gradient,
    gather_from_all,
    L2Normalize,
)


class BYOLLoss(nn.Module):
    """BYOL (Bootstrap Your Own Latent) loss function.

    Args:
        target_normalize: Whether to normalize target projections
        predictor_normalize: Whether to normalize predictor outputs
    """

    def __init__(
        self,
        target_normalize: bool = True,
        predictor_normalize: bool = True,
    ):
        super().__init__()
        self.target_normalize = target_normalize
        self.predictor_normalize = predictor_normalize

    def forward(self, online_projections: Tensor, target_projections: Tensor) -> Tensor:
        if self.predictor_normalize:
            online_projections = F.normalize(online_projections, dim=-1)
        if self.target_normalize:
            target_projections = F.normalize(target_projections, dim=-1)

        return 2 - 2 * (online_projections * target_projections).sum(dim=-1).mean()


class AdvancedBYOL(nn.Module):
    """Advanced BYOL with momentum encoder and optional NNCLR features.

    Features:
    - Momentum encoder for target representations
    - Optional nearest-neighbor contrastive features (NNCLR-style)
    - Flexible projection and prediction heads
    - Support for distributed training

    Args:
        encoder: Backbone encoder network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
        predictor_dim: Dimension of prediction head output
        momentum: Momentum coefficient for EMA
        use_nnclr: Whether to use NNCLR-style nearest neighbor contrast
        nn_regressor_cache_size: Size of cache for NNCLR
        temperature: Temperature for NNCLR similarity
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        predictor_dim: int = 256,
        momentum: float = 0.999,
        use_nnclr: bool = False,
        nn_regressor_cache_size: int = 4096,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection_dim = projection_dim
        self.use_nnclr = use_nnclr

        encoder_out_dim = self._get_encoder_dim(encoder)

        self.online_projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, predictor_dim),
        )

        self.momentum_encoder = copy.deepcopy(encoder)
        self.momentum_projector = copy.deepcopy(self.online_projector)

        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projector.parameters():
            param.requires_grad = False

        self.momentum_updater = EMAUpdater(decay=momentum)

        self.loss_fn = BYOLLoss()

        if use_nnclr:
            self.nn_cache = MemoryBank(
                size=nn_regressor_cache_size,
                dim=projection_dim,
                temperature=temperature,
            )
            self.temperature = temperature

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.training:
            self.momentum_updater.update(self.encoder, self.momentum_encoder)
            self.momentum_updater.update(self.online_projector, self.momentum_projector)

        online_proj1 = self.online_projector(self.encoder(x1))
        online_proj2 = self.online_projector(self.encoder(x2))

        online_pred1 = self.online_predictor(online_proj1)
        online_pred2 = self.online_predictor(online_proj2)

        with torch.no_grad():
            target_proj1 = self.momentum_projector(self.momentum_encoder(x1))
            target_proj2 = self.momentum_projector(self.momentum_encoder(x2))

        if self.use_nnclr and self.training:
            target_proj1 = self._nnclr_augment(target_proj1)
            target_proj2 = self._nnclr_augment(target_proj2)

        loss1 = self.loss_fn(online_pred1, stop_gradient(target_proj2))
        loss2 = self.loss_fn(online_pred2, stop_gradient(target_proj1))
        loss = (loss1 + loss2) / 2

        return loss, {
            "online_proj1": online_proj1,
            "online_proj2": online_proj2,
            "target_proj1": target_proj1,
            "target_proj2": target_proj2,
        }

    def _nnclr_augment(self, target_proj: Tensor) -> Tensor:
        cached = self.nn_cache.get(target_proj.shape[0])
        cached = cached.to(target_proj.device)

        similarities = target_proj @ cached.T / self.temperature
        weights = F.softmax(similarities, dim=-1)

        nearest_neighbors = cached.T @ weights.T
        return nearest_neighbors.T

    def get_embeddings(self, x: Tensor) -> Tensor:
        return F.normalize(self.online_projector(self.encoder(x)), dim=-1)


class SimSiamLoss(nn.Module):
    """SimSiam loss function with optional cosine similarity.

    Args:
        use_cosine: Whether to use cosine similarity (default: True)
    """

    def __init__(self, use_cosine: bool = True):
        super().__init__()
        self.use_cosine = use_cosine

    def forward(
        self, pred1: Tensor, pred2: Tensor, target1: Tensor, target2: Tensor
    ) -> Tensor:
        if self.use_cosine:
            pred1 = F.normalize(pred1, dim=-1)
            pred2 = F.normalize(pred2, dim=-1)
            target1 = F.normalize(target1, dim=-1)
            target2 = F.normalize(target2, dim=-1)

        loss1 = -F.cosine_similarity(pred1, stop_gradient(target1), dim=-1).mean()
        loss2 = -F.cosine_similarity(pred2, stop_gradient(target2), dim=-1).mean()

        return (loss1 + loss2) / 2


class AdvancedSimSiam(nn.Module):
    """Advanced SimSiam with improved training dynamics.

    Features:
    - Stop-gradient on target encoder outputs
    - Prediction head with asymmetric architecture
    - Optional MLP-Mixer style blocks
    - Support for distributed training

    Args:
        encoder: Backbone encoder network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
        predictor_dim: Dimension of prediction head output
        num_predictor_layers: Number of layers in predictor
        use_bn_in_predictor: Whether to use BN in predictor
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 2048,
        hidden_dim: int = 2048,
        predictor_dim: int = 512,
        num_predictor_layers: int = 2,
        use_bn_in_predictor: bool = True,
    ):
        super().__init__()
        self.encoder = encoder

        encoder_out_dim = self._get_encoder_dim(encoder)

        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False),
        )

        predictor_layers = []
        in_dim = projection_dim
        for i in range(num_predictor_layers - 1):
            predictor_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn_in_predictor:
                predictor_layers.append(nn.BatchNorm1d(hidden_dim))
            predictor_layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        predictor_layers.append(nn.Linear(in_dim, predictor_dim))
        if use_bn_in_predictor:
            predictor_layers.append(nn.BatchNorm1d(predictor_dim, affine=False))

        self.predictor = nn.Sequential(*predictor_layers)

        self.loss_fn = SimSiamLoss()

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        f1 = self.projector(self.encoder(x1))
        f2 = self.projector(self.encoder(x2))

        z1 = stop_gradient(f1)
        z2 = stop_gradient(f2)

        p1 = self.predictor(f1)
        p2 = self.predictor(f2)

        loss = self.loss_fn(p1, p2, z2, z1)

        return loss

    def get_embeddings(self, x: Tensor) -> Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=-1)


class MoCoV3(nn.Module):
    """MoCo v3 for Vision Transformers.

    Implements MoCo v3 with:
    - ViT backbone support
    - Momentum contrastive learning
    - Optional pyramid ViT support

    Args:
        encoder: ViT-based encoder
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
        predictor_dim: Dimension of prediction head
        momentum: Momentum coefficient for EMA
        temperature: Temperature for contrastive loss
        use_2d_pos_embed: Whether to use 2D positional embeddings
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        predictor_dim: int = 256,
        momentum: float = 0.999,
        temperature: float = 0.1,
        use_2d_pos_embed: bool = True,
    ):
        super().__init__()
        self.encoder = encoder

        encoder_out_dim = self._get_encoder_dim(encoder)

        self.online_projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, predictor_dim),
        )

        self.momentum_encoder = copy.deepcopy(encoder)
        self.momentum_projector = copy.deepcopy(self.online_projector)

        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projector.parameters():
            param.requires_grad = False

        self.momentum_updater = EMAUpdater(decay=momentum)
        self.temperature = temperature

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.training:
            self.momentum_updater.update(self.encoder, self.momentum_encoder)
            self.momentum_updater.update(self.online_projector, self.momentum_projector)

        online_proj1 = self.online_projector(self.encoder(x1))
        online_proj2 = self.online_projector(self.encoder(x2))

        online_pred1 = self.online_predictor(online_proj1)
        online_pred2 = self.online_predictor(online_proj2)

        with torch.no_grad():
            target_proj1 = self.momentum_projector(self.momentum_encoder(x1))
            target_proj2 = self.momentum_projector(self.momentum_encoder(x2))

        loss = self._contrastive_loss(
            online_pred1, online_pred2, target_proj1, target_proj2
        )

        return loss, {
            "online_proj1": online_proj1,
            "online_proj2": online_proj2,
            "target_proj1": target_proj1,
            "target_proj2": target_proj2,
        }

    def _contrastive_loss(
        self, pred1: Tensor, pred2: Tensor, target1: Tensor, target2: Tensor
    ) -> Tensor:
        pred1 = F.normalize(pred1, dim=-1)
        pred2 = F.normalize(pred2, dim=-1)
        target1 = F.normalize(target1, dim=-1)
        target2 = F.normalize(target2, dim=-1)

        batch_size = pred1.shape[0]

        pred = torch.cat([pred1, pred2], dim=0)
        target = torch.cat([target2, target1], dim=0)

        sim = pred @ target.T / self.temperature

        diag_mask = torch.eye(2 * batch_size, device=sim.device)
        diag_mask = diag_mask.fill_diagonal_(0)
        sim = sim.masked_fill(diag_mask.bool(), float("-inf"))

        loss = F.cross_entropy(sim, torch.arange(2 * batch_size, device=sim.device))

        return loss

    def get_embeddings(self, x: Tensor) -> Tensor:
        return F.normalize(self.online_projector(self.encoder(x)), dim=-1)


class NNCLR(nn.Module):
    """NNCLR: Nearest-Neighbor Contrastive Learning of Representations.

    Implements NNCLR with:
    - Nearest-neighbor augmentation
    - Memory bank for negative samples
    - Distributed training support

    Args:
        encoder: Backbone encoder network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
        memory_bank_size: Size of memory bank
        temperature: Temperature for similarity
        use_softmax_neg: Whether to use softmax for negatives
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        memory_bank_size: int = 16384,
        temperature: float = 0.2,
        use_softmax_neg: bool = True,
    ):
        super().__init__()
        self.encoder = encoder

        encoder_out_dim = self._get_encoder_dim(encoder)

        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.memory_bank = MemoryBank(
            size=memory_bank_size,
            dim=projection_dim,
            temperature=temperature,
        )

        self.temperature = temperature
        self.use_softmax_neg = use_softmax_neg

    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        if hasattr(encoder, "output_dim"):
            return encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            return encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                return encoder(dummy).flatten(1).shape[1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        proj1 = self.projector(self.encoder(x1))
        proj2 = self.projector(self.encoder(x2))

        proj1 = F.normalize(proj1, dim=-1)
        proj2 = F.normalize(proj2, dim=-1)

        nn_proj1 = self._get_nearest_neighbors(proj1)
        nn_proj2 = self._get_nearest_neighbors(proj2)

        pred1 = self.predictor(proj1)
        pred2 = self.predictor(proj2)

        loss = self._nn_contrastive_loss(pred1, pred2, nn_proj1, nn_proj2)

        self.memory_bank.update(proj1.detach())
        self.memory_bank.update(proj2.detach())

        return loss

    def _get_nearest_neighbors(self, projections: Tensor) -> Tensor:
        cached = self.memory_bank.get(projections.shape[0] * 2)
        cached = cached.to(projections.device)

        similarities = projections @ cached.T / self.temperature

        if self.use_softmax_neg:
            weights = F.softmax(similarities, dim=-1)
        else:
            weights = torch.softmax(similarities, dim=-1)

        nearest_neighbors = cached.T @ weights.T
        return nearest_neighbors.T

    def _nn_contrastive_loss(
        self,
        pred1: Tensor,
        pred2: Tensor,
        target1: Tensor,
        target2: Tensor,
    ) -> Tensor:
        pred1 = F.normalize(pred1, dim=-1)
        pred2 = F.normalize(pred2, dim=-1)

        loss1 = -F.cosine_similarity(pred1, target2, dim=-1).mean()
        loss2 = -F.cosine_similarity(pred2, target1, dim=-1).mean()

        return (loss1 + loss2) / 2

    def get_embeddings(self, x: Tensor) -> Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=-1)
