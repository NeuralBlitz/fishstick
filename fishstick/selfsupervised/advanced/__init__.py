"""
Additional Self-Supervised Learning Methods

DINO, SwAV, W-MSE implementations.
"""

from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class DINO(nn.Module):
    """DINO: Self-supervised Pretraining of Vision Transformers

    Knowledge distillation framework for self-supervised learning.

    Args:
        encoder: Vision Transformer encoder
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        freeze_last_layer: Number of layers to freeze (0 = none)
        student_temp: Temperature for student
        teacher_temp: Temperature for teacher
        warmup_teacher_temp: Initial teacher temperature
        teacher_temp_warmup_epochs: Warmup epochs for teacher temp
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        freeze_last_layer: int = -1,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        warmup_teacher_temp: float = 0.04,
        teacher_temp_warmup_epochs: int = 30,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_last_layer = freeze_last_layer
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # Teacher isEMA of student
        self.teacher = self._copy_encoder(encoder)
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.teacher_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        teacher = nn.Module()
        teacher.load_state_dict(encoder.state_dict())
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    @torch.no_grad()
    def _update_teacher(self, m: float = 0.999):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.teacher.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1 - m)

    def forward_student(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        z = self.projection_head(h)
        z = F.normalize(z, dim=-1)
        return z

    def forward_teacher(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            h = self.teacher(x)
            z = self.teacher_projection(h)
            z = F.normalize(z, dim=-1)
        return z

    def forward(
        self, x1: Tensor, x2: Tensor, m: float = 0.999
    ) -> Tuple[Tensor, Tensor]:
        self._update_teacher(m)

        z1 = self.forward_student(x1)
        z2 = self.forward_student(x2)

        with torch.no_grad():
            t1 = self.forward_teacher(x1)
            t2 = self.forward_teacher(x2)

        loss = self._dino_loss(z1, t2) + self._dino_loss(z2, t1)
        return loss / 2, z1

    def _dino_loss(self, z_student: Tensor, z_teacher: Tensor) -> Tensor:
        z_student = z_student / self.student_temp
        z_teacher = z_teacher / self.teacher_temp

        loss = -(z_teacher * z_student.logsumexp(dim=-1)).sum(-1).mean()
        return loss


class SwAV(nn.Module):
    """SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments

    Uses swapped prediction between views.

    Args:
        encoder: Backbone encoder
        embed_dim: Embedding dimension
        num_prototypes: Number of prototypes
        queue_size: Size of feature queue
        temperature: Temperature for softmax
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 256,
        num_prototypes: int = 3000,
        queue_size: int = 4096,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.prototypes = nn.Linear(embed_dim, num_prototypes, bias=False)

        self.register_buffer("queue", torch.randn(queue_size, embed_dim))
        self.queue = F.normalize(self.queue, dim=-1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        with torch.no_grad():
            p1 = self.prototypes(z1)
            p2 = self.prototypes(z2)

            q1 = self._sinkhorn(p1)
            q2 = self._sinkhorn(p2)

        loss = self._swav_loss(z1, z2, q2) + self._swav_loss(z2, z1, q1)
        return loss / 2, z1

    def _sinkhorn(self, z: Tensor, num_iters: int = 3) -> Tensor:
        z = z / 0.01

        Q = torch.exp(z.T / z.size(1))
        Q = Q / Q.sum(dim=0, keepdim=True)

        for _ in range(num_iters):
            Q = Q / Q.sum(dim=1, keepdim=True)
            Q = Q * z.size(1) / Q.sum(dim=0, keepdim=True)

        return Q.T

    def _swav_loss(self, z: Tensor, q: Tensor) -> Tensor:
        p = F.softmax(z / self.temperature, dim=-1)
        loss = -(p * q.log()).sum(-1).mean()
        return loss


class WMSE(nn.Module):
    """W-MSE: Whitened MSE for Self-Supervised Learning

    Uses whitening transformation for better representations.

    Args:
        encoder: Backbone encoder
        embed_dim: Embedding dimension
        momentum: Momentum for EMA updates
        warmup_steps: Warmup steps for normalization
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 256,
        momentum: float = 0.999,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.teacher = self._copy_encoder(encoder)
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(embed_dim))
        self.register_buffer("running_var", torch.ones(embed_dim))
        self.step = 0
        self.warmup_steps = warmup_steps

    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        teacher = nn.Module()
        teacher.load_state_dict(encoder.state_dict())
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    @torch.no_grad()
    def _update_teacher(self):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.teacher.parameters()
        ):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    def _whiten(self, z: Tensor) -> Tensor:
        if self.step < self.warmup_steps:
            return z

        mean = z.mean(dim=0, keepdim=True)
        var = z.var(dim=0, keepdim=True, unbiased=False)

        self.running_mean = 0.99 * self.running_mean + 0.01 * mean.detach()
        self.running_var = 0.99 * self.running_var + 0.01 * var.detach()

        z = (z - self.running_mean) / (self.running_var + 1e-5).sqrt()
        z = F.normalize(z, dim=-1)

        return z

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        self.step += 1
        self._update_teacher()

        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))

        z1 = self._whiten(z1)
        z2 = self._whiten(z2)

        loss = F.mse_loss(z1, z2)
        return loss


class MSN(nn.Module):
    """MSN: Masked Siamese Networks

    Self-supervised learning via masked image modeling and siamese matching.

    Args:
        encoder: Vision Transformer encoder
        embed_dim: Embedding dimension
        num_prototypes: Number of prototypes
        mask_ratio: Ratio of patches to mask
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 768,
        num_prototypes: int = 256,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.prototypes = nn.Linear(embed_dim, num_prototypes, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, N, C = x.shape
        num_masked = int(N * self.mask_ratio)

        rand_idx = torch.rand(B, N, device=x.device).argsort(dim=1)[:, :num_masked]
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, rand_idx, True)

        x_masked = x.clone()
        x_masked[mask] = 0

        z = self.projection(self.encoder(x_masked))
        p = self.prototypes(z)

        with torch.no_grad():
            z_anchor = self.projection(self.encoder(x))
            q = self.prototypes(z_anchor)
            q = q.detach()

        loss = -(
            p * q.logsumexp(dim=-1, keepdim=True) - (p * q).sum(-1, keepdim=True)
        ).mean()
        return loss, z
