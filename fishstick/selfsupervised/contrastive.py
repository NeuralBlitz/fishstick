"""
Contrastive Learning Implementations

Self-supervised contrastive learning algorithms:
- SimCLR: Simple Contrastive Learning of Representations
- BYOL: Bootstrap Your Own Latent
- SimSiam: Simple Siamese Representation Learning
- MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
"""

from typing import Optional, Tuple, Dict, Any
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributed import all_gather


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        use_bn: bool = True,
        bias_last: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(dims[-2], dims[-1], bias=bias_last))
        self.projection = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class ContrastiveHead(nn.Module):
    """Flexible contrastive head supporting multiple algorithms."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 256,
        num_layers: int = 3,
        algorithm: str = "simclr",
    ):
        super().__init__()
        self.algorithm = algorithm

        if algorithm in ["simclr", "simsiam"]:
            self.projector = ProjectionHead(
                input_dim, hidden_dim, output_dim, num_layers
            )
            if algorithm == "simsiam":
                self.predictor = ProjectionHead(
                    output_dim, hidden_dim, output_dim, num_layers, bias_last=False
                )
        elif algorithm == "byol":
            self.projector = ProjectionHead(
                input_dim, hidden_dim, output_dim, num_layers, use_bn=False
            )
            self.predictor = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        elif algorithm == "moco":
            self.projector = ProjectionHead(
                input_dim, hidden_dim, output_dim, num_layers
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def forward(self, x: Tensor, return_projection: bool = True) -> Tensor:
        if self.algorithm == "simsiam":
            z = self.projector(x)
            p = self.predictor(z)
            return (z, p) if return_projection else p
        elif self.algorithm == "byol":
            z = self.projector(x)
            p = self.predictor(z)
            return z
        return self.projector(x)


class SimCLR(nn.Module):
    """SimCLR: Simple Contrastive Learning of Representations.

    Args:
        encoder: Backbone neural network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers in projection head
        temperature: Temperature for softmax normalization
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        temperature: float = 0.07,
        encoder_output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        # Get encoder output dimension - try to infer or use provided value
        if encoder_output_dim is not None:
            encoder_out_dim = encoder_output_dim
        elif hasattr(encoder, "output_dim"):
            encoder_out_dim = encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            encoder_out_dim = encoder.embed_dim
        else:
            # Try a forward pass to get the dimension
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                encoder_out_dim = encoder(dummy).flatten(1).shape[1]

        self.projection_head = ProjectionHead(
            encoder_out_dim,
            hidden_dim,
            projection_dim,
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        return h1, h2, z1, z2

    def get_embeddings(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        z = self.projection_head(h)
        return F.normalize(z, dim=-1)


class BYOL(nn.Module):
    """BYOL: Bootstrap Your Own Latent.

    A self-supervised method that learns representations without negative samples
    by using an online network and a target network with momentum update.

    Args:
        encoder: Backbone neural network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
        momentum: Momentum for updating target network (default: 0.996)
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        momentum: float = 0.996,
        encoder_output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.momentum = momentum

        # Get encoder output dimension
        if encoder_output_dim is not None:
            encoder_out_dim = encoder_output_dim
        elif hasattr(encoder, "output_dim"):
            encoder_out_dim = encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            encoder_out_dim = encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                encoder_out_dim = encoder(dummy).flatten(1).shape[1]

        self.online_encoder = encoder
        self.online_projector = ProjectionHead(
            encoder_out_dim, hidden_dim, projection_dim, use_bn=False
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.target_encoder = self._copy_encoder(encoder)
        self.target_projector = self._copy_projector(self.online_projector)

        self._freeze_target()

    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        target = copy.deepcopy(encoder)
        for param in target.parameters():
            param.requires_grad = False
        return target

    def _copy_projector(self, projector: ProjectionHead) -> nn.Module:
        target = copy.deepcopy(projector)
        for param in target.parameters():
            param.requires_grad = False
        return target

    def _freeze_target(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.mul_(self.momentum).add_(
                online_param.data, alpha=1 - self.momentum
            )

        for online_param, target_param in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_param.data.mul_(self.momentum).add_(
                online_param.data, alpha=1 - self.momentum
            )

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        self._momentum_update()

        p1 = self.online_predictor(self.online_projector(self.online_encoder(x1)))
        p2 = self.online_predictor(self.online_projector(self.online_encoder(x2)))

        with torch.no_grad():
            z1 = self.target_projector(self.target_encoder(x1))
            z2 = self.target_projector(self.target_encoder(x2))

        return p1, p2, z1, z2

    def get_embeddings(self, x: Tensor) -> Tensor:
        z = self.online_projector(self.online_encoder(x))
        return F.normalize(z, dim=-1)


class SimSiam(nn.Module):
    """SimSiam: Simple Siamese Representation Learning.

    Explores siamese networks without negative samples, using a predictor
    and stop-gradient operation to prevent representation collapse.

    Args:
        encoder: Backbone neural network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 2048,
        hidden_dim: int = 2048,
    ):
        super().__init__()

        encoder_out_dim = (
            encoder.output_dim if hasattr(encoder, "output_dim") else encoder.embed_dim
        )

        self.encoder = encoder
        self.projector = ProjectionHead(encoder_out_dim, hidden_dim, projection_dim)
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return z1, z2, p1, p2

    def get_embeddings(self, x: Tensor) -> Tensor:
        z = self.projector(self.encoder(x))
        return F.normalize(z, dim=-1)


class MoCoQueue(nn.Module):
    """Momentum Contrast queue for maintaining negative samples."""

    def __init__(self, queue_size: int, dim: int):
        super().__init__()
        self.queue_size = queue_size
        self.dim = dim
        self.register_buffer("queue", torch.randn(queue_size, dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, k: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = k.shape[0]

        k = k.detach()

        ptr = int(self.queue_ptr)
        self.queue[ptr : ptr + batch_size] = k

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

        queue = self.queue.clone().t().contiguous()
        return k, queue

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        self.queue[ptr : ptr + batch_size] = keys

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


class MoCo(nn.Module):
    """MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.

    Uses a queue of negative samples and momentum-updated encoder for
    unsupervised contrastive learning.

    Args:
        encoder: Backbone neural network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
        queue_size: Size of the negative sample queue
        momentum: Momentum for updating key encoder
        temperature: Temperature for softmax normalization
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
        self.temperature = temperature

        encoder_out_dim = (
            encoder.output_dim if hasattr(encoder, "output_dim") else encoder.embed_dim
        )

        self.query_encoder = nn.ModuleDict(
            {
                "encoder": encoder,
                "projector": ProjectionHead(
                    encoder_out_dim, hidden_dim, projection_dim
                ),
            }
        )

        self.key_encoder = nn.ModuleDict(
            {
                "encoder": self._copy_encoder(encoder),
                "projector": self._copy_projector(
                    encoder_out_dim, hidden_dim, projection_dim
                ),
            }
        )

        for param in self.key_encoder.parameters():
            param.requires_grad = False

        self.queue = MoCoQueue(queue_size, projection_dim)

    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        new_encoder = nn.Module()
        if hasattr(encoder, "state_dict"):
            new_encoder.load_state_dict(encoder.state_dict())
        return new_encoder

    def _copy_projector(self, input_dim: int, hidden_dim: int, output_dim: int):
        return ProjectionHead(input_dim, hidden_dim, output_dim)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for q_param, k_param in zip(
            self.query_encoder.parameters(), self.key_encoder.parameters()
        ):
            k_param.data.mul_(0.999).add_(q_param.data, alpha=1 - 0.999)

    def forward(self, x_q: Tensor, x_k: Tensor) -> Dict[str, Tensor]:
        q = self.query_encoder.projector(self.query_encoder.encoder(x_q))

        with torch.no_grad():
            k = self.key_encoder.projector(self.key_encoder.encoder(x_k))

        k, queue = self.queue(k)

        self._momentum_update_key_encoder()

        return {
            "q": q,
            "k": k,
            "queue": queue,
        }

    def get_embeddings(self, x: Tensor) -> Tensor:
        z = self.query_encoder.projector(self.query_encoder.encoder(x))
        return F.normalize(z, dim=-1)


class MultiCropWrapper(nn.Module):
    """Wrapper for multi-crop augmentation in self-supervised learning."""

    def __init__(
        self,
        encoder: nn.Module,
        crop_size: int = 224,
        n_crops: int = 2,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.1, 0.4),
        local_crops_number: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.crop_size = crop_size
        self.n_crops = n_crops
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError("MultiCropWrapper requires specific crop handling")


def get_byol_loss(p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
    """Compute BYOL loss between predictions and targets."""
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    loss = 2 - 2 * (p1 * z2).sum(dim=-1).mean()
    return loss


def get_simsiam_loss(p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
    """Compute SimSiam loss with stop-gradient on targets."""
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)

    loss = 2 - 2 * (p1 * z2.detach()).sum(dim=-1).mean()
    loss += 2 - 2 * (p2 * z1.detach()).sum(dim=-1).mean()
    return loss / 2
