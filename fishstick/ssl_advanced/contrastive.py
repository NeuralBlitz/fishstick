from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer


@dataclass
class SimCLRConfig:
    proj_hidden_dim: int = 256
    proj_output_dim: int = 128
    temperature: float = 0.07
    batch_size: int = 256
    epochs: int = 100
    lr: float = 0.001
    momentum: float = 0.999


@dataclass
class MoCoConfig:
    proj_hidden_dim: int = 256
    proj_output_dim: int = 128
    queue_size: int = 65536
    momentum: float = 0.999
    temperature: float = 0.07
    batch_size: int = 256
    epochs: int = 100
    lr: float = 0.001


@dataclass
class BYOLConfig:
    proj_hidden_dim: int = 256
    proj_output_dim: int = 256
    momentum: float = 0.999
    batch_size: int = 256
    epochs: int = 100
    lr: float = 0.001


@dataclass
class SwAVConfig:
    proj_hidden_dim: int = 256
    proj_output_dim: int = 128
    queue_size: int = 0
    temperature: float = 0.1
    epsilon: float = 0.05
    n_crops: int = 2
    batch_size: int = 256
    epochs: int = 100
    lr: float = 0.001


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, projection_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections


class SimCLR(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 512,
        config: Optional[SimCLRConfig] = None,
    ):
        super().__init__()
        self.config = config or SimCLRConfig()
        self.encoder = encoder
        self.projection_head = SimCLRProjectionHead(
            input_dim, self.config.proj_hidden_dim, self.config.proj_output_dim
        )
        self.temperature = self.config.temperature

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.size(0)
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)).float()
        denominator = torch.sum(mask * similarity_matrix / self.temperature, dim=1)
        loss = -torch.log(nominator / denominator)
        return loss.mean()


class MoCo(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 512,
        config: Optional[MoCoConfig] = None,
    ):
        super().__init__()
        self.config = config or MoCoConfig()
        self.momentum = self.config.momentum
        self.temperature = self.config.temperature

        self.encoder_q = nn.Sequential(
            encoder,
            ProjectionHead(
                input_dim, self.config.proj_hidden_dim, self.config.proj_output_dim
            ),
        )
        self.encoder_k = nn.Sequential(
            encoder if hasattr(encoder, "copy") else self._copy_encoder(encoder),
            ProjectionHead(
                input_dim, self.config.proj_hidden_dim, self.config.proj_output_dim
            ),
        )

        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.register_buffer(
            "queue", torch.randn(self.config.queue_size, self.config.proj_output_dim)
        )
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        from copy import deepcopy

        return deepcopy(encoder)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.config.queue_size % batch_size == 0
        self.queue[ptr : ptr + batch_size] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.config.queue_size

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        q = self.encoder_q(x_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits / self.temperature, labels)

        self._dequeue_and_enqueue(k)
        return loss


class BYOL(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 512,
        config: Optional[BYOLConfig] = None,
    ):
        super().__init__()
        self.config = config or BYOLConfig()
        self.momentum = self.config.momentum

        self.online_encoder = nn.Sequential(
            encoder,
            ProjectionHead(
                input_dim, self.config.proj_hidden_dim, self.config.proj_output_dim
            ),
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(self.config.proj_output_dim, self.config.proj_hidden_dim),
            nn.BatchNorm1d(self.config.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.proj_hidden_dim, self.config.proj_output_dim),
        )

        self.target_encoder = self._create_target_encoder(encoder, input_dim)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def _create_target_encoder(self, encoder: nn.Module, input_dim: int) -> nn.Module:
        from copy import deepcopy

        target = nn.Sequential(
            deepcopy(encoder),
            ProjectionHead(
                input_dim, self.config.proj_hidden_dim, self.config.proj_output_dim
            ),
        )
        return target

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_online, param_target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_target.data.mul_(self.momentum).add_(
                param_online.data, alpha=1 - self.momentum
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        self._momentum_update_target_encoder()

        online_proj1 = self.online_encoder(x1)
        online_pred1 = self.online_predictor(online_proj1)
        online_proj2 = self.online_encoder(x2)
        online_pred2 = self.online_predictor(online_proj2)

        with torch.no_grad():
            target_proj1 = self.target_encoder(x1)
            target_proj2 = self.target_encoder(x2)

        online_pred1 = F.normalize(online_pred1, dim=1)
        online_pred2 = F.normalize(online_pred2, dim=1)
        target_proj1 = F.normalize(target_proj1, dim=1)
        target_proj2 = F.normalize(target_proj2, dim=1)

        loss1 = 2 - 2 * (online_pred1 * target_proj2).sum(dim=1).mean()
        loss2 = 2 - 2 * (online_pred2 * target_proj1).sum(dim=1).mean()
        return (loss1 + loss2) / 2


class SwAVPrototypes(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_prototypes: int):
        super().__init__()
        self.prototypes = nn.Linear(output_dim, n_prototypes, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.prototypes(z)


class SwAV(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 512,
        n_prototypes: int = 3000,
        config: Optional[SwAVConfig] = None,
    ):
        super().__init__()
        self.config = config or SwAVConfig()
        self.n_prototypes = n_prototypes
        self.temperature = self.config.temperature
        self.epsilon = self.config.epsilon
        self.n_crops = self.config.n_crops

        self.encoder = encoder
        self.projection_head = SimCLRProjectionHead(
            input_dim, self.config.proj_hidden_dim, self.config.proj_output_dim
        )
        self.prototypes = SwAVPrototypes(
            input_dim, self.config.proj_output_dim, n_prototypes
        )

    def forward(self, crops: List[torch.Tensor]) -> torch.Tensor:
        batch_size = crops[0].size(0)
        n_crops = len(crops)

        projections = []
        for crop in crops:
            z = self.encoder(crop)
            z = self.projection_head(z)
            z = F.normalize(z, dim=1)
            projections.append(z)

        logits = []
        for z in projections:
            logits.append(self.prototypes(z))

        logits = torch.stack(logits)
        logits = torch.flatten(logits, 0, 1)

        with torch.no_grad():
            targets = torch.argmax(logits, dim=1)
            targets = targets.view(n_crops, batch_size).T
            targets = targets.contiguous()
            targets = F.one_hot(targets, num_classes=self.n_prototypes).float()
            targets = self._sinkhorn_knopp(targets)

        loss = 0
        for i, z in enumerate(projections):
            q = F.softmax(
                logits[i * batch_size : (i + 1) * batch_size] / self.temperature, dim=-1
            )
            t = targets[:, i * batch_size : (i + 1) * batch_size]
            loss -= torch.sum(t * torch.log(q + self.epsilon), dim=-1).mean()

        return loss / n_crops

    def _sinkhorn_knopp(self, Q: torch.Tensor, max_iter: int = 3) -> torch.Tensor:
        K = Q.shape[1]
        Q = Q / K

        for _ in range(max_iter):
            sum_rows = Q.sum(dim=1, keepdim=True)
            Q = Q / (sum_rows + 1e-10)
            sum_cols = Q.sum(dim=0, keepdim=True)
            Q = Q / (sum_cols + 1e-10)

        return Q


def train_simclr(
    model: SimCLR,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = torch.cat(images, dim=0).to(device)
        x1 = images[: images.size(0) // 2]
        x2 = images[images.size(0) // 2 :]

        optimizer.zero_grad()
        loss = model(x1, x2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_moco(
    model: MoCo,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        x_q = images[: images.size(0) // 2]
        x_k = images[images.size(0) // 2 :]

        optimizer.zero_grad()
        loss = model(x_q, x_k)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_byol(
    model: BYOL,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = torch.cat(images, dim=0).to(device)
        x1 = images[: images.size(0) // 2]
        x2 = images[images.size(0) // 2 :]

        optimizer.zero_grad()
        loss = model(x1, x2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_swav(
    model: SwAV,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        crops = [img.to(device) for img in images]

        optimizer.zero_grad()
        loss = model(crops)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
