import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from enum import Enum


class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        distance: str = "euclidean",
        learnable_prototypes: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.distance = DistanceMetric(distance)
        self.learnable_prototypes = learnable_prototypes

        self.num_classes = 0
        self.prototype_dim = None

    def forward(
        self,
        x: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.encoder(x)

        if prototypes is None:
            return embeddings

        if self.distance == DistanceMetric.EUCLIDEAN:
            distances = self._euclidean_distance(embeddings, prototypes)
            return -distances
        elif self.distance == DistanceMetric.COSINE:
            similarities = self._cosine_similarity(embeddings, prototypes)
            return similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

    def _euclidean_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)

    def _cosine_similarity(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return torch.mm(x, y.t())

    def compute_prototypes(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            support_embeddings = self.encoder(support_x)

            classes = torch.unique(support_y)
            num_classes = len(classes)

            prototypes = []
            for c in classes:
                class_mask = support_y == c
                class_embeddings = support_embeddings[class_mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)

            prototypes = torch.stack(prototypes)

        return prototypes

    def predict(
        self,
        query_x: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(query_x, prototypes)
        predictions = logits.argmax(dim=-1)
        return predictions

    def episode_train_step(
        self,
        episode: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        support_x = episode["support_x"]
        support_y = episode["support_y"]
        query_x = episode["query_x"]
        query_y = episode["query_y"]

        prototypes = self.compute_prototypes(support_x, support_y)

        query_logits = self.forward(query_x, prototypes)

        loss = F.cross_entropy(query_logits, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            query_preds = query_logits.argmax(dim=-1)
            accuracy = (query_preds == query_y).float().mean().item()

        return {"loss": loss.item(), "accuracy": accuracy}

    def meta_train_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        total_loss = 0.0
        total_accuracy = 0.0

        for episode in tasks:
            support_x = episode["support_x"]
            support_y = episode["support_y"]
            query_x = episode["query_x"]
            query_y = episode["query_y"]

            prototypes = self.compute_prototypes(support_x, support_y)

            query_logits = self.forward(query_x, prototypes)

            loss = F.cross_entropy(query_logits, query_y)

            total_loss += loss.item()

            with torch.no_grad():
                query_preds = query_logits.argmax(dim=-1)
                accuracy = (query_preds == query_y).float().mean().item()
                total_accuracy += accuracy

        avg_loss = total_loss / len(tasks)
        avg_accuracy = total_accuracy / len(tasks)

        optimizer.zero_grad()
        torch.autograd.grad(
            avg_loss,
            self.parameters(),
            retain_graph=True,
            allow_unused=True,
        )
        optimizer.step()

        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_prototypes(support_x, support_y)


class PrototypicalNetworkFewShot(PrototypicalNetwork):
    def __init__(
        self,
        encoder: nn.Module,
        n_way: int,
        n_support: int,
        n_query: int,
        distance: str = "euclidean",
    ):
        super().__init__(encoder, distance)
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query


class ClassPrototypes(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        learnable: bool = False,
        init_method: str = "zeros",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.learnable = learnable

        if learnable:
            if init_method == "zeros":
                init = torch.zeros(num_classes, feature_dim)
            elif init_method == "random":
                init = torch.randn(num_classes, feature_dim) * 0.1
            elif init_method == "orthogonal":
                init = torch.nn.init.orthogonal_(torch.empty(num_classes, feature_dim))
            else:
                raise ValueError(f"Unknown init method: {init_method}")

            self.prototypes = nn.Parameter(init)
        else:
            self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))

    def update(self, new_prototypes: torch.Tensor):
        if self.learnable:
            self.prototypes.data.copy_(new_prototypes)
        else:
            self.prototypes.copy_(new_prototypes)

    def forward(self) -> torch.Tensor:
        if self.learnable:
            return self.prototypes
        return self.prototypes


class EpisodicTrainer:
    def __init__(
        self,
        model: PrototypicalNetwork,
        optimizer: torch.optim.Optimizer,
        n_way: int,
        n_support: int,
        n_query: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

    def create_episode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        classes = torch.unique(y)
        selected_classes = classes[torch.randperm(len(classes))[: self.n_way]]

        support_indices = []
        query_indices = []

        for c in selected_classes:
            class_mask = y == c
            class_indices = torch.where(class_mask)[0]

            perm = torch.randperm(len(class_indices))
            support_indices.append(class_indices[perm[: self.n_support]])
            query_indices.append(
                class_indices[perm[self.n_support : self.n_support + self.n_query]]
            )

        support_indices = torch.cat(support_indices)
        query_indices = torch.cat(query_indices)

        relabel_map = {c.item(): i for i, c in enumerate(selected_classes)}

        support_x = x[support_indices]
        support_y = torch.tensor(
            [relabel_map[y[i].item()] for i in support_indices],
            dtype=torch.long,
            device=x.device,
        )

        query_x = x[query_indices]
        query_y = torch.tensor(
            [relabel_map[y[i].item()] for i in query_indices],
            dtype=torch.long,
            device=x.device,
        )

        return {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y,
        }

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        episode = self.create_episode(x, y)
        return self.model.episode_train_step(episode, self.optimizer)


class MultiModalPrototypicalNetwork(PrototypicalNetwork):
    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        distance: str = "euclidean",
        fusion: str = "concat",
    ):
        super().__init__(nn.Identity(), distance)
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion

        if fusion == "concat":
            self.projection = None
        elif fusion == "attention":
            self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
            self.projection = nn.Linear(128, 128)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        prototypes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = []

        for modality, x in inputs.items():
            if modality in self.encoders:
                emb = self.encoders[modality](x)
                embeddings.append(emb)

        if self.fusion == "concat":
            embeddings = torch.cat(embeddings, dim=-1)
        elif self.fusion == "attention":
            embeddings = torch.stack(embeddings, dim=0)
            embeddings, _ = self.attention(embeddings, embeddings, embeddings)
            embeddings = embeddings.mean(dim=0)
            if self.projection:
                embeddings = self.projection(embeddings)

        if prototypes is None:
            return embeddings

        if self.distance == DistanceMetric.EUCLIDEAN:
            distances = self._euclidean_distance(embeddings, prototypes)
            return -distances
        elif self.distance == DistanceMetric.COSINE:
            similarities = self._cosine_similarity(embeddings, prototypes)
            return similarities


def compute_class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    prototypes = []

    for c in range(num_classes):
        class_mask = labels == c
        class_embeddings = embeddings[class_mask]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)

    return torch.stack(prototypes)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y)


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return 1 - torch.mm(x_norm, y_norm.t())


def prototypical_loss(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    distance: str = "euclidean",
) -> torch.Tensor:
    if distance == "euclidean":
        distances = euclidean_distance(embeddings, prototypes)
        logits = -distances
    elif distance == "cosine":
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1), prototypes.unsqueeze(0), dim=-1
        )
        logits = similarities
    else:
        raise ValueError(f"Unknown distance: {distance}")

    return F.cross_entropy(logits, labels)
