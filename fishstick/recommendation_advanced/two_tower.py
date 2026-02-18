import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class UserTower(nn.Module):
    def __init__(
        self,
        user_feature_dims: Dict[str, int],
        embedding_dims: Dict[str, int],
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_feature_dims = user_feature_dims
        self.embedding_dims = embedding_dims

        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_features, embedding_dim)
                for name, (num_features, embedding_dim) in zip(
                    user_feature_dims.keys(), embedding_dims.values()
                )
            }
        )

        total_embedding_dim = sum(embedding_dims.values())

        layers = []
        input_dim = total_embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for name, feature in user_features.items():
            emb = self.embeddings[name](feature)
            embeddings.append(emb)

        combined = torch.cat(embeddings, dim=-1)
        output = self.mlp(combined)
        return output


class ItemTower(nn.Module):
    def __init__(
        self,
        item_feature_dims: Dict[str, int],
        embedding_dims: Dict[str, int],
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.item_feature_dims = item_feature_dims
        self.embedding_dims = embedding_dims

        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_features, embedding_dim)
                for name, (num_features, embedding_dim) in zip(
                    item_feature_dims.keys(), embedding_dims.values()
                )
            }
        )

        total_embedding_dim = sum(embedding_dims.values())

        layers = []
        input_dim = total_embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for name, feature in item_features.items():
            emb = self.embeddings[name](feature)
            embeddings.append(emb)

        combined = torch.cat(embeddings, dim=-1)
        output = self.mlp(combined)
        return output


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_feature_dims: Dict[str, int],
        item_feature_dims: Dict[str, int],
        user_embedding_dims: Dict[str, int],
        item_embedding_dims: Dict[str, int],
        user_hidden_dims: List[int] = [256, 128, 64],
        item_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

        self.user_tower = UserTower(
            user_feature_dims,
            user_embedding_dims,
            user_hidden_dims,
            dropout,
        )

        self.item_tower = ItemTower(
            item_feature_dims,
            item_embedding_dims,
            item_hidden_dims,
            dropout,
        )

        self.output_dim = user_hidden_dims[-1]

        assert self.output_dim == item_hidden_dims[-1], (
            "User and item tower output dimensions must match"
        )

    def forward(
        self,
        user_features: Dict[str, torch.Tensor],
        item_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)

        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        item_embedding = F.normalize(item_embedding, p=2, dim=-1)

        logits = torch.matmul(user_embedding, item_embedding.transpose(-2, -1))
        logits = logits / self.temperature

        return logits

    def get_user_embedding(
        self, user_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.user_tower(user_features)

    def get_item_embedding(
        self, item_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.item_tower(item_features)

    def compute_similarity(
        self,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor,
    ) -> torch.Tensor:
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        item_embedding = F.normalize(item_embedding, p=2, dim=-1)
        return torch.matmul(user_embedding, item_embedding.transpose(-2, -1))


class CandidateGenerator:
    def __init__(
        self,
        item_embeddings: torch.Tensor,
        item_ids: List[int],
        device: str = "cpu",
    ):
        self.item_embeddings = item_embeddings.to(device)
        self.item_ids = item_ids
        self.device = device

        self.item_embeddings = F.normalize(self.item_embeddings, p=2, dim=-1)

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 100,
        batch_size: int = 1024,
    ) -> Tuple[List[int], torch.Tensor]:
        query_embedding = F.normalize(query_embedding, p=2, dim=-1)

        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        scores = torch.matmul(query_embedding, self.item_embeddings.T)

        top_scores, top_indices = torch.topk(
            scores, min(top_k, len(self.item_ids)), dim=-1
        )

        retrieved_ids = [
            [self.item_ids[idx] for idx in batch_indices]
            for batch_indices in top_indices
        ]
        retrieved_scores = top_scores

        return retrieved_ids[0] if len(
            retrieved_ids
        ) == 1 else retrieved_ids, retrieved_scores

    def search_approximate(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 100,
        num_clusters: int = 50,
    ) -> Tuple[List[int], torch.Tensor]:
        import random

        random.seed(42)
        torch.manual_seed(42)

        num_items = len(self.item_embeddings)
        cluster_size = num_items // num_clusters

        cluster_centers = []
        cluster_assignments = []

        indices = torch.randperm(num_items)
        for i in range(num_clusters):
            start_idx = i * cluster_size
            end_idx = min((i + 1) * cluster_size, num_items)
            if start_idx < num_items:
                cluster = self.item_embeddings[indices[start_idx:end_idx]]
                center = cluster.mean(dim=0)
                cluster_centers.append(center)
                cluster_assignments.extend([i] * len(cluster))

        cluster_centers = torch.stack(cluster_centers)

        query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        cluster_centers = F.normalize(cluster_centers, p=2, dim=-1)

        cluster_scores = torch.matmul(query_embedding, cluster_centers.T)
        top_clusters = torch.topk(cluster_scores, min(10, num_clusters), dim=-1).indices

        candidate_indices = []
        for cluster_idx in top_clusters:
            start_idx = cluster_idx * cluster_size
            end_idx = min((cluster_idx + 1) * cluster_size, num_items)
            candidate_indices.extend(range(start_idx, end_idx))

        candidate_embeddings = self.item_embeddings[candidate_indices]
        candidate_scores = torch.matmul(query_embedding, candidate_embeddings.T)

        top_scores, top_local_indices = torch.topk(
            candidate_scores.squeeze(0), min(top_k, len(candidate_indices))
        )
        global_indices = [candidate_indices[idx] for idx in top_local_indices]
        retrieved_ids = [self.item_ids[idx] for idx in global_indices]

        return retrieved_ids, top_scores


class RetrievalModel(nn.Module):
    def __init__(
        self,
        user_feature_dims: Dict[str, int],
        item_feature_dims: Dict[str, int],
        user_embedding_dims: Dict[str, int],
        item_embedding_dims: Dict[str, int],
        user_hidden_dims: List[int] = [256, 128, 64],
        item_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.two_tower = TwoTowerModel(
            user_feature_dims,
            item_feature_dims,
            user_embedding_dims,
            item_embedding_dims,
            user_hidden_dims,
            item_hidden_dims,
            dropout,
            temperature,
        )

        self.user_tower = self.two_tower.user_tower
        self.item_tower = self.two_tower.item_tower

    def forward(
        self,
        user_features: Dict[str, torch.Tensor],
        item_features: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.two_tower(user_features, item_features)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output

    def get_user_embedding(
        self, user_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.two_tower.get_user_embedding(user_features)

    def get_item_embedding(
        self, item_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.two_tower.get_item_embedding(item_features)

    def index_items(
        self,
        item_features: Dict[str, torch.Tensor],
        item_ids: List[int],
        device: str = "cpu",
    ) -> CandidateGenerator:
        self.eval()
        with torch.no_grad():
            item_embeddings = self.get_item_embedding(item_features)

        return CandidateGenerator(item_embeddings, item_ids, device)

    def recommend(
        self,
        user_features: Dict[str, torch.Tensor],
        candidate_generator: CandidateGenerator,
        top_k: int = 10,
        approximate: bool = False,
    ) -> Tuple[List[int], torch.Tensor]:
        self.eval()
        with torch.no_grad():
            user_embedding = self.get_user_embedding(user_features)

        if approximate:
            return candidate_generator.search_approximate(user_embedding, top_k)
        else:
            return candidate_generator.search(user_embedding, top_k)


class NegativeSamplingLoss(nn.Module):
    def __init__(self, num_negatives: int = 4, temperature: float = 0.1):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature

    def forward(
        self,
        user_embedding: torch.Tensor,
        pos_item_embedding: torch.Tensor,
        neg_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        pos_item_embedding = F.normalize(pos_item_embedding, p=2, dim=-1)
        neg_item_embeddings = F.normalize(neg_item_embeddings, p=2, dim=-1)

        pos_scores = torch.sum(
            user_embedding * pos_item_embedding, dim=-1, keepdim=True
        )

        neg_scores = torch.matmul(user_embedding, neg_item_embeddings.T)

        scores = torch.cat([pos_scores, neg_scores], dim=-1)
        scores = scores / self.temperature

        labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)

        loss = F.cross_entropy(scores, labels)
        return loss


class InBatchNegativeLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_embedding: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)

        logits = torch.matmul(user_embedding, item_embeddings.T) / self.temperature

        batch_size = user_embedding.size(0)
        labels = torch.arange(batch_size, device=user_embedding.device)

        loss = F.cross_entropy(logits, labels)
        return loss
