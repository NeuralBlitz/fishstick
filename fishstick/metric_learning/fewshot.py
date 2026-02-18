"""
Metric-Based Few-Shot Learning

Implementation of metric-based few-shot learning methods:
- Prototypical Networks
- Relation Networks
- MAML for metric learning
- Matching Networks
- FEAT (Feature Augmentation Transformation)
- FewShotClassifier wrapper
"""

from typing import Optional, Tuple, List, Dict, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module


class PrototypicalNetwork(Module):
    """Prototypical Networks for Few-Shot Learning.

    Computes class prototypes as the mean of support set embeddings,
    then classifies query samples based on distance to prototypes.

    Args:
        encoder: Feature encoder network
        distance: Distance function ('euclidean', 'cosine', or callable)
        num_prototypes: Number of prototypes per class
    """

    def __init__(
        self,
        encoder: nn.Module,
        distance: str = "euclidean",
        num_prototypes: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.distance = distance.lower()
        self.num_prototypes = num_prototypes

    def forward(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Classify query samples using prototypical networks.

        Args:
            support_features: Support set features (N_s, dim)
            support_labels: Support set labels (N_s,)
            query_features: Query set features (N_q, dim)

        Returns:
            Tuple of (logits, prototypes)
        """
        if self.encoder is not None:
            support_features = self.encoder(support_features)
            query_features = self.encoder(query_features)

        support_features = F.normalize(support_features, dim=-1)
        query_features = F.normalize(query_features, dim=-1)

        unique_labels = support_labels.unique()
        num_classes = len(unique_labels)

        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]

            if self.num_prototypes == 1:
                prototype = class_features.mean(dim=0)
            else:
                prototype = self._compute_multiple_prototypes(
                    class_features, self.num_prototypes
                )

            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        if self.distance == "euclidean":
            query_exp = query_features.unsqueeze(1)
            proto_exp = prototypes.unsqueeze(0)
            dists = torch.sum((query_exp - proto_exp) ** 2, dim=-1)
            logits = -dists
        elif self.distance == "cosine":
            query_exp = query_features.unsqueeze(1)
            proto_exp = prototypes.unsqueeze(0)
            logits = (query_exp * proto_exp).sum(dim=-1)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        return logits, prototypes

    def _compute_multiple_prototypes(
        self,
        features: Tensor,
        num_prototypes: int,
    ) -> Tensor:
        """Compute multiple prototypes using k-means clustering."""
        if features.shape[0] <= num_prototypes:
            return features.mean(dim=0).unsqueeze(0).repeat(num_prototypes, 1)

        from sklearn.cluster import KMeans

        with torch.no_grad():
            features_np = features.cpu().numpy()
            kmeans = KMeans(n_clusters=num_prototypes, n_init=10)
            kmeans.fit(features_np)
            centers = torch.tensor(
                kmeans.cluster_centers_, dtype=features.dtype, device=features.device
            )

        return centers

    def get_prototypes(
        self,
        support_features: Tensor,
        support_labels: Tensor,
    ) -> Tensor:
        """Get class prototypes from support set."""
        if self.encoder is not None:
            support_features = self.encoder(support_features)

        support_features = F.normalize(support_features, dim=-1)

        unique_labels = support_labels.unique()
        prototypes = []

        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)


class RelationNetwork(Module):
    """Relation Networks for Few-Shot Learning.

    Uses a relation module to compute relation scores between
    query samples and class prototypes.

    Args:
        encoder: Feature encoder network
        relation_module: Network that computes relation scores
        distance: Distance function
    """

    def __init__(
        self,
        encoder: nn.Module,
        relation_module: Optional[nn.Module] = None,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim

        if relation_module is None:
            self.relation_module = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
            )
        else:
            self.relation_module = relation_module

    def forward(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
    ) -> Tuple[Tensor, Dict]:
        """Classify query samples using relation networks.

        Args:
            support_features: Support set features
            support_labels: Support set labels
            query_features: Query set features

        Returns:
            Tuple of (logits, info_dict)
        """
        if self.encoder is not None:
            support_features = self.encoder(support_features)
            query_features = self.encoder(query_features)

        support_features = F.normalize(support_features, dim=-1)
        query_features = F.normalize(query_features, dim=-1)

        unique_labels = support_labels.unique()
        num_classes = len(unique_labels)

        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        query_exp = query_features.unsqueeze(1)
        proto_exp = prototypes.unsqueeze(0)

        concat = torch.cat([query_exp, proto_exp], dim=-1)

        relations = self.relation_module(concat).squeeze(-1)

        return relations, {"prototypes": prototypes}

    def create_relation_module(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
        )


class MatchingNetwork(Module):
    """Matching Networks for Few-Shot Learning.

    Uses attention over support set to classify query samples.

    Args:
        encoder: Feature encoder network
        attention_fn: Attention function ('cosine' or 'bilinear')
    """

    def __init__(
        self,
        encoder: nn.Module,
        attention_fn: str = "cosine",
    ):
        super().__init__()
        self.encoder = encoder
        self.attention_fn = attention_fn.lower()

    def forward(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
    ) -> Tuple[Tensor, Dict]:
        """Classify query samples using matching networks."""
        if self.encoder is not None:
            support_features = self.encoder(support_features)
            query_features = self.encoder(query_features)

        support_features = F.normalize(support_features, dim=-1)
        query_features = F.normalize(query_features, dim=-1)

        unique_labels = support_labels.unique()

        if self.attention_fn == "cosine":
            similarities = torch.mm(query_features, support_features.T)
        elif self.attention_fn == "bilinear":
            weight = nn.Parameter(
                torch.randn(query_features.shape[-1], support_features.shape[-1])
            )
            similarities = torch.mm(query_features @ weight, support_features.T)
        else:
            raise ValueError(f"Unknown attention: {self.attention_fn}")

        attention = F.softmax(similarities, dim=-1)

        class_logits = []
        for label in unique_labels:
            mask = support_labels == label
            class_attention = attention[:, mask].sum(dim=-1, keepdim=True)
            class_logits.append(class_attention)

        logits = torch.cat(class_logits, dim=-1)

        return logits, {"attention": attention}

    def get_attention(
        self,
        support_features: Tensor,
        query_features: Tensor,
    ) -> Tensor:
        """Compute attention weights."""
        if self.encoder is not None:
            support_features = self.encoder(support_features)
            query_features = self.encoder(query_features)

        support_features = F.normalize(support_features, dim=-1)
        query_features = F.normalize(query_features, dim=-1)

        similarities = torch.mm(query_features, support_features.T)
        attention = F.softmax(similarities, dim=-1)

        return attention


class FEAT(Module):
    """Feature Augmentation Transformation (FEAT) for Few-Shot Learning.

    Uses a transformation network to adapt class prototypes for each episode.

    Args:
        encoder: Feature encoder network
        transform_network: Network to transform prototypes
        num_classes: Number of classes
    """

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim

        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
    ) -> Tuple[Tensor, Dict]:
        """Classify query samples using FEAT."""
        if self.encoder is not None:
            support_features = self.encoder(support_features)
            query_features = self.encoder(query_features)

        support_features = F.normalize(support_features, dim=-1)
        query_features = F.normalize(query_features, dim=-1)

        unique_labels = support_labels.unique()

        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        transformed_prototypes = self.transform(prototypes)
        transformed_prototypes = F.normalize(transformed_prototypes, dim=-1)

        query_exp = query_features.unsqueeze(1)
        proto_exp = transformed_prototypes.unsqueeze(0)
        logits = (query_exp * proto_exp).sum(dim=-1)

        return logits, {"prototypes": transformed_prototypes}


class MAMLFewShot(Module):
    """MAML for Few-Shot Learning.

    Model-Agnostic Meta-Learning adapted for few-shot classification.

    Args:
        encoder: Feature encoder network
        num_classes: Number of classes in few-shot task
        inner_lr: Inner loop learning rate
        inner_steps: Number of inner loop gradient steps
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 5,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        self.classifier = nn.Linear(
            encoder.output_dim if hasattr(encoder, "output_dim") else 64, num_classes
        )

    def forward(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
        labels: Optional[Tensor] = None,
        return_logits: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """MAML forward pass for few-shot learning."""
        if self.encoder is not None:
            support_features = self.encoder(support_features)
            query_features = self.encoder(query_features)

        if return_logits:
            return self._forward_with_finetuning(
                support_features, support_labels, query_features
            )
        else:
            return support_features, query_features

    def _forward_with_finetuning(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Perform inner loop adaptation and compute query predictions."""
        support_logits = self.classifier(support_features)

        criterion = nn.CrossEntropyLoss()

        params = {
            name: param.clone() for name, param in self.classifier.named_parameters()
        }

        for _ in range(self.inner_steps):
            support_preds = F.linear(support_features, params["weight"], params["bias"])
            loss = criterion(support_preds, support_labels)

            grads = torch.autograd.grad(loss, params.values(), create_graph=True)

            params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(params.items(), grads)
            }

        query_logits = F.linear(query_features, params["weight"], params["bias"])

        return query_logits, support_logits


class FewShotClassifier(Module):
    """Wrapper for few-shot classification models.

    Provides a unified interface for various few-shot learning methods.

    Args:
        encoder: Feature encoder network
        method: Few-shot method ('protonet', 'relationnet', 'matchingnet', 'feat')
        distance: Distance function for distance-based methods
    """

    def __init__(
        self,
        encoder: nn.Module,
        method: str = "protonet",
        distance: str = "euclidean",
    ):
        super().__init__()
        self.encoder = encoder
        self.method = method.lower()
        self.distance = distance

        if self.method == "protonet":
            self.model = PrototypicalNetwork(encoder, distance=distance)
        elif self.method == "relationnet":
            self.model = RelationNetwork(encoder)
        elif self.method == "matchingnet":
            self.model = MatchingNetwork(encoder)
        elif self.method == "feat":
            self.model = FEAT(encoder)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forward(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
    ) -> Tensor:
        """Forward pass for few-shot classification.

        Args:
            support_features: Support set features
            support_labels: Support set labels
            query_features: Query set features

        Returns:
            Classification logits for query samples
        """
        logits, _ = self.model(support_features, support_labels, query_features)
        return logits

    def get_prototypes(
        self,
        support_features: Tensor,
        support_labels: Tensor,
    ) -> Tensor:
        """Get class prototypes from support set."""
        if hasattr(self.model, "get_prototypes"):
            return self.model.get_prototypes(support_features, support_labels)
        else:
            raise NotImplementedError(f"get_prototypes not available for {self.method}")


class EpisodicSampler:
    """Sampler for creating episodic few-shot learning batches.

    Args:
        num_classes: Total number of classes
        num_support: Number of support samples per class
        num_query: Number of query samples per class
    """

    def __init__(
        self,
        num_classes: int,
        num_support: int = 5,
        num_query: int = 15,
    ):
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query

    def sample_episode(
        self,
        features: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample an episode from the data.

        Args:
            features: All features
            labels: All labels

        Returns:
            Tuple of (support_features, support_labels, query_features, query_labels)
        """
        unique_labels = labels.unique()
        selected_labels = unique_labels[
            torch.randperm(len(unique_labels))[: self.num_classes]
        ]

        support_features_list = []
        support_labels_list = []
        query_features_list = []
        query_labels_list = []

        for i, label in enumerate(selected_labels):
            mask = labels == label
            class_features = features[mask]

            indices = torch.randperm(len(class_features))

            support_indices = indices[: self.num_support]
            query_indices = indices[
                self.num_support : self.num_support + self.num_query
            ]

            support_features_list.append(class_features[support_indices])
            support_labels_list.append(
                torch.full((self.num_support,), i, dtype=torch.long)
            )

            query_features_list.append(class_features[query_indices])
            query_labels_list.append(torch.full((self.num_query,), i, dtype=torch.long))

        support_features = torch.cat(support_features_list)
        support_labels = torch.cat(support_labels_list)
        query_features = torch.cat(query_features_list)
        query_labels = torch.cat(query_labels_list)

        return support_features, support_labels, query_features, query_labels


__all__ = [
    "PrototypicalNetwork",
    "RelationNetwork",
    "MatchingNetwork",
    "FEAT",
    "MAMLFewShot",
    "FewShotClassifier",
    "EpisodicSampler",
]
