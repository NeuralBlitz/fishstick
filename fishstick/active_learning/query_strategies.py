import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, List
from abc import ABC, abstractmethod


class QueryStrategy(ABC):
    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def query(self, unlabeled_features: torch.Tensor, n_query: int) -> torch.Tensor:
        pass

    def get_model_predictions(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


class UncertaintyStrategy(QueryStrategy):
    def __init__(self, model: nn.Module):
        super().__init__(model)


class EntropyStrategy(UncertaintyStrategy):
    def query(self, unlabeled_features: torch.Tensor, n_query: int) -> torch.Tensor:
        probs = self.get_model_predictions(unlabeled_features)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        _, indices = torch.topk(entropy, n_query)
        return indices


class MarginStrategy(UncertaintyStrategy):
    def query(self, unlabeled_features: torch.Tensor, n_query: int) -> torch.Tensor:
        probs = self.get_model_predictions(unlabeled_features)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        _, indices = torch.topk(margins, n_query, largest=False)
        return indices


class LeastConfidenceStrategy(UncertaintyStrategy):
    def query(self, unlabeled_features: torch.Tensor, n_query: int) -> torch.Tensor:
        probs = self.get_model_predictions(unlabeled_features)
        max_probs, _ = torch.max(probs, dim=-1)
        lc_scores = 1 - max_probs
        _, indices = torch.topk(lc_scores, n_query)
        return indices


class DiversityStrategy(QueryStrategy):
    def __init__(self, model: nn.Module, feature_extractor: Optional[nn.Module] = None):
        super().__init__(model)
        self.feature_extractor = feature_extractor or model

    def query(self, unlabeled_features: torch.Tensor, n_query: int) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor(unlabeled_features)

        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        features = torch.nn.functional.normalize(features, p=2, dim=1)

        selected_indices = []
        selected_features = []

        distances = torch.cdist(features, features, p=2)

        for _ in range(n_query):
            if len(selected_indices) == 0:
                idx = torch.randint(0, features.size(0), (1,)).item()
            else:
                min_distances = distances[:, selected_indices].min(dim=1)[0]
                min_distances[selected_indices] = -1
                _, idx = torch.max(min_distances, dim=0)
                idx = idx.item()

            selected_indices.append(idx)
            selected_features.append(features[idx])

        return torch.tensor(selected_indices, dtype=torch.long)


class ExpectedModelChangeStrategy(QueryStrategy):
    def __init__(self, model: nn.Module, feature_extractor: Optional[nn.Module] = None):
        super().__init__(model)
        self.feature_extractor = feature_extractor or model

    def compute_gradient_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        features = self.feature_extractor(x)

        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        features.requires_grad_(True)
        output = self.model(features)

        if output.dim() > 2:
            output = output.view(output.size(0), -1)

        num_classes = output.size(-1)
        grad_embeddings = []

        for i in range(num_classes):
            self.model.zero_grad()
            grad = torch.autograd.grad(output[:, i].sum(), features, retain_graph=True)[
                0
            ]
            grad_embeddings.append(grad)

        grad_embeddings = torch.stack(grad_embeddings, dim=0)
        grad_embeddings = grad_embeddings.mean(dim=0)

        return grad_embeddings

    def query(self, unlabeled_features: torch.Tensor, n_query: int) -> torch.Tensor:
        grad_embeddings = self.compute_gradient_embeddings(unlabeled_features)

        grad_norms = torch.norm(grad_embeddings, p=2, dim=1)
        _, indices = torch.topk(grad_norms, n_query)

        return indices


class ExpectedGradientLengthStrategy(ExpectedModelChangeStrategy):
    def query(self, unlabeled_features: torch.Tensor, n_query: int) -> torch.Tensor:
        return super().query(unlabeled_features, n_query)


class BadgeStrategy(QueryStrategy):
    def __init__(self, model: nn.Module, feature_extractor: Optional[nn.Module] = None):
        super().__init__(model)
        self.feature_extractor = feature_extractor or model

    def compute_badge_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        if hasattr(self.feature_extractor, "forward_features"):
            features = self.feature_extractor.forward_features(x)
        else:
            features = self.feature_extractor(x)

        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        features = features.detach()

        logits = self.model(features)
        probs = torch.softmax(logits, dim=-1)

        one_hot = torch.zeros_like(probs)
        labels = probs.argmax(dim=-1)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        grad_embeddings = probs - one_hot

        combined = features * grad_embeddings

        return combined

    def query(self, unlabeled_features: torch.Tensor, n_query: int) -> torch.Tensor:
        badge_embeddings = self.compute_badge_embeddings(unlabeled_features)

        if badge_embeddings.dim() > 2:
            badge_embeddings = badge_embeddings.view(badge_embeddings.size(0), -1)

        kmeans = KMeans(n_clusters=n_query, random_state=42)
        cluster_labels = kmeans.fit_predict(badge_embeddings.cpu().numpy())

        indices = []
        for i in range(n_query):
            cluster_idx = np.where(cluster_labels == i)[0]
            if len(cluster_idx) > 0:
                farthest = cluster_idx[0]
                max_dist = 0
                for j in cluster_idx:
                    dist = torch.norm(
                        badge_embeddings[j] - badge_embeddings[cluster_idx[0]]
                    ).item()
                    if dist > max_dist:
                        max_dist = dist
                        farthest = j
                indices.append(farthest)

        return torch.tensor(indices, dtype=torch.long)


class KMeans:
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(100):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [
                    X[labels == i].mean(axis=0)
                    if np.sum(labels == i) > 0
                    else self.centroids[i]
                    for i in range(self.n_clusters)
                ]
            )

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return labels
