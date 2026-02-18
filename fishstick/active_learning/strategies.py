import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


class AdvancedQueryStrategy(ABC):
    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def query(
        self,
        unlabeled_features: torch.Tensor,
        n_query: int,
        labeled_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass


class BADGEStrategy(AdvancedQueryStrategy):
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Optional[nn.Module] = None,
        batch_size: int = 64,
    ):
        super().__init__(model)
        self.feature_extractor = feature_extractor or model
        self.batch_size = batch_size

    def get_embedding_dim(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            features = self.feature_extractor(x)
        if features.dim() > 2:
            return features.view(features.size(0), -1).size(1)
        return features.size(1)

    def query(
        self,
        unlabeled_features: torch.Tensor,
        n_query: int,
        labeled_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.model.eval()

        if unlabeled_features.dim() > 2:
            unlabeled_features = unlabeled_features.view(unlabeled_features.size(0), -1)

        all_embeddings = []
        all_gradients = []

        for i in range(0, unlabeled_features.size(0), self.batch_size):
            batch = unlabeled_features[i : i + self.batch_size]
            batch = batch.detach().requires_grad_(True)

            features = self.feature_extractor(batch)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)

            logits = self.model(features)
            probs = F.softmax(logits, dim=-1)

            loss = probs.sum(dim=1).mean()
            self.model.zero_grad()

            if features.grad is not None:
                features.grad.zero_()

            loss.backward(retain_graph=True)

            gradients = batch.grad
            if gradients is not None:
                all_gradients.append(gradients.cpu())

            all_embeddings.append(features.detach().cpu())

        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
        else:
            embeddings = unlabeled_features

        if all_gradients:
            gradients = torch.cat(all_gradients, dim=0)
            badge_embeddings = embeddings * gradients
        else:
            badge_embeddings = embeddings

        if badge_embeddings.dim() > 2:
            badge_embeddings = badge_embeddings.view(badge_embeddings.size(0), -1)

        indices = self._kmeans_plus_plus(badge_embeddings, n_query)

        return indices

    def _kmeans_plus_plus(
        self, embeddings: torch.Tensor, n_clusters: int
    ) -> torch.Tensor:
        n_samples = embeddings.size(0)

        indices = []
        idx = np.random.randint(0, n_samples)
        indices.append(idx)

        centroids = embeddings[idx : idx + 1]

        for _ in range(n_clusters - 1):
            distances = torch.cdist(embeddings, centroids, p=2)
            min_distances = distances.min(dim=1)[0]

            probs = min_distances / min_distances.sum()
            idx = torch.multinomial(probs, 1).item()

            while idx in indices:
                idx = torch.multinomial(probs, 1).item()

            indices.append(idx)
            centroids = torch.cat([centroids, embeddings[idx : idx + 1]], dim=0)

        return torch.tensor(indices, dtype=torch.long)


class CoreSetStrategy(AdvancedQueryStrategy):
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Optional[nn.Module] = None,
        greedy_geo: bool = True,
    ):
        super().__init__(model)
        self.feature_extractor = feature_extractor or model
        self.greedy_geo = greedy_geo

    def query(
        self,
        unlabeled_features: torch.Tensor,
        n_query: int,
        labeled_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.model.eval()

        unlabeled_embeddings = self._get_embeddings(unlabeled_features)

        if labeled_features is not None and labeled_features.size(0) > 0:
            labeled_embeddings = self._get_embeddings(labeled_features)
            all_embeddings = torch.cat(
                [labeled_embeddings, unlabeled_embeddings], dim=0
            )
            labeled_count = labeled_embeddings.size(0)
        else:
            all_embeddings = unlabeled_embeddings
            labeled_count = 0

        indices = self._greedy_sampling(all_embeddings, n_query, labeled_count)

        if labeled_count > 0:
            indices = indices - labeled_count
            indices = indices[indices >= 0]

        return indices

    def _get_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.feature_extractor(features)
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
        return embeddings

    def _greedy_sampling(
        self, embeddings: torch.Tensor, n_query: int, offset: int = 0
    ) -> torch.Tensor:
        n_samples = embeddings.size(0)

        distances = torch.cdist(embeddings, embeddings, p=2)

        selected_indices = []

        if offset > 0:
            for i in range(offset):
                selected_indices.append(i)

        remaining_indices = list(range(offset, n_samples))

        for _ in range(n_query):
            if len(remaining_indices) == 0:
                break

            if len(selected_indices) == 0:
                idx = remaining_indices[np.random.randint(0, len(remaining_indices))]
            else:
                min_distances = distances[remaining_indices][:, selected_indices].min(
                    dim=1
                )
                max_min_idx = min_distances[0].argmax().item()
                idx = remaining_indices[max_min_idx]

            selected_indices.append(idx)
            remaining_indices.remove(idx)

        return torch.tensor(selected_indices, dtype=torch.long)


class VAALStrategy(AdvancedQueryStrategy):
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        discriminator: nn.Module,
        feature_extractor: Optional[nn.Module] = None,
        beta: float = 1.0,
        gamma: float = 0.1,
    ):
        super().__init__(model)
        self.vae = vae
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor or model
        self.beta = beta
        self.gamma = gamma

    def query(
        self,
        unlabeled_features: torch.Tensor,
        n_query: int,
        labeled_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.model.eval()
        self.vae.eval()
        self.discriminator.eval()

        unlabeled_embeddings = self._get_embeddings(unlabeled_features)

        if labeled_features is not None and labeled_features.size(0) > 0:
            labeled_embeddings = self._get_embeddings(labeled_features)

            labeled_recon, labeled_mu, labeled_logvar = self.vae(labeled_embeddings)
            unlabeled_recon, unlabeled_mu, unlabeled_logvar = self.vae(
                unlabeled_embeddings
            )

            labeled_scores = self.discriminator(labeled_mu)
            unlabeled_scores = self.discriminator(unlabeled_mu)

            labeled_loss = F.binary_cross_entropy_with_logits(
                labeled_scores, torch.ones_like(labeled_scores)
            )
            unlabeled_loss = F.binary_cross_entropy_with_logits(
                unlabeled_scores, torch.zeros_like(unlabeled_scores)
            )

            disc_loss = labeled_loss + unlabeled_loss

            unlabeled_kl = -0.5 * torch.mean(
                1 + labeled_logvar - unlabeled_mu.pow(2) - labeled_logvar.exp()
            )

            scores = unlabeled_scores.squeeze()
        else:
            mu, logvar = self.vae(unlabeled_embeddings)
            scores = self.discriminator(mu).squeeze()

        _, indices = torch.topk(scores, n_query)

        return indices

    def _get_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.feature_extractor(features)
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
        return embeddings


class AdversarialDeepFoolingStrategy(AdvancedQueryStrategy):
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Optional[nn.Module] = None,
        max_iter: int = 50,
        overshoot: float = 0.02,
        epsilon: float = 1e-4,
    ):
        super().__init__(model)
        self.feature_extractor = feature_extractor or model
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.epsilon = epsilon

    def query(
        self,
        unlabeled_features: torch.Tensor,
        n_query: int,
        labeled_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.model.eval()

        perturbations = []

        for i in range(unlabeled_features.size(0)):
            x = unlabeled_features[i : i + 1].clone().detach().requires_grad_(True)

            pert = self._deep_fool(x)
            perturbations.append(pert)

        perturbations = torch.cat(perturbations, dim=0)

        perturbation_norms = torch.norm(
            perturbations.view(perturbations.size(0), -1), p=2, dim=1
        )

        _, indices = torch.topk(perturbation_norms, n_query)

        return indices

    def _deep_fool(self, x: torch.Tensor) -> torch.Tensor:
        input_dim = x.view(1, -1).size(1)

        x_adv = x.clone().detach().requires_grad_(True)

        self.model.eval()

        for iteration in range(self.max_iter):
            output = self.model(x_adv)

            if output.dim() > 1:
                output = output.squeeze()

            num_classes = output.size(-1)

            if num_classes < 2:
                break

            current_class = output.argmax(dim=-1)

            perturbations = []

            for k in range(num_classes):
                if k == current_class:
                    continue

                self.model.zero_grad()

                one_hot = torch.zeros_like(output)
                one_hot[0, k] = 1

                (output * one_hot).sum().backward(retain_graph=True)

                if x_adv.grad is not None:
                    grad_k = x_adv.grad.data.clone()
                else:
                    grad_k = torch.zeros_like(x_adv)

                pert_k = torch.abs(output[0, k] - output[0, current_class]) / (
                    torch.norm(grad_k.view(1, -1)) + 1e-10
                )

                perturbations.append((pert_k, grad_k))

            if not perturbations:
                break

            perturbations.sort(key=lambda x: x[0])

            min_pert, min_grad = perturbations[0]

            if min_pert < self.epsilon:
                break

            r = (min_pert + self.overshoot) * min_grad / (torch.norm(min_grad) + 1e-10)

            x_adv = x_adv + r.view_as(x_adv)
            x_adv = torch.clamp(x_adv, 0, 1)

        perturbation = x_adv - x
        return perturbation

    def _get_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.feature_extractor(features)
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
        return embeddings
