"""
Active Learning Module

Query strategies for efficient labeling.
"""

from typing import Optional, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


class QueryStrategy:
    """Base class for query strategies."""

    def __init__(self, model: nn.Module):
        self.model = model

    def score(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class UncertaintySampling(QueryStrategy):
    """Uncertainty sampling query strategy."""

    def score(self, x: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            uncertainties = 1 - probs.max(dim=-1)[0]
        return uncertainties


class MarginSampling(QueryStrategy):
    """Margin sampling: difference between top 2 class probabilities."""

    def score(self, x: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            sorted_probs, _ = probs.sort(dim=-1, descending=True)
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        return -margins


class EntropySampling(QueryStrategy):
    """Entropy-based uncertainty sampling."""

    def score(self, x: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropies


class BALD:
    """Bayesian Active Learning by Disagreement (BALD).

    Args:
        model: Model with dropout for Bayesian approximation
        n_samples: Number of MC samples
    """

    def __init__(self, model: nn.Module, n_samples: int = 10):
        self.model = model
        self.n_samples = n_samples

    def score(self, x: Tensor) -> Tensor:
        self.model.eval()
        probs_list = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)

        probs = torch.stack(probs_list)
        mean_probs = probs.mean(dim=0)

        entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        mean_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean(dim=0)

        bald = entropy_mean - mean_entropy
        return -bald


class CoreSet:
    """CoreSet: greedy selection based on feature distances.

    Args:
        model: Feature extractor
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.selected_indices = []
        self.features = []

    def score(self, x: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            features = self.model(x)

        if len(self.selected_indices) == 0:
            return torch.zeros(len(x))

        selected_features = torch.stack(
            [self.features[i] for i in self.selected_indices]
        )
        distances = torch.cdist(features, selected_features)
        min_distances = distances.min(dim=1)[0]

        return -min_distances

    def select(self, x: Tensor, n: int = 1) -> List[int]:
        scores = self.score(x)
        indices = scores.argsort()[-n:].tolist()
        self.selected_indices.extend(indices)

        with torch.no_grad():
            self.features.append(self.model(x[indices[0] : indices[0] + 1]))

        return indices
