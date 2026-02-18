"""
BALD (Bayesian Active Learning by Disagreement) Implementation

Active learning using Bayesian uncertainty estimation.
"""

from typing import Optional, Tuple, List, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax


class BALD:
    """Bayesian Active Learning by Disagreement.

    Args:
        model: Base model
        n_samples: Number of Monte Carlo samples for dropout
        dropout_rate: Dropout rate for MC sampling
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
        dropout_rate: float = 0.5,
    ):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout at inference time."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def score(self, x: Tensor) -> Tensor:
        """Compute BALD score for each sample.

        Args:
            x: Input data

        Returns:
            BALD scores (higher = more informative)
        """
        self.model.eval()

        all_probs = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)

        expected_entropy = (
            -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1).mean(dim=0)
        )

        mean_probs = all_probs.mean(dim=0)
        entropy_expected = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        bald_score = entropy_expected - expected_entropy

        return bald_score

    def query(self, unlabeled_x: Tensor, n_query: int) -> Tuple[Tensor, Tensor]:
        """Query most informative samples.

        Args:
            unlabeled_x: Unlabeled input data
            n_query: Number of samples to query

        Returns:
            Tuple of (indices, scores)
        """
        scores = self.score(unlabeled_x)

        _, top_indices = scores.topk(min(n_query, scores.size(0)))

        return top_indices, scores[top_indices]


class BatchBALD:
    """Batch Bayesian Active Learning by Disagreement.

    Efficient batch selection for active learning.

    Args:
        model: Base model
        n_samples: Number of MC samples
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
        dropout_rate: float = 0.5,
    ):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        self._enable_dropout()

    def _enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def compute_bald_score(self, probs: Tensor) -> Tensor:
        """Compute BALD score from probability samples.

        Args:
            probs: Tensor of shape (n_samples, batch_size, n_classes)

        Returns:
            BALD scores
        """
        expected_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean(dim=0)

        mean_probs = probs.mean(dim=0)
        entropy_expected = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        bald_score = entropy_expected - expected_entropy

        return bald_score

    def get_probs(self, x: Tensor) -> Tensor:
        """Get probability samples using MC dropout.

        Args:
            x: Input data

        Returns:
            Tensor of shape (n_samples, batch_size, n_classes)
        """
        self.model.eval()

        all_probs = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        return torch.stack(all_probs, dim=0)

    def greedy_select(
        self,
        unlabeled_x: Tensor,
        n_query: int,
        batch_size: int = 100,
    ) -> Tuple[Tensor, List[int]]:
        """Greedy BatchBALD selection.

        Args:
            unlabeled_x: Unlabeled data
            n_query: Number of samples to select
            batch_size: Batch size for computation

        Returns:
            Tuple of (selected indices, selected probs)
        """
        n_unlabeled = unlabeled_x.size(0)
        selected_indices: List[int] = []
        selected_probs: List[Tensor] = []

        remaining_mask = torch.ones(n_unlabeled, dtype=torch.bool)

        for i in range(n_query):
            remaining_indices = torch.where(remaining_mask)[0]

            if remaining_indices.size(0) == 0:
                break

            batch_remaining = unlabeled_x[remaining_indices]
            probs_remaining = self.get_probs(batch_remaining)

            mutual_infos = []

            for idx_in_batch in range(len(remaining_indices)):
                test_indices = selected_indices + [
                    remaining_indices[idx_in_batch].item()
                ]

                test_probs_list = []
                for s in range(self.n_samples):
                    sample_probs = probs_remaining[idx_in_batch].unsqueeze(0)
                    test_probs_list.append(sample_probs)

                if len(selected_probs) > 0:
                    combined_probs = torch.cat(
                        [
                            torch.stack(selected_probs, dim=0),
                            test_probs_list[0].unsqueeze(0),
                        ],
                        dim=0,
                    )
                else:
                    combined_probs = torch.stack(test_probs_list, dim=0)

                mi = self.compute_bald_score(combined_probs)
                mutual_infos.append(mi[0].item())

            mutual_infos = torch.tensor(mutual_infos)
            best_local_idx = mutual_infos.argmax()
            best_global_idx = remaining_indices[best_local_idx].item()

            selected_indices.append(best_global_idx)
            remaining_mask[best_global_idx] = False

            probs_for_idx = self.get_probs(
                unlabeled_x[best_global_idx : best_global_idx + 1]
            )
            selected_probs.append(probs_for_idx.squeeze(0))

        return torch.tensor(selected_indices), selected_probs


class ExpectedEntropySearch:
    """Expected Entropy Search (EES) for active learning.

    Args:
        model: Base model
        n_samples: Number of MC samples
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
    ):
        self.model = model
        self.n_samples = n_samples

        self._enable_dropout()

    def _enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def compute_expected_entropy(self, probs: Tensor) -> Tensor:
        """Compute expected entropy of predictions.

        Args:
            probs: Tensor of shape (n_samples, batch_size, n_classes)

        Returns:
            Expected entropy
        """
        return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean(dim=0)

    def score(self, x: Tensor) -> Tensor:
        """Compute EES score (lower is better).

        Args:
            x: Input data

        Returns:
            EES scores
        """
        self.model.eval()

        all_probs = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)

        expected_entropy = self.compute_expected_entropy(all_probs)

        return expected_entropy

    def query(self, unlabeled_x: Tensor, n_query: int) -> Tuple[Tensor, Tensor]:
        """Query samples with lowest expected entropy.

        Args:
            unlabeled_x: Unlabeled input data
            n_query: Number of samples to query

        Returns:
            Tuple of (indices, scores)
        """
        scores = self.score(unlabeled_x)

        _, top_indices = scores.topk(n_query, largest=False)

        return top_indices, scores[top_indices]


class VarianceReduction:
    """Variance reduction based active learning.

    Args:
        model: Base model
        n_samples: Number of MC samples
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
    ):
        self.model = model
        self.n_samples = n_samples

    def score(self, x: Tensor) -> Tensor:
        """Compute variance reduction score.

        Args:
            x: Input data

        Returns:
            Variance scores (higher = more informative)
        """
        self.model.eval()

        all_probs = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)

        variance = all_probs.var(dim=0).mean(dim=-1)

        return variance

    def query(self, unlabeled_x: Tensor, n_query: int) -> Tuple[Tensor, Tensor]:
        """Query samples with highest variance.

        Args:
            unlabeled_x: Unlabeled input data
            n_query: Number of samples to query

        Returns:
            Tuple of (indices, scores)
        """
        scores = self.score(unlabeled_x)

        _, top_indices = scores.topk(n_query)

        return top_indices, scores[top_indices]


class BALDWithAbstention:
    """BALD with option to abstain from prediction.

    Args:
        model: Base model
        n_samples: Number of MC samples
        abstention_threshold: Threshold for abstention
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
        abstention_threshold: float = 0.5,
    ):
        self.model = model
        self.n_samples = n_samples
        self.abstention_threshold = abstention_threshold

        self._enable_dropout()

    def _enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_abstention(self, x: Tensor) -> Dict[str, Tensor]:
        """Predict with option to abstain.

        Args:
            x: Input data

        Returns:
            Dictionary with predictions and abstention flags
        """
        self.model.eval()

        all_probs = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)

        mean_probs = all_probs.mean(dim=0)
        max_probs, predictions = mean_probs.max(dim=-1)

        expected_entropy = (
            -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1).mean(dim=0)
        )

        mean_probs_2 = all_probs.mean(dim=0)
        entropy_expected = -(mean_probs_2 * torch.log(mean_probs_2 + 1e-10)).sum(dim=-1)

        bald_score = entropy_expected - expected_entropy

        abstain = bald_score > self.abstention_threshold

        return {
            "predictions": predictions,
            "max_probs": max_probs,
            "confidence": max_probs,
            "bald_score": bald_score,
            "abstain": abstain,
        }


class CoreSetSelection:
    """Core set based active learning selection.

    Args:
        model: Feature extractor model
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_distance_matrix(self, features: Tensor) -> Tensor:
        """Compute pairwise distance matrix.

        Args:
            features: Feature embeddings

        Returns:
            Distance matrix
        """
        features_norm = features / features.norm(dim=1, keepdim=True)

        dist_matrix = 1 - torch.mm(features_norm, features_norm.t())

        return dist_matrix

    def greedy_k_center(
        self,
        labeled_features: Tensor,
        unlabeled_features: Tensor,
        n_query: int,
    ) -> Tensor:
        """Greedy k-center selection.

        Args:
            labeled_features: Features of labeled data
            unlabeled_features: Features of unlabeled data
            n_query: Number of samples to select

        Returns:
            Selected indices
        """
        all_features = torch.cat([labeled_features, unlabeled_features], dim=0)
        n_labeled = labeled_features.size(0)
        n_unlabeled = unlabeled_features.size(0)

        dist_matrix = self.compute_distance_matrix(all_features)

        selected = torch.zeros(n_labeled + n_unlabeled, dtype=torch.bool)
        selected[:n_labeled] = True

        min_distances = dist_matrix[:n_labeled, n_labeled:].min(dim=0)[0]

        selected_indices = []

        for _ in range(n_query):
            furthest_idx = min_distances.argmax()
            selected_indices.append(furthest_idx)

            new_dist = dist_matrix[n_labeled + furthest_idx, n_labeled:]
            min_distances = torch.min(min_distances, new_dist)

        return torch.tensor(selected_indices)

    def query(
        self,
        labeled_x: Tensor,
        unlabeled_x: Tensor,
        n_query: int,
    ) -> Tuple[Tensor, Tensor]:
        """Query using core set selection.

        Args:
            labeled_x: Labeled input data
            unlabeled_x: Unlabeled input data
            n_query: Number of samples to query

        Returns:
            Tuple of (indices, distances)
        """
        self.model.eval()

        with torch.no_grad():
            labeled_features = self.model(labeled_x)
            unlabeled_features = self.model(unlabeled_x)

        selected = self.greedy_k_center(labeled_features, unlabeled_features, n_query)

        return selected, torch.zeros(n_query)


class AdaptiveBALD:
    """Adaptive BALD that adjusts exploration vs exploitation.

    Args:
        model: Base model
        n_samples: Number of MC samples
        exploration_weight: Weight for exploration term
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
        exploration_weight: float = 0.1,
    ):
        self.model = model
        self.n_samples = n_samples
        self.exploration_weight = exploration_weight

        self._enable_dropout()

    def _enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def score(self, x: Tensor) -> Tensor:
        """Compute adaptive BALD score.

        Args:
            x: Input data

        Returns:
            Adaptive scores
        """
        self.model.eval()

        all_probs = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)

        expected_entropy = (
            -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1).mean(dim=0)
        )

        mean_probs = all_probs.mean(dim=0)
        entropy_expected = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        bald_score = entropy_expected - expected_entropy

        confidence = mean_probs.max(dim=-1)[0]

        adaptive_score = bald_score + self.exploration_weight * (1 - confidence)

        return adaptive_score

    def query(self, unlabeled_x: Tensor, n_query: int) -> Tuple[Tensor, Tensor]:
        """Query using adaptive BALD.

        Args:
            unlabeled_x: Unlabeled input data
            n_query: Number of samples to query

        Returns:
            Tuple of (indices, scores)
        """
        scores = self.score(unlabeled_x)

        _, top_indices = scores.topk(n_query)

        return top_indices, scores[top_indices]
