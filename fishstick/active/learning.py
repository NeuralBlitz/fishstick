"""
Comprehensive Active Learning Module

This module implements state-of-the-art active learning algorithms including:
- Query strategies (uncertainty, margin, entropy, diversity)
- Uncertainty estimation (MC dropout, ensembles, Bayesian, evidential)
- Batch active learning (BatchBALD, CoreSet, BADGE)
- Diversity sampling (K-center, K-means, adversarial)
- Expected model change (EGL, BALD, variation ratio)
- Pool-based and stream-based active learning
- Multi-task active learning
- Evaluation and visualization tools

References:
    [1] Settles, B. (2009). Active learning literature survey.
    [2] Gal, Y., et al. (2017). Deep Bayesian Active Learning with Image Data.
    [3] Kirsch, A., et al. (2019). BatchBALD: Efficient and Diverse Batch Acquisition.
    [4] Sener, O., & Savarese, S. (2018). Active Learning for Convolutional Neural Networks.
    [5] Ash, J. T., et al. (2020). Deep Batch Active Learning by Diverse, Uncertain Gradient.
"""

from typing import (
    Optional,
    List,
    Tuple,
    Callable,
    Dict,
    Union,
    Any,
    Protocol,
    Iterator,
    Sequence,
    NamedTuple,
    Set,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import warnings
from collections import defaultdict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment


# =============================================================================
# Type Definitions
# =============================================================================


class UncertaintyType(Enum):
    """Types of uncertainty estimation methods."""

    LEAST_CONFIDENT = "least_confident"
    MARGIN = "margin"
    ENTROPY = "entropy"
    BALD = "bald"
    VARIATION_RATIO = "variation_ratio"
    MEAN_STD = "mean_std"


class QueryType(Enum):
    """Types of query strategies."""

    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"
    EXPECTED_MODEL_CHANGE = "expected_model_change"


@dataclass
class QueryResult:
    """Result of a query operation."""

    indices: List[int]
    scores: np.ndarray
    query_time: float
    strategy_name: str
    batch_size: int


@dataclass
class ActiveLearningState:
    """State of the active learning process."""

    labeled_indices: Set[int]
    unlabeled_indices: Set[int]
    iteration: int
    labeled_size: int
    unlabeled_size: int
    performance_history: List[Dict[str, float]]


class ScoreFunction(Protocol):
    """Protocol for score functions."""

    def __call__(self, model: nn.Module, data: Tensor) -> Tensor: ...


# =============================================================================
# Base Classes
# =============================================================================


class QueryStrategy(ABC):
    """Abstract base class for query strategies.

    All query strategies must implement the `score` and `select` methods.

    Args:
        model: The neural network model to use for querying
        device: Device to run computations on
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.name = self.__class__.__name__

    @abstractmethod
    def score(self, x: Tensor) -> Tensor:
        """Compute acquisition scores for input samples.

        Args:
            x: Input tensor of shape (batch_size, ...)

        Returns:
            Tensor of scores with shape (batch_size,)
        """
        pass

    def select(
        self, x: Tensor, n: int = 1, exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[int], Tensor]:
        """Select top-n samples based on scores.

        Args:
            x: Input tensor
            n: Number of samples to select
            exclude_indices: Indices to exclude from selection

        Returns:
            Tuple of (selected indices, scores tensor)
        """
        scores = self.score(x)

        if exclude_indices:
            mask = torch.ones(len(scores), dtype=torch.bool)
            mask[list(exclude_indices)] = False
            scores_filtered = scores[mask]
            indices_filtered = torch.where(mask)[0]

            if len(scores_filtered) < n:
                n = len(scores_filtered)

            top_indices = scores_filtered.argsort(descending=True)[:n]
            selected = indices_filtered[top_indices].tolist()
        else:
            selected = scores.argsort(descending=True)[:n].tolist()

        return selected, scores

    def reset(self) -> None:
        """Reset the strategy state."""
        pass


class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimation.

    Uncertainty estimators compute various forms of uncertainty
    from model predictions.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def estimate(self, x: Tensor) -> Dict[str, Tensor]:
        """Estimate uncertainty for input samples.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing uncertainty metrics
        """
        pass


class BatchQueryStrategy(ABC):
    """Abstract base class for batch active learning strategies.

    Batch strategies consider interactions between samples when selecting
    a batch, promoting diversity and reducing redundancy.
    """

    def __init__(self, model: nn.Module, batch_size: int = 10):
        self.model = model
        self.batch_size = batch_size
        self.name = self.__class__.__name__

    @abstractmethod
    def select_batch(self, x: Tensor, n: int) -> Tuple[List[int], Tensor]:
        """Select a batch of samples.

        Args:
            x: Input tensor of all unlabeled samples
            n: Number of samples to select

        Returns:
            Tuple of (selected indices, batch scores)
        """
        pass


# =============================================================================
# Query Strategies
# =============================================================================


class UncertaintySampling(QueryStrategy):
    """Least confidence uncertainty sampling.

    Selects samples where the model is least confident in its prediction.
    Score = 1 - max(P(y|x))

    Args:
        model: Neural network classifier
        device: Computation device
    """

    def score(self, x: Tensor) -> Tensor:
        """Compute uncertainty scores using least confidence."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]
            uncertainties = 1.0 - max_probs

        return uncertainties.cpu()

    def get_predictions(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Get both predictions and confidence scores.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predicted classes, confidence scores)
        """
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)

        return predictions.cpu(), confidences.cpu()


class MarginSampling(QueryStrategy):
    """Margin sampling strategy.

    Selects samples with smallest margin between top two class probabilities.
    Score = - (P(y_1|x) - P(y_2|x))

    Args:
        model: Neural network classifier
        device: Computation device
    """

    def score(self, x: Tensor) -> Tensor:
        """Compute margin scores (negative margin for sorting)."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            sorted_probs, _ = probs.sort(dim=-1, descending=True)

            if sorted_probs.shape[-1] < 2:
                # Only one class
                margins = torch.zeros(len(x), device=x.device)
            else:
                margins = sorted_probs[:, 0] - sorted_probs[:, 1]

        return (-margins).cpu()

    def get_margins(self, x: Tensor) -> Tensor:
        """Get actual margin values (positive).

        Args:
            x: Input tensor

        Returns:
            Margin values
        """
        scores = self.score(x)
        return -scores


class EntropySampling(QueryStrategy):
    """Entropy-based uncertainty sampling.

    Selects samples with highest predictive entropy.
    Score = -sum_i P(y_i|x) * log(P(y_i|x))

    Args:
        model: Neural network classifier
        device: Computation device
    """

    def __init__(
        self, model: nn.Module, device: Optional[str] = None, epsilon: float = 1e-10
    ):
        super().__init__(model, device)
        self.epsilon = epsilon

    def score(self, x: Tensor) -> Tensor:
        """Compute entropy-based uncertainty scores."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + self.epsilon)
            entropies = -(probs * log_probs).sum(dim=-1)

        return entropies.cpu()

    def get_entropy_distribution(self, x: Tensor) -> Dict[str, float]:
        """Get statistics of entropy distribution.

        Args:
            x: Input tensor

        Returns:
            Dictionary with entropy statistics
        """
        scores = self.score(x)
        return {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "median": float(scores.median()),
        }


class RandomSampling(QueryStrategy):
    """Random sampling baseline.

    Selects samples uniformly at random. Useful as a baseline for comparison.

    Args:
        model: Neural network classifier (not used, but kept for API consistency)
        device: Computation device
        seed: Random seed for reproducibility
    """

    def __init__(
        self, model: nn.Module, device: Optional[str] = None, seed: Optional[int] = None
    ):
        super().__init__(model, device)
        self.rng = np.random.RandomState(seed)

    def score(self, x: Tensor) -> Tensor:
        """Return random scores."""
        return torch.from_numpy(self.rng.random(len(x))).float()

    def select(
        self, x: Tensor, n: int = 1, exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[int], Tensor]:
        """Select random samples."""
        scores = self.score(x)

        available_indices = list(set(range(len(x))) - (exclude_indices or set()))

        if len(available_indices) < n:
            n = len(available_indices)

        selected = self.rng.choice(available_indices, size=n, replace=False).tolist()
        return selected, scores


class ClusterBasedSampling(QueryStrategy):
    """Diversity sampling using clustering.

    Selects samples that represent different clusters in the feature space,
    promoting diversity in the labeled set.

    Args:
        model: Feature extractor (last layer before classification)
        n_clusters: Number of clusters for diversity
        device: Computation device
    """

    def __init__(
        self, model: nn.Module, n_clusters: int = 10, device: Optional[str] = None
    ):
        super().__init__(model, device)
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.cluster_centers: Optional[np.ndarray] = None

    def extract_features(self, x: Tensor) -> np.ndarray:
        """Extract features from input using the model."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            # Try to get features before the final layer
            if hasattr(self.model, "get_features"):
                features = self.model.get_features(x)
            elif hasattr(self.model, "features"):
                features = self.model.features(x)
            else:
                # Use penultimate layer activations
                features = x.view(x.size(0), -1)  # Fallback to flattened input

        return features.cpu().numpy()

    def score(self, x: Tensor) -> Tensor:
        """Score samples by distance to cluster centers."""
        features = self.extract_features(x)

        if self.kmeans is None:
            # Initialize k-means
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.kmeans.fit(features)
            self.cluster_centers = self.kmeans.cluster_centers_

        # Score by negative distance to nearest cluster center
        distances = pairwise_distances(features, self.cluster_centers)
        min_distances = distances.min(axis=1)

        return torch.from_numpy(min_distances).float()

    def select_diverse_batch(self, x: Tensor, n: int) -> Tuple[List[int], Tensor]:
        """Select diverse samples from different clusters.

        Args:
            x: Input tensor
            n: Number of samples to select

        Returns:
            Tuple of (selected indices, scores)
        """
        features = self.extract_features(x)

        # Cluster the data
        kmeans = KMeans(n_clusters=min(self.n_clusters, n), random_state=42)
        labels = kmeans.fit_predict(features)

        selected_indices = []
        for cluster_id in range(kmeans.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Select sample closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(
                    features[cluster_indices] - cluster_center, axis=1
                )
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)

            if len(selected_indices) >= n:
                break

        scores = self.score(x)
        return selected_indices[:n], scores


class DensityWeightedSampling(QueryStrategy):
    """Density-weighted uncertainty sampling.

    Combines uncertainty with density to avoid outliers.
    Score = Uncertainty(x) * Density(x)

    Args:
        model: Neural network classifier
        density_estimator: Method to estimate density ('kde' or 'knn')
        k: Number of neighbors for density estimation
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        density_estimator: str = "knn",
        k: int = 5,
        device: Optional[str] = None,
    ):
        super().__init__(model, device)
        self.density_estimator = density_estimator
        self.k = k
        self.uncertainty_strategy = EntropySampling(model, device)

    def estimate_density(self, x: Tensor) -> Tensor:
        """Estimate local density of samples.

        Args:
            x: Input tensor

        Returns:
            Density scores (higher = denser region)
        """
        x_flat = x.view(x.size(0), -1).cpu().numpy()

        # Compute k-nearest neighbor distances
        distances = pairwise_distances(x_flat)
        k_distances = np.partition(distances, self.k + 1, axis=1)[:, 1 : self.k + 1]
        mean_k_distances = k_distances.mean(axis=1)

        # Density is inverse of distance
        density = 1.0 / (mean_k_distances + 1e-10)
        density = density / density.sum()  # Normalize

        return torch.from_numpy(density).float()

    def score(self, x: Tensor) -> Tensor:
        """Compute density-weighted uncertainty scores."""
        uncertainty = self.uncertainty_strategy.score(x)
        density = self.estimate_density(x)

        # Weight uncertainty by density
        weighted_scores = uncertainty * density

        return weighted_scores


# =============================================================================
# Uncertainty Estimation
# =============================================================================


class MCDropoutUncertainty(UncertaintyEstimator):
    """Monte Carlo Dropout uncertainty estimation.

    Uses dropout at test time to approximate Bayesian inference.

    Reference:
        Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.

    Args:
        model: Neural network with dropout layers
        n_samples: Number of MC forward passes
        dropout_rate: Dropout rate for MC sampling
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__(model)
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def enable_dropout(self, module: nn.Module) -> None:
        """Enable dropout layers during evaluation."""
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()

    def estimate(self, x: Tensor) -> Dict[str, Tensor]:
        """Estimate uncertainty using MC dropout.

        Args:
            x: Input tensor

        Returns:
            Dictionary with:
                - mean: Mean prediction probabilities
                - variance: Predictive variance
                - entropy: Predictive entropy
                - mutual_info: Mutual information (epistemic uncertainty)
        """
        self.model.eval()
        self.model.apply(self.enable_dropout)

        x = x.to(next(self.model.parameters()).device)
        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions)  # (n_samples, batch, n_classes)

        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        # Total uncertainty (predictive entropy)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)

        # Expected entropy (aleatoric uncertainty)
        expected_entropy = (
            -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1).mean(dim=0)
        )

        # Mutual information (epistemic uncertainty)
        mutual_info = entropy - expected_entropy

        return {
            "mean": mean_pred,
            "variance": variance,
            "entropy": entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_info,
            "epistemic": mutual_info,
            "aleatoric": expected_entropy,
            "all_predictions": predictions,
        }

    def get_uncertainty_scores(
        self, x: Tensor, uncertainty_type: str = "entropy"
    ) -> Tensor:
        """Get specific uncertainty scores.

        Args:
            x: Input tensor
            uncertainty_type: Type of uncertainty ('entropy', 'mutual_info', 'variance')

        Returns:
            Uncertainty scores
        """
        estimates = self.estimate(x)
        return estimates[uncertainty_type]


class EnsembleUncertainty(UncertaintyEstimator):
    """Deep ensemble uncertainty estimation.

    Uses multiple independently trained models to estimate uncertainty.

    Reference:
        Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive
        Uncertainty Estimation using Deep Ensembles.

    Args:
        models: List of neural network models
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__(models[0])
        self.models = models
        self.n_models = len(models)

    def estimate(self, x: Tensor) -> Dict[str, Tensor]:
        """Estimate uncertainty using deep ensemble.

        Args:
            x: Input tensor

        Returns:
            Dictionary with uncertainty estimates
        """
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions)

        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)

        # Epistemic uncertainty (mutual information)
        expected_entropy = (
            -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1).mean(dim=0)
        )
        mutual_info = entropy - expected_entropy

        return {
            "mean": mean_pred,
            "variance": variance,
            "entropy": entropy,
            "mutual_information": mutual_info,
            "epistemic": mutual_info,
            "aleatoric": expected_entropy,
            "all_predictions": predictions,
            "disagreement": predictions.std(dim=0).mean(dim=-1),
        }

    def compute_disagreement(self, x: Tensor) -> Tensor:
        """Compute prediction disagreement among ensemble members.

        Args:
            x: Input tensor

        Returns:
            Disagreement scores
        """
        estimates = self.estimate(x)
        return estimates["disagreement"]


class BayesianUncertainty(UncertaintyEstimator):
    """Bayesian Neural Network uncertainty.

    Uses variational inference to learn weight distributions.

    Args:
        model: Bayesian neural network with probabilistic layers
        n_samples: Number of weight samples
    """

    def __init__(self, model: nn.Module, n_samples: int = 100):
        super().__init__(model)
        self.n_samples = n_samples

    def estimate(self, x: Tensor) -> Dict[str, Tensor]:
        """Estimate uncertainty using BNN."""
        predictions = []

        self.model.train()  # Enable stochastic forward passes

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions)

        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)

        return {
            "mean": mean_pred,
            "variance": variance,
            "entropy": entropy,
            "epistemic": variance.sum(dim=-1),
            "all_predictions": predictions,
        }


class EvidentialUncertainty(UncertaintyEstimator):
    """Evidential Deep Learning uncertainty.

    Uses subjective logic to model uncertainty through Dirichlet distributions.

    Reference:
        Sensoy, M., et al. (2018). Evidential Deep Learning to Quantify
        Classification Uncertainty.

    Args:
        model: Neural network outputting Dirichlet parameters (alpha)
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def estimate(self, x: Tensor) -> Dict[str, Tensor]:
        """Estimate uncertainty using evidential learning.

        Args:
            x: Input tensor

        Returns:
            Dictionary with evidential uncertainty estimates
        """
        self.model.eval()

        with torch.no_grad():
            alpha = self.model(x)
            alpha = F.softplus(alpha) + 1.0  # Ensure alpha > 1

        # Dirichlet strength
        strength = alpha.sum(dim=-1)

        # Expected probability
        expected_prob = alpha / strength.unsqueeze(-1)

        # Uncertainty measures
        entropy = -(expected_prob * torch.log(expected_prob + 1e-10)).sum(dim=-1)

        # Vacuity (uncertainty due to lack of evidence)
        n_classes = alpha.shape[-1]
        vacuity = n_classes / strength

        # Dissonance (uncertainty due to conflicting evidence)
        belief = (alpha - 1) / strength.unsqueeze(-1)
        dissonance = self._compute_dissonance(belief)

        return {
            "alpha": alpha,
            "expected_probability": expected_prob,
            "strength": strength,
            "entropy": entropy,
            "vacuity": vacuity,
            "dissonance": dissonance,
            "uncertainty": vacuity + dissonance,
        }

    def _compute_dissonance(self, belief: Tensor) -> Tensor:
        """Compute dissonance from belief distribution."""
        n_classes = belief.shape[-1]

        dissonance = torch.zeros(belief.shape[0], device=belief.device)

        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    # Similarity between beliefs
                    similarity = 1 - torch.abs(belief[:, i] - belief[:, j])
                    dissonance += belief[:, i] * belief[:, j] * similarity

        return dissonance


# =============================================================================
# Batch Active Learning
# =============================================================================


class BatchBALD(BatchQueryStrategy):
    """Batch Bayesian Active Learning by Disagreement.

    Selects batches that maximize information gain about model parameters.

    Reference:
        Kirsch, A., et al. (2019). BatchBALD: Efficient and Diverse
        Batch Acquisition for Deep Bayesian Active Learning.

    Args:
        model: Neural network with MC dropout
        n_samples: Number of MC dropout samples
        num_subsamples: Number of samples for approximation
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 10,
        n_samples: int = 100,
        num_subsamples: int = 10000,
    ):
        super().__init__(model, batch_size)
        self.n_samples = n_samples
        self.num_subsamples = num_subsamples

    def compute_bald_score(self, x: Tensor) -> Tensor:
        """Compute BALD scores for individual samples."""
        mc_dropout = MCDropoutUncertainty(self.model, self.n_samples)
        estimates = mc_dropout.estimate(x)
        return estimates["mutual_information"]

    def select_batch(self, x: Tensor, n: int) -> Tuple[List[int], Tensor]:
        """Select batch using BatchBALD.

        Args:
            x: Input tensor of all unlabeled samples
            n: Batch size

        Returns:
            Tuple of (selected indices, scores)
        """
        # Get MC dropout predictions
        mc_dropout = MCDropoutUncertainty(self.model, self.n_samples)
        estimates = mc_dropout.estimate(x)
        predictions = estimates["all_predictions"]  # (n_samples, batch, n_classes)

        # Greedy batch selection
        selected_indices = []
        available_indices = list(range(len(x)))

        for _ in range(n):
            best_score = -float("inf")
            best_idx = -1

            for idx in available_indices:
                # Compute joint BALD for current selection + candidate
                candidate_indices = selected_indices + [idx]
                score = self._compute_joint_bald(predictions, candidate_indices)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected_indices.append(best_idx)
            available_indices.remove(best_idx)

        # Compute final scores
        final_scores = self.compute_bald_score(x)

        return selected_indices, final_scores

    def _compute_joint_bald(self, predictions: Tensor, indices: List[int]) -> float:
        """Compute joint BALD score for a set of indices."""
        if len(indices) == 0:
            return 0.0

        # Get predictions for selected indices
        selected_preds = predictions[:, indices, :]  # (n_samples, k, n_classes)

        # Compute joint entropy
        joint_preds = selected_preds.reshape(
            selected_preds.shape[0], -1
        )  # (n_samples, k * n_classes)

        mean_joint = joint_preds.mean(dim=0)
        entropy_mean = -(mean_joint * torch.log(mean_joint + 1e-10)).sum()

        mean_entropy = -(joint_preds * torch.log(joint_preds + 1e-10)).sum(dim=1).mean()

        bald = entropy_mean - mean_entropy

        return float(bald)


class CoreSet(BatchQueryStrategy):
    """Core-Set selection for active learning.

    Selects samples that cover the data distribution using k-center greedy.

    Reference:
        Sener, O., & Savarese, S. (2018). Active Learning for Convolutional
        Neural Networks: A Core-Set Approach.

    Args:
        model: Feature extractor
        batch_size: Selection batch size
        distance_metric: Distance metric ('euclidean' or 'cosine')
    """

    def __init__(
        self, model: nn.Module, batch_size: int = 10, distance_metric: str = "euclidean"
    ):
        super().__init__(model, batch_size)
        self.distance_metric = distance_metric
        self.selected_indices: Set[int] = set()
        self.features_history: List[Tensor] = []

    def extract_features(self, x: Tensor) -> np.ndarray:
        """Extract features from input."""
        self.model.eval()

        with torch.no_grad():
            # Try different feature extraction methods
            if hasattr(self.model, "get_features"):
                features = self.model.get_features(x)
            elif hasattr(self.model, "features"):
                features = self.model.features(x)
            elif hasattr(self.model, "encoder"):
                features = self.model.encoder(x)
            else:
                # Use flattened activations
                features = x.view(x.size(0), -1)

        return features.cpu().numpy()

    def compute_distances(self, features: np.ndarray) -> np.ndarray:
        """Compute pairwise distances."""
        return pairwise_distances(features, features, metric=self.distance_metric)

    def select_batch(self, x: Tensor, n: int) -> Tuple[List[int], Tensor]:
        """Select batch using k-center greedy algorithm.

        Args:
            x: Input tensor
            n: Number of samples to select

        Returns:
            Tuple of (selected indices, minimum distances)
        """
        features = self.extract_features(x)
        distances = self.compute_distances(features)

        n_samples = len(x)
        available = set(range(n_samples)) - self.selected_indices

        if len(self.selected_indices) == 0:
            # First selection: pick the sample with maximum distance to others
            avg_distances = distances.mean(axis=1)
            first_idx = int(np.argmax(avg_distances))
            self.selected_indices.add(first_idx)
            available.remove(first_idx)

        # K-center greedy selection
        while len(self.selected_indices) < n and available:
            # For each available point, compute distance to nearest selected point
            min_distances = []
            for i in available:
                min_dist = min(distances[i, j] for j in self.selected_indices)
                min_distances.append((i, min_dist))

            # Select point with maximum minimum distance
            best_idx, best_dist = max(min_distances, key=lambda x: x[1])
            self.selected_indices.add(best_idx)
            available.remove(best_idx)

        # Return newly selected indices
        new_indices = list(self.selected_indices)[-n:]

        # Compute scores (negative min distances for sorting)
        scores = torch.zeros(len(x))
        for i in range(len(x)):
            if self.selected_indices:
                min_dist = min(distances[i, j] for j in self.selected_indices)
                scores[i] = -min_dist

        return new_indices, scores

    def reset(self) -> None:
        """Reset selection state."""
        self.selected_indices = set()
        self.features_history = []


class BADGE(BatchQueryStrategy):
    """Batch Active learning by Diverse Gradient Embeddings.

    Combines uncertainty and diversity by clustering gradient embeddings.

    Reference:
        Ash, J. T., et al. (2020). Deep Batch Active Learning by Diverse,
        Uncertain Gradient Lower Bounds.

    Args:
        model: Neural network model
        batch_size: Selection batch size
        num_clusters: Number of clusters for diversity
    """

    def __init__(
        self, model: nn.Module, batch_size: int = 10, num_clusters: Optional[int] = None
    ):
        super().__init__(model, batch_size)
        self.num_clusters = num_clusters

    def compute_gradient_embeddings(
        self, x: Tensor, y: Optional[Tensor] = None
    ) -> np.ndarray:
        """Compute gradient embeddings for uncertainty and diversity.

        Args:
            x: Input tensor
            y: Optional pseudo-labels (if None, use model predictions)

        Returns:
            Gradient embeddings
        """
        self.model.eval()
        x = x.to(next(self.model.parameters()).device)

        # Get model predictions
        with torch.no_grad():
            logits = self.model(x)
            if y is None:
                y = logits.argmax(dim=-1)

        # Compute gradient embeddings
        embeddings = []

        for i in range(len(x)):
            self.model.zero_grad()

            # Forward pass
            logit = self.model(x[i : i + 1])

            # Cross-entropy loss
            loss = F.cross_entropy(logit, y[i : i + 1])

            # Compute gradients w.r.t. penultimate layer
            loss.backward()

            # Extract gradient embedding
            grad_embed = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_embed.append(param.grad.view(-1))

            if grad_embed:
                grad_embed = torch.cat(grad_embed).detach().cpu().numpy()
                embeddings.append(grad_embed)

        return np.array(embeddings)

    def select_batch(self, x: Tensor, n: int) -> Tuple[List[int], Tensor]:
        """Select batch using BADGE.

        Args:
            x: Input tensor
            n: Number of samples to select

        Returns:
            Tuple of (selected indices, scores)
        """
        # Compute gradient embeddings
        embeddings = self.compute_gradient_embeddings(x)

        # Use k-means++ initialization for diversity
        n_clusters = self.num_clusters or n
        n_clusters = min(n_clusters, len(x))

        # K-means++ initialization
        centers = self._kmeans_plus_plus(embeddings, n)

        # Assign samples to nearest centers
        distances = pairwise_distances(embeddings, embeddings[centers])
        selected = distances.argmin(axis=1)

        # Select one sample per cluster
        selected_indices = []
        for i in range(n):
            cluster_samples = np.where(selected == i)[0]
            if len(cluster_samples) > 0:
                # Select sample with highest uncertainty in cluster
                selected_indices.append(cluster_samples[0])

        # Dummy scores
        scores = torch.ones(len(x))

        return selected_indices[:n], scores

    def _kmeans_plus_plus(self, X: np.ndarray, k: int) -> List[int]:
        """K-means++ initialization."""
        n_samples = X.shape[0]
        centers = [np.random.randint(n_samples)]

        for _ in range(1, k):
            # Compute distances to nearest center
            dists = np.min(pairwise_distances(X, X[centers]), axis=1)

            # Select next center with probability proportional to distance^2
            probs = dists**2 / np.sum(dists**2)
            next_center = np.random.choice(n_samples, p=probs)
            centers.append(next_center)

        return centers


class BatchActive(BatchQueryStrategy):
    """Generic batch active learning with customizable components.

    Combines uncertainty and diversity objectives with configurable weights.

    Args:
        model: Neural network model
        batch_size: Selection batch size
        uncertainty_weight: Weight for uncertainty objective
        diversity_weight: Weight for diversity objective
        uncertainty_strategy: Strategy for uncertainty estimation
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 10,
        uncertainty_weight: float = 0.5,
        diversity_weight: float = 0.5,
        uncertainty_strategy: Optional[QueryStrategy] = None,
    ):
        super().__init__(model, batch_size)
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.uncertainty_strategy = uncertainty_strategy or EntropySampling(model)

    def select_batch(self, x: Tensor, n: int) -> Tuple[List[int], Tensor]:
        """Select batch combining uncertainty and diversity."""
        # Get uncertainty scores
        uncertainty_scores = self.uncertainty_strategy.score(x)

        # Get diversity features
        features = self._extract_features(x)

        # Greedy selection
        selected_indices = []
        available = set(range(len(x)))

        for _ in range(n):
            best_score = -float("inf")
            best_idx = -1

            for idx in available:
                # Uncertainty component
                unc_score = uncertainty_scores[idx].item()

                # Diversity component (distance to nearest selected)
                if selected_indices:
                    dists = pairwise_distances(
                        features[idx : idx + 1], features[selected_indices]
                    )
                    div_score = float(dists.min())
                else:
                    div_score = 1.0

                # Combined score
                combined = (
                    self.uncertainty_weight * unc_score
                    + self.diversity_weight * div_score
                )

                if combined > best_score:
                    best_score = combined
                    best_idx = idx

            selected_indices.append(best_idx)
            available.remove(best_idx)

        return selected_indices, uncertainty_scores

    def _extract_features(self, x: Tensor) -> np.ndarray:
        """Extract features for diversity computation."""
        self.model.eval()

        with torch.no_grad():
            if hasattr(self.model, "get_features"):
                features = self.model.get_features(x)
            else:
                features = x.view(x.size(0), -1)

        return features.cpu().numpy()


class GreedyBatch(BatchQueryStrategy):
    """Greedy batch selection with submodular maximization.

    Uses greedy algorithm to maximize submodular acquisition function.

    Args:
        model: Neural network model
        batch_size: Selection batch size
        objective: Objective function ('uncertainty', 'coverage', 'hybrid')
    """

    def __init__(
        self, model: nn.Module, batch_size: int = 10, objective: str = "hybrid"
    ):
        super().__init__(model, batch_size)
        self.objective = objective

    def select_batch(self, x: Tensor, n: int) -> Tuple[List[int], Tensor]:
        """Select batch using greedy submodular maximization."""
        selected_indices = []
        available = set(range(len(x)))

        # Pre-compute features
        features = self._get_features(x)
        uncertainty_scores = self._get_uncertainty(x)

        # Greedy selection
        for _ in range(n):
            best_marginal = -float("inf")
            best_idx = -1

            for idx in available:
                marginal = self._compute_marginal_gain(
                    idx, selected_indices, features, uncertainty_scores
                )

                if marginal > best_marginal:
                    best_marginal = marginal
                    best_idx = idx

            selected_indices.append(best_idx)
            available.remove(best_idx)

        scores = torch.tensor(uncertainty_scores)
        return selected_indices, scores

    def _get_features(self, x: Tensor) -> np.ndarray:
        """Extract features from input."""
        self.model.eval()

        with torch.no_grad():
            if hasattr(self.model, "get_features"):
                features = self.model.get_features(x)
            else:
                features = x.view(x.size(0), -1)

        return features.cpu().numpy()

    def _get_uncertainty(self, x: Tensor) -> np.ndarray:
        """Compute uncertainty scores."""
        strategy = EntropySampling(self.model)
        scores = strategy.score(x)
        return scores.numpy()

    def _compute_marginal_gain(
        self,
        idx: int,
        selected: List[int],
        features: np.ndarray,
        uncertainty: np.ndarray,
    ) -> float:
        """Compute marginal gain of adding a sample."""
        if self.objective == "uncertainty":
            return uncertainty[idx]

        elif self.objective == "coverage":
            if not selected:
                return 1.0

            # Distance to nearest selected point
            dists = pairwise_distances(features[idx : idx + 1], features[selected])
            return float(dists.min())

        else:  # hybrid
            unc_gain = uncertainty[idx]

            if not selected:
                div_gain = 1.0
            else:
                dists = pairwise_distances(features[idx : idx + 1], features[selected])
                div_gain = float(dists.min())

            return 0.5 * unc_gain + 0.5 * div_gain


# =============================================================================
# Diversity Sampling
# =============================================================================


class KCenterSampling(QueryStrategy):
    """K-center greedy sampling for diversity.

    Selects samples to minimize the maximum distance to any sample.

    Args:
        model: Feature extractor
        device: Computation device
        distance_metric: Distance metric for diversity
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        distance_metric: str = "euclidean",
    ):
        super().__init__(model, device)
        self.distance_metric = distance_metric
        self.selected_features: List[np.ndarray] = []

    def extract_features(self, x: Tensor) -> np.ndarray:
        """Extract features from input."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "get_features"):
                features = self.model.get_features(x)
            else:
                features = x.view(x.size(0), -1)

        return features.cpu().numpy()

    def score(self, x: Tensor) -> Tensor:
        """Score by distance to nearest selected point."""
        features = self.extract_features(x)

        if len(self.selected_features) == 0:
            # First selection - use farthest from origin
            distances = np.linalg.norm(features, axis=1)
            return torch.from_numpy(distances).float()

        # Distance to nearest selected
        selected_array = np.array(self.selected_features)
        distances = pairwise_distances(
            features, selected_array, metric=self.distance_metric
        )
        min_distances = distances.min(axis=1)

        return torch.from_numpy(min_distances).float()

    def select(
        self, x: Tensor, n: int = 1, exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[int], Tensor]:
        """Select samples using k-center greedy."""
        selected = []

        for _ in range(n):
            scores = self.score(x)

            if exclude_indices:
                mask = torch.ones(len(scores), dtype=torch.bool)
                mask[list(exclude_indices)] = False
                scores = scores.clone()
                scores[~mask] = -float("inf")

            # Select point with maximum minimum distance
            idx = scores.argmax().item()
            selected.append(idx)

            # Update selected features
            features = self.extract_features(x[idx : idx + 1])
            self.selected_features.append(features[0])

            if exclude_indices is None:
                exclude_indices = set()
            exclude_indices.add(idx)

        return selected, self.score(x)


class KMeansSampling(QueryStrategy):
    """K-means based diversity sampling.

    Uses k-means clustering to select representative samples.

    Args:
        model: Feature extractor
        n_clusters: Number of clusters
        device: Computation device
    """

    def __init__(
        self, model: nn.Module, n_clusters: int = 10, device: Optional[str] = None
    ):
        super().__init__(model, device)
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None

    def extract_features(self, x: Tensor) -> np.ndarray:
        """Extract features from input."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "get_features"):
                features = self.model.get_features(x)
            else:
                features = x.view(x.size(0), -1)

        return features.cpu().numpy()

    def score(self, x: Tensor) -> Tensor:
        """Score samples by distance to cluster centers."""
        features = self.extract_features(x)

        if self.kmeans is None or not hasattr(self.kmeans, "cluster_centers_"):
            return torch.zeros(len(x))

        distances = pairwise_distances(features, self.kmeans.cluster_centers_)
        min_distances = distances.min(axis=1)

        return torch.from_numpy(min_distances).float()

    def select(
        self, x: Tensor, n: int = 1, exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[int], Tensor]:
        """Select samples closest to k-means centers."""
        features = self.extract_features(x)

        # Fit k-means
        n_clusters = min(self.n_clusters, n, len(x))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(features)

        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]

            if exclude_indices:
                cluster_indices = np.array(
                    [i for i in cluster_indices if i not in exclude_indices]
                )

            if len(cluster_indices) > 0:
                # Select sample closest to cluster center
                center = self.kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(features[cluster_indices] - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected.append(closest_idx)

            if len(selected) >= n:
                break

        scores = self.score(x)
        return selected[:n], scores


class RepresentativeSampling(QueryStrategy):
    """Prototype-based representative sampling.

    Selects samples that are representative of the data distribution
    using prototype learning.

    Args:
        model: Feature extractor
        n_prototypes: Number of prototypes
        device: Computation device
    """

    def __init__(
        self, model: nn.Module, n_prototypes: int = 10, device: Optional[str] = None
    ):
        super().__init__(model, device)
        self.n_prototypes = n_prototypes
        self.prototypes: Optional[np.ndarray] = None

    def extract_features(self, x: Tensor) -> np.ndarray:
        """Extract features from input."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "get_features"):
                features = self.model.get_features(x)
            else:
                features = x.view(x.size(0), -1)

        return features.cpu().numpy()

    def learn_prototypes(self, x: Tensor) -> None:
        """Learn representative prototypes."""
        features = self.extract_features(x)

        # Use k-means to find prototypes
        n_prototypes = min(self.n_prototypes, len(x))
        kmeans = KMeans(n_clusters=n_prototypes, random_state=42, n_init=10)
        kmeans.fit(features)

        self.prototypes = kmeans.cluster_centers_

    def score(self, x: Tensor) -> Tensor:
        """Score by proximity to prototypes."""
        if self.prototypes is None:
            self.learn_prototypes(x)

        features = self.extract_features(x)

        # Distance to nearest prototype (closer is more representative)
        distances = pairwise_distances(features, self.prototypes)
        min_distances = distances.min(axis=1)

        # Negative distance for scoring (higher = more representative)
        scores = -min_distances

        return torch.from_numpy(scores).float()

    def select(
        self, x: Tensor, n: int = 1, exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[int], Tensor]:
        """Select most representative samples."""
        scores = self.score(x)

        if exclude_indices:
            mask = torch.ones(len(scores), dtype=torch.bool)
            mask[list(exclude_indices)] = False
            scores = scores.clone()
            scores[~mask] = -float("inf")

        # Select top n by score
        selected = scores.argsort(descending=True)[:n].tolist()

        return selected, scores


class DiversityAwareSampling(QueryStrategy):
    """Diversity-aware sampling with adaptive weighting.

    Dynamically balances uncertainty and diversity based on labeled set size.

    Args:
        model: Neural network model
        initial_diversity_weight: Initial weight for diversity
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        initial_diversity_weight: float = 0.3,
        device: Optional[str] = None,
    ):
        super().__init__(model, device)
        self.initial_diversity_weight = initial_diversity_weight
        self.labeled_size = 0
        self.uncertainty_strategy = EntropySampling(model, device)

    def update_labeled_size(self, size: int) -> None:
        """Update labeled set size for adaptive weighting."""
        self.labeled_size = size

    def get_diversity_weight(self) -> float:
        """Compute adaptive diversity weight."""
        # Increase diversity weight as more labels are acquired
        adaptive_weight = self.initial_diversity_weight * (1 + self.labeled_size / 1000)
        return min(adaptive_weight, 0.8)

    def extract_features(self, x: Tensor) -> np.ndarray:
        """Extract features from input."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "get_features"):
                features = self.model.get_features(x)
            else:
                features = x.view(x.size(0), -1)

        return features.cpu().numpy()

    def score(self, x: Tensor) -> Tensor:
        """Compute diversity-aware scores."""
        uncertainty = self.uncertainty_strategy.score(x)
        features = self.extract_features(x)

        # Normalize features
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

        # Diversity score: distance from mean feature
        mean_feature = features.mean(axis=0, keepdims=True)
        diversity = np.linalg.norm(features - mean_feature, axis=1)
        diversity = torch.from_numpy(diversity).float()

        # Normalize scores
        uncertainty = (uncertainty - uncertainty.min()) / (
            uncertainty.max() - uncertainty.min() + 1e-10
        )
        diversity = (diversity - diversity.min()) / (
            diversity.max() - diversity.min() + 1e-10
        )

        # Combine scores
        div_weight = self.get_diversity_weight()
        combined = (1 - div_weight) * uncertainty + div_weight * diversity

        return combined


class AdversarialSampling(QueryStrategy):
    """Adversarial sampling for active learning.

    Uses adversarial examples to find decision boundary samples.

    Args:
        model: Neural network model
        epsilon: Perturbation magnitude
        num_steps: Number of adversarial steps
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.01,
        num_steps: int = 10,
        device: Optional[str] = None,
    ):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.num_steps = num_steps

    def generate_adversarial(self, x: Tensor) -> Tensor:
        """Generate adversarial examples using FGSM.

        Args:
            x: Input tensor

        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach().requires_grad_(True)

        for _ in range(self.num_steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            logits = self.model(x_adv)

            # Maximize entropy (uncertainty)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()

            # Maximize entropy
            loss = -entropy
            loss.backward()

            # Update adversarial example
            with torch.no_grad():
                x_adv = x_adv + self.epsilon * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
                x_adv = torch.clamp(x_adv, 0, 1)  # Assume normalized input

            x_adv.requires_grad_(True)

        return x_adv.detach()

    def score(self, x: Tensor) -> Tensor:
        """Score by adversarial distance."""
        self.model.eval()
        x = x.to(self.device)

        # Get original predictions
        with torch.no_grad():
            logits_orig = self.model(x)
            probs_orig = F.softmax(logits_orig, dim=-1)

        # Generate adversarial examples
        x_adv = self.generate_adversarial(x)

        # Get adversarial predictions
        with torch.no_grad():
            logits_adv = self.model(x_adv)
            probs_adv = F.softmax(logits_adv, dim=-1)

        # Score by prediction change (KL divergence)
        kl_div = (
            probs_orig * (torch.log(probs_orig + 1e-10) - torch.log(probs_adv + 1e-10))
        ).sum(dim=-1)

        return kl_div.cpu()


# =============================================================================
# Expected Model Change
# =============================================================================


class EGL(QueryStrategy):
    """Expected Gradient Length.

    Selects samples expected to cause the largest gradient update.

    Reference:
        Settles, B., et al. (2008). Active Learning with Real Annotation Costs.

    Args:
        model: Neural network model
        loss_fn: Loss function
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model, device)
        self.loss_fn = loss_fn or F.cross_entropy

    def score(self, x: Tensor) -> Tensor:
        """Compute expected gradient length.

        Args:
            x: Input tensor

        Returns:
            Expected gradient lengths
        """
        self.model.eval()
        x = x.to(self.device)

        # Get predictions
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

        egl_scores = []

        for i in range(len(x)):
            expected_grad_norm = 0.0

            # Iterate over all possible labels
            for y in range(probs.shape[-1]):
                # Compute gradient for this label
                self.model.zero_grad()

                xi = x[i : i + 1].clone().requires_grad_(False)
                yi = torch.tensor([y], device=x.device)

                logit = self.model(xi)
                loss = self.loss_fn(logit, yi)
                loss.backward()

                # Compute gradient norm
                grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2

                grad_norm = grad_norm**0.5

                # Weight by predicted probability
                expected_grad_norm += probs[i, y].item() * grad_norm

            egl_scores.append(expected_grad_norm)

        return torch.tensor(egl_scores)


class BALD(QueryStrategy):
    """Bayesian Active Learning by Disagreement.

    Selects samples that maximize information gain about model parameters.

    Reference:
        Houlsby, N., et al. (2011). Bayesian Active Learning for Classification.

    Args:
        model: Neural network with dropout or ensemble
        n_samples: Number of MC samples
        device: Computation device
    """

    def __init__(
        self, model: nn.Module, n_samples: int = 10, device: Optional[str] = None
    ):
        super().__init__(model, device)
        self.n_samples = n_samples

    def enable_dropout(self, module: nn.Module) -> None:
        """Enable dropout during evaluation."""
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()

    def score(self, x: Tensor) -> Tensor:
        """Compute BALD scores.

        BALD = H[y|x,D] - E_{θ~p(θ|D)}[H[y|x,θ]]
             = Mutual information between predictions and model parameters

        Args:
            x: Input tensor

        Returns:
            BALD scores
        """
        self.model.eval()
        self.model.apply(self.enable_dropout)

        x = x.to(self.device)
        predictions = []

        # Collect MC dropout predictions
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions)  # (n_samples, batch, n_classes)

        # Mean prediction (approximate posterior predictive)
        mean_probs = predictions.mean(dim=0)

        # Entropy of mean (total uncertainty)
        entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        # Mean of entropies (aleatoric uncertainty)
        entropies = -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1)
        mean_entropy = entropies.mean(dim=0)

        # BALD = mutual information
        bald = entropy_mean - mean_entropy

        return bald.cpu()

    def get_uncertainty_components(self, x: Tensor) -> Dict[str, Tensor]:
        """Get uncertainty components.

        Args:
            x: Input tensor

        Returns:
            Dictionary with uncertainty components
        """
        scores = self.score(x)

        self.model.eval()
        self.model.apply(self.enable_dropout)

        x = x.to(self.device)
        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)

        entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        entropies = -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1)
        mean_entropy = entropies.mean(dim=0)

        return {
            "bald": scores,
            "total_uncertainty": entropy_mean.cpu(),
            "aleatoric_uncertainty": mean_entropy.cpu(),
            "epistemic_uncertainty": scores,
        }


class VariationRatio(QueryStrategy):
    """Variation Ratio sampling.

    Measures disagreement among model predictions.

    Args:
        model: Neural network with MC dropout
        n_samples: Number of MC samples
        device: Computation device
    """

    def __init__(
        self, model: nn.Module, n_samples: int = 10, device: Optional[str] = None
    ):
        super().__init__(model, device)
        self.n_samples = n_samples

    def enable_dropout(self, module: nn.Module) -> None:
        """Enable dropout during evaluation."""
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()

    def score(self, x: Tensor) -> Tensor:
        """Compute variation ratio.

        Variation Ratio = 1 - (frequency of mode class / total samples)

        Args:
            x: Input tensor

        Returns:
            Variation ratios
        """
        self.model.eval()
        self.model.apply(self.enable_dropout)

        x = x.to(self.device)
        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                pred = logits.argmax(dim=-1)
                predictions.append(pred)

        predictions = torch.stack(predictions).cpu().numpy()  # (n_samples, batch)

        variation_ratios = []
        for i in range(len(x)):
            # Get predictions for sample i
            sample_preds = predictions[:, i]

            # Count frequency of each class
            unique, counts = np.unique(sample_preds, return_counts=True)
            mode_freq = counts.max()

            # Variation ratio
            vr = 1.0 - (mode_freq / self.n_samples)
            variation_ratios.append(vr)

        return torch.tensor(variation_ratios)


class InformationGain(QueryStrategy):
    """Information Gain / Mutual Information sampling.

    General mutual information between labels and model parameters.

    Args:
        model: Neural network model
        n_samples: Number of posterior samples
        device: Computation device
    """

    def __init__(
        self, model: nn.Module, n_samples: int = 10, device: Optional[str] = None
    ):
        super().__init__(model, device)
        self.n_samples = n_samples

    def score(self, x: Tensor) -> Tensor:
        """Compute mutual information.

        This is equivalent to BALD for classification tasks.

        Args:
            x: Input tensor

        Returns:
            Mutual information scores
        """
        # Use BALD implementation
        bald = BALD(self.model, self.n_samples, self.device)
        return bald.score(x)

    def compute_predictive_entropy(self, x: Tensor) -> Tensor:
        """Compute predictive entropy (total uncertainty).

        Args:
            x: Input tensor

        Returns:
            Predictive entropies
        """
        self.model.eval()

        # Enable dropout if present
        def enable_dropout(m):
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

        self.model.apply(enable_dropout)

        x = x.to(self.device)
        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)

        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return entropy.cpu()


# =============================================================================
# Pool-Based vs Stream-Based Active Learning
# =============================================================================


class PoolBasedAL:
    """Pool-based active learning.

    Selects samples from a large unlabeled pool for annotation.

    Args:
        model: Neural network model
        query_strategy: Query strategy to use
        initial_labeled_size: Initial labeled set size
        batch_size: Selection batch size
    """

    def __init__(
        self,
        model: nn.Module,
        query_strategy: QueryStrategy,
        initial_labeled_size: int = 100,
        batch_size: int = 10,
    ):
        self.model = model
        self.query_strategy = query_strategy
        self.initial_labeled_size = initial_labeled_size
        self.batch_size = batch_size

        self.labeled_indices: Set[int] = set()
        self.unlabeled_indices: Set[int] = set()
        self.iteration = 0
        self.history: List[Dict[str, Any]] = []

    def initialize_pool(self, dataset_size: int) -> None:
        """Initialize labeled and unlabeled pools.

        Args:
            dataset_size: Total size of the dataset
        """
        all_indices = set(range(dataset_size))

        # Random initial labeling
        initial_indices = np.random.choice(
            list(all_indices),
            size=min(self.initial_labeled_size, dataset_size),
            replace=False,
        )

        self.labeled_indices = set(initial_indices)
        self.unlabeled_indices = all_indices - self.labeled_indices

    def query(self, dataset: Dataset) -> List[int]:
        """Query next batch of samples for labeling.

        Args:
            dataset: Dataset containing all samples

        Returns:
            List of indices to label
        """
        if len(self.unlabeled_indices) == 0:
            return []

        # Get unlabeled data
        unlabeled_indices = list(self.unlabeled_indices)
        unlabeled_data = torch.stack([dataset[i][0] for i in unlabeled_indices])

        # Query samples
        selected_local, scores = self.query_strategy.select(
            unlabeled_data,
            n=min(self.batch_size, len(unlabeled_indices)),
            exclude_indices=set(),
        )

        # Map back to global indices
        selected_global = [unlabeled_indices[i] for i in selected_local]

        # Update pools
        self.labeled_indices.update(selected_global)
        self.unlabeled_indices -= set(selected_global)

        # Record history
        self.history.append(
            {
                "iteration": self.iteration,
                "selected_indices": selected_global,
                "scores": scores.numpy(),
                "labeled_size": len(self.labeled_indices),
            }
        )

        self.iteration += 1

        return selected_global

    def get_state(self) -> ActiveLearningState:
        """Get current active learning state."""
        return ActiveLearningState(
            labeled_indices=self.labeled_indices,
            unlabeled_indices=self.unlabeled_indices,
            iteration=self.iteration,
            labeled_size=len(self.labeled_indices),
            unlabeled_size=len(self.unlabeled_indices),
            performance_history=self.history,
        )


class StreamBasedAL:
    """Stream-based active learning.

    Decides whether to label samples as they arrive in a stream.

    Args:
        model: Neural network model
        query_strategy: Query strategy for scoring
        threshold: Uncertainty threshold for labeling decision
        budget: Maximum labeling budget
    """

    def __init__(
        self,
        model: nn.Module,
        query_strategy: QueryStrategy,
        threshold: float = 0.5,
        budget: Optional[int] = None,
    ):
        self.model = model
        self.query_strategy = query_strategy
        self.threshold = threshold
        self.budget = budget

        self.labeled_count = 0
        self.seen_count = 0
        self.labeled_indices: List[int] = []
        self.history: List[Dict[str, Any]] = []

    def decide_label(
        self, x: Tensor, index: Optional[int] = None
    ) -> Tuple[bool, float]:
        """Decide whether to label a sample from the stream.

        Args:
            x: Input sample
            index: Optional index of the sample

        Returns:
            Tuple of (should_label, uncertainty_score)
        """
        self.seen_count += 1

        # Check budget
        if self.budget is not None and self.labeled_count >= self.budget:
            return False, 0.0

        # Compute uncertainty
        score = self.query_strategy.score(x.unsqueeze(0))[0].item()

        # Decision based on threshold
        should_label = score > self.threshold

        if should_label:
            self.labeled_count += 1
            if index is not None:
                self.labeled_indices.append(index)

        # Record history
        self.history.append(
            {
                "index": index,
                "score": score,
                "labeled": should_label,
                "seen_count": self.seen_count,
            }
        )

        return should_label, score

    def process_stream(
        self, stream: Iterator[Tuple[Tensor, Optional[int]]]
    ) -> List[int]:
        """Process an entire stream of samples.

        Args:
            stream: Iterator yielding (sample, index) pairs

        Returns:
            List of indices selected for labeling
        """
        selected = []

        for x, idx in stream:
            should_label, _ = self.decide_label(x, idx)
            if should_label:
                selected.append(idx)

        return selected

    def adapt_threshold(self, target_rate: float = 0.1) -> None:
        """Adapt threshold to achieve target labeling rate.

        Args:
            target_rate: Target fraction of samples to label
        """
        if len(self.history) == 0:
            return

        actual_rate = self.labeled_count / self.seen_count

        if actual_rate > target_rate:
            # Labeling too much, increase threshold
            self.threshold *= 1.1
        else:
            # Labeling too little, decrease threshold
            self.threshold *= 0.9

        # Clamp threshold
        self.threshold = max(0.01, min(0.99, self.threshold))


class MembershipQuerySynthesis:
    """Membership Query Synthesis.

    Generates synthetic samples for annotation rather than selecting
    from a pool.

    Args:
        model: Neural network model
        input_shape: Shape of input samples
        synthesis_method: Method for generating queries ('gradient', 'gan', 'random')
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        synthesis_method: str = "gradient",
    ):
        self.model = model
        self.input_shape = input_shape
        self.synthesis_method = synthesis_method

    def synthesize_query(self, n: int = 1) -> Tensor:
        """Generate synthetic samples for querying.

        Args:
            n: Number of samples to generate

        Returns:
            Synthetic samples
        """
        if self.synthesis_method == "random":
            return self._random_synthesis(n)
        elif self.synthesis_method == "gradient":
            return self._gradient_synthesis(n)
        elif self.synthesis_method == "adversarial":
            return self._adversarial_synthesis(n)
        else:
            raise ValueError(f"Unknown synthesis method: {self.synthesis_method}")

    def _random_synthesis(self, n: int) -> Tensor:
        """Generate random samples."""
        return torch.randn(n, *self.input_shape)

    def _gradient_synthesis(self, n: int) -> Tensor:
        """Generate samples by maximizing uncertainty."""
        # Start from random noise
        x = torch.randn(n, *self.input_shape, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.01)

        for _ in range(100):
            optimizer.zero_grad()

            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

            # Maximize entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            loss = -entropy  # Negative because we maximize

            loss.backward()
            optimizer.step()

        return x.detach()

    def _adversarial_synthesis(self, n: int) -> Tensor:
        """Generate adversarial examples."""
        # Start from random samples
        x = torch.randn(n, *self.input_shape, requires_grad=True)

        for _ in range(50):
            if x.grad is not None:
                x.grad.zero_()

            logits = self.model(x)

            # Maximize entropy (uncertainty)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

            (-entropy).backward()

            with torch.no_grad():
                x = x + 0.01 * x.grad.sign()
                x = torch.clamp(x, -3, 3)  # Reasonable range

            x.requires_grad_(True)

        return x.detach()


# =============================================================================
# Multi-Task Active Learning
# =============================================================================


class MultiTaskAL:
    """Multi-task active learning.

    Active learning across multiple related tasks.

    Args:
        models: Dictionary of task-specific models
        query_strategy: Base query strategy
        task_weights: Weights for each task
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        query_strategy: Type[QueryStrategy],
        task_weights: Optional[Dict[str, float]] = None,
    ):
        self.models = models
        self.query_strategy_class = query_strategy
        self.task_weights = task_weights or {task: 1.0 for task in models.keys()}

        # Create query strategies for each task
        self.query_strategies = {
            task: query_strategy(model) for task, model in models.items()
        }

    def multi_task_query(
        self, x: Tensor, n: int = 1, aggregation: str = "mean"
    ) -> Tuple[List[int], Dict[str, Tensor]]:
        """Query samples considering multiple tasks.

        Args:
            x: Input tensor
            n: Number of samples to select
            aggregation: Method to aggregate task scores ('mean', 'max', 'weighted')

        Returns:
            Tuple of (selected indices, task scores)
        """
        # Get scores from each task
        task_scores = {}

        for task, strategy in self.query_strategies.items():
            task_scores[task] = strategy.score(x)

        # Aggregate scores
        if aggregation == "mean":
            combined_scores = torch.stack(list(task_scores.values())).mean(dim=0)
        elif aggregation == "max":
            combined_scores = torch.stack(list(task_scores.values())).max(dim=0)[0]
        elif aggregation == "weighted":
            weights = torch.tensor(
                [self.task_weights[task] for task in task_scores.keys()]
            )
            scores_stack = torch.stack(list(task_scores.values()))
            combined_scores = (scores_stack * weights.unsqueeze(1)).sum(
                dim=0
            ) / weights.sum()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Select top samples
        selected = combined_scores.argsort(descending=True)[:n].tolist()

        return selected, task_scores

    def get_task_uncertainty(self, x: Tensor) -> Dict[str, float]:
        """Get average uncertainty for each task.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping task names to average uncertainty
        """
        uncertainties = {}

        for task, strategy in self.query_strategies.items():
            scores = strategy.score(x)
            uncertainties[task] = float(scores.mean())

        return uncertainties


class TransferActive:
    """Transfer learning with active learning.

    Uses knowledge from source tasks to improve active learning on target.

    Args:
        source_model: Model trained on source domain
        target_model: Model for target domain
        query_strategy: Query strategy
        transfer_weight: Weight for transferred knowledge
    """

    def __init__(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        query_strategy: QueryStrategy,
        transfer_weight: float = 0.3,
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.query_strategy = query_strategy
        self.transfer_weight = transfer_weight

    def transfer_weighted_score(self, x: Tensor) -> Tensor:
        """Compute scores combining source and target models.

        Args:
            x: Input tensor

        Returns:
            Combined scores
        """
        # Target uncertainty
        target_scores = self.query_strategy.score(x)

        # Source uncertainty
        source_strategy = EntropySampling(self.source_model)
        source_scores = source_strategy.score(x)

        # Combine scores
        combined = (
            1 - self.transfer_weight
        ) * target_scores + self.transfer_weight * source_scores

        return combined

    def select(
        self, x: Tensor, n: int = 1, exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[int], Tensor]:
        """Select samples using transfer-weighted scores."""
        scores = self.transfer_weighted_score(x)

        if exclude_indices:
            mask = torch.ones(len(scores), dtype=torch.bool)
            mask[list(exclude_indices)] = False
            scores = scores.clone()
            scores[~mask] = -float("inf")

        selected = scores.argsort(descending=True)[:n].tolist()

        return selected, scores


class DomainAdaptiveAL:
    """Domain-adaptive active learning.

    Active learning that adapts to domain shift between labeled and unlabeled data.

    Args:
        model: Neural network model
        query_strategy: Base query strategy
        domain_classifier: Optional domain classifier
    """

    def __init__(
        self,
        model: nn.Module,
        query_strategy: QueryStrategy,
        domain_classifier: Optional[nn.Module] = None,
    ):
        self.model = model
        self.query_strategy = query_strategy
        self.domain_classifier = domain_classifier

    def compute_domain_weight(self, x: Tensor) -> Tensor:
        """Compute domain adaptation weights.

        Args:
            x: Input tensor

        Returns:
            Domain weights
        """
        if self.domain_classifier is None:
            # Uniform weights if no domain classifier
            return torch.ones(len(x))

        self.domain_classifier.eval()

        with torch.no_grad():
            domain_pred = self.domain_classifier(x)
            # Weight by distance to target domain
            weights = (
                1.0 - F.softmax(domain_pred, dim=-1)[:, 1]
            )  # Assuming binary domain

        return weights

    def domain_aware_score(self, x: Tensor) -> Tensor:
        """Compute domain-aware uncertainty scores.

        Args:
            x: Input tensor

        Returns:
            Domain-aware scores
        """
        # Base uncertainty
        uncertainty = self.query_strategy.score(x)

        # Domain weights
        domain_weights = self.compute_domain_weight(x)

        # Weight uncertainty by domain relevance
        weighted_scores = uncertainty * domain_weights

        return weighted_scores

    def select(
        self, x: Tensor, n: int = 1, exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[int], Tensor]:
        """Select samples using domain-aware scores."""
        scores = self.domain_aware_score(x)

        if exclude_indices:
            mask = torch.ones(len(scores), dtype=torch.bool)
            mask[list(exclude_indices)] = False
            scores = scores.clone()
            scores[~mask] = -float("inf")

        selected = scores.argsort(descending=True)[:n].tolist()

        return selected, scores


# =============================================================================
# Evaluation
# =============================================================================


class LearningCurve:
    """Track and analyze learning curves for active learning.

    Monitors model performance as more labels are acquired.

    Args:
        metrics: List of metric names to track
    """

    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ["accuracy", "loss"]
        self.history: Dict[str, List[float]] = {"labeled_size": [], "iteration": []}
        for metric in self.metrics:
            self.history[metric] = []

    def update(self, labeled_size: int, iteration: int, **metric_values: float) -> None:
        """Update learning curve with new measurements.

        Args:
            labeled_size: Current size of labeled set
            iteration: Current iteration
            **metric_values: Metric values to record
        """
        self.history["labeled_size"].append(labeled_size)
        self.history["iteration"].append(iteration)

        for metric, value in metric_values.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(value)

    def compute_auc(self, metric: str = "accuracy") -> float:
        """Compute area under learning curve.

        Args:
            metric: Metric to compute AUC for

        Returns:
            AUC score
        """
        if metric not in self.history or len(self.history[metric]) < 2:
            return 0.0

        x = np.array(self.history["labeled_size"])
        y = np.array(self.history[metric])

        # Compute AUC using trapezoidal rule
        auc = np.trapz(y, x) / (x[-1] - x[0])

        return float(auc)

    def get_improvement_rate(self, metric: str = "accuracy") -> float:
        """Compute rate of improvement.

        Args:
            metric: Metric to analyze

        Returns:
            Average improvement per labeled sample
        """
        if metric not in self.history or len(self.history[metric]) < 2:
            return 0.0

        y = np.array(self.history[metric])
        x = np.array(self.history["labeled_size"])

        # Compute slope
        improvement = (y[-1] - y[0]) / (x[-1] - x[0])

        return float(improvement)

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert history to dictionary."""
        return self.history.copy()


class AnnotationCost:
    """Analyze annotation costs for active learning.

    Tracks and optimizes labeling costs.

    Args:
        cost_model: Cost model function or dictionary
        budget: Total annotation budget
    """

    def __init__(
        self,
        cost_model: Optional[Union[Callable, Dict[int, float]]] = None,
        budget: Optional[float] = None,
    ):
        self.cost_model = cost_model or self._default_cost_model
        self.budget = budget

        self.total_cost = 0.0
        self.cost_history: List[Dict[str, Any]] = []

    def _default_cost_model(self, index: int) -> float:
        """Default uniform cost model."""
        return 1.0

    def compute_cost(self, indices: List[int]) -> float:
        """Compute cost for labeling indices.

        Args:
            indices: Indices to label

        Returns:
            Total cost
        """
        if callable(self.cost_model):
            cost = sum(self.cost_model(i) for i in indices)
        else:
            cost = sum(self.cost_model.get(i, 1.0) for i in indices)

        return cost

    def record_labeling(self, indices: List[int], iteration: int) -> None:
        """Record labeling event.

        Args:
            indices: Labeled indices
            iteration: Current iteration
        """
        cost = self.compute_cost(indices)
        self.total_cost += cost

        self.cost_history.append(
            {
                "iteration": iteration,
                "indices": indices,
                "cost": cost,
                "total_cost": self.total_cost,
            }
        )

    def within_budget(self, indices: List[int]) -> bool:
        """Check if labeling is within budget.

        Args:
            indices: Proposed indices to label

        Returns:
            Whether labeling is within budget
        """
        if self.budget is None:
            return True

        additional_cost = self.compute_cost(indices)
        return (self.total_cost + additional_cost) <= self.budget

    def get_cost_efficiency(self, performance: float) -> float:
        """Compute cost efficiency (performance per unit cost).

        Args:
            performance: Model performance metric

        Returns:
            Cost efficiency
        """
        if self.total_cost == 0:
            return 0.0

        return performance / self.total_cost


class ALBenchmark:
    """Standard benchmarks for active learning evaluation.

    Provides standardized evaluation protocols.

    Args:
        dataset: Dataset to benchmark on
        initial_size: Initial labeled set size
        budget: Total labeling budget
        batch_size: Selection batch size
    """

    def __init__(
        self,
        dataset: Dataset,
        initial_size: int = 100,
        budget: int = 1000,
        batch_size: int = 10,
    ):
        self.dataset = dataset
        self.initial_size = initial_size
        self.budget = budget
        self.batch_size = batch_size

        self.results: Dict[str, LearningCurve] = {}

    def run(
        self,
        model_builder: Callable[[], nn.Module],
        strategies: Dict[str, QueryStrategy],
        train_fn: Callable[[nn.Module, Dataset], Dict[str, float]],
        test_fn: Callable[[nn.Module, Dataset], Dict[str, float]],
    ) -> Dict[str, LearningCurve]:
        """Run benchmark for multiple strategies.

        Args:
            model_builder: Function to build model
            strategies: Dictionary of strategies to compare
            train_fn: Training function
            test_fn: Testing function

        Returns:
            Dictionary of learning curves for each strategy
        """
        for strategy_name, strategy in strategies.items():
            print(f"Running benchmark for {strategy_name}...")

            # Build fresh model
            model = model_builder()

            # Initialize pool-based AL
            al = PoolBasedAL(
                model,
                strategy,
                initial_labeled_size=self.initial_size,
                batch_size=self.batch_size,
            )
            al.initialize_pool(len(self.dataset))

            # Create learning curve tracker
            curve = LearningCurve()

            # Active learning loop
            while len(al.labeled_indices) < self.budget:
                # Train on current labeled set
                labeled_dataset = Subset(self.dataset, list(al.labeled_indices))
                train_metrics = train_fn(model, labeled_dataset)

                # Evaluate
                test_metrics = test_fn(model, self.dataset)

                # Update curve
                curve.update(len(al.labeled_indices), al.iteration, **test_metrics)

                # Query next batch
                if len(al.unlabeled_indices) > 0:
                    al.query(self.dataset)

            self.results[strategy_name] = curve

        return self.results

    def compare_strategies(self, metric: str = "accuracy") -> Dict[str, float]:
        """Compare strategies by final performance.

        Args:
            metric: Metric to compare

        Returns:
            Dictionary of final scores
        """
        comparison = {}

        for strategy_name, curve in self.results.items():
            if metric in curve.history and len(curve.history[metric]) > 0:
                comparison[strategy_name] = curve.history[metric][-1]

        return comparison


class ALVisualization:
    """Visualization utilities for active learning.

    Provides plotting functions for AL analysis.

    Args:
        figsize: Default figure size
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize
        self.has_matplotlib = False

        try:
            import matplotlib.pyplot as plt

            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            warnings.warn("matplotlib not available for visualization")

    def plot_learning_curves(
        self,
        curves: Dict[str, LearningCurve],
        metric: str = "accuracy",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot learning curves for multiple strategies.

        Args:
            curves: Dictionary of learning curves
            metric: Metric to plot
            save_path: Optional path to save figure
        """
        if not self.has_matplotlib:
            return

        self.plt.figure(figsize=self.figsize)

        for strategy_name, curve in curves.items():
            if metric in curve.history:
                x = curve.history["labeled_size"]
                y = curve.history[metric]
                self.plt.plot(x, y, marker="o", label=strategy_name)

        self.plt.xlabel("Number of Labeled Samples")
        self.plt.ylabel(metric.capitalize())
        self.plt.title("Active Learning Learning Curves")
        self.plt.legend()
        self.plt.grid(True, alpha=0.3)

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")

        self.plt.show()

    def plot_query_distribution(
        self,
        scores: Tensor,
        selected_indices: List[int],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot distribution of query scores.

        Args:
            scores: Query scores
            selected_indices: Indices selected for labeling
            save_path: Optional path to save figure
        """
        if not self.has_matplotlib:
            return

        self.plt.figure(figsize=self.figsize)

        # Histogram of all scores
        self.plt.hist(scores.numpy(), bins=50, alpha=0.5, label="All samples")

        # Selected samples
        selected_scores = scores[selected_indices].numpy()
        self.plt.hist(selected_scores, bins=20, alpha=0.7, label="Selected")

        self.plt.xlabel("Query Score")
        self.plt.ylabel("Frequency")
        self.plt.title("Query Score Distribution")
        self.plt.legend()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")

        self.plt.show()

    def plot_uncertainty_components(
        self, aleatoric: Tensor, epistemic: Tensor, save_path: Optional[str] = None
    ) -> None:
        """Plot aleatoric vs epistemic uncertainty.

        Args:
            aleatoric: Aleatoric uncertainty scores
            epistemic: Epistemic uncertainty scores
            save_path: Optional path to save figure
        """
        if not self.has_matplotlib:
            return

        self.plt.figure(figsize=self.figsize)

        self.plt.scatter(aleatoric.numpy(), epistemic.numpy(), alpha=0.5)

        self.plt.xlabel("Aleatoric Uncertainty")
        self.plt.ylabel("Epistemic Uncertainty")
        self.plt.title("Uncertainty Decomposition")
        self.plt.grid(True, alpha=0.3)

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")

        self.plt.show()


# =============================================================================
# Integration
# =============================================================================


class ActiveDataset:
    """Dataset manager for active learning.

    Manages labeled and unlabeled data pools.

    Args:
        dataset: Full dataset
        initial_labeled_size: Initial labeled set size
    """

    def __init__(self, dataset: Dataset, initial_labeled_size: int = 100):
        self.dataset = dataset
        self.initial_labeled_size = initial_labeled_size

        self.labeled_indices: Set[int] = set()
        self.unlabeled_indices: Set[int] = set()
        self.labels: Dict[int, Any] = {}

        self._initialize_pools()

    def _initialize_pools(self) -> None:
        """Initialize labeled and unlabeled pools."""
        n = len(self.dataset)

        # Random initial labeling
        initial = np.random.choice(
            n, size=min(self.initial_labeled_size, n), replace=False
        )

        self.labeled_indices = set(initial)
        self.unlabeled_indices = set(range(n)) - self.labeled_indices

        # Load initial labels
        for idx in self.labeled_indices:
            _, label = self.dataset[idx]
            self.labels[idx] = label

    def label_sample(self, index: int, label: Any) -> None:
        """Add a label for a sample.

        Args:
            index: Sample index
            label: Label value
        """
        if index not in self.unlabeled_indices:
            warnings.warn(f"Index {index} is not in unlabeled pool")
            return

        self.labels[index] = label
        self.labeled_indices.add(index)
        self.unlabeled_indices.remove(index)

    def get_labeled_data(self) -> Tuple[Tensor, Tensor]:
        """Get labeled data as tensors.

        Returns:
            Tuple of (data, labels)
        """
        data_list = []
        label_list = []

        for idx in self.labeled_indices:
            x, _ = self.dataset[idx]
            data_list.append(x)
            label_list.append(self.labels[idx])

        data = torch.stack(data_list)
        labels = torch.tensor(label_list)

        return data, labels

    def get_unlabeled_data(self) -> Tensor:
        """Get unlabeled data as tensor.

        Returns:
            Unlabeled data tensor
        """
        data_list = []

        for idx in self.unlabeled_indices:
            x, _ = self.dataset[idx]
            data_list.append(x)

        return torch.stack(data_list)

    def get_unlabeled_indices(self) -> List[int]:
        """Get list of unlabeled indices."""
        return list(self.unlabeled_indices)

    def labeled_size(self) -> int:
        """Get size of labeled pool."""
        return len(self.labeled_indices)

    def unlabeled_size(self) -> int:
        """Get size of unlabeled pool."""
        return len(self.unlabeled_indices)


class ActiveTrainer:
    """Combined trainer for active learning.

    Integrates model training with active learning queries.

    Args:
        model: Neural network model
        active_dataset: Active dataset manager
        query_strategy: Query strategy
        trainer_config: Training configuration
    """

    def __init__(
        self,
        model: nn.Module,
        active_dataset: ActiveDataset,
        query_strategy: QueryStrategy,
        trainer_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.active_dataset = active_dataset
        self.query_strategy = query_strategy
        self.config = trainer_config or {}

        self.learning_curve = LearningCurve()
        self.iteration = 0

    def train_epoch(self, epochs: int = 1) -> Dict[str, float]:
        """Train model for epochs.

        Args:
            epochs: Number of epochs

        Returns:
            Training metrics
        """
        # Get labeled data
        data, labels = self.active_dataset.get_labeled_data()

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(data, labels)
        loader = DataLoader(
            dataset, batch_size=self.config.get("batch_size", 32), shuffle=True
        )

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.get("lr", 0.001)
        )

        # Training loop
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for batch_data, batch_labels in loader:
                optimizer.zero_grad()

                logits = self.model(batch_data)
                loss = F.cross_entropy(logits, batch_labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

        metrics = {
            "loss": total_loss / len(loader),
            "accuracy": correct / total if total > 0 else 0.0,
        }

        return metrics

    def query_and_label(self, n: int = 10) -> List[int]:
        """Query new samples and simulate labeling.

        Args:
            n: Number of samples to query

        Returns:
            Selected indices
        """
        # Get unlabeled data
        unlabeled_indices = self.active_dataset.get_unlabeled_indices()

        if len(unlabeled_indices) == 0:
            return []

        unlabeled_data = self.active_dataset.get_unlabeled_data()

        # Query samples
        selected_local, scores = self.query_strategy.select(
            unlabeled_data, n=min(n, len(unlabeled_indices)), exclude_indices=set()
        )

        # Map to global indices
        selected_global = [unlabeled_indices[i] for i in selected_local]

        # Label samples (using ground truth from dataset)
        for idx in selected_global:
            _, label = self.active_dataset.dataset[idx]
            self.active_dataset.label_sample(idx, label)

        return selected_global

    def run_iteration(self) -> Dict[str, Any]:
        """Run one active learning iteration.

        Returns:
            Dictionary with iteration results
        """
        # Train
        train_metrics = self.train_epoch(epochs=self.config.get("train_epochs", 1))

        # Query
        selected = self.query_and_label(n=self.config.get("query_batch_size", 10))

        # Update learning curve
        self.learning_curve.update(
            self.active_dataset.labeled_size(), self.iteration, **train_metrics
        )

        self.iteration += 1

        return {
            "iteration": self.iteration - 1,
            "selected_indices": selected,
            "labeled_size": self.active_dataset.labeled_size(),
            **train_metrics,
        }


class ActiveLoop:
    """Full active learning pipeline.

    Complete end-to-end active learning system.

    Args:
        model: Neural network model
        dataset: Full dataset
        query_strategy: Query strategy
        config: Pipeline configuration
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        query_strategy: QueryStrategy,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.query_strategy = query_strategy
        self.config = config or {}

        # Initialize components
        self.active_dataset = ActiveDataset(
            dataset, initial_labeled_size=self.config.get("initial_size", 100)
        )

        self.trainer = ActiveTrainer(
            model,
            self.active_dataset,
            query_strategy,
            trainer_config=config.get("training", {}),
        )

        self.cost_tracker = AnnotationCost(budget=self.config.get("budget", None))

        self.visualizer = ALVisualization()

        self.history: List[Dict[str, Any]] = []

    def run(
        self,
        max_iterations: Optional[int] = None,
        target_performance: Optional[float] = None,
    ) -> LearningCurve:
        """Run active learning loop.

        Args:
            max_iterations: Maximum number of iterations
            target_performance: Stop when reaching this performance

        Returns:
            Learning curve
        """
        max_iter = max_iterations or self.config.get("max_iterations", 100)

        for iteration in range(max_iter):
            print(f"Iteration {iteration + 1}/{max_iter}")

            # Run iteration
            result = self.trainer.run_iteration()
            self.history.append(result)

            print(
                f"  Labeled: {result['labeled_size']}, "
                f"Accuracy: {result.get('accuracy', 0):.4f}"
            )

            # Check stopping criteria
            if target_performance and result.get("accuracy", 0) >= target_performance:
                print(f"Reached target performance: {target_performance}")
                break

            if self.active_dataset.unlabeled_size() == 0:
                print("No more unlabeled data")
                break

        return self.trainer.learning_curve

    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """Evaluate on test set.

        Args:
            test_dataset: Test dataset

        Returns:
            Test metrics
        """
        self.model.eval()

        loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in loader:
                logits = self.model(data)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        accuracy = correct / total if total > 0 else 0.0

        return {"accuracy": accuracy, "total": total, "correct": correct}

    def save_checkpoint(self, path: str) -> None:
        """Save pipeline checkpoint.

        Args:
            path: Save path
        """
        checkpoint = {
            "model_state": self.model.state_dict(),
            "iteration": self.trainer.iteration,
            "labeled_indices": list(self.active_dataset.labeled_indices),
            "history": self.history,
            "learning_curve": self.trainer.learning_curve.to_dict(),
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load pipeline checkpoint.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state"])
        self.trainer.iteration = checkpoint["iteration"]
        self.active_dataset.labeled_indices = set(checkpoint["labeled_indices"])
        self.history = checkpoint["history"]


# =============================================================================
# Factory Functions
# =============================================================================


def create_query_strategy(
    strategy_name: str, model: nn.Module, **kwargs
) -> QueryStrategy:
    """Factory function to create query strategies.

    Args:
        strategy_name: Name of the strategy
        model: Neural network model
        **kwargs: Additional arguments for the strategy

    Returns:
        Query strategy instance

    Example:
        >>> strategy = create_query_strategy("uncertainty", model)
        >>> strategy = create_query_strategy("entropy", model, epsilon=1e-8)
    """
    strategies = {
        "uncertainty": UncertaintySampling,
        "margin": MarginSampling,
        "entropy": EntropySampling,
        "random": RandomSampling,
        "cluster": ClusterBasedSampling,
        "density": DensityWeightedSampling,
        "kcenter": KCenterSampling,
        "kmeans": KMeansSampling,
        "representative": RepresentativeSampling,
        "diversity": DiversityAwareSampling,
        "adversarial": AdversarialSampling,
        "egl": EGL,
        "bald": BALD,
        "variation_ratio": VariationRatio,
        "information_gain": InformationGain,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}"
        )

    return strategies[strategy_name](model, **kwargs)


def create_batch_strategy(
    strategy_name: str, model: nn.Module, batch_size: int = 10, **kwargs
) -> BatchQueryStrategy:
    """Factory function to create batch query strategies.

    Args:
        strategy_name: Name of the strategy
        model: Neural network model
        batch_size: Batch size for selection
        **kwargs: Additional arguments

    Returns:
        Batch query strategy instance
    """
    strategies = {
        "batchbald": BatchBALD,
        "coreset": CoreSet,
        "badge": BADGE,
        "batch": BatchActive,
        "greedy": GreedyBatch,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown batch strategy: {strategy_name}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy_name](model, batch_size=batch_size, **kwargs)


def create_uncertainty_estimator(
    estimator_name: str, model: nn.Module, **kwargs
) -> UncertaintyEstimator:
    """Factory function to create uncertainty estimators.

    Args:
        estimator_name: Name of the estimator
        model: Neural network model
        **kwargs: Additional arguments

    Returns:
        Uncertainty estimator instance
    """
    estimators = {
        "mc_dropout": MCDropoutUncertainty,
        "ensemble": EnsembleUncertainty,
        "bayesian": BayesianUncertainty,
        "evidential": EvidentialUncertainty,
    }

    if estimator_name not in estimators:
        raise ValueError(
            f"Unknown estimator: {estimator_name}. Available: {list(estimators.keys())}"
        )

    return estimators[estimator_name](model, **kwargs)


# =============================================================================
# Utility Functions
# =============================================================================


def compute_query_diversity(indices: List[int], features: np.ndarray) -> float:
    """Compute diversity of selected queries.

    Args:
        indices: Selected indices
        features: Feature matrix

    Returns:
        Diversity score (average pairwise distance)
    """
    if len(indices) < 2:
        return 0.0

    selected_features = features[indices]
    distances = pairwise_distances(selected_features)

    # Average pairwise distance (excluding diagonal)
    n = len(indices)
    diversity = distances.sum() / (n * (n - 1))

    return float(diversity)


def compute_coverage(indices: List[int], features: np.ndarray) -> float:
    """Compute coverage of selected queries.

    Args:
        indices: Selected indices
        features: Feature matrix

    Returns:
        Coverage score (fraction of samples within radius)
    """
    if len(indices) == 0:
        return 0.0

    selected_features = features[indices]
    distances = pairwise_distances(features, selected_features)
    min_distances = distances.min(axis=1)

    # Coverage as inverse of mean minimum distance
    coverage = 1.0 / (1.0 + min_distances.mean())

    return float(coverage)


def active_learning_summary(results: Dict[str, LearningCurve]) -> str:
    """Generate text summary of active learning results.

    Args:
        results: Dictionary of learning curves

    Returns:
        Summary string
    """
    lines = ["=" * 60, "Active Learning Results Summary", "=" * 60]

    for strategy_name, curve in results.items():
        lines.append(f"\n{strategy_name}:")
        lines.append(
            f"  - Final accuracy: {curve.history.get('accuracy', [0])[-1]:.4f}"
        )
        lines.append(f"  - AUC: {curve.compute_auc('accuracy'):.4f}")
        lines.append(
            f"  - Improvement rate: {curve.get_improvement_rate('accuracy'):.6f}"
        )

    return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "QueryStrategy",
    "UncertaintyEstimator",
    "BatchQueryStrategy",
    "QueryResult",
    "ActiveLearningState",
    # Query strategies
    "UncertaintySampling",
    "MarginSampling",
    "EntropySampling",
    "RandomSampling",
    "ClusterBasedSampling",
    "DensityWeightedSampling",
    # Uncertainty estimation
    "MCDropoutUncertainty",
    "EnsembleUncertainty",
    "BayesianUncertainty",
    "EvidentialUncertainty",
    # Batch active learning
    "BatchBALD",
    "CoreSet",
    "BADGE",
    "BatchActive",
    "GreedyBatch",
    # Diversity sampling
    "KCenterSampling",
    "KMeansSampling",
    "RepresentativeSampling",
    "DiversityAwareSampling",
    "AdversarialSampling",
    # Expected model change
    "EGL",
    "BALD",
    "VariationRatio",
    "InformationGain",
    # Pool and stream
    "PoolBasedAL",
    "StreamBasedAL",
    "MembershipQuerySynthesis",
    # Multi-task
    "MultiTaskAL",
    "TransferActive",
    "DomainAdaptiveAL",
    # Evaluation
    "LearningCurve",
    "AnnotationCost",
    "ALBenchmark",
    "ALVisualization",
    # Integration
    "ActiveDataset",
    "ActiveTrainer",
    "ActiveLoop",
    # Factories
    "create_query_strategy",
    "create_batch_strategy",
    "create_uncertainty_estimator",
    # Utilities
    "compute_query_diversity",
    "compute_coverage",
    "active_learning_summary",
]
