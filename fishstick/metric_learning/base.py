"""
Base Module for Metric Learning

Core utilities and base classes for metric learning including:
- Distance metric abstractions
- Similarity metric abstractions
- Base classes for metric learning models
"""

from typing import Optional, Tuple, Union, Callable, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
import torch.nn.functional as F


@runtime_checkable
class DistanceMetric(Protocol):
    """Protocol for distance metrics."""

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute distance between x and y."""
        ...


class MetricSpace(ABC):
    """Abstract base class for metric spaces.

    A metric space defines a distance function that satisfies:
    - Non-negativity: d(x, y) >= 0
    - Identity of indiscernibles: d(x, y) = 0 iff x = y
    - Symmetry: d(x, y) = d(y, x)
    - Triangle inequality: d(x, z) <= d(x, y) + d(y, z)

    Args:
        name: Name of the metric space
    """

    def __init__(self, name: str = "metric_space"):
        self.name = name

    @abstractmethod
    def compute_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute distance between two tensors.

        Args:
            x: First tensor of shape (..., d)
            y: Second tensor of shape (..., d)

        Returns:
            Distance tensor of shape (...)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}()"


class EuclideanMetric(MetricSpace):
    """Euclidean distance metric.

    d(x, y) = sqrt(sum((x_i - y_i)^2))

    Args:
        normalize: Whether to L2-normalize inputs before computing distance
    """

    def __init__(self, normalize: bool = False):
        super().__init__(name="EuclideanMetric")
        self.normalize = normalize

    def compute_distance(self, x: Tensor, y: Tensor) -> Tensor:
        if self.normalize:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        return torch.cdist(x, y, p=2)


class CosineMetric(MetricSpace):
    """Cosine distance metric.

    d(x, y) = 1 - cos(x, y) = 1 - (x · y) / (||x|| ||y||)

    Args:
        eps: Small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(name="CosineMetric")
        self.eps = eps

    def compute_distance(self, x: Tensor, y: Tensor) -> Tensor:
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        cosine = torch.sum(x_norm * y_norm, dim=-1)
        return 1.0 - cosine.clamp(-1 + self.eps, 1 - self.eps)


class ManhattanMetric(MetricSpace):
    """Manhattan (L1) distance metric.

    d(x, y) = sum(|x_i - y_i|)

    Args:
        normalize: Whether to L2-normalize inputs before computing distance
    """

    def __init__(self, normalize: bool = False):
        super().__init__(name="ManhattanMetric")
        self.normalize = normalize

    def compute_distance(self, x: Tensor, y: Tensor) -> Tensor:
        if self.normalize:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        return torch.cdist(x, y, p=1)


class MahalanobisMetric(MetricSpace):
    """Mahalanobis distance metric.

    d(x, y) = sqrt((x - y)^T M (x - y))

    where M is a positive semi-definite matrix.

    Args:
        dim: Input dimension
        learnable: Whether to make the metric matrix learnable
        init: Initial value for the metric matrix (identity if None)
    """

    def __init__(
        self,
        dim: int,
        learnable: bool = True,
        init: Optional[Tensor] = None,
    ):
        super().__init__(name="MahalanobisMetric")
        self.dim = dim

        if init is None:
            init = torch.eye(dim)

        if learnable:
            self.metric_matrix = nn.Parameter(init)
        else:
            self.register_buffer("metric_matrix", init)

    def compute_distance(self, x: Tensor, y: Tensor) -> Tensor:
        diff = x - y
        mahal = torch.sqrt(torch.sum(diff @ self.metric_matrix * diff, dim=-1) + 1e-10)
        return mahal


class LearnableDistance(nn.Module):
    """Learnable distance metric module.

    A neural network that learns a distance metric from data.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers in the MLP
        activation: Activation function
        learn_scale: Whether to learn a learnable scale
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
        learn_scale: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim

        activation_fn = self._get_activation(activation)

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(activation_fn)
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        if learn_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.ones(1))

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_enc = self.encoder(x)
        y_enc = self.encoder(y)
        dist = torch.norm(x_enc - y_enc, dim=-1) * self.scale
        return dist

    def get_embedding(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class AttentionDistance(nn.Module):
    """Attention-weighted distance metric.

    Uses attention to weight different dimensions/features for distance computation.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden attention dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.scale = self.head_dim**-0.5

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        batch_size = x.shape[0]

        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(y).view(batch_size, self.num_heads, self.head_dim)
        v = self.value(y).view(batch_size, self.num_heads, self.head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        attn_out = torch.matmul(attn, v).view(batch_size, -1)
        attn_out = self.out_proj(attn_out)

        diff = x - y
        weighted_diff = diff * attn_out

        distance = torch.norm(weighted_diff, dim=-1)
        return distance


def compute_distance_matrix(
    features: Tensor,
    metric: Union[str, MetricSpace, Callable] = "euclidean",
    normalize: bool = False,
) -> Tensor:
    """Compute pairwise distance matrix.

    Args:
        features: Input features of shape (n, d)
        metric: Distance metric to use ('euclidean', 'cosine', 'manhattan', or callable)
        normalize: Whether to L2-normalize features

    Returns:
        Distance matrix of shape (n, n)
    """
    if normalize:
        features = F.normalize(features, dim=-1)

    if isinstance(metric, str):
        metric = metric.lower()
        if metric == "euclidean":
            return torch.cdist(features, features, p=2)
        elif metric == "cosine":
            features_norm = F.normalize(features, dim=-1)
            similarity = torch.mm(features_norm, features_norm.T)
            return 1.0 - similarity
        elif metric == "manhattan":
            return torch.cdist(features, features, p=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        n = features.shape[0]
        dist_matrix = torch.zeros(n, n, device=features.device)
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = metric(features[i], features[j])
        return dist_matrix
    else:
        raise TypeError(f"metric must be str or callable, got {type(metric)}")


def compute_similarity_matrix(
    features: Tensor,
    metric: str = "cosine",
    temperature: float = 1.0,
    normalize: bool = True,
) -> Tensor:
    """Compute pairwise similarity matrix.

    Args:
        features: Input features of shape (n, d)
        metric: Similarity metric ('cosine', 'dot')
        temperature: Temperature scaling for similarity
        normalize: Whether to L2-normalize features

    Returns:
        Similarity matrix of shape (n, n)
    """
    if normalize:
        features = F.normalize(features, dim=-1)

    if metric == "cosine":
        similarity = torch.mm(features, features.T)
    elif metric == "dot":
        similarity = torch.mm(features, features.T)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarity / temperature


__all__ = [
    "MetricSpace",
    "EuclideanMetric",
    "CosineMetric",
    "ManhattanMetric",
    "MahalanobisMetric",
    "LearnableDistance",
    "AttentionDistance",
    "compute_distance_matrix",
    "compute_similarity_matrix",
]
