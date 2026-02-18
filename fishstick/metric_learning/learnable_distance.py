"""
Learnable Distance Functions for Metric Learning

Implementation of learnable distance metrics:
- Learnable Euclidean distance with learnable scaling
- Learnable Mahalanobis with parameterized covariance
- Neural network-based distance
- Attention-weighted distance
- Hyperbolic distance (Poincaré ball)
- Learnable metric with learnable transformations
"""

from typing import Optional, Tuple, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LearnableEuclidean(nn.Module):
    """Learnable Euclidean distance with learnable scaling.

    Adds learnable scale and bias parameters to Euclidean distance.

    Args:
        dim: Input dimension
        learn_scale: Whether to learn a learnable scale
        learn_bias: Whether to learn a learnable bias
    """

    def __init__(
        self,
        dim: int,
        learn_scale: bool = True,
        learn_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim

        if learn_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.ones(1))

        if learn_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("bias", torch.zeros(1))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        dist = torch.norm(x - y, dim=-1)
        return dist * self.scale + self.bias

    def get_distance_matrix(self, features: Tensor) -> Tensor:
        """Compute pairwise distance matrix.

        Args:
            features: Input features (n, dim)

        Returns:
            Distance matrix (n, n)
        """
        n = features.shape[0]
        dist_matrix = torch.zeros(n, n, device=features.device)

        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = self.forward(features[i], features[j])

        return dist_matrix


class LearnableMahalanobis(nn.Module):
    """Learnable Mahalanobis distance.

    Parameterizes the Mahalanobis metric matrix M = L^T L.

    Args:
        dim: Input dimension
        num_layers: Number of transformation layers
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        if hidden_dim is None:
            hidden_dim = dim * 2

        layers = []
        in_dim = dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, dim))

        self.transform = nn.Sequential(*layers)

        self.metric = nn.Parameter(torch.eye(dim))

    def _get_metric_matrix(self) -> Tensor:
        """Get the positive semi-definite metric matrix."""
        return self.metric @ self.metric.T + 1e-5 * torch.eye(
            self.dim, device=self.metric.device
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_trans = self.transform(x)
        y_trans = self.transform(y)

        diff = x_trans - y_trans

        M = self._get_metric_matrix()

        dist = torch.sqrt(torch.sum(diff @ M * diff, dim=-1) + 1e-10)
        return dist

    def get_metric(self) -> Tensor:
        """Get the current metric matrix."""
        return self._get_metric_matrix()


class NeuralDistance(nn.Module):
    """Neural network-based learnable distance.

    Uses a neural network to compute a learnable distance function.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        activation: Activation function
        output_dim: Output embedding dimension (if different from input)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim

        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim

        activation_fn = self._get_activation(activation)

        layers = []
        in_dim = input_dim * 2

        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(activation_fn)
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

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
        diff = x - y
        concat = torch.cat([x, y], dim=-1)

        output = self.net(concat)

        dist = torch.norm(output, dim=-1)
        return dist

    def get_embedding(self, x: Tensor) -> Tensor:
        return x


class BilinearDistance(nn.Module):
    """Bilinear learnable distance.

    Uses a learnable bilinear form for computing distance.

    Args:
        dim: Input dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.weight)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        diff = x - y
        weighted = diff @ self.weight
        dist = torch.norm(weighted, dim=-1)
        return dist


class AttentionDistance(nn.Module):
    """Attention-weighted learnable distance.

    Uses multi-head attention to weight feature dimensions.

    Args:
        input_dim: Input feature dimension
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for attention
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        if hidden_dim is None:
            hidden_dim = input_dim

        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, input_dim)

        self.distance_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        batch_size = x.shape[0]

        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(y).view(batch_size, self.num_heads, self.head_dim)
        v = self.value(y).view(batch_size, self.num_heads, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.contiguous().view(batch_size, -1)
        attn_output = self.out_proj(attn_output)

        combined = torch.cat([x, y, x - y, x * y], dim=-1)
        dist_weights = torch.sigmoid(self.distance_net(combined))

        weighted_diff = (x - y) * dist_weights

        distance = torch.norm(weighted_diff, dim=-1)
        return distance


class HyperbolicDistance(nn.Module):
    """Hyperbolic distance in Poincaré ball model.

    Computes distance in hyperbolic space (Poincaré ball).

    Args:
        curvature: Curvature of the hyperbolic space
    """

    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature

    def forward(self, x: Tensor, y: Tensor, c: Optional[float] = None) -> Tensor:
        if c is None:
            c = self.curvature

        x = self._project_to_ball(x, c)
        y = self._project_to_ball(y, c)

        diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)

        x_norm_sq = torch.sum(x**2, dim=-1).clamp(max=1 - 1e-5)
        y_norm_sq = torch.sum(y**2, dim=-1).clamp(max=1 - 1e-5)

        numerator = diff_norm_sq
        denominator = (1 - x_norm_sq) * (1 - y_norm_sq)

        arcosh_input = 1 + 2 * numerator / (denominator + 1e-10)
        arcosh_input = arcosh_input.clamp(min=1 + 1e-8)

        distance = torch.acosh(arcosh_input) / torch.sqrt(
            torch.tensor(c, device=x.device)
        )
        return distance

    def _project_to_ball(self, x: Tensor, c: float) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = (1 - 1e-5) / (norm + 1e-10)
        scale = torch.minimum(scale, torch.ones_like(scale))
        return x * scale


class LearnableMetric(nn.Module):
    """General learnable metric module.

    Combines multiple distance functions with learnable weights.

    Args:
        input_dim: Input feature dimension
        distance_types: List of distance types to combine
        learn_weights: Whether to learn combination weights
    """

    def __init__(
        self,
        input_dim: int,
        distance_types: Optional[List[str]] = None,
        learn_weights: bool = True,
    ):
        super().__init__()

        if distance_types is None:
            distance_types = ["euclidean", "cosine"]

        self.distance_types = distance_types

        self.learn_weights = learn_weights

        if learn_weights:
            self.weights = nn.Parameter(torch.ones(len(distance_types)))
        else:
            self.register_buffer("weights", torch.ones(len(distance_types)))

    def _compute_distances(self, x: Tensor, y: Tensor) -> List[Tensor]:
        distances = []

        for dtype in self.distance_types:
            if dtype == "euclidean":
                dist = torch.norm(x - y, dim=-1)
            elif dtype == "cosine":
                x_norm = F.normalize(x, dim=-1)
                y_norm = F.normalize(y, dim=-1)
                dist = 1 - (x_norm * y_norm).sum(dim=-1)
            elif dtype == "manhattan":
                dist = torch.sum(torch.abs(x - y), dim=-1)
            else:
                dist = torch.norm(x - y, dim=-1)

            distances.append(dist)

        return distances

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        distances = self._compute_distances(x, y)

        distances_tensor = torch.stack(distances, dim=-1)

        weights = F.softmax(self.weights, dim=0)

        weighted_dist = (distances_tensor * weights).sum(dim=-1)
        return weighted_dist


class LearnableBregmanDistance(nn.Module):
    """Learnable Bregman distance.

    Implements learnable Bregman divergences.

    Args:
        dim: Input dimension
        num_centroids: Number of learnable centroids
    """

    def __init__(
        self,
        dim: int,
        num_centroids: int = 10,
    ):
        super().__init__()
        self.dim = dim
        self.num_centroids = num_centroids

        self.centroids = nn.Parameter(torch.randn(num_centroids, dim))

        self.phi = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        phi_x = self.phi(x)
        phi_y = self.phi(y)

        diff = (phi_x - phi_y).squeeze(-1)

        dist = torch.abs(diff)
        return dist

    def get_centroids(self) -> Tensor:
        return self.centroids


class GaussianKernelDistance(nn.Module):
    """Gaussian kernel-based learnable distance.

    Uses a learnable Gaussian kernel for computing similarity/distance.

    Args:
        dim: Input dimension
        learn_sigma: Whether to learn the sigma parameter
    """

    def __init__(
        self,
        dim: int,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.dim = dim

        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("sigma", torch.ones(1))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if hasattr(self, "log_sigma"):
            sigma = torch.exp(self.log_sigma)
        else:
            sigma = self.sigma

        diff = x - y
        dist_sq = torch.sum(diff**2, dim=-1)

        distance = 1 - torch.exp(-dist_sq / (2 * sigma**2 + 1e-10))
        return distance


class CosineLearnableDistance(nn.Module):
    """Learnable cosine distance with learnable angle margins.

    Args:
        dim: Input dimension
        learn_margin: Whether to learn angle margin
    """

    def __init__(
        self,
        dim: int,
        learn_margin: bool = True,
    ):
        super().__init__()
        self.dim = dim

        if learn_margin:
            self.margin = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("margin", torch.zeros(1))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)

        cosine = (x_norm * y_norm).sum(dim=-1)

        dist = 1 - cosine + self.margin
        return dist.clamp(min=0)


def create_learnable_distance(
    distance_type: str,
    input_dim: int,
    **kwargs,
) -> nn.Module:
    """Create a learnable distance module.

    Args:
        distance_type: Type of learnable distance
        input_dim: Input feature dimension
        **kwargs: Additional arguments

    Returns:
        Learnable distance module
    """
    distance_types = {
        "euclidean": LearnableEuclidean,
        "mahalanobis": LearnableMahalanobis,
        "neural": NeuralDistance,
        "bilinear": BilinearDistance,
        "attention": AttentionDistance,
        "hyperbolic": HyperbolicDistance,
        "metric": LearnableMetric,
        "gaussian": GaussianKernelDistance,
        "cosine": CosineLearnableDistance,
    }

    distance_type = distance_type.lower()
    if distance_type not in distance_types:
        raise ValueError(f"Unknown distance type: {distance_type}")

    return distance_types[distance_type](input_dim=input_dim, **kwargs)


__all__ = [
    "LearnableEuclidean",
    "LearnableMahalanobis",
    "NeuralDistance",
    "BilinearDistance",
    "AttentionDistance",
    "HyperbolicDistance",
    "LearnableMetric",
    "LearnableBregmanDistance",
    "GaussianKernelDistance",
    "CosineLearnableDistance",
    "create_learnable_distance",
]
