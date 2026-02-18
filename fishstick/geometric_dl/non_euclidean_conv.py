"""
Non-Euclidean Convolutions for Graph Neural Networks.

Implements convolutions on non-Euclidean manifolds:
- Hyperbolic space (Poincaré ball model)
- Lorentz model of hyperbolic space
- Riemannian manifolds

Based on:
- Ganea et al. (2018): Hyperbolic Neural Networks
- Chami et al. (2019): Hyperbolic Graph Convolutional Networks
- Bose et al. (2019): Harmonic Networks for Graph Representation
"""

from typing import Optional, Tuple, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class PoincareEmbedding(nn.Module):
    """
    Embeddings in the Poincaré ball model of hyperbolic space.

    Maps discrete inputs to hyperbolic space for hierarchical data.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.curvature = curvature

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize in Poincaré ball."""
        nn.init.uniform_(self.weight, -1e-3, 1e-3)

    def forward(self, indices: Tensor) -> Tensor:
        """
        Get hyperbolic embeddings.

        Args:
            indices: Token indices [B, ...] or [...]

        Returns:
            Hyperbolic embeddings [..., embedding_dim]
        """
        embeds = F.normalize(self.weight[indices], dim=-1) * 0.1

        return embeds

    def project_to_hyperbolic(self, x: Tensor) -> Tensor:
        """
        Project Euclidean points to Poincaré ball.

        Args:
            x: Points in Euclidean space [..., dim]

        Returns:
            Points in Poincaré ball
        """
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (1 - torch.exp(-norm * self.curvature)) / norm
        return x * scale

    def project_to_euclidean(self, x: Tensor) -> Tensor:
        """
        Project Poincaré ball points to Euclidean (tangent space at origin).

        Args:
            x: Points in Poincaré ball [..., dim]

        Returns:
            Euclidean approximations
        """
        norm_sq = torch.sum(x**2, dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        scale = (1 - norm_sq) / (4 * self.curvature)
        return x * scale


class LorentzEmbedding(nn.Module):
    """
    Embeddings in the Lorentz model (hyperboloid) of hyperbolic space.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.curvature = curvature

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim + 1))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize on hyperboloid."""
        nn.init.uniform_(self.weight, -1e-3, 1e-3)
        with torch.no_grad():
            self.weight[:, 0] = torch.sqrt(
                1 + torch.sum(self.weight[:, 1:] ** 2, dim=-1)
            )

    def forward(self, indices: Tensor) -> Tensor:
        """
        Get Lorentz embeddings.

        Args:
            indices: Token indices

        Returns:
            Lorentz embeddings [..., dim+1]
        """
        return self.weight[indices]

    def project_to_lorentz(self, x: Tensor) -> Tensor:
        """
        Project to Lorentz hyperboloid.

        Args:
            x: Points in R^(dim+1)

        Returns:
            Points on hyperboloid
        """
        x0 = torch.sqrt(1 + torch.sum(x[..., 1:] ** 2, dim=-1, keepdim=True))
        return torch.cat([x0, x[..., 1:]], dim=-1)

    def lorentz_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute Lorentz distance.

        Args:
            x: Points on hyperboloid [..., dim+1]
            y: Points on hyperboloid [..., dim+1]

        Returns:
            Hyperbolic distances
        """
        inner = -x[..., 0:1] * y[..., 0:1] + torch.sum(
            x[..., 1:] * y[..., 1:], dim=-1, keepdim=True
        )
        return torch.acosh(-inner / self.curvature + 1e-5)


class HyperbolicGraphConv(nn.Module):
    """
    Graph convolution in hyperbolic space.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        curvature: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.curvature = curvature

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        self.poincare = PoincareEmbedding(1, in_channels, curvature)

    def exp_map(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Exponential map from tangent space to manifold.

        Args:
            x: Base points on Poincaré ball [..., dim]
            v: Tangent vectors [..., dim]

        Returns:
            Points on manifold
        """
        norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
        second_term = (
            v * torch.tanh(self.curvature * norm_v) / (self.curvature * norm_v)
        )
        return (x + second_term) / (
            1 + self.curvature * torch.sum(x * v, dim=-1, keepdim=True)
        )

    def log_map(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithmic map from manifold to tangent space.

        Args:
            x: Base points on Poincaré ball [..., dim]
            y: Target points on Poincaré ball [..., dim]

        Returns:
            Tangent vectors
        """
        diff = y - x
        norm_diff = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
        coef = torch.atanh(self.curvature * norm_diff) / (self.curvature * norm_diff)
        return coef * diff

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Apply hyperbolic graph convolution.

        Args:
            features: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]

        Returns:
            Updated features [N, out_channels]
        """
        row, col = edge_index

        neighbors = features[col]

        messages = self.linear(neighbors)

        aggregated = torch.zeros_like(features)
        aggregated.index_add_(0, row, messages)

        out = self.linear(aggregated)

        return out


class HyperbolicMLP(nn.Module):
    """
    Multi-layer perceptron in hyperbolic space.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 2,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.curvature = curvature

        layers = []

        layers.append(HyperbolicGraphConv(in_channels, hidden_channels, curvature))
        layers.append(nn.SiLU())

        for _ in range(n_layers - 2):
            layers.append(
                HyperbolicGraphConv(hidden_channels, hidden_channels, curvature)
            )
            layers.append(nn.SiLU())

        layers.append(HyperbolicGraphConv(hidden_channels, out_channels, curvature))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Apply hyperbolic MLP."""
        return self.layers(x, edge_index)


class HyperbolicAttention(nn.Module):
    """
    Attention mechanism in hyperbolic space.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.curvature = curvature

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        self.out_proj = nn.Linear(channels, channels)

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Apply hyperbolic attention.

        Args:
            features: Node features [N, channels]
            edge_index: Edge connectivity [2, E]

        Returns:
            Attended features [N, channels]
        """
        row, col = edge_index

        q = self.q_proj(features)
        k = self.k_proj(features)
        v = self.v_proj(features)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        attn = (q[row] * k[col]).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=0)

        out = torch.zeros_like(v)
        out.index_add_(0, row, attn.unsqueeze(-1) * v[col])

        out = out.view(-1, self.channels)

        return self.out_proj(out)


class RiemannianGNN(nn.Module):
    """
    Graph Neural Network on Riemannian manifolds.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 3,
        manifold: str = "hyperbolic",
        curvature: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = manifold
        self.curvature = curvature

        if manifold == "hyperbolic":
            conv_class = HyperbolicGraphConv
        else:
            conv_class = HyperbolicGraphConv

        self.embedding = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [
                conv_class(hidden_channels, hidden_channels, curvature)
                for _ in range(n_layers)
            ]
        )

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass."""
        x = self.embedding(features)

        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.silu(x)

        return self.classifier(x)


class HyperbolicGraphPooling(nn.Module):
    """
    Graph pooling in hyperbolic space.
    """

    def __init__(
        self,
        channels: int,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.curvature = curvature

        self.attention = nn.Sequential(
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Pool graph-level features in hyperbolic space.

        Args:
            features: Node features [N, channels]
            batch: Batch assignment [N]

        Returns:
            Graph-level features
        """
        weights = self.attention(features)

        if batch is not None:
            batch_size = batch.max().item() + 1
            out = torch.zeros(batch_size, self.channels, device=features.device)

            for i in range(batch_size):
                mask = batch == i
                nodes = features[mask]
                node_weights = weights[mask]

                weighted = nodes * node_weights
                out[i] = weighted.sum(dim=0)

            return out
        else:
            return (features * weights).sum(dim=0, keepdim=True)


class HyperbolicDistance(nn.Module):
    """
    Compute distances in hyperbolic space.
    """

    def __init__(self, model: str = "poincare", curvature: float = 1.0):
        super().__init__()
        self.model = model
        self.curvature = curvature

    def poincare_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute distance in Poincaré ball.

        Args:
            x: Points [N, D]
            y: Points [N, D]

        Returns:
            Distances
        """
        diff_norm = torch.norm(x - y, dim=-1)
        x_norm = torch.norm(x, dim=-1).clamp(max=1 - 1e-5)
        y_norm = torch.norm(y, dim=-1).clamp(max=1 - 1e-5)

        numerator = diff_norm**2
        denominator = (1 - x_norm**2) * (1 - y_norm**2)

        return torch.acosh(1 + 2 * numerator / denominator.clamp(min=1e-8))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute hyperbolic distance."""
        if self.model == "poincare":
            return self.poincare_distance(x, y)
        else:
            return self.poincare_distance(x, y)


class LorentzModelConv(nn.Module):
    """
    Graph convolution in Lorentz (hyperboloid) model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.curvature = curvature

        self.linear = nn.Linear(in_channels + 1, out_channels + 1)

        self.lorentz = LorentzEmbedding(1, in_channels, curvature)

    def lorentz_add(self, x: Tensor, y: Tensor) -> Tensor:
        """Addition in Lorentz model."""
        return x + y

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Apply Lorentz graph convolution.

        Args:
            features: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]

        Returns:
            Updated features [N, out_channels+1]
        """
        row, col = edge_index

        x = self.lorentz.project_to_lorentz(
            torch.cat([torch.zeros_like(features[:, :1]), features], dim=-1)
        )

        neighbors = x[col]

        messages = self.linear(neighbors)

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, row, messages)

        out = self.linear(aggregated)

        return out[:, 1:]


class HyperbolicBatchNorm(nn.Module):
    """
    Batch normalization in hyperbolic space.
    """

    def __init__(self, channels: int, curvature: float = 1.0):
        super().__init__()
        self.channels = channels
        self.curvature = curvature

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        """Apply hyperbolic batch normalization."""
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)

        x_norm = (x - mean) / (var + 1e-8).sqrt()

        return x_norm * self.weight + self.bias


__all__ = [
    "PoincareEmbedding",
    "LorentzEmbedding",
    "HyperbolicGraphConv",
    "HyperbolicMLP",
    "HyperbolicAttention",
    "RiemannianGNN",
    "HyperbolicGraphPooling",
    "HyperbolicDistance",
    "LorentzModelConv",
    "HyperbolicBatchNorm",
]
