"""
Equivariant Neural Networks.

Implements E(n) and SE(3) equivariant layers for:
- Point cloud processing
- Molecular/crystalline systems
- 3D vision tasks

Based on:
- E(n) Equivariant Graph Neural Networks (Satorras et al., 2021)
- SE(3)-Transformers (Fuchs et al., 2020)
"""

from typing import Optional, Tuple, Dict, List, Callable
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class SE3EquivariantLayer(nn.Module):
    """
    SE(3)-equivariant layer for point clouds.

    Maintains equivariance to rotations and translations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 64,
        edge_dim: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Message network (invariant)
        edge_input_dim = in_features * 2 + edge_dim + 1  # +1 for distance
        self.message_net = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update (equivariant)
        self.coord_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Feature update
        self.feature_net = nn.Sequential(
            nn.Linear(in_features + hidden_dim, out_features),
            nn.SiLU(),
        )

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            features: Node features [n_nodes, in_features]
            coords: Node coordinates [n_nodes, 3]
            edge_index: Edge indices [2, n_edges]
            edge_attr: Edge attributes [n_edges, edge_dim]

        Returns:
            new_features: Updated features [n_nodes, out_features]
            new_coords: Updated coords [n_nodes, 3] (equivariant)
        """
        row, col = edge_index

        # Compute edge features (invariant to SE(3))
        edge_diff = coords[row] - coords[col]  # [n_edges, 3]
        edge_dist = (edge_diff**2).sum(dim=-1, keepdim=True)  # [n_edges, 1]

        edge_features = [features[row], features[col], edge_dist]
        if edge_attr is not None:
            edge_features.append(edge_attr)
        edge_features = torch.cat(edge_features, dim=-1)

        # Messages (invariant)
        messages = self.message_net(edge_features)  # [n_edges, hidden_dim]

        # Coordinate update (equivariant)
        coord_weights = self.coord_net(messages)  # [n_edges, 1]
        coord_update = coord_weights * edge_diff  # [n_edges, 3]

        # Aggregate for coordinates
        new_coords = coords.clone()
        new_coords.index_add_(0, row, coord_update)

        # Aggregate for features
        aggregated = torch.zeros(
            features.size(0), messages.size(-1), device=features.device
        )
        aggregated.index_add_(0, row, messages)

        # Update features
        new_features = self.feature_net(torch.cat([features, aggregated], dim=-1))

        return new_features, new_coords


class SE3Transformer(nn.Module):
    """
    SE(3)-equivariant transformer.

    Attention mechanism respecting SE(3) symmetries.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # SE(3)-aware attention weight
        self.attention_weight = nn.Sequential(
            nn.Linear(1, num_heads),  # Distance to attention bias
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            features: Node features [n_nodes, feature_dim]
            coords: Node coordinates [n_nodes, 3]
            edge_index: Edge indices [2, n_edges]

        Returns:
            Updated features [n_nodes, feature_dim]
        """
        n_nodes = features.size(0)
        row, col = edge_index

        # Project to Q, K, V
        q = self.q_proj(features).view(n_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(features).view(n_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(features).view(n_nodes, self.num_heads, self.head_dim)

        # Compute attention scores
        attn = (q[row] * k[col]).sum(dim=-1) / math.sqrt(self.head_dim)

        # SE(3)-aware bias from distances
        distances = torch.norm(coords[row] - coords[col], dim=-1, keepdim=True)
        geom_bias = self.attention_weight(distances)
        attn = attn * geom_bias

        # Softmax
        attn = F.softmax(attn, dim=0)

        # Apply attention - aggregate per target node
        out = torch.zeros(
            n_nodes, self.num_heads, self.head_dim, device=features.device
        )
        out.index_add_(0, row, attn.unsqueeze(-1) * v[col])
        out = out.view(n_nodes, self.feature_dim)

        return self.out_proj(out)


class E3Conv(nn.Module):
    """
    E(3)-equivariant convolution.

    Handles both rotations and reflections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Filter generating network
        self.filter_net = nn.Sequential(
            nn.Linear(1, hidden_channels),  # Input: distance
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )

        # Mixing coefficients
        self.coeff_net = nn.Linear(hidden_channels, in_channels * out_channels)

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            features: Node features [n_nodes, in_channels]
            coords: Node coordinates [n_nodes, 3]
            edge_index: Edge indices [2, n_edges]

        Returns:
            Updated features [n_nodes, out_channels]
        """
        row, col = edge_index

        # Relative positions
        rel_pos = coords[row] - coords[col]  # [n_edges, 3]
        distances = torch.norm(rel_pos, dim=-1, keepdim=True)  # [n_edges, 1]

        # Generate filters
        filters = self.filter_net(distances)  # [n_edges, hidden_channels]
        coeffs = self.coeff_net(filters)  # [n_edges, in_channels * out_channels]
        coeffs = coeffs.view(-1, self.out_channels, self.in_channels)

        # Apply to neighbor features
        neighbor_features = features[col]  # [n_edges, in_channels]
        messages = torch.einsum("noi,ei->no", coeffs, neighbor_features)

        # Aggregate
        out = torch.zeros(features.size(0), self.out_channels, device=features.device)
        out.index_add_(0, row, messages)

        return out


class RadialBasisFunctions(nn.Module):
    """Radial basis functions for encoding distances."""

    def __init__(
        self,
        n_rbf: int = 50,
        cutoff: float = 10.0,
        rbf_type: str = "gaussian",
    ):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        # Centers and widths for Gaussian RBF
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centers", centers)

        width = (cutoff / n_rbf) * torch.ones(n_rbf)
        self.register_buffer("width", width)

    def forward(self, distances: Tensor) -> Tensor:
        """
        Compute RBF features.

        Args:
            distances: Pairwise distances [n_edges]

        Returns:
            RBF features [n_edges, n_rbf]
        """
        distances = distances.unsqueeze(-1)  # [n_edges, 1]

        # Gaussian RBF
        rbf = torch.exp(-((distances - self.centers) ** 2) / self.width)

        # Cosine cutoff
        cutoff_vals = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1)
        cutoff_vals = cutoff_vals * (distances < self.cutoff).float()

        return rbf * cutoff_vals


class SphericalBasisLayer(nn.Module):
    """
    Spherical harmonics basis for angular features.

    For encoding directional information.
    """

    def __init__(self, l_max: int = 2):
        super().__init__()
        self.l_max = l_max
        self.n_harmonics = (l_max + 1) ** 2

    def forward(self, vectors: Tensor) -> Tensor:
        """
        Compute spherical harmonics up to l_max.

        Args:
            vectors: Direction vectors [n_edges, 3]

        Returns:
            Spherical harmonic features [n_edges, n_harmonics]
        """
        # Normalize
        vectors = vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-8)
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

        features = []

        # l = 0
        features.append(torch.ones_like(x) * 0.5 / math.sqrt(math.pi))

        # l = 1
        features.append(-0.5 * math.sqrt(3 / (2 * math.pi)) * (x + 1j * y))
        features.append(0.5 * math.sqrt(3 / math.pi) * z)
        features.append(0.5 * math.sqrt(3 / (2 * math.pi)) * (x - 1j * y))

        if self.l_max >= 2:
            # l = 2
            features.append(0.25 * math.sqrt(15 / (2 * math.pi)) * ((x + 1j * y) ** 2))
            features.append(-0.5 * math.sqrt(15 / (2 * math.pi)) * z * (x + 1j * y))
            features.append(0.25 * math.sqrt(5 / math.pi) * (2 * z**2 - x**2 - y**2))
            features.append(0.5 * math.sqrt(15 / (2 * math.pi)) * z * (x - 1j * y))
            features.append(0.25 * math.sqrt(15 / (2 * math.pi)) * ((x - 1j * y) ** 2))

        # Take real part for now (simplified)
        return torch.stack([f.real for f in features], dim=-1)


class EquivariantPointCloudNetwork(nn.Module):
    """
    Complete SE(3)-equivariant network for point clouds.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 1,
        n_layers: int = 4,
    ):
        super().__init__()

        self.embedding = nn.Linear(in_features, hidden_features)

        self.layers = nn.ModuleList(
            [
                SE3EquivariantLayer(
                    in_features=hidden_features if i == 0 else hidden_features,
                    out_features=hidden_features,
                    hidden_dim=hidden_features,
                )
                for i in range(n_layers)
            ]
        )

        # Readout (invariant to global position)
        self.readout = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            features: Node features [n_nodes, in_features]
            coords: Node coordinates [n_nodes, 3]
            edge_index: Edge connectivity [2, n_edges]

        Returns:
            Output [n_nodes, out_features] or scalar
        """
        h = self.embedding(features)

        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)

        return self.readout(h)


class TetrisNetwork(nn.Module):
    """
    Network for Tetris-like block classification (E(3)-invariant).
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()

        self.network = EquivariantPointCloudNetwork(
            in_features=1,
            hidden_features=64,
            out_features=64,
            n_layers=4,
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            features: Block type features [n_blocks, 1]
            coords: Block coordinates [n_blocks, 3]
            edge_index: Connectivity [2, n_edges]
            batch: Batch assignment [n_blocks]

        Returns:
            Class logits [batch_size, num_classes]
        """
        h = self.network(features, coords, edge_index)

        if batch is not None:
            # Global pooling per graph
            batch_size = batch.max().item() + 1
            pooled = torch.zeros(batch_size, h.size(-1), device=h.device)
            pooled.index_add_(0, batch, h)
            counts = torch.bincount(batch, minlength=batch_size).float()
            pooled = pooled / counts.unsqueeze(-1)
        else:
            pooled = h.mean(dim=0, keepdim=True)

        return self.classifier(pooled)


class EquivariantMolecularEnergy(nn.Module):
    """
    Predict molecular energy with SE(3) invariance.
    """

    def __init__(
        self,
        n_atom_types: int = 100,
        hidden_dim: int = 128,
        n_layers: int = 6,
        max_z: float = 100.0,
    ):
        super().__init__()

        # Atom type embedding
        self.atom_embedding = nn.Embedding(n_atom_types, hidden_dim)

        # RBF for distances
        self.rbf = RadialBasisFunctions(n_rbf=50, cutoff=10.0)

        # SE(3) layers
        self.layers = nn.ModuleList(
            [
                SE3EquivariantLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    hidden_dim=hidden_dim,
                    edge_dim=50,  # RBF dimension
                )
                for _ in range(n_layers)
            ]
        )

        # Energy prediction (invariant)
        self.energy_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        atomic_numbers: Tensor,
        coords: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Predict energy.

        Args:
            atomic_numbers: Atom types [n_atoms]
            coords: Atom positions [n_atoms, 3]
            edge_index: Neighbor list [2, n_edges]

        Returns:
            Energy scalar [1]
        """
        # Embed atom types
        features = self.atom_embedding(atomic_numbers)

        # Compute edge features (RBF)
        row, col = edge_index
        distances = torch.norm(coords[row] - coords[col], dim=-1)
        edge_attr = self.rbf(distances)

        # Message passing
        for layer in self.layers:
            features, coords = layer(features, coords, edge_index, edge_attr)

        # Predict atom-wise energies
        atom_energies = self.energy_predictor(features)

        # Total energy (sum, invariant to permutation)
        return atom_energies.sum()


class EquivariantVectorField(nn.Module):
    """
    Learn SE(3)-equivariant vector fields.

    Useful for predicting forces on atoms.
    """

    def __init__(self, feature_dim: int = 64, n_layers: int = 4):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                SE3EquivariantLayer(
                    in_features=feature_dim,
                    out_features=feature_dim,
                    hidden_dim=feature_dim,
                )
                for _ in range(n_layers)
            ]
        )

        # Force prediction (equivariant to coords)
        self.force_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Predict forces (vectors).

        Returns:
            Forces [n_nodes, 3] (equivariant to input coords)
        """
        h = features

        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)

        # Predict force magnitudes (invariant)
        force_magnitudes = self.force_net(h)  # [n_nodes, 1]

        # Multiply by normalized coordinates to get vectors
        # (This is a simplified version - proper implementation uses edge features)
        row, col = edge_index
        rel_pos = coords[row] - coords[col]

        # Aggregate forces
        forces = torch.zeros_like(coords)
        forces.index_add_(0, row, force_magnitudes[row] * rel_pos)

        return forces
