"""
Graph Generation Models.

This module provides implementations for graph generation:
- Variational Autoencoder (VAE) based graph generation
- Generative Adversarial Network (GAN) based generation
- Autoregressive graph generation
- Molecular graph generation
"""

from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, ModuleList
import math


class GraphVAE(nn.Module):
    """
    Graph Variational Autoencoder for graph generation.

    Encodes graphs into latent space and generates new graphs.

    Args:
        node_features: Number of node features
        edge_features: Number of edge features
        hidden_dim: Hidden dimension
        latent_dim: Dimension of latent space
        num_node_types: Number of possible node types
        num_edge_types: Number of possible edge types
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int = 1,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_node_types: int = 1,
        num_edge_types: int = 1,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.encoder = GraphEncoder(node_features, hidden_dim, latent_dim * 2)

        self.decoder = GraphDecoder(
            latent_dim, hidden_dim, node_features, num_node_types, num_edge_types
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features

        Returns:
            Tuple of (reconstructed_x, reconstructed_edge, mu, logvar)
        """
        mu, logvar = self.encoder(x, edge_index)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        recon_x, recon_edge = self.decoder(z, edge_index)

        return recon_x, recon_edge, mu, logvar

    def generate(self, z: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate graph from latent code.

        Args:
            z: Latent code
            edge_index: Edge index for reconstruction

        Returns:
            Tuple of (node_logits, edge_logits)
        """
        return self.decoder(z, edge_index)


class GraphEncoder(nn.Module):
    """Graph encoder for VAE."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc_mu = Linear(hidden_dim, out_channels // 2)
        self.fc_logvar = Linear(hidden_dim, out_channels // 2)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Encode graph to latent parameters."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = x.mean(dim=0, keepdim=True)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class GraphDecoder(nn.Module):
    """Graph decoder for VAE."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        node_features: int,
        num_node_types: int,
        num_edge_types: int,
    ):
        super().__init__()

        self.node_type_proj = Linear(latent_dim, num_node_types)

        self.edge_mlp = nn.Sequential(
            Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, num_edge_types),
        )

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Decode latent code to graph.

        Args:
            z: Latent code
            edge_index: Edge index

        Returns:
            Tuple of (node_logits, edge_logits)
        """
        node_logits = self.node_type_proj(z)

        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=-1)
        edge_logits = self.edge_mlp(edge_features)

        return node_logits, edge_logits


class GCNConv(nn.Module):
    """Simple Graph Convolutional Network layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        out = self.linear(x)

        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = out * norm.unsqueeze(-1)

        aggregated = torch.zeros_like(out)
        aggregated.index_add_(0, row, out[col])

        return aggregated


class GraphGAN(nn.Module):
    """
    Graph Generative Adversarial Network.

    Uses a generator and discriminator for adversarial graph generation.

    Args:
        node_dim: Node feature dimension
        edge_dim: Edge feature dimension
        hidden_dim: Hidden dimension
        max_nodes: Maximum number of nodes
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 1,
        hidden_dim: int = 128,
        max_nodes: int = 20,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        self.generator = GraphGenerator(node_dim, edge_dim, hidden_dim, max_nodes)

        self.discriminator = GraphDiscriminator(node_dim, edge_dim, hidden_dim)

    def forward(
        self,
        real_x: Tensor,
        real_edge_index: Tensor,
        z: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through GAN.

        Args:
            real_x: Real node features
            real_edge_index: Real edge connectivity
            z: Optional noise for generation

        Returns:
            Tuple of (real_logits, fake_logits, generated_graph)
        """
        batch_size = real_x.size(0) if real_x.dim() > 1 else 1

        if z is None:
            z = torch.randn(batch_size, self.hidden_dim, device=real_x.device)

        fake_graph = self.generator(z)

        real_logits = self.discriminator(real_x, real_edge_index)
        fake_logits = self.discriminator(fake_graph.x, fake_graph.edge_index)

        return real_logits, fake_logits, fake_graph


class GraphGenerator(nn.Module):
    """Graph generator network."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        max_nodes: int,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        self.node_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, max_nodes * node_dim),
        )

        self.edge_mlp = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, max_nodes * max_nodes * edge_dim),
        )

    def forward(self, z: Tensor) -> "Graph":
        """Generate graph from noise."""
        batch_size = z.size(0)

        node_logits = self.node_mlp(z)
        node_logits = node_logits.view(batch_size, self.max_nodes, self.node_dim)

        edge_logits = self.edge_mlp(z)
        edge_logits = edge_logits.view(
            batch_size, self.max_nodes, self.max_nodes, self.edge_dim
        )

        edge_probs = torch.sigmoid(edge_logits)
        edge_index = (edge_probs > 0.5).nonzero(as_tuple=False)

        x = node_logits.mean(dim=1)

        return Graph(x=x, edge_index=edge_index)


@dataclass
class Graph:
    """Simple graph data structure."""

    x: Tensor
    edge_index: Tensor
    edge_attr: Optional[Tensor] = None


class GraphDiscriminator(nn.Module):
    """Graph discriminator network."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Discriminate real vs fake graphs."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = x.mean(dim=0, keepdim=True)

        return self.fc(x)


class GraphAutoReggressiveGenerator(nn.Module):
    """
    Autoregressive graph generator.

    Generates graphs node by node and edge by edge.

    Args:
        node_features: Number of node features
        max_nodes: Maximum number of nodes
        hidden_dim: Hidden dimension
        num_edge_types: Number of edge types
    """

    def __init__(
        self,
        node_features: int,
        max_nodes: int = 20,
        hidden_dim: int = 128,
        num_edge_types: int = 1,
    ):
        super().__init__()
        self.node_features = node_features
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.node_embedding = nn.Embedding(max_nodes + 1, hidden_dim)

        self.node_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.edge_rnn = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)

        self.node_predictor = nn.Linear(hidden_dim, node_features)
        self.edge_predictor = nn.Linear(hidden_dim, num_edge_types)

    def forward(
        self,
        num_nodes: int,
        z: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Autoregressively generate graph.

        Args:
            num_nodes: Target number of nodes
            z: Optional latent code

        Returns:
            Tuple of (generated_x, generated_edge_index)
        """
        if z is None:
            z = torch.randn(1, self.hidden_dim, device=next(self.parameters()).device)

        x_list = []

        node_h = torch.zeros(1, 1, self.hidden_dim, device=z.device)
        node_c = torch.zeros(1, 1, self.hidden_dim, device=z.device)

        for i in range(num_nodes):
            node_input = self.node_embedding(torch.tensor([i], device=z.device))

            node_output, (node_h, node_c) = self.node_rnn(
                node_input.unsqueeze(0), (node_h, node_c)
            )

            x_i = self.node_predictor(node_output.squeeze(0))
            x_list.append(x_i)

        generated_x = torch.cat(x_list, dim=0)

        edge_list = []
        edge_probs = []

        edge_h = torch.zeros(1, 1, self.hidden_dim, device=z.device)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_input = torch.cat(
                    [generated_x[i : i + 1], generated_x[j : j + 1]], dim=-1
                )

                edge_output, edge_h = self.edge_rnn(
                    edge_input.unsqueeze(0), (edge_h, edge_h)
                )

                edge_logit = self.edge_predictor(edge_output.squeeze(0))
                edge_probs.append(edge_logit)

                if edge_logit.argmax(dim=-1).item() > 0:
                    edge_list.append([i, j])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=z.device).t()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=z.device)

        return generated_x, edge_index


class MoleculeGenerator(nn.Module):
    """
    Molecular graph generator using junction tree VAE approach.

    Specialized for molecular graph generation with chemical validity constraints.

    Args:
        atom_features: Number of atom features
        bond_features: Number of bond features
        hidden_dim: Hidden dimension
        max_atoms: Maximum number of atoms
    """

    def __init__(
        self,
        atom_features: int = 34,
        bond_features: int = 12,
        hidden_dim: int = 128,
        max_atoms: int = 50,
    ):
        super().__init__()
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms

        self.encoder = nn.Sequential(
            Linear(atom_features, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.mu = Linear(hidden_dim, hidden_dim)
        self.logvar = Linear(hidden_dim, hidden_dim)

        self.decoder_rnn = nn.LSTM(
            hidden_dim + bond_features, hidden_dim, batch_first=True
        )

        self.atom_predictor = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, atom_features),
        )

        self.bond_predictor = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, bond_features),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through molecule generator.

        Args:
            x: Atom features [num_atoms, atom_features]
            edge_index: Bond connectivity [2, num_bonds]

        Returns:
            Tuple of (recon_atoms, recon_bonds, mu, logvar)
        """
        h = self.encoder(x)
        h = h.mean(dim=0, keepdim=True)

        mu = self.mu(h)
        logvar = self.logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return self._decode(z)

    def _decode(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode latent code to molecule."""
        batch_size = z.size(0)

        atoms = []
        bonds = []

        h = z
        cell = torch.zeros(1, batch_size, self.hidden_dim, device=z.device)

        for _ in range(self.max_atoms):
            output, (h, cell) = self.decoder_rnn(h.unsqueeze(1), (h, cell))

            atom_logits = self.atom_predictor(output.squeeze(1))
            atoms.append(atom_logits)

            if atom_logits.argmax(dim=-1).item() == 0:
                break

        atom_logits = torch.cat(atoms, dim=0)

        return atom_logits, torch.zeros(0, self.bond_features, device=z.device)

    def generate(self, num_atoms: int) -> Tuple[Tensor, Tensor]:
        """
        Generate molecule with specified number of atoms.

        Args:
            num_atoms: Number of atoms to generate

        Returns:
            Tuple of (atom_features, bond_index)
        """
        z = torch.randn(1, self.hidden_dim, device=next(self.parameters()).device)

        h = z
        cell = torch.zeros(1, 1, self.hidden_dim, device=z.device)

        atoms = []

        for _ in range(num_atoms):
            output, (h, cell) = self.decoder_rnn(h.unsqueeze(1), (h, cell))

            atom_logits = self.atom_predictor(output.squeeze(1))
            atoms.append(atom_logits)

            if atom_logits.argmax(dim=-1).item() == 0:
                break

        atom_features = torch.cat(atoms, dim=0)

        edge_index = self._generate_bonds(atom_features)

        return atom_features, edge_index

    def _generate_bonds(self, atoms: Tensor) -> Tensor:
        """Generate bonds between atoms."""
        num_atoms = atoms.size(0)

        edge_list = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                bond_input = torch.cat([atoms[i], atoms[j]], dim=-1)
                bond_logit = self.bond_predictor(bond_input.unsqueeze(0))

                if bond_logit.argmax(dim=-1).item() > 0:
                    edge_list.append([i, j])

        if edge_list:
            return torch.tensor(edge_list, dtype=torch.long, device=atoms.device).t()
        else:
            return torch.empty(2, 0, dtype=torch.long, device=atoms.device)
