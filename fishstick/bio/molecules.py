"""
Comprehensive Molecular Generation Module for Fishstick

This module provides tools for molecular generation, property prediction,
and optimization using deep learning approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict

# RDKit imports for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, QED, RDConfig
    from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles
    from rdkit.Chem import RWMol, BondType
    from rdkit.Chem import Descriptors3D, Crippen
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import rdRascalMCES

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Some molecular features will be limited.")

# SELFIES support
try:
    import selfies as sf

    SELFIES_AVAILABLE = True
except ImportError:
    SELFIES_AVAILABLE = False
    warnings.warn("SELFIES not available. SELFIES representation will be disabled.")

# Graph neural network support
try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing, GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    warnings.warn(
        "PyTorch Geometric not available. Graph-based models will be limited."
    )

# =============================================================================
# MOLECULAR REPRESENTATIONS
# =============================================================================


@dataclass
class MolecularRepresentation(ABC):
    """Base class for molecular representations."""

    @abstractmethod
    def to_smiles(self) -> str:
        """Convert to SMILES string."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if representation is valid."""
        pass


@dataclass
class SMILES(MolecularRepresentation):
    """SMILES molecular representation."""

    smiles: str

    def __post_init__(self):
        self.smiles = self.smiles.strip()

    def to_smiles(self) -> str:
        return self.smiles

    def is_valid(self) -> bool:
        if not RDKIT_AVAILABLE:
            return True  # Assume valid if RDKit unavailable
        mol = MolFromSmiles(self.smiles)
        return mol is not None

    def to_mol(self):
        """Convert to RDKit molecule object."""
        if not RDKIT_AVAILABLE:
            return None
        return MolFromSmiles(self.smiles)

    def to_selfies(self) -> "SELFIES":
        """Convert to SELFIES representation."""
        if not SELFIES_AVAILABLE:
            raise ImportError("SELFIES library required")
        encoded = sf.encoder(self.smiles)
        return SELFIES(encoded)

    def to_graph(self) -> "Graph":
        """Convert to graph representation."""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required")
        return smiles_to_graph(self.smiles)


@dataclass
class SELFIES(MolecularRepresentation):
    """SELFIES molecular representation."""

    selfies: str

    def __post_init__(self):
        self.selfies = self.selfies.strip()

    def to_smiles(self) -> str:
        if not SELFIES_AVAILABLE:
            return self.selfies
        return sf.decoder(self.selfies)

    def is_valid(self) -> bool:
        if not SELFIES_AVAILABLE:
            return True
        try:
            sf.decoder(self.selfies)
            return True
        except:
            return False

    def to_mol(self):
        """Convert to RDKit molecule object."""
        if not RDKIT_AVAILABLE:
            return None
        smiles = self.to_smiles()
        return MolFromSmiles(smiles)


@dataclass
class Graph(MolecularRepresentation):
    """Graph molecular representation."""

    atom_features: torch.Tensor  # [num_atoms, atom_feature_dim]
    bond_features: torch.Tensor  # [num_bonds, bond_feature_dim]
    edge_index: torch.Tensor  # [2, num_bonds]
    smiles: Optional[str] = None

    def to_smiles(self) -> str:
        if self.smiles is not None:
            return self.smiles
        if RDKIT_AVAILABLE:
            return graph_to_smiles(self)
        raise ValueError("Cannot convert graph to SMILES without RDKit")

    def is_valid(self) -> bool:
        return self.atom_features.shape[0] > 0

    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required")
        return Data(
            x=self.atom_features,
            edge_index=self.edge_index,
            edge_attr=self.bond_features,
        )


@dataclass
class Conformer3D:
    """3D molecular conformer."""

    coordinates: torch.Tensor  # [num_atoms, 3]
    atom_types: List[str]  # Atom element symbols
    smiles: Optional[str] = None
    energy: Optional[float] = None

    def to_mol(self):
        """Convert to RDKit molecule with 3D coordinates."""
        if not RDKIT_AVAILABLE:
            return None

        mol = Chem.RWMol()
        for atom_type in self.atom_types:
            mol.AddAtom(Chem.Atom(atom_type))

        # Create conformer and set coordinates
        conf = Chem.Conformer(len(self.atom_types))
        for i, coord in enumerate(self.coordinates):
            conf.SetAtomPosition(i, coord.tolist())
        mol.AddConformer(conf)

        return mol.GetMol()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def smiles_to_graph(
    smiles: str,
    atom_features_fn: Optional[Callable] = None,
    bond_features_fn: Optional[Callable] = None,
) -> Graph:
    """
    Convert SMILES string to graph representation.

    Args:
        smiles: SMILES string
        atom_features_fn: Function to compute atom features
        bond_features_fn: Function to compute bond features

    Returns:
        Graph representation
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required for SMILES to graph conversion")

    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Default atom features
    if atom_features_fn is None:

        def atom_features_fn(atom):
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic()),
                atom.GetHybridization().real,
            ]
            return features

    # Default bond features
    if bond_features_fn is None:

        def bond_features_fn(bond):
            features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
            ]
            return features

    # Extract atoms
    atom_list = []
    for atom in mol.GetAtoms():
        atom_list.append(atom_features_fn(atom))
    atom_features = torch.tensor(atom_list, dtype=torch.float)

    # Extract bonds
    bond_list = []
    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = bond_features_fn(bond)
        bond_list.append(bond_feat)
        edge_list.append([i, j])
        edge_list.append([j, i])  # Undirected

    if len(bond_list) > 0:
        bond_features = torch.tensor(bond_list * 2, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        bond_features = torch.zeros((0, 3))
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Graph(
        atom_features=atom_features,
        bond_features=bond_features,
        edge_index=edge_index,
        smiles=smiles,
    )


def graph_to_smiles(graph: Graph) -> str:
    """
    Convert graph representation back to SMILES.

    Note: This is a heuristic reconstruction and may not always produce
    the exact original molecule.

    Args:
        graph: Graph representation

    Returns:
        SMILES string
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required for graph to SMILES conversion")

    mol = Chem.RWMol()

    # Add atoms
    atom_types = []
    for feat in graph.atom_features:
        # Heuristic: first feature is often atomic number
        atomic_num = int(feat[0].item()) if feat[0] > 0 else 6
        atom_types.append(atomic_num)
        mol.AddAtom(Chem.Atom(atomic_num))

    # Add bonds
    edge_index = graph.edge_index.numpy()
    added_bonds = set()

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < dst and (src, dst) not in added_bonds:
            # Heuristic: first bond feature is bond type
            bond_type_val = (
                graph.bond_features[i, 0].item()
                if len(graph.bond_features) > 0
                else 1.0
            )

            if bond_type_val == 2.0:
                bond_type = BondType.DOUBLE
            elif bond_type_val == 3.0:
                bond_type = BondType.TRIPLE
            elif bond_type_val == 1.5:
                bond_type = BondType.AROMATIC
            else:
                bond_type = BondType.SINGLE

            mol.AddBond(int(src), int(dst), bond_type)
            added_bonds.add((src, dst))

    try:
        smiles = MolToSmiles(mol.GetMol())
        return smiles
    except:
        return ""


def rdkit_converter(input_mol, output_format: str = "smiles"):
    """
    Universal RDKit converter for various molecular formats.

    Args:
        input_mol: Input molecule (SMILES, RDKit Mol, Graph, etc.)
        output_format: Desired output format ('smiles', 'mol', 'graph', 'inchi')

    Returns:
        Converted molecule
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required")

    # Convert input to RDKit Mol
    if isinstance(input_mol, str):
        mol = MolFromSmiles(input_mol)
    elif isinstance(input_mol, Chem.Mol):
        mol = input_mol
    elif isinstance(input_mol, Graph):
        mol = MolFromSmiles(input_mol.to_smiles())
    elif isinstance(input_mol, SMILES):
        mol = input_mol.to_mol()
    else:
        raise ValueError(f"Unsupported input type: {type(input_mol)}")

    # Convert to output format
    if output_format == "smiles":
        return MolToSmiles(mol) if mol else None
    elif output_format == "mol":
        return mol
    elif output_format == "graph":
        return smiles_to_graph(MolToSmiles(mol)) if mol else None
    elif output_format == "inchi":
        return Chem.inchi.MolToInchi(mol) if mol else None
    else:
        raise ValueError(f"Unknown output format: {output_format}")


# =============================================================================
# GENERATION MODELS
# =============================================================================


class VAE_Molecule(nn.Module):
    """
    Variational Autoencoder for molecular generation.

    Encodes molecules into a latent space and decodes back.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution."""
        embedded = self.embedding(x)
        _, (h_n, _) = self.encoder_lstm(embedded)

        # Concatenate forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, max_length: int = 120) -> torch.Tensor:
        """Decode latent vector to sequence."""
        batch_size = z.size(0)

        # Initialize hidden state
        h = (
            self.latent_to_hidden(z)
            .unsqueeze(0)
            .repeat(self.decoder_lstm.num_layers, 1, 1)
        )
        c = torch.zeros_like(h)

        # Start token
        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        outputs = []

        for _ in range(max_length):
            embedded = self.embedding(input_token)
            output, (h, c) = self.decoder_lstm(embedded, (h, c))
            logits = self.output_layer(output.squeeze(1))
            outputs.append(logits)

            # Greedy decoding
            input_token = logits.argmax(dim=-1, keepdim=True)

        return torch.stack(outputs, dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, max_length=x.size(1))
        return recon_x, mu, logvar


class AAE_Molecule(nn.Module):
    """
    Adversarial Autoencoder for molecular generation.

    Uses adversarial training to match latent space to prior distribution.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # Encoder (same as VAE)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.fc_latent = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder (same as VAE)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        # Discriminator for adversarial training
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        embedded = self.embedding(x)
        _, (h_n, _) = self.encoder_lstm(embedded)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc_latent(h)

    def decode(self, z: torch.Tensor, max_length: int = 120) -> torch.Tensor:
        """Decode latent vector to sequence."""
        batch_size = z.size(0)
        h = (
            self.latent_to_hidden(z)
            .unsqueeze(0)
            .repeat(self.decoder_lstm.num_layers, 1, 1)
        )
        c = torch.zeros_like(h)

        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        outputs = []

        for _ in range(max_length):
            embedded = self.embedding(input_token)
            output, (h, c) = self.decoder_lstm(embedded, (h, c))
            logits = self.output_layer(output.squeeze(1))
            outputs.append(logits)
            input_token = logits.argmax(dim=-1, keepdim=True)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        z = self.encode(x)
        return self.decode(z, max_length=x.size(1))


class CharRNN(nn.Module):
    """
    Character-level RNN for sequential molecular generation.

    Generates molecules one character at a time.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass."""
        embedded = self.dropout(self.embedding(x))

        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)

        output = self.dropout(output)
        logits = self.fc(output)

        return logits, hidden

    def generate(
        self,
        start_token: int,
        max_length: int = 100,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> List[int]:
        """Generate sequence."""
        self.eval()
        with torch.no_grad():
            tokens = [start_token]
            hidden = None

            for _ in range(max_length):
                input_tensor = torch.tensor([[tokens[-1]]], device=device)
                logits, hidden = self.forward(input_tensor, hidden)

                # Apply temperature
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                # Sample from distribution
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)

                # Stop if end token
                if next_token == 0:
                    break

            return tokens


class TransformerMolecule(nn.Module):
    """
    Transformer-based molecular generation model.

    Uses self-attention for capturing long-range dependencies.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        src_emb = self.pos_encoder(self.embedding(src) * np.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * np.sqrt(self.d_model))

        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.fc_out(output)

    def generate(
        self, start_sequence: torch.Tensor, max_length: int = 100, device: str = "cpu"
    ) -> torch.Tensor:
        """Generate sequence autoregressively."""
        self.eval()
        with torch.no_grad():
            generated = start_sequence.clone()

            for _ in range(max_length):
                output = self.forward(generated, generated)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == 0:  # End token
                    break

            return generated


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class JTNN(nn.Module):
    """
    Junction Tree Variational Autoencoder.

    Generates molecules by first generating a scaffold (junction tree)
    then assembling molecular graph from tree nodes.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 450,
        latent_size: int = 56,
        depth: int = 3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth

        # Tree encoder
        self.tree_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Graph encoder
        self.graph_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Latent spaces
        self.tree_latent = nn.Linear(hidden_size, latent_size * 2)
        self.graph_latent = nn.Linear(hidden_size, latent_size * 2)

        # Decoders
        self.tree_decoder = JTNNDecoder(vocab_size, hidden_size, latent_size)
        self.graph_decoder = nn.ModuleList(
            [nn.Linear(latent_size, hidden_size) for _ in range(depth)]
        )

    def encode(self, tree_tensors, graph_tensors):
        """Encode tree and graph."""
        tree_vecs, _ = self.tree_encoder(tree_tensors)
        graph_vecs, _ = self.graph_encoder(graph_tensors)

        tree_vecs = tree_vecs[:, -1, :]
        graph_vecs = graph_vecs[:, -1, :]

        tree_latent = self.tree_latent(tree_vecs)
        graph_latent = self.graph_latent(graph_vecs)

        return tree_latent, graph_latent

    def forward(self, tree_tensors, graph_tensors):
        """Forward pass."""
        tree_latent, graph_latent = self.encode(tree_tensors, graph_tensors)
        return tree_latent, graph_latent


class JTNNDecoder(nn.Module):
    """Decoder for junction tree."""

    def __init__(self, vocab_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, z: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Decode latent vector."""
        # Simplified implementation
        batch_size = z.size(0)
        h = z.unsqueeze(0)

        if targets is not None:
            embedded = self.embedding(targets)
            output, _ = self.gru(embedded, h)
            logits = self.fc(output)
            return logits
        else:
            # Generation mode
            return None


class HierVAE(nn.Module):
    """
    Hierarchical VAE for molecular generation.

    Uses hierarchical latent variables to capture multi-scale structure.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        latent_dims: List[int] = [128, 64, 32],
        num_layers: int = 3,
    ):
        super().__init__()

        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Hierarchical encoders
        self.encoders = nn.ModuleList()
        for i in range(self.num_levels):
            input_dim = embed_dim if i == 0 else hidden_dim
            self.encoders.append(
                nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # Latent layers
        self.latent_layers = nn.ModuleList()
        for dim in latent_dims:
            self.latent_layers.append(
                nn.ModuleDict(
                    {
                        "mu": nn.Linear(hidden_dim * 2, dim),
                        "logvar": nn.Linear(hidden_dim * 2, dim),
                    }
                )
            )

        # Hierarchical decoders
        self.decoder_init = nn.ModuleList()
        total_latent = sum(latent_dims)
        self.decoder_init.append(nn.Linear(total_latent, hidden_dim))

        self.decoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Hierarchical encoding."""
        embedded = self.embedding(x)
        latents = []

        for i, (encoder, latent_layer) in enumerate(
            zip(self.encoders, self.latent_layers)
        ):
            if i == 0:
                output, (h_n, _) = encoder(embedded)
            else:
                # Use previous latent to condition next level
                output, (h_n, _) = encoder(output)

            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
            mu = latent_layer["mu"](h)
            logvar = latent_layer["logvar"](h)
            latents.append((mu, logvar))

        return latents

    def reparameterize(
        self, latents: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Sample from hierarchical latents."""
        samples = []
        for mu, logvar in latents:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            samples.append(mu + eps * std)
        return torch.cat(samples, dim=1)

    def decode(self, z: torch.Tensor, max_length: int = 120) -> torch.Tensor:
        """Decode hierarchical latent."""
        batch_size = z.size(0)
        h = self.decoder_init[0](z).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c = torch.zeros_like(h)

        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        outputs = []

        for _ in range(max_length):
            embedded = self.embedding(input_token)
            output, (h, c) = self.decoder(embedded, (h, c))
            logits = self.fc_out(output.squeeze(1))
            outputs.append(logits)
            input_token = logits.argmax(dim=-1, keepdim=True)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        latents = self.encode(x)
        z = self.reparameterize(latents)
        recon_x = self.decode(z, max_length=x.size(1))
        return recon_x, latents


# =============================================================================
# PROPERTY PREDICTION MODELS
# =============================================================================


class GCN_Property(nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.
    """

    def __init__(
        self,
        node_feature_dim: int = 9,
        hidden_dim: int = 64,
        num_classes: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GCN")

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolution layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MPNN_Property(nn.Module):
    """
    Message Passing Neural Network for molecular property prediction.

    Implements the MPNN framework for learning on graphs.
    """

    def __init__(
        self,
        node_feature_dim: int = 9,
        edge_feature_dim: int = 3,
        hidden_dim: int = 64,
        num_classes: int = 1,
        num_steps: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required for MPNN")

        self.num_steps = num_steps
        self.hidden_dim = hidden_dim

        # Message passing layers
        self.message_function = nn.Sequential(
            nn.Linear(node_feature_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update function
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Readout function
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass with message passing."""
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initialize hidden states
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        # Message passing steps
        for _ in range(self.num_steps):
            # Compute messages
            src, dst = edge_index
            messages = self.message_function(torch.cat([x[src], edge_attr], dim=1))

            # Aggregate messages
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, dst, messages)

            # Update hidden states
            h = self.gru(aggregated, h)

        # Readout
        # Global mean pooling
        out = global_mean_pool(h, batch)
        return self.readout(out)


class GAT_Property(nn.Module):
    """
    Graph Attention Network for molecular property prediction.

    Uses attention mechanisms to weight neighbor contributions.
    """

    def __init__(
        self,
        node_feature_dim: int = 9,
        hidden_dim: int = 64,
        num_classes: int = 1,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        concat: bool = False,
    ):
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GAT")

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(
            GATConv(
                node_feature_dim,
                hidden_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
            )
        )
        out_dim = hidden_dim * heads if concat else hidden_dim
        self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(
                    out_dim, hidden_dim, heads=heads, concat=concat, dropout=dropout
                )
            )
            out_dim = hidden_dim * heads if concat else hidden_dim
            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.dropout = nn.Dropout(dropout)

        # Output layers
        fc_input = out_dim
        self.fc1 = nn.Linear(fc_input, fc_input // 2)
        self.fc2 = nn.Linear(fc_input // 2, num_classes)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph attention layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SchNet(nn.Module):
    """
    SchNet - Continuous-filter convolutional neural network.

    Designed for modeling quantum interactions in molecules.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_filters: int = 64,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_z: int = 100,
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        # Embedding
        self.embedding = nn.Embedding(max_z, hidden_channels)

        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(
                SchNetInteraction(hidden_channels, num_gaussians, num_filters, cutoff)
            )

        # Output layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Atomic numbers [num_atoms]
            pos: Atomic positions [num_atoms, 3]
            batch: Batch assignment [num_atoms]
        """
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        # Embedding
        h = self.embedding(z)

        # Compute distances
        edge_index, edge_weight = self.compute_edges(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        # Interaction blocks
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Global pooling
        h = global_mean_pool(h, batch)

        # Output
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)

    def compute_edges(self, pos: torch.Tensor, batch: torch.Tensor):
        """Compute edges within cutoff."""
        # Compute pairwise distances
        edge_index = []
        edge_weight = []

        for b in torch.unique(batch):
            mask = batch == b
            pos_batch = pos[mask]

            # Compute distance matrix
            dist_matrix = torch.cdist(pos_batch, pos_batch)

            # Get edges within cutoff
            src, dst = torch.where((dist_matrix < self.cutoff) & (dist_matrix > 0))

            # Adjust indices for batch
            batch_offset = torch.where(mask)[0][0]
            src = src + batch_offset
            dst = dst + batch_offset

            edge_index.append(torch.stack([src, dst], dim=0))
            edge_weight.append(
                dist_matrix[(dist_matrix < self.cutoff) & (dist_matrix > 0)]
            )

        if len(edge_index) > 0:
            edge_index = torch.cat(edge_index, dim=1)
            edge_weight = torch.cat(edge_weight)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos.device)
            edge_weight = torch.zeros(0, device=pos.device)

        return edge_index, edge_weight


class GaussianSmearing(nn.Module):
    """Gaussian smearing of distances."""

    def __init__(self, start: float, stop: float, num_gaussians: int):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SchNetInteraction(nn.Module):
    """SchNet interaction block."""

    def __init__(
        self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.Tanh(),
            nn.Linear(num_filters, num_filters),
        )

        self.lin = nn.Linear(hidden_channels, num_filters, bias=False)
        self.lin_out = nn.Linear(num_filters, hidden_channels)

        self.cutoff = cutoff

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Continuous filter
        W = self.mlp(edge_attr) * self.cutoff_cosine(
            edge_weight, self.cutoff
        ).unsqueeze(-1)

        # Message passing
        h_j = self.lin(h)
        src, dst = edge_index
        messages = h_j[src] * W

        # Aggregate
        out = torch.zeros_like(h)
        out.index_add_(0, dst, messages)

        return self.lin_out(out)

    def cutoff_cosine(self, distances: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Cosine cutoff function."""
        return (
            0.5
            * (torch.cos(distances * np.pi / cutoff) + 1.0)
            * (distances < cutoff).float()
        )


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================


class ScoringFunction(ABC):
    """Base class for molecular scoring functions."""

    @abstractmethod
    def score(self, molecule) -> float:
        """Score a molecule. Higher is better."""
        pass

    def score_batch(self, molecules: List) -> List[float]:
        """Score a batch of molecules."""
        return [self.score(mol) for mol in molecules]


class QSAR(ScoringFunction):
    """
    QSAR-based scoring function.

    Quantitative Structure-Activity Relationship models for predicting
    biological activity.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        target: str = "activity",
        descriptors: Optional[List[str]] = None,
    ):
        self.model = model
        self.target = target
        self.descriptors = descriptors or ["MolLogP", "MolWt", "TPSA"]

        if model is None and RDKIT_AVAILABLE:
            # Use simple heuristic model
            self.use_heuristic = True
        else:
            self.use_heuristic = False

    def compute_descriptors(self, mol) -> np.ndarray:
        """Compute molecular descriptors."""
        if not RDKIT_AVAILABLE:
            return np.zeros(len(self.descriptors))

        if isinstance(mol, str):
            mol = MolFromSmiles(mol)
        if mol is None:
            return np.zeros(len(self.descriptors))

        desc_values = []
        for desc_name in self.descriptors:
            try:
                desc_func = getattr(Descriptors, desc_name)
                desc_values.append(desc_func(mol))
            except:
                desc_values.append(0.0)

        return np.array(desc_values)

    def score(self, molecule) -> float:
        """Score molecule using QSAR model."""
        if self.use_heuristic:
            # Simple heuristic: favor drug-like properties
            desc = self.compute_descriptors(molecule)
            # Lipinski's Rule of Five approximation
            score = 0.0
            if len(desc) >= 3:
                mw_score = 1.0 if 180 <= desc[1] <= 500 else 0.5
                logp_score = 1.0 if -0.4 <= desc[0] <= 5.6 else 0.5
                score = (mw_score + logp_score) / 2.0
            return score

        # Use neural network model
        if self.model is not None:
            desc = self.compute_descriptors(molecule)
            with torch.no_grad():
                score = self.model(torch.tensor(desc, dtype=torch.float32).unsqueeze(0))
                return score.item()

        return 0.0


class DockingScore(ScoringFunction):
    """
    Molecular docking score.

    Scores molecules based on predicted binding affinity to target protein.
    """

    def __init__(
        self,
        receptor_file: Optional[str] = None,
        docking_program: str = "vina",
        scoring_function: str = "vina",
    ):
        self.receptor_file = receptor_file
        self.docking_program = docking_program
        self.scoring_function = scoring_function

    def score(self, molecule) -> float:
        """
        Compute docking score.

        Note: This is a placeholder. Actual docking requires external tools.
        """
        if not RDKIT_AVAILABLE:
            return 0.0

        if isinstance(molecule, str):
            mol = MolFromSmiles(molecule)
        else:
            mol = molecule

        if mol is None:
            return -100.0  # Invalid molecule

        # Placeholder: use simple heuristic based on molecular properties
        # In practice, this would call AutoDock Vina or similar
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)

        # Heuristic: smaller, more polar molecules often dock better
        score = -(mw / 500.0) - abs(logp - 2.0)

        return score


class ADMET(ScoringFunction):
    """
    ADMET property scoring.

    Absorption, Distribution, Metabolism, Excretion, Toxicity.
    """

    def __init__(
        self,
        properties: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.properties = properties or ["solubility", "permeability", "toxicity"]
        self.weights = weights or {
            "solubility": 0.3,
            "permeability": 0.3,
            "toxicity": 0.4,
        }

    def score(self, molecule) -> float:
        """Compute ADMET score."""
        if not RDKIT_AVAILABLE:
            return 0.5

        if isinstance(molecule, str):
            mol = MolFromSmiles(molecule)
        else:
            mol = molecule

        if mol is None:
            return 0.0

        scores = {}

        # Solubility (approximated by LogP)
        if "solubility" in self.properties:
            logp = Crippen.MolLogP(mol)
            # Optimal LogP for solubility is around 2
            scores["solubility"] = max(0.0, 1.0 - abs(logp - 2.0) / 3.0)

        # Permeability (approximated by TPSA)
        if "permeability" in self.properties:
            tpsa = Descriptors.TPSA(mol)
            # Lower TPSA generally means better permeability
            scores["permeability"] = max(0.0, 1.0 - tpsa / 140.0)

        # Toxicity (simplified - check for toxic groups)
        if "toxicity" in self.properties:
            # Simple heuristic based on molecular weight and complexity
            mw = Descriptors.MolWt(mol)
            num_rings = Descriptors.RingCount(mol)
            # Smaller, simpler molecules are generally less toxic
            scores["toxicity"] = max(0.0, 1.0 - (mw / 1000.0) - (num_rings / 10.0))

        # Weighted sum
        total_score = sum(
            scores.get(prop, 0.0) * self.weights.get(prop, 1.0)
            for prop in self.properties
        )
        total_weight = sum(self.weights.get(prop, 1.0) for prop in self.properties)

        return total_score / total_weight if total_weight > 0 else 0.0


class SAScore(ScoringFunction):
    """
    Synthetic Accessibility Score.

    Estimates how difficult a molecule is to synthesize.
    """

    def __init__(self):
        if RDKIT_AVAILABLE:
            try:
                from rdkit.Chem import RDConfig
                import os
                import pickle

                # Load fragment contributions if available
                self.fscores = self._load_fragment_scores()
            except:
                self.fscores = {}
        else:
            self.fscores = {}

    def _load_fragment_scores(self) -> Dict:
        """Load fragment score contributions."""
        # Simplified version - in practice would load from RDKit data
        return {}

    def score(self, molecule) -> float:
        """
        Compute SA score.

        Returns score between 0 (difficult) and 1 (easy).
        """
        if not RDKIT_AVAILABLE:
            return 0.5

        if isinstance(molecule, str):
            mol = MolFromSmiles(molecule)
        else:
            mol = molecule

        if mol is None:
            return 0.0

        # Compute complexity score
        # 1. Number of atoms
        num_atoms = mol.GetNumAtoms()

        # 2. Number of rings
        num_rings = Descriptors.RingCount(mol)

        # 3. Number of stereocenters
        num_stereo = len(Chem.FindMolChiralCenters(mol))

        # 4. Molecular weight
        mw = Descriptors.MolWt(mol)

        # Complexity score (higher is more complex)
        complexity = (
            num_atoms / 100.0 + num_rings / 10.0 + num_stereo / 5.0 + mw / 1000.0
        )

        # Convert to SA score (1 = easy, 0 = difficult)
        sa_score = max(0.0, 1.0 - complexity / 5.0)

        return sa_score


# =============================================================================
# OPTIMIZATION ALGORITHMS
# =============================================================================


class MolecularOptimizer(ABC):
    """Base class for molecular optimization algorithms."""

    def __init__(
        self,
        scoring_function: ScoringFunction,
        constraints: Optional[List[Callable]] = None,
    ):
        self.scoring_function = scoring_function
        self.constraints = constraints or []

    def is_valid(self, molecule) -> bool:
        """Check if molecule satisfies constraints."""
        return all(constraint(molecule) for constraint in self.constraints)

    @abstractmethod
    def optimize(self, initial_population: List, num_steps: int) -> List:
        """Optimize molecules."""
        pass


class ReinforcementLearning(MolecularOptimizer):
    """
    Reinforcement Learning for molecular optimization.

    Uses RL to train a generative model to produce molecules with desired properties.
    """

    def __init__(
        self,
        generator: nn.Module,
        scoring_function: ScoringFunction,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        super().__init__(scoring_function)
        self.generator = generator
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def generate_episode(
        self, batch_size: int = 32, max_length: int = 100
    ) -> Tuple[List, List, List]:
        """Generate an episode of molecules."""
        self.generator.eval()

        molecules = []
        log_probs = []
        rewards = []

        with torch.no_grad():
            for _ in range(batch_size):
                # Start with a random token or SMILES prefix
                tokens = [0]  # Start token
                log_prob_sum = 0.0

                for _ in range(max_length):
                    input_tensor = torch.tensor([tokens[-1]]).unsqueeze(0)
                    logits, _ = self.generator(input_tensor)
                    probs = F.softmax(logits[:, -1, :], dim=-1)

                    # Sample action
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob_sum += dist.log_prob(action).item()

                    tokens.append(action.item())

                    if action.item() == 0:  # End token
                        break

                # Convert to SMILES and score
                # This is simplified - actual implementation would use tokenizer
                smiles = "".join([chr(65 + t % 26) for t in tokens])
                score = self.scoring_function.score(smiles)

                molecules.append(smiles)
                log_probs.append(log_prob_sum)
                rewards.append(score)

        return molecules, log_probs, rewards

    def optimize(self, initial_population: List, num_steps: int = 1000) -> List:
        """Train using REINFORCE algorithm."""
        best_molecules = []
        best_scores = []

        for step in range(num_steps):
            molecules, log_probs, rewards = self.generate_episode()

            # Compute returns
            returns = torch.tensor(rewards, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Compute loss
            log_probs_tensor = torch.tensor(log_probs, requires_grad=True)
            loss = -(log_probs_tensor * returns).mean()

            # Update generator
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.optimizer.step()

            # Track best molecules
            for mol, score in zip(molecules, rewards):
                if score > 0.7:
                    best_molecules.append(mol)
                    best_scores.append(score)

            if step % 100 == 0:
                avg_reward = np.mean(rewards)
                print(f"Step {step}: Avg Reward = {avg_reward:.4f}")

        return best_molecules


class BayesianOptimization(MolecularOptimizer):
    """
    Bayesian Optimization for molecular optimization.

    Uses a Gaussian Process surrogate model to efficiently explore chemical space.
    """

    def __init__(
        self,
        scoring_function: ScoringFunction,
        initial_samples: int = 10,
        acquisition_function: str = "ei",
    ):
        super().__init__(scoring_function)
        self.initial_samples = initial_samples
        self.acquisition_function = acquisition_function

        # Placeholder for GP model - in practice would use GPyTorch or scikit-learn
        self.X_observed = []
        self.y_observed = []

    def surrogate_model(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Gaussian Process surrogate model.

        Returns mean and uncertainty (std) predictions.
        """
        if len(self.y_observed) < 2:
            return 0.0, 1.0

        # Simplified: use nearest neighbor average
        # In practice, would use a proper GP
        distances = np.array([np.linalg.norm(x - xi) for xi in self.X_observed])
        k = min(3, len(self.y_observed))
        nearest_idx = np.argsort(distances)[:k]

        mean = np.mean([self.y_observed[i] for i in nearest_idx])
        std = np.std([self.y_observed[i] for i in nearest_idx]) + 0.1

        return mean, std

    def acquisition(self, x: np.ndarray) -> float:
        """Compute acquisition function value."""
        mean, std = self.surrogate_model(x)

        if self.acquisition_function == "ei":  # Expected Improvement
            if len(self.y_observed) == 0:
                return mean
            y_best = max(self.y_observed)
            z = (mean - y_best) / (std + 1e-8)
            ei = (mean - y_best) * (0.5 * (1 + np.sign(z))) + std * np.exp(
                -(z**2) / 2
            ) / np.sqrt(2 * np.pi)
            return ei
        elif self.acquisition_function == "ucb":  # Upper Confidence Bound
            kappa = 2.0
            return mean + kappa * std
        else:
            return mean

    def smiles_to_vector(self, smiles: str) -> np.ndarray:
        """Convert SMILES to feature vector."""
        # Simplified: use character counts
        vec = np.zeros(128)
        for char in smiles:
            vec[ord(char) % 128] += 1
        return vec / (len(smiles) + 1)

    def optimize(self, initial_population: List, num_steps: int = 100) -> List:
        """Run Bayesian Optimization."""
        # Initialize with random samples
        population = initial_population[: self.initial_samples]

        for smiles in population:
            x = self.smiles_to_vector(smiles)
            y = self.scoring_function.score(smiles)
            self.X_observed.append(x)
            self.y_observed.append(y)

        best_molecules = []

        for step in range(num_steps):
            # Optimize acquisition function (simplified random search)
            candidates = []
            for smiles in population:
                # Generate neighbors
                neighbors = self._generate_neighbors(smiles)
                candidates.extend(neighbors)

            # Evaluate acquisition function for candidates
            if len(candidates) == 0:
                break

            acq_values = [
                self.acquisition(self.smiles_to_vector(c)) for c in candidates
            ]
            best_idx = np.argmax(acq_values)
            next_smiles = candidates[best_idx]

            # Evaluate true objective
            next_y = self.scoring_function.score(next_smiles)

            # Update observations
            self.X_observed.append(self.smiles_to_vector(next_smiles))
            self.y_observed.append(next_y)

            if next_y > 0.7:
                best_molecules.append(next_smiles)

            if step % 10 == 0:
                print(f"Step {step}: Best Score = {max(self.y_observed):.4f}")

        return best_molecules

    def _generate_neighbors(self, smiles: str, n_neighbors: int = 5) -> List[str]:
        """Generate neighboring molecules."""
        neighbors = []
        for i in range(min(n_neighbors, len(smiles))):
            # Simple mutation: change character
            for c in "CNOF" if i < 5 else "0123456789=()[]#":
                neighbor = smiles[:i] + c + smiles[i + 1 :]
                neighbors.append(neighbor)
        return neighbors


class GeneticAlgorithm(MolecularOptimizer):
    """
    Genetic Algorithm for molecular optimization.

    Evolves a population of molecules through selection, crossover, and mutation.
    """

    def __init__(
        self,
        scoring_function: ScoringFunction,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: int = 5,
    ):
        super().__init__(scoring_function)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

    def select_parents(self, population: List, scores: List[float]) -> Tuple[str, str]:
        """Select two parents using tournament selection."""
        tournament_size = 3

        def tournament():
            contestants = np.random.choice(
                len(population), tournament_size, replace=False
            )
            best = max(contestants, key=lambda i: scores[i])
            return population[best]

        return tournament(), tournament()

    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Perform crossover between two SMILES strings."""
        if np.random.random() > self.crossover_rate:
            return parent1, parent2

        # Find common substructures (simplified)
        min_len = min(len(parent1), len(parent2))
        if min_len < 2:
            return parent1, parent2

        point = np.random.randint(1, min_len)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    def mutate(self, smiles: str) -> str:
        """Mutate a SMILES string."""
        if np.random.random() > self.mutation_rate:
            return smiles

        if len(smiles) == 0:
            return smiles

        # Random mutation
        mutations = [
            lambda s: s[: np.random.randint(0, len(s))]
            + np.random.choice(list("CNOS"))
            + s[np.random.randint(0, len(s)) + 1 :],
            lambda s: s + np.random.choice(list("0123456789")),
            lambda s: s[:-1] if len(s) > 1 else s,
        ]

        mutation = np.random.choice(mutations)
        return mutation(smiles)

    def optimize(self, initial_population: List, num_steps: int = 100) -> List:
        """Run genetic algorithm."""
        population = initial_population[: self.population_size]

        # Pad population if needed
        while len(population) < self.population_size:
            population.append(
                initial_population[len(population) % len(initial_population)]
            )

        best_molecules = []

        for generation in range(num_steps):
            # Evaluate population
            scores = self.scoring_function.score_batch(population)

            # Track best
            best_idx = np.argmax(scores)
            if scores[best_idx] > 0.7:
                best_molecules.append(population[best_idx])

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            sorted_indices = np.argsort(scores)[::-1]
            for i in range(self.elitism):
                new_population.append(population[sorted_indices[i]])

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                if self.is_valid(child1):
                    new_population.append(child1)
                if len(new_population) < self.population_size and self.is_valid(child2):
                    new_population.append(child2)

            population = new_population[: self.population_size]

            if generation % 10 == 0:
                print(f"Generation {generation}: Best Score = {max(scores):.4f}")

        return best_molecules


class HillClimbing(MolecularOptimizer):
    """
    Hill Climbing for molecular optimization.

    Iteratively improves molecules by making small modifications.
    """

    def __init__(
        self,
        scoring_function: ScoringFunction,
        max_iterations: int = 1000,
        num_neighbors: int = 10,
    ):
        super().__init__(scoring_function)
        self.max_iterations = max_iterations
        self.num_neighbors = num_neighbors

    def generate_neighbors(self, smiles: str) -> List[str]:
        """Generate neighboring molecules."""
        neighbors = []

        # Character-level mutations
        for i in range(min(len(smiles), 10)):
            for char in "CNOF123456789=()#@":
                neighbor = smiles[:i] + char + smiles[i + 1 :]
                if self.is_valid(neighbor):
                    neighbors.append(neighbor)

        # Insertions
        for i in range(min(len(smiles), 5)):
            for char in "CNO":
                neighbor = smiles[:i] + char + smiles[i:]
                if self.is_valid(neighbor):
                    neighbors.append(neighbor)

        # Deletions
        for i in range(min(len(smiles), 5)):
            neighbor = smiles[:i] + smiles[i + 1 :]
            if len(neighbor) > 2 and self.is_valid(neighbor):
                neighbors.append(neighbor)

        return neighbors[: self.num_neighbors]

    def optimize(self, initial_population: List, num_steps: int = None) -> List:
        """Run hill climbing from multiple starting points."""
        if num_steps is None:
            num_steps = self.max_iterations

        best_molecules = []

        for start_mol in initial_population:
            current = start_mol
            current_score = self.scoring_function.score(current)

            for iteration in range(num_steps):
                neighbors = self.generate_neighbors(current)

                if len(neighbors) == 0:
                    break

                neighbor_scores = self.scoring_function.score_batch(neighbors)
                best_neighbor_idx = np.argmax(neighbor_scores)
                best_neighbor_score = neighbor_scores[best_neighbor_idx]

                if best_neighbor_score > current_score:
                    current = neighbors[best_neighbor_idx]
                    current_score = best_neighbor_score
                else:
                    # Local maximum reached
                    break

            if current_score > 0.7:
                best_molecules.append(current)

        return best_molecules


# =============================================================================
# DATASETS
# =============================================================================


class MoleculeDataset(Dataset):
    """
    Generic dataset for molecular data.

    Supports ZINC, ChEMBL, and custom molecule collections.
    """

    def __init__(
        self,
        smiles_list: List[str],
        properties: Optional[Dict[str, List]] = None,
        tokenizer=None,
        max_length: int = 120,
    ):
        self.smiles_list = smiles_list
        self.properties = properties or {}
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Filter valid molecules
        self.valid_indices = []
        for i, smiles in enumerate(smiles_list):
            if RDKIT_AVAILABLE:
                mol = MolFromSmiles(smiles)
                if mol is not None:
                    self.valid_indices.append(i)
            else:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        smiles = self.smiles_list[real_idx]

        item = {"smiles": smiles}

        # Add properties
        for key, values in self.properties.items():
            item[key] = values[real_idx]

        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(smiles, max_length=self.max_length)
            item["tokens"] = tokens

        return item

    def get_graph(self, idx: int) -> Optional[Graph]:
        """Get graph representation of molecule."""
        smiles = self.smiles_list[self.valid_indices[idx]]
        try:
            return smiles_to_graph(smiles)
        except:
            return None


def load_zinc(
    filepath: Optional[str] = None, subset: str = "250K", split: str = "train"
) -> MoleculeDataset:
    """
    Load ZINC dataset.

    Args:
        filepath: Path to ZINC data file
        subset: Dataset subset ('250K', '1M', etc.)
        split: Data split ('train', 'valid', 'test')

    Returns:
        MoleculeDataset
    """
    # In practice, would download and load actual ZINC data
    # This is a placeholder with sample molecules
    sample_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen-like
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "c1ccc(cc1)c2ccccc2",  # Biphenyl
        "CC(C)NCC(COc1ccccc1)O",  # Propranolol-like
    ] * 100  # Repeat to simulate larger dataset

    print(f"Loading ZINC {subset} {split} subset...")
    print(f"Note: Using placeholder data. Load actual ZINC dataset for production use.")

    return MoleculeDataset(sample_smiles)


def load_chembl(
    filepath: Optional[str] = None,
    version: str = "31",
    bioactivity_threshold: Optional[float] = None,
) -> MoleculeDataset:
    """
    Load ChEMBL dataset.

    Args:
        filepath: Path to ChEMBL data file
        version: ChEMBL version
        bioactivity_threshold: Filter by bioactivity

    Returns:
        MoleculeDataset
    """
    # Placeholder with sample molecules
    sample_smiles = [
        "COc1ccc2nc(sc2c1)N3CCN(CC3)C",  # Promethazine
        "CC(C)NCC(COc1ccc(cc1)CC(=O)N)O",  # Atenolol
        "CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3",  # Atropine
        "Cc1ccc(cc1)S(=O)(=O)NC(=O)NN2CCCCCC2",  # Gliclazide
        "CCN(CC)C(=O)C1(c2ccccc2)CCCCC1",  # Bencyclane
    ] * 100

    print(f"Loading ChEMBL {version}...")
    print(
        f"Note: Using placeholder data. Load actual ChEMBL dataset for production use."
    )

    return MoleculeDataset(sample_smiles)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================


class MoleculeTrainer:
    """
    Trainer for molecular generation models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader, loss_fn: Callable) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Compute loss
            loss = loss_fn(self.model, batch, self.device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.history["train_loss"].append(avg_loss)

        return avg_loss

    def validate(self, dataloader: DataLoader, loss_fn: Callable) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                loss = loss_fn(self.model, batch, self.device)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.history["val_loss"].append(avg_loss)

        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        loss_fn: Optional[Callable] = None,
    ) -> Dict:
        """Train model."""
        if loss_fn is None:
            loss_fn = MoleculeLoss()

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, loss_fn)

            if val_loader is not None:
                val_loss = self.validate(val_loader, loss_fn)
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

        return self.history

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})


class MoleculeLoss:
    """
    Combined loss function for molecular generation.

    Includes reconstruction loss and validity loss.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        validity_weight: float = 0.1,
        property_weight: float = 0.1,
        property_predictor: Optional[nn.Module] = None,
    ):
        self.recon_weight = recon_weight
        self.validity_weight = validity_weight
        self.property_weight = property_weight
        self.property_predictor = property_predictor
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self, model: nn.Module, batch: Dict, device: str) -> torch.Tensor:
        """Compute combined loss."""
        # This is a simplified interface - actual implementation depends on model type

        if "tokens" in batch:
            tokens = batch["tokens"].to(device)

            # Forward pass
            if hasattr(model, "forward"):
                output = model(tokens)

                # Handle different model outputs
                if isinstance(output, tuple):
                    # VAE output: (recon, mu, logvar)
                    recon, mu, logvar = output
                    recon_loss = self.ce_loss(
                        recon.view(-1, recon.size(-1)), tokens.view(-1)
                    )

                    # KL divergence
                    kl_loss = (
                        -0.5
                        * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        / mu.size(0)
                    )

                    loss = recon_loss + 0.1 * kl_loss
                else:
                    # Standard output
                    loss = self.ce_loss(
                        output.view(-1, output.size(-1)), tokens.view(-1)
                    )
            else:
                loss = torch.tensor(0.0, device=device)
        else:
            loss = torch.tensor(0.0, device=device)

        return loss


class ValidityLoss(nn.Module):
    """
    Loss function that encourages valid SMILES generation.
    """

    def __init__(self, validity_checker: Optional[Callable] = None):
        super().__init__()
        self.validity_checker = validity_checker or self._default_checker

    def _default_checker(self, smiles: str) -> bool:
        """Default validity checker using RDKit."""
        if not RDKIT_AVAILABLE:
            return True
        mol = MolFromSmiles(smiles)
        return mol is not None

    def forward(self, generated_smiles: List[str]) -> torch.Tensor:
        """Compute validity loss."""
        valid_count = sum(1 for smi in generated_smiles if self.validity_checker(smi))
        validity_ratio = (
            valid_count / len(generated_smiles) if generated_smiles else 0.0
        )

        # Loss is 1 - validity ratio (we want to minimize this)
        loss = 1.0 - validity_ratio
        return torch.tensor(loss, requires_grad=True)


# =============================================================================
# ADDITIONAL UTILITIES
# =============================================================================


def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string."""
    if not RDKIT_AVAILABLE:
        return True
    mol = MolFromSmiles(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form."""
    if not RDKIT_AVAILABLE:
        return smiles
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    return MolToSmiles(mol, canonical=True)


def compute_molecular_properties(smiles: str) -> Dict[str, float]:
    """Compute common molecular properties."""
    if not RDKIT_AVAILABLE:
        return {}

    mol = MolFromSmiles(smiles)
    if mol is None:
        return {}

    properties = {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_hbd": Descriptors.NumHDonors(mol),
        "num_hba": Descriptors.NumHAcceptors(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_rings": Descriptors.RingCount(mol),
        "qed": QED.qed(mol) if hasattr(QED, "qed") else 0.0,
    }

    return properties


def filter_drug_like(
    smiles_list: List[str],
    mw_range: Tuple[float, float] = (150, 500),
    logp_range: Tuple[float, float] = (-0.4, 5.6),
) -> List[str]:
    """Filter molecules by drug-like properties (Lipinski's Rule of Five)."""
    if not RDKIT_AVAILABLE:
        return smiles_list

    filtered = []
    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        # Lipinski's Rule of Five
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        if violations <= 1:
            filtered.append(smiles)

    return filtered


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Fishstick Molecular Generation Module")
    print("=" * 50)

    # Example: Load dataset
    print("\n1. Loading dataset...")
    dataset = load_zinc(subset="250K")
    print(f"   Dataset size: {len(dataset)}")

    # Example: Create model
    print("\n2. Creating VAE model...")
    vocab_size = 100  # Simplified
    vae = VAE_Molecule(vocab_size=vocab_size, latent_dim=128)
    print(f"   Model parameters: {sum(p.numel() for p in vae.parameters()):,}")

    # Example: Property prediction
    print("\n3. Testing property prediction...")
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    props = compute_molecular_properties(test_smiles)
    print(f"   Properties of {test_smiles}:")
    for key, value in props.items():
        print(f"     {key}: {value:.2f}")

    # Example: Scoring
    print("\n4. Testing scoring functions...")
    qsar = QSAR()
    admet = ADMET()
    sa = SAScore()

    score_qsar = qsar.score(test_smiles)
    score_admet = admet.score(test_smiles)
    score_sa = sa.score(test_smiles)

    print(f"   QSAR Score: {score_qsar:.4f}")
    print(f"   ADMET Score: {score_admet:.4f}")
    print(f"   SA Score: {score_sa:.4f}")

    print("\n" + "=" * 50)
    print("Module loaded successfully!")
