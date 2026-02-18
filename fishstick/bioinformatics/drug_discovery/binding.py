"""Binding affinity prediction for drug discovery."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class BindingAffinityPredictor(nn.Module):
    """Predicts protein-ligand binding affinity.

    Uses graph neural networks to predict binding affinity from
    molecular structures.

    Attributes:
        num_atom_features: Number of atom features
        num_bond_features: Number of bond features
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
    """

    def __init__(
        self,
        num_atom_features: int = 62,
        num_bond_features: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.num_atom_features = num_atom_features
        self.num_bond_features = num_bond_features
        self.hidden_dim = hidden_dim

        self.atom_embedding = nn.Linear(num_atom_features, hidden_dim)

        self.message_passing = nn.ModuleList(
            [
                MessagePassingLayer(hidden_dim, num_bond_features)
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        atom_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict binding affinity.

        Args:
            atom_features: Atom feature matrix
            edge_index: Edge connectivity
            edge_features: Optional edge features

        Returns:
            Predicted binding affinity
        """
        x = self.atom_embedding(atom_features)

        for layer in self.message_passing:
            x = layer(x, edge_index, edge_features)

        x = self.readout(x)

        return x.mean(dim=0)


class MessagePassingLayer(nn.Module):
    """Message passing layer for molecular graph."""

    def __init__(self, hidden_dim: int, num_bond_features: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_bond_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Perform message passing.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_features: Edge features

        Returns:
            Updated node features
        """
        src, dst = edge_index

        messages = []

        for i in range(len(src)):
            if edge_features is not None:
                edge_feat = edge_features[i]
            else:
                edge_feat = torch.zeros(self.hidden_dim)

            message = torch.cat([x[src[i]], x[dst[i]], edge_feat])
            messages.append(message)

        messages = torch.stack(messages)
        messages = self.message_net(messages)

        new_messages = torch.zeros_like(x)
        for i, idx in enumerate(dst):
            new_messages[idx] += messages[i]

        combined = torch.cat([x, new_messages], dim=-1)
        x = self.update_net(combined)

        return x


def compute_sap_score(
    protein_sequence: str,
    ligand_smiles: str,
) -> float:
    """Compute SAP (Shape-Atom Pairwise) similarity score.

    Args:
        protein_sequence: Protein sequence
        ligand_smiles: Ligand SMILES

    Returns:
        SAP score
    """
    from .molecules import SMILESParser, compute_molecular_properties

    parser = SMILESParser()
    atoms, bonds = parser.parse(ligand_smiles)

    num_hbd = sum(1 for a in atoms if a.element in ["N", "O"] and a.is_aromatic)
    num_hba = sum(1 for a in atoms if a.element in ["N", "O"])

    props = compute_molecular_properties(ligand_smiles)

    score = 0.0
    score -= props.get("tpsa", 0) / 10
    score += props.get("num_aromatic", 0) * 0.5
    score -= num_hbd * 0.5

    return score


def predict_pKa(smiles: str) -> float:
    """Predict molecular pKa.

    Args:
        smiles: SMILES string

    Returns:
        Predicted pKa
    """
    from .molecules import SMILESParser

    parser = SMILESParser()
    atoms, _ = parser.parse(smiles)

    num_heteroatoms = sum(1 for a in atoms if a.element in ["N", "O", "S"])
    num_positive = sum(1 for a in atoms if a.element == "N")
    num_negative = sum(1 for a in atoms if a.element == "O")

    pka = 7.0

    if num_positive > 0:
        pka = 9.0 + num_positive * 0.5

    if num_negative > 0:
        pka = 4.5 - num_negative * 0.3

    return pka


def compute_solubility(smiles: str) -> float:
    """Predict aqueous solubility.

    Args:
        smiles: SMILES string

    Returns:
        Log solubility (mol/L)
    """
    from .molecules import compute_molecular_properties

    props = compute_molecular_properties(smiles)

    mw = props.get("molecular_weight", 500)
    tpsa = props.get("tpsa", 0)
    num_rotatable = props.get("num_rotatable_bonds", 0)

    log_s = 0.0

    log_s -= mw / 500

    log_s += tpsa / 30

    log_s -= num_rotatable * 0.1

    return log_s
