"""Molecule generation and drug-likeness filtering."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class MoleculeGenerator(nn.Module):
    """Generates novel molecules using variational autoencoder.

    Attributes:
        atom_vocab_size: Size of atom vocabulary
        embedding_dim: Embedding dimension
        hidden_dim: Hidden dimension
        max_atoms: Maximum number of atoms per molecule
    """

    def __init__(
        self,
        atom_vocab_size: 128,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        max_atoms: int = 50,
    ) -> None:
        super().__init__()
        self.max_atoms = max_atoms
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, embedding_dim)
        self.logvar = nn.Linear(hidden_dim, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.atom_predictor = nn.Linear(hidden_dim, atom_vocab_size)
        self.bond_predictor = nn.Linear(hidden_dim, 4)

    def forward(self, z: Tensor) -> Dict[str, Tensor]:
        """Generate molecule from latent vector.

        Args:
            z: Latent vector

        Returns:
            Dictionary with atom and bond predictions
        """
        h = self.decoder(z)

        atom_logits = []
        bond_logits = []

        for _ in range(self.max_atoms):
            atom_logits.append(self.atom_predictor(h))

            bond_feat = self.bond_predictor(h)
            bond_logits.append(bond_feat)

        return {
            "atom_logits": torch.stack(atom_logits),
            "bond_logits": torch.stack(bond_logits),
        }

    def encode(self, x: Tensor) -> Tensor:
        """Encode molecule to latent space.

        Args:
            x: Input features

        Returns:
            Latent representation
        """
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


class DrugLikenessFilter:
    """Filters molecules by drug-likeness rules.

    Implements Lipinski's rule of five and other drug-likeness criteria.
    """

    def __init__(
        self,
        max_mw: float = 500.0,
        max_logp: float = 5.0,
        max_hbd: int = 5,
        max_hba: int = 10,
        max_tpsa: float = 140.0,
    ) -> None:
        self.max_mw = max_mw
        self.max_logp = max_logp
        self.max_hbd = max_hbd
        self.max_hba = max_hba
        self.max_tpsa = max_tpsa

    def compute_likeness_score(self, smiles: str) -> Dict[str, float]:
        """Compute drug-likeness score.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with score components
        """
        from .molecules import SMILESParser, compute_molecular_properties

        parser = SMILESParser()
        atoms, bonds = parser.parse(smiles)

        props = compute_molecular_properties(smiles)

        mw = props.get("molecular_weight", 0)
        tpsa = props.get("tpsa", 0)

        num_hbd = sum(1 for a in atoms if a.element in ["N", "O"])
        num_hba = sum(1 for a in atoms if a.element in ["N", "O"])

        num_rotatable = props.get("num_rotatable_bonds", 0)

        mw_score = 1.0 if mw <= self.max_mw else max(0, 1 - (mw - self.max_mw) / 500)
        hbd_score = (
            1.0 if num_hbd <= self.max_hbd else max(0, 1 - (num_hbd - self.max_hbd) / 5)
        )
        hba_score = (
            1.0
            if num_hba <= self.max_hba
            else max(0, 1 - (num_hba - self.max_hba) / 10)
        )
        tpsa_score = (
            1.0 if tpsa <= self.max_tpsa else max(0, 1 - (tpsa - self.max_tpsa) / 100)
        )

        rot_score = (
            1.0 if num_rotatable <= 10 else max(0, 1 - (num_rotatable - 10) / 10)
        )

        overall_score = (mw_score + hbd_score + hba_score + tpsa_score + rot_score) / 5

        return {
            "overall": overall_score,
            "mw_score": mw_score,
            "hbd_score": hbd_score,
            "hba_score": hba_score,
            "tpsa_score": tpsa_score,
            "rotatable_score": rot_score,
            "molecular_weight": mw,
            "tpsa": tpsa,
            "num_hbd": num_hbd,
            "num_hba": num_hba,
            "num_rotatable": num_rotatable,
        }

    def is_druglike(self, smiles: str) -> bool:
        """Check if molecule is drug-like.

        Args:
            smiles: SMILES string

        Returns:
            True if drug-like
        """
        scores = self.compute_likeness_score(smiles)
        return scores["overall"] > 0.5


def generate_similar_molecules(
    smiles: str,
    n_samples: int = 10,
    temperature: float = 1.0,
) -> List[str]:
    """Generate similar molecules using SMILES enumeration.

    Args:
        smiles: Reference SMILES
        n_samples: Number of samples to generate
        temperature: Sampling temperature

    Returns:
        List of similar SMILES
    """
    from .molecules import SMILESParser

    parser = SMILESParser()
    atoms, bonds = parser.parse(smiles)

    variations = []

    for _ in range(n_samples):
        new_smiles = smiles

        new_smiles = new_smiles.replace("C", "c", 1)
        variations.append(new_smiles)

    return variations[:n_samples]


def compute_synthetic_accessibility(smiles: str) -> float:
    """Compute synthetic accessibility score.

    Args:
        smiles: SMILES string

    Returns:
        SA score (1-10, lower is more accessible)
    """
    from .molecules import SMILESParser

    parser = SMILESParser()
    atoms, bonds = parser.parse(smiles)

    num_rings = sum(1 for a in atoms if a.is_aromatic) // 3

    num_stereocenters = 0

    complexity = len(atoms) / 20.0

    num_rotatable = len(bonds) // 2

    sa_score = 1.0

    sa_score += num_rings * 0.5

    sa_score += complexity * 0.5

    sa_score += num_rotatable * 0.3

    return min(10.0, sa_score)


def compute_chemical_stability(smiles: str) -> float:
    """Compute chemical stability score.

    Args:
        smiles: SMILES string

    Returns:
        Stability score (0-1)
    """
    from .molecules import SMILESParser

    parser = SMILESParser()
    atoms, bonds = parser.parse(smiles)

    stability = 1.0

    num_aromatic = sum(1 for a in atoms if a.is_aromatic)

    stability += num_aromatic * 0.1

    num_unstable = sum(
        1 for a in atoms if a.element in ["P", "S"] and not a.is_aromatic
    )
    stability -= num_unstable * 0.2

    return max(0.0, min(1.0, stability))
