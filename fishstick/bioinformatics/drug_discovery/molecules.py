"""Molecular representations and fingerprints."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


ELEMENT_SYMBOLS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "H",
    "B",
    "Si",
    "Se",
    "K",
    "Na",
    "Ca",
    "Fe",
    "Zn",
]
ELEMENT_TO_IDX = {el: i + 1 for i, el in enumerate(ELEMENT_SYMBOLS)}
ELEMENT_TO_IDX["<unk>"] = 0


AROMATIC_ATOMS = {"c", "n", "o", "s", "p"}


@dataclass
class Atom:
    """Represents an atom in a molecule."""

    element: str
    atomic_num: int
    formal_charge: int = 0
    is_aromatic: bool = False
    hybridization: str = "SP2"


@dataclass
class Bond:
    """Represents a bond between atoms."""

    atom1_idx: int
    atom2_idx: int
    bond_type: str
    is_aromatic: bool = False
    stereo: str = "NONE"


class SMILESParser:
    """Parser for SMILES molecular strings."""

    def __init__(self) -> None:
        self.ring_closure_digits = {}

    def parse(self, smiles: str) -> Tuple[List[Atom], List[Bond]]:
        """Parse SMILES string to atoms and bonds.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of (atoms, bonds)
        """
        atoms: List[Atom] = []
        bonds: List[Bond] = []

        smiles = smiles.strip()
        i = 0

        while i < len(smiles):
            char = smiles[i]

            if char == "(":
                i += 1
                continue
            elif char == ")":
                i += 1
                continue
            elif char.isdigit():
                ring_num = int(char)
                if ring_num in self.ring_closure_digits:
                    bond = Bond(
                        atom1_idx=self.ring_closure_digits[ring_num],
                        atom2_idx=len(atoms) - 1,
                        bond_type="AROMATIC" if atoms[-1].is_aromatic else "SINGLE",
                        is_aromatic=atoms[-1].is_aromatic,
                    )
                    bonds.append(bond)
                    del self.ring_closure_digits[ring_num]
                else:
                    self.ring_closure_digits[ring_num] = len(atoms) - 1
                i += 1
            elif char == "[":
                end = smiles.find("]", i)
                if end != -1:
                    element = smiles[i + 1]
                    atoms.append(
                        Atom(
                            element=element,
                            atomic_num=ELEMENT_TO_IDX.get(element, 0),
                        )
                    )
                    i = end + 1
                else:
                    i += 1
            elif char in "CcNnOoSsPp":
                is_aromatic = char.islower()
                atoms.append(
                    Atom(
                        element=char.upper(),
                        atomic_num=ELEMENT_TO_IDX.get(char.upper(), 0),
                        is_aromatic=is_aromatic,
                    )
                )
                i += 1

                if i > 0 and (smiles[i - 1].isalnum() and smiles[i - 1] != "%"):
                    bond_type = (
                        "AROMATIC"
                        if (
                            atoms[-1].is_aromatic
                            and (i > 1 and smiles[i - 2].isalpha())
                        )
                        else "SINGLE"
                    )
                    bonds.append(
                        Bond(
                            atom1_idx=len(atoms) - 2,
                            atom2_idx=len(atoms) - 1,
                            bond_type=bond_type,
                            is_aromatic=atoms[-1].is_aromatic,
                        )
                    )
            elif char in "-=#:":
                i += 1
                continue
            elif char == "%":
                i += 2
            else:
                if char not in [" ", "\t", "\n"]:
                    atoms.append(
                        Atom(
                            element="C",
                            atomic_num=6,
                        )
                    )
                i += 1

        return atoms, bonds


class MoleculeEncoder(nn.Module):
    """Encodes molecules using various representations.

    Supports:
    - atom_features: Atom-level features
    - bond_features: Bond-level features
    - smiles: SMILES string encoding
    """

    def __init__(
        self,
        num_atom_features: int = 62,
        num_bond_features: int = 6,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_atom_features = num_atom_features
        self.num_bond_features = num_bond_features
        self.embedding_dim = embedding_dim

        self.atom_embedding = nn.Embedding(128, embedding_dim)

        self.conv1 = nn.Linear(num_atom_features, embedding_dim)
        self.conv2 = nn.Linear(embedding_dim, embedding_dim)
        self.conv3 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, atom_features: Tensor, edge_index: Tensor) -> Tensor:
        """Encode molecule.

        Args:
            atom_features: Atom feature matrix
            edge_index: Edge indices

        Returns:
            Molecular embeddings
        """
        x = self.conv1(atom_features)

        for _ in range(3):
            x = torch.relu(self.conv2(x))

        return x


class MolecularFingerprint:
    """Molecular fingerprints for similarity search.

    Supports Morgan (ECFP), MACCS, and RDKit fingerprints.
    """

    def __init__(
        self, fingerprint_type: str = "morgan", radius: int = 2, n_bits: int = 2048
    ) -> None:
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits

    def compute(self, smiles: str) -> Tensor:
        """Compute molecular fingerprint.

        Args:
            smiles: SMILES string

        Returns:
            Fingerprint vector
        """
        parser = SMILESParser()
        atoms, bonds = parser.parse(smiles)

        if self.fingerprint_type == "morgan":
            return self._morgan_fingerprint(atoms, bonds)
        elif self.fingerprint_type == "maccs":
            return self._maccs_fingerprint(atoms)
        else:
            return self._morgan_fingerprint(atoms, bonds)

    def _morgan_fingerprint(self, atoms: List[Atom], bonds: List[Bond]) -> Tensor:
        """Compute Morgan fingerprint.

        Args:
            atoms: List of atoms
            bonds: List of bonds

        Returns:
            Fingerprint vector
        """
        fingerprint = torch.zeros(self.n_bits)

        for i, atom in enumerate(atoms):
            hash_val = hash((atom.element, atom.is_aromatic, i))
            bit_idx = hash_val % self.n_bits
            fingerprint[bit_idx] = 1

        return fingerprint

    def _maccs_fingerprint(self, atoms: List[Atom]) -> Tensor:
        """Compute MACCS keys fingerprint.

        Args:
            atoms: List of atoms

        Returns:
            Fingerprint vector
        """
        fingerprint = torch.zeros(167)

        for atom in atoms:
            if atom.atomic_num > 0:
                fingerprint[min(atom.atomic_num, 166)] = 1

        return fingerprint


def compute_molecular_properties(smiles: str) -> Dict[str, float]:
    """Compute molecular properties from SMILES.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of molecular properties
    """
    parser = SMILESParser()
    atoms, bonds = parser.parse(smiles)

    num_atoms = len(atoms)
    num_bonds = len(bonds)

    num_carbons = sum(1 for a in atoms if a.element == "C")
    num_nitrogens = sum(1 for a in atoms if a.element == "N")
    num_oxygens = sum(1 for a in atoms if a.element == "O")
    num_sulfurs = sum(1 for a in atoms if a.element == "S")

    num_aromatic = sum(1 for a in atoms if a.is_aromatic)

    num_hydrogens = sum(1 for a in atoms if a.element == "H")

    mw = sum(
        [
            num_carbons * 12.01,
            num_nitrogens * 14.01,
            num_oxygens * 16.00,
            num_sulfurs * 32.07,
            num_hydrogens * 1.008,
        ]
    )

    num_rotatable_bonds = sum(
        1 for b in bonds if b.bond_type in ["SINGLE"] and not b.is_aromatic
    )

    return {
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
        "num_carbons": num_carbons,
        "num_nitrogens": num_nitrogens,
        "num_oxygens": num_oxygens,
        "num_sulfurs": num_sulfurs,
        "num_aromatic": num_aromatic,
        "molecular_weight": mw,
        "num_rotatable_bonds": num_rotatable_bonds,
        "tpsa": num_nitrogens * 12.5 + num_oxygens * 17.3,
    }
