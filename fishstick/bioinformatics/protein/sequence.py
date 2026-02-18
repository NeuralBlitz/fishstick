"""Protein sequence encoding and utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AMINO_ACID = {i: aa for aa, i in AMINO_ACID_TO_IDX.items()}


BLOSUM62 = {
    "A": [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    "R": [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    "N": [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    "D": [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    "C": [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    "Q": [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    "E": [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    "G": [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    "H": [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    "K": [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    "P": [
        -1,
        -2,
        -2,
        -1,
        -3,
        -1,
        -1,
        -2,
        -2,
        -3,
        -3,
        -1,
        -2,
        -4,
        7,
        -1,
        -1,
        -4,
        -3,
        -2,
    ],
    "S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    "T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    "Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    "V": [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
}


@dataclass
class ProteinSequence:
    """Represents a protein sequence with metadata.

    Attributes:
        sequence: The amino acid sequence string
        name: Optional name/identifier
        species: Optional species of origin
    """

    sequence: str
    name: Optional[str] = None
    species: Optional[str] = None

    def __post_init__(self) -> None:
        self.sequence = self.sequence.upper()
        valid_chars = set(AMINO_ACIDS)
        if not all(c in valid_chars or c == "X" for c in self.sequence):
            raise ValueError(f"Invalid amino acid characters in sequence")

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def length(self) -> int:
        """Return sequence length."""
        return len(self)

    def to_indices(self) -> List[int]:
        """Convert sequence to list of integer indices."""
        return [AMINO_ACID_TO_IDX.get(aa, 20) for aa in self.sequence]


class ProteinEncoder(nn.Module):
    """Encodes protein sequences using various encoding schemes.

    Supports multiple encoding methods:
    - onehot: One-hot encoding of amino acids
    - blosum: BLOSUM62 substitution matrix encoding
    - physicochemical: Physicochemical properties
    - kmer: K-mer embedding

    Attributes:
        encoding: The encoding scheme to use
        kmer_size: Size of k-mers for kmer encoding
        embedding_dim: Dimension of embeddings
    """

    def __init__(
        self,
        encoding: str = "onehot",
        kmer_size: int = 3,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoding = encoding
        self.kmer_size = kmer_size
        self.embedding_dim = embedding_dim

        if encoding == "onehot":
            self.register_buffer(
                "onehot_matrix",
                torch.eye(21),
            )
        elif encoding == "blosum":
            self.blosum_matrix = torch.tensor(
                [BLOSUM62[aa] for aa in AMINO_ACIDS] + [[0] * 20],
                dtype=torch.float32,
            )
        elif encoding == "physicochemical":
            self.physicochemical = self._init_physicochemical()
        elif encoding == "kmer":
            num_kmers = 20**kmer_size
            self.kmer_embedding = nn.Embedding(num_kmers, embedding_dim)

        self._num_kmers = 20**kmer_size

    def _init_physicochemical(self) -> Tensor:
        physicochemical_props = {
            "A": [1.8, 0, 0.5],
            "R": [-4.5, 3, 0.75],
            "N": [-3.5, 2, 0.62],
            "D": [-3.5, 2, 0.62],
            "C": [2.5, 1, 0.5],
            "Q": [-3.5, 2, 0.62],
            "E": [-3.5, 2, 0.62],
            "G": [-0.4, 0, 0.5],
            "H": [-3.2, 2, 0.62],
            "I": [4.5, 0, 0.5],
            "L": [3.8, 0, 0.5],
            "K": [-3.9, 3, 0.75],
            "M": [1.9, 0, 0.5],
            "F": [2.8, 0, 0.5],
            "P": [-1.6, 0, 0.5],
            "S": [-0.8, 1, 0.62],
            "T": [-0.7, 1, 0.62],
            "W": [-0.9, 0, 0.5],
            "Y": [-1.3, 1, 0.62],
            "V": [4.2, 0, 0.5],
        }
        matrix = []
        for aa in AMINO_ACIDS:
            matrix.append(physicochemical_props.get(aa, [0, 0, 0.5]))
        matrix.append([0, 0, 0])
        return torch.tensor(matrix, dtype=torch.float32)

    def _kmer_to_idx(self, kmer: str) -> int:
        idx = 0
        for aa in kmer:
            if aa in AMINO_ACID_TO_IDX:
                idx = idx * 20 + AMINO_ACID_TO_IDX[aa]
            else:
                idx = idx * 20 + 20
        return idx

    def forward(self, sequences: List[str]) -> Tensor:
        """Encode a batch of protein sequences.

        Args:
            sequences: List of amino acid sequences

        Returns:
            Encoded sequences tensor of shape (batch, seq_len, encoding_dim)
        """
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)

        if self.encoding == "onehot":
            encoded = torch.zeros(batch_size, max_len, 21)
            for i, seq in enumerate(sequences):
                indices = [AMINO_ACID_TO_IDX.get(aa, 20) for aa in seq]
                encoded[i, : len(indices)] = self.onehot_matrix[indices]

        elif self.encoding == "blosum":
            encoded = torch.zeros(batch_size, max_len, 20)
            for i, seq in enumerate(sequences):
                for j, aa in enumerate(seq):
                    if aa in BLOSUM62:
                        encoded[i, j] = torch.tensor(BLOSUM62[aa])

        elif self.encoding == "physicochemical":
            encoded = torch.zeros(batch_size, max_len, 3)
            for i, seq in enumerate(sequences):
                indices = [AMINO_ACID_TO_IDX.get(aa, 20) for aa in seq]
                encoded[i, : len(indices)] = self.physicochemical[indices]

        elif self.encoding == "kmer":
            encoded = torch.zeros(batch_size, max_len, self.embedding_dim)
            for i, seq in enumerate(sequences):
                for j in range(len(seq)):
                    kmer = seq[max(0, j - self.kmer_size + 1) : j + 1]
                    if len(kmer) == self.kmer_size:
                        kmer_idx = self._kmer_to_idx(kmer)
                        encoded[i, j] = self.kmer_embedding(
                            torch.tensor(
                                kmer_idx, device=self.kmer_embedding.weight.device
                            )
                        )

        return encoded

    def get_physicochemical_features(self, sequence: str) -> Tensor:
        """Extract physicochemical features for a sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            Feature tensor of shape (seq_len, 3)
        """
        indices = [AMINO_ACID_TO_IDX.get(aa, 20) for aa in sequence]
        return self.physicochemical[indices]


def compute_kmer_frequencies(
    sequence: str,
    k: int = 3,
    normalize: bool = True,
) -> Dict[str, float]:
    """Compute k-mer frequencies for a protein sequence.

    Args:
        sequence: Amino acid sequence
        k: Size of k-mers
        normalize: Whether to normalize by sequence length

    Returns:
        Dictionary mapping k-mer to frequency
    """
    kmers = {}
    seq_len = len(sequence)

    for i in range(seq_len - k + 1):
        kmer = sequence[i : i + k]
        kmers[kmer] = kmers.get(kmer, 0) + 1

    if normalize:
        total = sum(kmers.values())
        if total > 0:
            kmers = {k: v / total for k, v in kmers.items()}

    return kmers
