"""DNA/RNA nucleotide encoding, k-mer counting, and motif finding."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import re


DNA_ALPHABET = "ACGT"
RNA_ALPHABET = "ACGU"
DNA_RNA_ALPHABET = "ACGTU"

NUCLEOTIDE_TO_IDX = {nt: i for i, nt in enumerate(DNA_RNA_ALPHABET)}
IDX_TO_NUCLEOTIDE = {i: nt for nt, i in NUCLEOTIDE_TO_IDX.items()}

COMPLEMENT = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "U": "A",
    "N": "N",
}

DNA_CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


class NucleotideEncoder(nn.Module):
    """Encodes DNA/RNA sequences using various encoding schemes.

    Supports:
    - onehot: One-hot encoding
    - binary: Binary encoding (2-bit)
    - physicochemical: Physicochemical properties

    Attributes:
        encoding: The encoding scheme to use
        is_rna: Whether the sequence is RNA
    """

    def __init__(
        self,
        encoding: str = "onehot",
        is_rna: bool = False,
    ) -> None:
        super().__init__()
        self.encoding = encoding
        self.is_rna = is_rna

        if encoding == "onehot":
            self.alphabet_size = 5
            self.register_buffer(
                "onehot_matrix",
                torch.eye(self.alphabet_size),
            )
        elif encoding == "binary":
            self.alphabet_size = 4

    def _nucleotide_to_idx(self, nt: str) -> int:
        nt = nt.upper()
        if nt in NUCLEOTIDE_TO_IDX:
            return NUCLEOTIDE_TO_IDX[nt]
        return 4

    def forward(self, sequences: List[str]) -> Tensor:
        """Encode a batch of DNA/RNA sequences.

        Args:
            sequences: List of nucleotide sequences

        Returns:
            Encoded sequences tensor
        """
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)

        if self.encoding == "onehot":
            encoded = torch.zeros(batch_size, max_len, self.alphabet_size)
            for i, seq in enumerate(sequences):
                indices = [self._nucleotide_to_idx(nt) for nt in seq]
                encoded[i, : len(indices)] = self.onehot_matrix[indices]

        elif self.encoding == "binary":
            encoded = torch.zeros(batch_size, max_len, 2)
            for i, seq in enumerate(sequences):
                for j, nt in enumerate(seq):
                    idx = self._nucleotide_to_idx(nt)
                    encoded[i, j, 0] = (idx >> 1) & 1
                    encoded[i, j, 1] = idx & 1

        return encoded


class KMerCounter:
    """Counts k-mers in DNA/RNA sequences.

    Attributes:
        k: Size of k-mers to count
        alphabet: Nucleotide alphabet to use
    """

    def __init__(self, k: int = 3, alphabet: str = DNA_ALPHABET) -> None:
        self.k = k
        self.alphabet = alphabet.upper()

    def count(self, sequence: str) -> Dict[str, int]:
        """Count k-mers in a sequence.

        Args:
            sequence: Nucleotide sequence

        Returns:
            Dictionary mapping k-mer to count
        """
        sequence = sequence.upper()
        kmers = {}

        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i : i + self.k]
            if all(c in self.alphabet for c in kmer):
                kmers[kmer] = kmers.get(kmer, 0) + 1

        return kmers

    def count_reverse_complement(self, sequence: str) -> Dict[str, int]:
        """Count k-mers including reverse complement.

        Args:
            sequence: Nucleotide sequence

        Returns:
            Dictionary with combined k-mer counts
        """
        forward_counts = self.count(sequence)
        reverse_seq = self.reverse_complement(sequence)
        reverse_counts = self.count(reverse_seq)

        combined = {}
        for kmer, count in forward_counts.items():
            combined[kmer] = combined.get(kmer, 0) + count

        for kmer, count in reverse_counts.items():
            combined[kmer] = combined.get(kmer, 0) + count

        return combined

    def normalize_counts(
        self,
        counts: Dict[str, int],
        method: str = "frequency",
    ) -> Dict[str, float]:
        """Normalize k-mer counts.

        Args:
            counts: Raw k-mer counts
            method: Normalization method

        Returns:
            Normalized counts
        """
        total = sum(counts.values())

        if method == "frequency" and total > 0:
            return {k: v / total for k, v in counts.items()}

        elif method == "log":
            return {k: np.log1p(v) for k, v in counts.items()}

        return counts

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Compute reverse complement of DNA sequence.

        Args:
            sequence: DNA sequence

        Returns:
            Reverse complement sequence
        """
        sequence = sequence.upper()
        complement = "".join(COMPLEMENT.get(nt, "N") for nt in sequence)
        return complement[::-1]

    def get_all_kmers(self) -> List[str]:
        """Get all possible k-mers for the alphabet.

        Returns:
            List of all possible k-mers
        """
        import itertools

        return ["".join(p) for p in itertools.product(self.alphabet, repeat=self.k)]


class MotifFinder:
    """Finds known sequence motifs in DNA/RNA sequences.

    Supports both exact matching and position weight matrix (PWM) matching.

    Attributes:
        motifs: Dictionary of motif patterns or PWMs
    """

    def __init__(self, motifs: Optional[Dict[str, str]] = None) -> None:
        self.motifs = motifs or {}

    def add_motif(self, name: str, pattern: str) -> None:
        """Add a motif pattern.

        Args:
            name: Motif name
            pattern: Motif pattern (regex or IUPAC notation)
        """
        self.motifs[name] = pattern

    def find_matches(
        self,
        sequence: str,
        motif_name: str,
        use_regex: bool = True,
    ) -> List[Dict]:
        """Find motif matches in sequence.

        Args:
            sequence: DNA/RNA sequence
            motif_name: Name of motif to search
            use_regex: Whether to use regex matching

        Returns:
            List of match dictionaries with position and sequence
        """
        if motif_name not in self.motifs:
            return []

        pattern = self.motifs[motif_name]
        matches = []

        if use_regex:
            try:
                regex = re.compile(pattern)
                for match in regex.finditer(sequence.upper()):
                    matches.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "sequence": match.group(),
                            "motif": motif_name,
                        }
                    )
            except re.error:
                pass
        else:
            seq = sequence.upper()
            pattern_upper = pattern.upper()
            start = 0
            while True:
                pos = seq.find(pattern_upper, start)
                if pos == -1:
                    break
                matches.append(
                    {
                        "start": pos,
                        "end": pos + len(pattern_upper),
                        "sequence": pattern_upper,
                        "motif": motif_name,
                    }
                )
                start = pos + 1

        return matches

    def find_all_matches(
        self,
        sequence: str,
    ) -> Dict[str, List[Dict]]:
        """Find all motif matches.

        Args:
            sequence: DNA/RNA sequence

        Returns:
            Dictionary mapping motif name to matches
        """
        results = {}
        for motif_name in self.motifs:
            matches = self.find_matches(sequence, motif_name)
            if matches:
                results[motif_name] = matches
        return results


def translate_sequence(sequence: str) -> str:
    """Translate DNA sequence to protein.

    Args:
        sequence: DNA sequence

    Returns:
        Protein sequence
    """
    sequence = sequence.upper().replace("U", "T")

    if len(sequence) < 3:
        return ""

    protein = []
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i : i + 3]
        aa = DNA_CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)

    return "".join(protein)


def gc_content(sequence: str) -> float:
    """Calculate GC content of a sequence.

    Args:
        sequence: DNA/RNA sequence

    Returns:
        GC content as fraction
    """
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    if len(sequence) == 0:
        return 0.0
    return gc_count / len(sequence)


def compute_complexity_profile(
    sequence: str,
    window_size: int = 50,
) -> List[float]:
    """Compute sequence complexity profile.

    Uses Shannon entropy in sliding windows.

    Args:
        sequence: DNA/RNA sequence
        window_size: Size of sliding window

    Returns:
        List of complexity values
    """
    sequence = sequence.upper()
    complexity = []

    for i in range(len(sequence) - window_size + 1):
        window = sequence[i : i + window_size]
        counts = {}
        for nt in window:
            counts[nt] = counts.get(nt, 0) + 1

        entropy = 0.0
        total = len(window)
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        complexity.append(entropy)

    return complexity
