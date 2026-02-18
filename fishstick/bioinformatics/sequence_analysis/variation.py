"""DNA variation analysis and variant effect prediction."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class VariantType(Enum):
    """Types of genetic variants."""

    SNP = 0
    INSERTION = 1
    DELETION = 2
    MNV = 3
    STRUCTURAL = 4


class VariantEffect(Enum):
    """Predicted variant effects."""

    SYNONYMOUS = 0
    MISSENSE = 1
    NONSENSE = 2
    SPLICE_SITE = 3
    INTRON = 4
    INTERGENIC = 5
    UTR = 6
    PROMOTER = 7


@dataclass
class SNP:
    """Represents a single nucleotide polymorphism.

    Attributes:
        chrom: Chromosome
        position: Position (1-indexed)
        ref: Reference allele
        alt: Alternative allele
        rsid: dbSNP identifier
    """

    chrom: str
    position: int
    ref: str
    alt: str
    rsid: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.ref) == 1 and len(self.alt) == 1:
            self.variant_type = VariantType.SNP
        elif len(self.alt) > len(self.ref):
            self.variant_type = VariantType.INSERTION
        elif len(self.ref) > len(self.alt):
            self.variant_type = VariantType.DELETION
        else:
            self.variant_type = VariantType.MNV


CODON_TABLE = {
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

AA_PROPERTIES = {
    "A": "hydrophobic",
    "R": "positive",
    "N": "polar",
    "D": "negative",
    "C": "cysteine",
    "E": "negative",
    "Q": "polar",
    "G": "glycine",
    "H": "positive",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "K": "positive",
    "M": "hydrophobic",
    "F": "hydrophobic",
    "P": "proline",
    "S": "polar",
    "T": "polar",
    "W": "hydrophobic",
    "Y": "polar",
    "V": "hydrophobic",
}


class VariantEffectPredictor(nn.Module):
    """Predicts functional effects of genetic variants.

    Uses a neural network to predict variant effects on protein function,
    splicing, and regulation.

    Attributes:
        num_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of effect classes
    """

    def __init__(
        self,
        num_features: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Predict variant effects.

        Args:
            x: Input features

        Returns:
            Class probabilities
        """
        features = self.encoder(x)
        return self.classifier(features)

    def predict_effect(
        self,
        ref_sequence: str,
        alt_sequence: str,
        position: int,
    ) -> Dict:
        """Predict variant effect from sequences.

        Args:
            ref_sequence: Reference sequence
            alt_sequence: Alternative sequence
            position: Variant position

        Returns:
            Effect prediction dictionary
        """
        ref_aa = translate_dna_to_aa(ref_sequence, position)
        alt_aa = translate_dna_to_aa(alt_sequence, position)

        if ref_aa is None or alt_aa is None:
            return {"effect": VariantEffect.INTRON, "confidence": 0.0}

        if ref_aa == alt_aa:
            effect = VariantEffect.SYNONYMOUS
        elif alt_aa == "*":
            effect = VariantEffect.NONSENSE
        else:
            effect = VariantEffect.MISSENSE

        return {
            "effect": effect,
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "confidence": 1.0,
        }


def translate_dna_to_aa(sequence: str, position: int) -> Optional[str]:
    """Translate DNA at specific position to amino acid.

    Args:
        sequence: DNA sequence
        position: Position (0-indexed)

    Returns:
        Amino acid or None
    """
    codon_start = (position // 3) * 3

    if codon_start + 3 > len(sequence):
        return None

    codon = sequence[codon_start : codon_start + 3].upper()
    return CODON_TABLE.get(codon)


def compute_sift_score(
    ref_aa: str,
    alt_aa: str,
) -> float:
    """Compute SIFT-like score for variant.

    Args:
        ref_aa: Reference amino acid
        alt_aa: Alternative amino acid

    Returns:
        SIFT score (0-1, lower is more damaging)
    """
    if ref_aa == alt_aa:
        return 1.0

    if "*" in [ref_aa, alt_aa]:
        return 0.0

    ref_prop = AA_PROPERTIES.get(ref_aa, "unknown")
    alt_prop = AA_PROPERTIES.get(alt_aa, "unknown")

    if ref_prop == alt_prop:
        return 0.8
    elif ref_prop in ["hydrophobic", "polar", "positive", "negative"]:
        return 0.3

    return 0.5


def compute_polyphen_score(
    ref_aa: str,
    alt_aa: str,
) -> float:
    """Compute PolyPhen-like score for variant.

    Args:
        ref_aa: Reference amino acid
        alt_aa: Alternative amino acid

    Returns:
        PolyPhen score (0-1, higher is more damaging)
    """
    if ref_aa == alt_aa:
        return 0.0

    if "*" in [ref_aa, alt_aa]:
        return 0.95

    ref_prop = AA_PROPERTIES.get(ref_aa, "unknown")
    alt_prop = AA_PROPERTIES.get(alt_aa, "unknown")

    if ref_prop == "hydrophobic" and alt_prop == "polar":
        return 0.7
    elif ref_prop == "positive" and alt_prop == "negative":
        return 0.9

    return 0.4


def vcf_to_tensor(vcf_records: List[Dict]) -> Tensor:
    """Convert VCF records to tensor.

    Args:
        vcf_records: List of VCF record dictionaries

    Returns:
        Feature tensor
    """
    features = []

    for record in vcf_records:
        ref_len = len(record.get("ref", ""))
        alt_len = len(record.get("alt", ""))

        qual = record.get("QUAL", 0)

        is_snp = 1 if ref_len == 1 and alt_len == 1 else 0
        is_indel = 1 if ref_len != alt_len else 0

        features.append([ref_len, alt_len, qual, is_snp, is_indel])

    return torch.tensor(features, dtype=torch.float32)


def compute_af_from_genotypes(
    genotypes: List[str],
) -> Dict[str, float]:
    """Compute allele frequencies from genotypes.

    Args:
        genotypes: List of genotype strings

    Returns:
        Dictionary of allele frequencies
    """
    allele_counts: Dict[str, int] = {}

    for genotype in genotypes:
        alleles = genotype.replace("|", "/").split("/")
        for allele in alleles:
            if allele not in [".", "0"]:
                allele_counts[allele] = allele_counts.get(allele, 0) + 1

    total = sum(allele_counts.values())
    if total == 0:
        return {}

    return {allele: count / total for allele, count in allele_counts.items()}
