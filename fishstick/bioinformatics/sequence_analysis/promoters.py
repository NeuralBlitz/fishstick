"""Promoter prediction and transcription factor binding analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TranscriptionFactor:
    """Represents a transcription factor.

    Attributes:
        name: TF name
        gene_id: Gene identifier
        pwm: Position weight matrix (4 x motif_length)
        binding_threshold: Score threshold for binding
    """

    name: str
    gene_id: str
    pwm: Optional[Tensor] = None
    binding_threshold: float = 0.0


class PromoterPredictor(nn.Module):
    """Predicts promoter regions and transcription factor binding sites.

    Uses a CNN-based model to identify promoter sequences and predict
    transcription factor binding.

    Attributes:
        num_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_tfs: Number of transcription factors to predict
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_features: int = 4,
        hidden_dim: int = 128,
        num_tfs: int = 20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_tfs = num_tfs

        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )

        self.promoter_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.tf_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tfs),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Predict promoter regions and TF binding.

        Args:
            x: Input sequences (batch, seq_len, features)

        Returns:
            Tuple of (promoter_probs, tf_scores)
        """
        x = x.transpose(1, 2)

        features = self.conv_layers(x)

        seq_len = features.shape[2]
        features_pooled = features.mean(dim=2)

        promoter_scores = self.promoter_classifier(features_pooled)

        tf_scores = self.tf_predictor(features.transpose(1, 2))

        return promoter_scores, tf_scores

    def predict_promoter(
        self,
        x: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        """Predict if regions are promoters.

        Args:
            x: Input sequences
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        scores, _ = self.forward(x)
        return (scores > threshold).long()

    def predict_tf_binding(
        self,
        x: Tensor,
        threshold: float = 0.0,
    ) -> Tensor:
        """Predict transcription factor binding.

        Args:
            x: Input sequences
            threshold: Binding threshold

        Returns:
            Binding predictions
        """
        _, tf_scores = self.forward(x)
        return torch.sigmoid(tf_scores) > threshold


class TranscriptionFactorBinding:
    """Analyzes transcription factor binding sites.

    Uses position weight matrices to score TF binding.
    """

    def __init__(self) -> None:
        self.pwms: Dict[str, Tensor] = {}

    def add_pwm(self, tf_name: str, pwm: Tensor) -> None:
        """Add a position weight matrix.

        Args:
            tf_name: Transcription factor name
            pwm: PWM tensor of shape (4, motif_length)
        """
        self.pwms[tf_name] = pwm

    def score_sequence(
        self,
        sequence: Tensor,
        tf_name: str,
    ) -> Tensor:
        """Score TF binding to a sequence.

        Args:
            sequence: One-hot encoded sequence (4, seq_len)
            tf_name: TF name

        Returns:
            Binding scores per position
        """
        if tf_name not in self.pwms:
            return torch.zeros(sequence.shape[1])

        pwm = self.pwms[tf_name]

        motif_len = pwm.shape[1]
        scores = []

        for i in range(sequence.shape[1] - motif_len + 1):
            segment = sequence[:, i : i + motif_len]
            score = torch.sum(segment * pwm)
            scores.append(score.item())

        return torch.tensor(scores)

    def find_binding_sites(
        self,
        sequence: Tensor,
        tf_name: str,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """Find binding sites in sequence.

        Args:
            sequence: One-hot encoded sequence
            tf_name: TF name
            threshold: Score threshold

        Returns:
            List of binding site dictionaries
        """
        scores = self.score_sequence(sequence, tf_name)

        sites = []
        for i, score in enumerate(scores):
            if score > threshold:
                sites.append(
                    {
                        "position": i,
                        "score": score.item(),
                        "tf": tf_name,
                    }
                )

        return sites


def compute_tata_box_score(sequence: str) -> float:
    """Compute TATA box binding score.

    Args:
        sequence: DNA sequence

    Returns:
        TATA box score
    """
    sequence = sequence.upper()

    tata_consensus = "TATAAA"

    best_score = 0.0
    for i in range(len(sequence) - len(tata_consensus) + 1):
        segment = sequence[i : i + len(tata_consensus)]
        score = sum(
            1.0 if segment[j] == tata_consensus[j] else 0.5
            for j in range(len(tata_consensus))
        )
        best_score = max(best_score, score)

    return best_score / len(tata_consensus)


def compute_gc_richness(sequence: str) -> float:
    """Compute GC-richness score.

    Args:
        sequence: DNA sequence

    Returns:
        GC-richness score
    """
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    if len(sequence) == 0:
        return 0.0
    return gc_count / len(sequence)


def find_cpg_islands(sequence: str) -> List[Dict]:
    """Find CpG islands in sequence.

    Args:
        sequence: DNA sequence

    Returns:
        List of CpG island dictionaries
    """
    sequence = sequence.upper()
    islands = []

    window_size = 200
    min_gc = 0.5
    min_obsexp = 0.6

    for i in range(0, len(sequence) - window_size + 1, 50):
        window = sequence[i : i + window_size]

        c_count = window.count("C")
        g_count = window.count("G")
        cg_count = window.count("CG")

        if c_count == 0 or g_count == 0:
            continue

        gc_content = (c_count + g_count) / window_size

        if gc_content < min_gc:
            continue

        if cg_count > 0:
            obs_exp = (cg_count * window_size) / (c_count * g_count)
        else:
            obs_exp = 0

        if obs_exp >= min_obsexp:
            islands.append(
                {
                    "start": i,
                    "end": i + window_size,
                    "gc_content": gc_content,
                    "obs_exp": obs_exp,
                }
            )

    return islands
