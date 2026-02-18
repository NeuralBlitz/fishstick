"""Protein residue contact prediction."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ContactMapPredictor(nn.Module):
    """Predicts residue-residue contacts from protein sequences.

    Uses a deep residual convolutional network with attention to predict
    inter-residue contact probability maps.

    Attributes:
        num_features: Input feature dimension per residue
        hidden_dim: Hidden layer dimension
        num_layers: Number of residual blocks
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_features: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 30,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_projection = nn.Linear(num_features, hidden_dim)

        self.residual_blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: Tensor,
        sequence_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict contact map.

        Args:
            x: Input features of shape (batch, seq_len, num_features)
            sequence_mask: Optional mask for valid positions

        Returns:
            Contact probabilities of shape (batch, seq_len, seq_len)
        """
        x = self.input_projection(x)

        for block in self.residual_blocks:
            x = block(x, sequence_mask)

        x1, _ = self.attention(x, x, x, key_padding_mask=sequence_mask)
        x = x + x1

        batch_size, seq_len, _ = x.shape

        x_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        x_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)

        pairwise = torch.cat([x_i, x_j, x_i * x_j], dim=-1)

        pairwise = self.output_projection(pairwise).squeeze(-1)

        contacts = torch.sigmoid(pairwise)

        if sequence_mask is not None:
            mask_i = sequence_mask.unsqueeze(2)
            mask_j = sequence_mask.unsqueeze(1)
            contacts = contacts * mask_i * mask_j

        return contacts

    def predict_contacts(
        self,
        x: Tensor,
        threshold: float = 0.5,
        min_separation: int = 6,
    ) -> Tensor:
        """Predict contacts above threshold.

        Args:
            x: Input features
            threshold: Contact probability threshold
            min_separation: Minimum sequence separation for contacts

        Returns:
            Binary contact map
        """
        contacts = self.forward(x)

        batch_size, seq_len, _ = contacts.shape

        diagonal = torch.eye(seq_len, device=contacts.device).bool()
        separation_mask = ~diagonal

        sep_range = torch.arange(seq_len, device=contacts.device)
        sep_matrix = torch.abs(sep_range.unsqueeze(0) - sep_range.unsqueeze(1))
        separation_mask = sep_matrix >= min_separation

        contacts = contacts * separation_mask.float()

        return (contacts > threshold).long()


class ResidualConvBlock(nn.Module):
    """Residual convolution block with layer normalization."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.conv1(x).transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = x + residual
        return x


def compute_contact_map_from_coordinates(
    coordinates: Tensor,
    cutoff: float = 8.0,
) -> Tensor:
    """Compute contact map from atomic coordinates.

    A contact is defined when any two non-adjacent CA atoms
    are within the cutoff distance.

    Args:
        coordinates: CA atom coordinates (N, 3)
        cutoff: Distance cutoff in Angstroms

    Returns:
        Binary contact map (N, N)
    """
    num_residues = coordinates.shape[0]

    coords_i = coordinates.unsqueeze(1)
    coords_j = coordinates.unsqueeze(0)

    distances = torch.sqrt(torch.sum((coords_i - coords_j) ** 2, dim=-1))

    contacts = distances < cutoff

    contacts = contacts & (torch.eye(num_residues, device=coordinates.device) == 0)

    return contacts.long()


def evaluate_contact_prediction(
    predicted: Tensor,
    ground_truth: Tensor,
    min_separation: int = 6,
) -> dict:
    """Evaluate contact prediction accuracy.

    Args:
        predicted: Predicted contact probabilities
        ground_truth: True contact map
        min_separation: Minimum sequence separation

    Returns:
        Dictionary with evaluation metrics
    """
    seq_len = predicted.shape[0]

    sep_range = torch.arange(seq_len, device=predicted.device)
    sep_matrix = torch.abs(sep_range.unsqueeze(0) - sep_range.unsqueeze(1))
    mask = sep_matrix >= min_separation

    predicted = predicted[mask]
    ground_truth = ground_truth[mask]

    predicted_binary = (predicted > 0.5).float()
    correct = (predicted_binary == ground_truth).float()

    accuracy = correct.mean()

    precision = (predicted_binary * ground_truth).sum() / (
        predicted_binary.sum() + 1e-8
    )
    recall = (predicted_binary * ground_truth).sum() / (ground_truth.sum() + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }
