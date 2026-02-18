"""Protein structure prediction: secondary structure and torsion angles."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class SecondaryStructure(Enum):
    """Secondary structure element types."""

    HELIX = 0
    SHEET = 1
    COIL = 2


SECONDARY_STRUCTURE_LABELS = {
    "H": SecondaryStructure.HELIX,
    "G": SecondaryStructure.HELIX,
    "I": SecondaryStructure.HELIX,
    "E": SecondaryStructure.SHEET,
    "B": SecondaryStructure.SHEET,
    "S": SecondaryStructure.COIL,
    "C": SecondaryStructure.COIL,
    "T": SecondaryStructure.COIL,
    "-": SecondaryStructure.COIL,
}


class SecondaryStructurePredictor(nn.Module):
    """Predicts secondary structure from protein sequence.

    Uses a convolutional neural network architecture with attention
    to predict 3-class secondary structure (helix, sheet, coil).

    Attributes:
        num_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes (3 for helix/sheet/coil)
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_features: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(num_features, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Predict secondary structure.

        Args:
            x: Input features of shape (batch, seq_len, num_features)
            mask: Optional attention mask

        Returns:
            Predictions of shape (batch, seq_len, num_classes)
        """
        x = x.transpose(1, 2)

        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)

        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        x = x.transpose(1, 2)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)

        x = x.transpose(1, 2)
        x = torch.relu(self.conv3(x.transpose(1, 2))).transpose(1, 2)
        x = self.norm2(x + x)

        x = self.classifier(x)

        return x

    def predict_psipred(self, x: Tensor) -> List[str]:
        """Convert predictions to PSIPRED format.

        Args:
            x: Predictions of shape (seq_len, num_classes)

        Returns:
            List of predicted secondary structure labels
        """
        predictions = torch.argmax(x, dim=-1)
        labels = ["C"] * len(predictions)
        for i, pred in enumerate(predictions):
            if pred == 0:
                labels[i] = "H"
            elif pred == 1:
                labels[i] = "E"
        return labels


@dataclass
class TorsionAngles:
    """Represents protein backbone torsion angles.

    Attributes:
        phi: Phi angle (C-N-CA-C)
        psi: Psi angle (N-CA-C-N)
        omega: Omega angle (CA-C-N-CA)
    """

    phi: Optional[Tensor] = None
    psi: Optional[Tensor] = None
    omega: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.phi is not None and self.psi is not None:
            self._validate_angles()

    def _validate_angles(self) -> None:
        if self.phi is not None:
            assert torch.all((self.phi >= -np.pi) & (self.phi <= np.pi))
        if self.psi is not None:
            assert torch.all((self.psi >= -np.pi) & (self.psi <= np.pi))
        if self.omega is not None:
            assert torch.all((self.omega >= -np.pi) & (self.omega <= np.pi))

    @classmethod
    def from_coordinates(
        cls,
        coordinates: Tensor,
        atom_names: Optional[List[str]] = None,
    ) -> "TorsionAngles":
        """Compute torsion angles from atomic coordinates.

        Args:
            coordinates: Atomic coordinates (N, 3)
            atom_names: Optional list of atom names

        Returns:
            TorsionAngles object with computed angles
        """
        if atom_names is None:
            atom_names = ["N", "CA", "C"] * (len(coordinates) // 3)

        phi_angles = []
        psi_angles = []
        omega_angles = []

        for i in range(1, len(coordinates) - 1):
            if (
                atom_names[i - 1] == "C"
                and atom_names[i] == "N"
                and atom_names[i + 1] == "CA"
            ):
                if i >= 2 and i < len(coordinates) - 1:
                    prev_c = coordinates[i - 2]
                    n = coordinates[i]
                    ca = coordinates[i + 1]
                    c = coordinates[i + 2] if i + 2 < len(coordinates) else None

                    if c is not None:
                        phi = cls._compute_dihedral(prev_c, n, ca, c)
                        phi_angles.append(phi)

                        if i + 3 < len(coordinates):
                            next_n = coordinates[i + 3]
                            psi = cls._compute_dihedral(n, ca, c, next_n)
                            psi_angles.append(psi)

                            omega = cls._compute_dihedral(
                                ca,
                                c,
                                next_n,
                                coordinates[i + 4]
                                if i + 4 < len(coordinates)
                                else None,
                            )
                            if omega is not None:
                                omega_angles.append(omega)

        return cls(
            phi=torch.tensor(phi_angles) if phi_angles else None,
            psi=torch.tensor(psi_angles) if psi_angles else None,
            omega=torch.tensor(omega_angles) if omega_angles else None,
        )

    @staticmethod
    def _compute_dihedral(
        p0: Tensor,
        p1: Tensor,
        p2: Tensor,
        p3: Optional[Tensor],
    ) -> Optional[float]:
        if p3 is None:
            return None

        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2

        n1 = torch.cross(b1, b2)
        n2 = torch.cross(b2, b3)

        n1_norm = n1 / torch.norm(n1)
        n2_norm = n2 / torch.norm(n2)

        m1 = torch.cross(n1_norm, b2 / torch.norm(b2))

        x = torch.dot(n1_norm, n2_norm)
        y = torch.dot(m1, n2_norm)

        angle = torch.atan2(y, x)
        return angle.item()

    def to_sin_cos(self) -> Tensor:
        """Convert angles to sin/cos representation.

        Returns:
            Tensor of shape (3, 2) or (seq_len, 6)
        """
        angles = []
        for angle in [self.phi, self.psi, self.omega]:
            if angle is not None:
                angles.append(torch.sin(angle))
                angles.append(torch.cos(angle))
            else:
                angles.append(torch.zeros(len(self.phi) if self.phi is not None else 1))
                angles.append(torch.zeros(len(self.psi) if self.psi is not None else 1))
        return torch.stack(angles)

    @classmethod
    def from_sin_cos(cls, sin_cos: Tensor) -> "TorsionAngles":
        """Reconstruct angles from sin/cos representation.

        Args:
            sin_cos: Tensor of shape (3, 2) or (seq_len, 6)

        Returns:
            TorsionAngles object
        """
        phi = torch.atan2(sin_cos[0, 0], sin_cos[0, 1])
        psi = torch.atan2(sin_cos[1, 0], sin_cos[1, 1])
        omega = torch.atan2(sin_cos[2, 0], sin_cos[2, 1])
        return cls(phi=phi.unsqueeze(0), psi=psi.unsqueeze(0), omega=omega.unsqueeze(0))


def predict_secondary_structure_from_sequence(
    sequence: str,
    model: SecondaryStructurePredictor,
    encoder: nn.Module,
) -> List[str]:
    """Predict secondary structure for a protein sequence.

    Args:
        sequence: Amino acid sequence
        model: Trained secondary structure predictor
        encoder: Sequence encoder

    Returns:
        List of predicted secondary structure labels
    """
    model.eval()
    with torch.no_grad():
        encoded = encoder([sequence])
        predictions = model(encoded)
        return model.predict_psipred(predictions[0])
