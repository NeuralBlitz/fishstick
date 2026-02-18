"""
Sparse Coding and Dictionary Learning

Sparse representation learning with learnable dictionaries
for signal decomposition.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DictionaryLearning(nn.Module):
    """Learnable dictionary for sparse coding.

    Implements online dictionary learning with sparse coding
    for signal decomposition.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_atoms: int = 64,
        code_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_atoms = num_atoms
        self.code_dim = code_dim

        self.dictionary = nn.Parameter(torch.randn(num_atoms, input_dim))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, code_dim * 2),
            nn.ReLU(),
            nn.Linear(code_dim * 2, code_dim),
        )

        self.decoder = nn.Linear(code_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode with dictionary learning.

        Args:
            x: Input signal

        Returns:
            Tuple of (reconstructed signal, sparse code)
        """
        code = self.encoder(x)

        code_sparse = F.relu(code)

        reconstructed = self.decoder(code_sparse)

        return reconstructed, code_sparse

    def get_atoms(self) -> torch.Tensor:
        """Get dictionary atoms."""
        return F.normalize(self.dictionary, dim=-1)

    def sparse_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse coding with dictionary.

        Args:
            x: Input signal

        Returns:
            Sparse code
        """
        D = self.get_atoms()

        code = x @ D.T

        code_sparse = F.relu(code)

        return code_sparse


class SparseCodingLayer(nn.Module):
    """Single sparse coding layer with ISTA algorithm."""

    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        num_iterations: int = 10,
        lambda_: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_iterations = num_iterations
        self.lambda_ = lambda_

        self.dictionary = nn.Parameter(torch.randn(code_dim, input_dim))

        self._init_dictionary()

    def _init_dictionary(self):
        """Initialize dictionary using K-SVD like initialization."""
        with torch.no_grad():
            nn.init.orthogonal_(self.dictionary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply iterative shrinkage (ISTA) for sparse coding.

        Args:
            x: Input signal (batch, input_dim)

        Returns:
            Sparse code (batch, code_dim)
        """
        D = F.normalize(self.dictionary, dim=-1)

        code = torch.zeros(x.shape[0], self.code_dim, device=x.device)

        for _ in range(self.num_iterations):
            residual = x - code @ D

            gradient = residual @ D.T

            code = code + 0.5 * gradient

            code = self._soft_threshold(code, self.lambda_)

        return code

    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft thresholding operator."""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class KSVDLearner(nn.Module):
    """K-SVD inspired dictionary learning layer."""

    def __init__(
        self,
        signal_size: int = 256,
        atom_size: int = 64,
        num_atoms: int = 32,
    ):
        super().__init__()
        self.signal_size = signal_size
        self.atom_size = atom_size
        self.num_atoms = num_atoms

        self.atoms = nn.Parameter(torch.randn(num_atoms, atom_size))

        self.projection = nn.Linear(signal_size, atom_size)

        self.reconstruction = nn.Linear(atom_size, signal_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with dictionary learning.

        Args:
            x: Input signal

        Returns:
            Reconstructed signal
        """
        x_proj = self.projection(x)

        atoms = F.normalize(self.atoms, dim=-1)

        code = x_proj @ atoms.T

        code_sparse = F.relu(code)

        reconstructed = self.reconstruction(code_sparse)

        return reconstructed

    def get_sparse_code(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse code from input."""
        x_proj = self.projection(x)

        atoms = F.normalize(self.atoms, dim=-1)

        code = x_proj @ atoms.T

        return F.relu(code)


class ConvSparseCoding(nn.Module):
    """Convolutional sparse coding."""

    def __init__(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_filters: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        self.filters = nn.Parameter(torch.randn(num_filters, in_channels, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional sparse coding.

        Args:
            x: Input signal

        Returns:
            Sparse representation
        """
        batch_size = x.shape[0]

        feature_maps = F.conv1d(
            x, self.filters, padding=self.kernel_size // 2, groups=1
        )

        sparse_maps = F.relu(feature_maps)

        return sparse_maps


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder with dictionary learning."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        code_dim: int = 32,
        sparsity_weight: float = 0.1,
    ):
        super().__init__()
        self.sparsity_weight = sparsity_weight

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, code_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)
        )

        self.dictionary = nn.Parameter(torch.randn(code_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparsity penalty.

        Args:
            x: Input signal

        Returns:
            Tuple of (reconstructed, code)
        """
        code = self.encoder(x)

        code_sparse = F.relu(code)

        reconstructed = self.decoder(code_sparse)

        return reconstructed, code_sparse

    def get_sparsity_loss(self, code: torch.Tensor) -> torch.Tensor:
        """Compute sparsity regularization loss."""
        return torch.mean(torch.abs(code))


class OnlineDictionaryLearning(nn.Module):
    """Online dictionary learning for streaming data."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_atoms: int = 32,
        batch_size: int = 32,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_atoms = num_atoms
        self.batch_size = batch_size

        self.dictionary = nn.Parameter(torch.randn(num_atoms, feature_dim))

        self._init_dictionary()

    def _init_dictionary(self):
        """Initialize dictionary."""
        with torch.no_grad():
            for i in range(self.num_atoms):
                self.dictionary[i] = F.normalize(torch.randn(self.feature_dim), dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply online dictionary learning.

        Args:
            x: Input features (batch, feature_dim)

        Returns:
            Tuple of (sparse codes, reconstructed)
        """
        D = F.normalize(self.dictionary, dim=-1)

        codes = self._omp_sparse_encode(x, D)

        reconstructed = codes @ D

        return codes, reconstructed

    def _omp_sparse_encode(
        self,
        x: torch.Tensor,
        D: torch.Tensor,
        sparsity: int = 5,
    ) -> torch.Tensor:
        """Orthogonal Matching Pursuit for sparse coding."""
        batch_size, n_samples = x.shape[0], x.shape[1] if x.dim() > 1 else 1

        if x.dim() == 1:
            x = x.unsqueeze(0)

        codes = torch.zeros(x.shape[0], self.num_atoms, device=x.device)

        for b in range(x.shape[0]):
            signal = x[b]
            residual = signal.clone()
            selected_indices = []

            for _ in range(sparsity):
                correlations = torch.abs(D @ residual)
                idx = torch.argmax(correlations)
                selected_indices.append(idx)

                D_subset = D[selected_indices]

                try:
                    coeffs = torch.linalg.lstsq(
                        D_subset, signal.unsqueeze(-1)
                    ).solution.squeeze(-1)
                except:
                    coeffs = torch.zeros(len(selected_indices), device=x.device)

                residual = signal - D_subset.T @ coeffs

            for idx, coeff in zip(selected_indices, coeffs):
                codes[b, idx] = coeff

        return codes


class DictionaryConv1D(nn.Module):
    """1D convolution with dictionary learning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_atoms: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_atoms = num_atoms

        self.dictionary = nn.Parameter(torch.randn(num_atoms, in_channels, kernel_size))

        self.atten_coefficients = nn.Sequential(
            nn.Conv1d(num_atoms, out_channels, 1), nn.ReLU()
        )

        self.reconstruction = nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dictionary-based convolution.

        Args:
            x: Input signal

        Returns:
            Output signal
        """
        batch_size = x.shape[0]

        sparse_codes = []

        for i in range(self.num_atoms):
            response = F.conv1d(
                x, self.dictionary[i : i + 1], padding=self.dictionary.shape[-1] // 2
            )
            sparse_codes.append(response)

        sparse_maps = torch.cat(sparse_codes, dim=1)

        features = self.atten_coefficients(sparse_maps)

        output = self.reconstruction(features)

        return output


class SparseCodingLoss(nn.Module):
    """Combined loss for sparse coding training."""

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        sparsity_weight: float = 0.1,
        diversity_weight: float = 0.01,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        code: torch.Tensor,
        dictionary: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sparse coding loss.

        Args:
            reconstructed: Reconstructed signal
            original: Original signal
            code: Sparse code
            dictionary: Dictionary matrix

        Returns:
            Combined loss
        """
        recon_loss = F.mse_loss(reconstructed, original)

        sparsity_loss = torch.mean(torch.abs(code))

        D = F.normalize(dictionary, dim=-1)
        DDT = D @ D.T
        diversity_loss = torch.mean(
            (DDT - torch.eye(DDT.shape[0], device=DDT.device)) ** 2
        )

        total_loss = (
            self.reconstruction_weight * recon_loss
            + self.sparsity_weight * sparsity_loss
            + self.diversity_weight * diversity_loss
        )

        return total_loss


class ScatteringSparseCoding(nn.Module):
    """Sparse coding on wavelet scattering features."""

    def __init__(
        self,
        input_dim: int = 128,
        num_atoms: int = 32,
        code_dim: int = 16,
    ):
        super().__init__()

        self.dictionary = nn.Parameter(torch.randn(num_atoms, input_dim))

        self.code_proj = nn.Sequential(
            nn.Linear(num_atoms, code_dim), nn.ReLU(), nn.Linear(code_dim, code_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sparse coding on scattering features.

        Args:
            x: Input features

        Returns:
            Sparse code
        """
        D = F.normalize(self.dictionary, dim=-1)

        code = x @ D.T

        code_sparse = F.relu(code)

        code_proj = self.code_proj(code_sparse)

        return code_proj


class LearnedSparsify(nn.Module):
    """Learnable sparsification layer."""

    def __init__(
        self,
        input_dim: int,
        code_dim: int,
    ):
        super().__init__()

        self.thresholds = nn.Parameter(torch.ones(code_dim) * 0.1)

        self.encoder_weights = nn.Parameter(torch.randn(input_dim, code_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable sparsification.

        Args:
            x: Input signal

        Returns:
            Sparse output
        """
        code = x @ self.encoder_weights

        thresholds = torch.abs(self.thresholds)

        code_sparse = self._soft_threshold(code, thresholds)

        return code_sparse

    def _soft_threshold(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Learnable soft thresholding."""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class MultiScaleSparseCoding(nn.Module):
    """Sparse coding at multiple scales."""

    def __init__(
        self,
        input_dim: int = 256,
        scales: list = [1, 2, 4],
        atoms_per_scale: int = 16,
    ):
        super().__init__()
        self.scales = scales

        self.sparse_layers = nn.ModuleList(
            [
                SparseCodingLayer(
                    input_dim // scale, atoms_per_scale, code_dim=atoms_per_scale
                )
                for scale in scales
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Apply multi-scale sparse coding.

        Args:
            x: Input signal

        Returns:
            List of sparse codes at each scale
        """
        codes = []

        for scale, layer in zip(self.scales, self.sparse_layers):
            if scale == 1:
                x_scaled = x
            else:
                x_scaled = F.avg_pool1d(x, kernel_size=scale, stride=scale)

            code = layer(x_scaled)
            codes.append(code)

        return codes
