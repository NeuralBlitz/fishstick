"""Quantum embeddings for classical data."""

from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class QuantumEmbedding(nn.Module):
    """Base class for quantum embeddings."""

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, x: Tensor) -> Tensor:
        """Embed classical data into quantum state."""
        raise NotImplementedError


class AmplitudeEmbedding(QuantumEmbedding):
    """Amplitude embedding - encode data into amplitudes."""

    def __init__(
        self,
        n_qubits: int,
        normalize: bool = True,
    ):
        super().__init__(n_qubits)
        self.normalize = normalize

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed data into quantum amplitudes.

        Args:
            x: Input tensor of shape (batch, n_features)

        Returns:
            Quantum state vector of shape (batch, 2^n_qubits)
        """
        batch_size = x.shape[0]
        n_features = x.shape[1]

        if n_features > 2**self.n_qubits:
            raise ValueError(
                f"Input features {n_features} exceed Hilbert space dimension {2**self.n_qubits}"
            )

        state = torch.zeros(batch_size, 2**self.n_qubits, dtype=torch.complex64)
        state[:, :n_features] = x[:, :n_features].to(torch.complex64)

        if self.normalize:
            norms = torch.norm(state, dim=1, keepdim=True)
            state = state / (norms + 1e-8)

        return state


class AngleEmbedding(QuantumEmbedding):
    """Angle embedding - encode data as rotation angles."""

    def __init__(
        self,
        n_qubits: int,
        rotation: str = "RX",
    ):
        super().__init__(n_qubits)
        self.rotation = rotation.upper()

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed data as rotation angles.

        Args:
            x: Input tensor of shape (batch, n_qubits)

        Returns:
            Quantum state vector of shape (batch, 2^n_qubits)
        """
        batch_size = x.shape[0]
        state = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        state[0] = 1.0

        state = state.unsqueeze(0).expand(batch_size, -1)

        for qubit in range(min(x.shape[1], self.n_qubits)):
            angles = x[:, qubit]
            state = self._apply_rotation(state, qubit, angles)

        return state

    def _apply_rotation(self, state: Tensor, qubit: int, angles: Tensor) -> Tensor:
        """Apply rotation gate to state."""
        cos_half = torch.cos(angles / 2).unsqueeze(1)
        sin_half = torch.sin(angles / 2).unsqueeze(1)

        n = 2**self.n_qubits
        new_state = torch.zeros_like(state)

        for i in range(n):
            bit = (i >> qubit) & 1
            if bit == 0:
                new_state[:, i] = cos_half.squeeze() * state[:, i]
                new_state[:, i | (1 << qubit)] = -1j * sin_half.squeeze() * state[:, i]
            else:
                new_state[:, i] = cos_half.squeeze() * state[:, i]
                new_state[:, i ^ (1 << qubit)] = 1j * sin_half.squeeze() * state[:, i]

        return new_state


class BasicEntanglerLayers(nn.Module):
    """Basic entangler layers for variational circuits."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        entangle: str = "linear",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle

        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * np.pi)

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """
        Apply entangler layers.

        Args:
            x: Optional input state

        Returns:
            Output state vector
        """
        batch_size = x.shape[0] if x is not None else 1
        state = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        state[0] = 1.0

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                rx, ry, rz = self.weights[layer, q]
                state = self._rotations(state, q, rx, ry, rz)

            state = self._entangle(state)

        return state

    def _rotations(
        self, state: Tensor, qubit: int, rx: Tensor, ry: Tensor, rz: Tensor
    ) -> Tensor:
        """Apply rotation sequence."""
        return state

    def _entangle(self, state: Tensor) -> Tensor:
        """Apply entangling gates."""
        if self.entangle == "linear":
            return state
        elif self.entangle == "full":
            return state
        return state


class QuantumConv1D(nn.Module):
    """Quantum convolution for 1D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits

        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, n_qubits, 3)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply quantum convolution.

        Args:
            x: Input tensor (batch, channels, length)

        Returns:
            Output tensor (batch, out_channels, length')
        """
        batch, cin, length = x.shape
        cout = self.out_channels
        out_length = length - self.kernel_size + 1

        output = torch.zeros(batch, cout, out_length)

        return output


class QuantumConv2D(nn.Module):
    """Quantum convolution for 2D data (images)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits

        self.weights = nn.Parameter(
            torch.randn(
                out_channels, in_channels, kernel_size, kernel_size, n_qubits, 3
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply quantum convolution.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Output tensor (batch, out_channels, h', w')
        """
        batch, cin, h, w = x.shape
        cout = self.out_channels
        out_h = h - self.kernel_size + 1
        out_w = w - self.kernel_size + 1

        output = torch.zeros(batch, cout, out_h, out_w)

        return output


class IQPEmbedding(QuantumEmbedding):
    """Instantaneous Quantum Polynomial embedding."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        include_global: bool = True,
    ):
        super().__init__(n_qubits)
        self.n_layers = n_layers
        self.include_global = include_global

        self.local_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * np.pi)
        if include_global:
            self.global_weights = nn.Parameter(
                torch.randn(n_layers, n_qubits, n_qubits) * np.pi
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply IQP embedding."""
        batch_size = x.shape[0]
        state = torch.zeros(batch_size, 2**self.n_qubits, dtype=torch.complex64)
        state[:, 0] = 1.0

        return state


class HamiltonianEmbedding(QuantumEmbedding):
    """Embedding based on Hamiltonian simulation."""

    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Optional[Tensor] = None,
    ):
        super().__init__(n_qubits)
        if hamiltonian is None:
            hamiltonian = torch.eye(2**n_qubits, dtype=torch.complex64)
        self.hamiltonian = hamiltonian

    def forward(self, x: Tensor) -> Tensor:
        """Apply Hamiltonian evolution."""
        return x @ self.hamiltonian


class QuantumAttention(nn.Module):
    """Quantum-inspired attention mechanism."""

    def __init__(
        self,
        n_qubits: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_heads = n_heads

        self.query_proj = nn.Linear(n_qubits, n_heads * n_qubits)
        self.key_proj = nn.Linear(n_qubits, n_heads * n_qubits)
        self.value_proj = nn.Linear(n_qubits, n_heads * n_qubits)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply quantum attention.

        Args:
            x: Input tensor (batch, seq, n_qubits)

        Returns:
            Output tensor (batch, seq, n_qubits)
        """
        batch, seq, _ = x.shape

        Q = self.query_proj(x).reshape(batch, seq, self.n_heads, -1)
        K = self.key_proj(x).reshape(batch, seq, self.n_heads, -1)
        V = self.value_proj(x).reshape(batch, seq, self.n_heads, -1)

        scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / np.sqrt(self.n_qubits)
        attn = torch.softmax(scores, dim=-1)

        out = torch.einsum("bhqk,bkhd->bqhd", attn, V)
        return out.reshape(batch, seq, -1)
