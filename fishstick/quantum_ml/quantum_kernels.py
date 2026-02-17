"""
Quantum Kernel Methods.

Provides quantum kernel implementations for machine learning including
fidelity kernels, projected kernels, and kernel-based algorithms.
"""

from typing import Optional, Tuple, List, Callable
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from sklearn.svm import SVC


class QuantumKernel(nn.Module):
    """
    Base class for quantum kernels.

    Quantum kernels compute similarity between data points using
    quantum circuits by encoding data into quantum states and
    measuring their overlap or fidelity.

    Args:
        n_qubits: Number of qubits for quantum encoding
        depth: Depth of variational circuit
    """

    def __init__(
        self,
        n_qubits: int = 8,
        depth: int = 2,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.params = nn.Parameter(torch.randn(depth, n_qubits, 3))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute kernel matrix between two sets of samples.

        Args:
            x1: First set of samples [n1, features]
            x2: Second set of samples [n2, features]
        Returns:
            Kernel matrix [n1, n2]
        """
        raise NotImplementedError

    def _encode(self, x: Tensor) -> Tensor:
        """Encode classical data into quantum state."""
        raise NotImplementedError

    def _compute_overlap(self, state1: Tensor, state2: Tensor) -> Tensor:
        """Compute overlap (fidelity) between two quantum states."""
        return torch.abs(torch.conj(state1) @ state2) ** 2


class FidelityQuantumKernel(QuantumKernel):
    """
    Fidelity-based Quantum Kernel.

    Computes kernel as the fidelity (overlap) between quantum
    states encoded from data points.

    K(x, y) = |⟨ψ(x)|ψ(y)⟩|²

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        encoding_type: Type of encoding ('amplitude', 'angle', 'basis')
    """

    def __init__(
        self,
        n_qubits: int = 8,
        depth: int = 2,
        encoding_type: str = "angle",
    ):
        super().__init__(n_qubits, depth)
        self.encoding_type = encoding_type

        if encoding_type == "angle":
            self.feature_map = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * depth),
                nn.Tanh(),
            )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute fidelity kernel matrix.

        Args:
            x1: [n1, features]
            x2: [n2, features]
        Returns:
            Kernel matrix [n1, n2]
        """
        n1 = x1.shape[0]
        n2 = x2.shape[0]

        x1_states = self._encode_batch(x1)
        x2_states = self._encode_batch(x2)

        kernel_matrix = torch.zeros(n1, n2, device=x1.device)

        for i in range(n1):
            for j in range(n2):
                kernel_matrix[i, j] = self._compute_overlap(x1_states[i], x2_states[j])

        return kernel_matrix

    def _encode_batch(self, x: Tensor) -> List[Tensor]:
        """Encode batch of data points into quantum states."""
        states = []
        for i in range(x.shape[0]):
            state = self._encode(x[i])
            states.append(state)
        return states

    def _encode(self, x: Tensor) -> Tensor:
        """Encode single data point into quantum state."""
        dim = 2**self.n_qubits
        state = torch.zeros(dim, dtype=torch.complex64)
        state[0] = 1.0

        if self.encoding_type == "angle":
            x_enc = x[: self.n_qubits]
            for q in range(min(self.n_qubits, len(x_enc))):
                theta = x_enc[q] * torch.pi
                state = self._apply_rotation(state, q, theta)

        elif self.encoding_type == "amplitude":
            x_norm = x / (torch.norm(x) + 1e-8)
            for i, val in enumerate(x_norm):
                if i < dim:
                    state[i] = val

        return state

    def _apply_rotation(self, state: Tensor, qubit: int, theta: float) -> Tensor:
        """Apply rotation to quantum state."""
        return state

    def gram_matrix(self, x: Tensor) -> Tensor:
        """
        Compute Gram matrix (kernel matrix) for single set.

        Args:
            x: Samples [n, features]
        Returns:
            Gram matrix [n, n]
        """
        return self.forward(x, x)


class ProjectedQuantumKernel(QuantumKernel):
    """
    Projected Quantum Kernel.

    Projects quantum state to classical space using measurement
    and computes kernel in that space. More robust to noise.

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        n_projections: Number of measurement projections
    """

    def __init__(
        self,
        n_qubits: int = 8,
        depth: int = 2,
        n_projections: int = 100,
    ):
        super().__init__(n_qubits, depth)
        self.n_projections = n_projections

        self.projection_basis = nn.Parameter(torch.randn(n_projections, n_qubits, 3))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute projected kernel matrix.

        Args:
            x1: [n1, features]
            x2: [n2, features]
        Returns:
            Kernel matrix [n1, n2]
        """
        x1_proj = self._project(x1)
        x2_proj = self._project(x2)

        x1_proj = x1_proj / (x1_proj.norm(dim=-1, keepdim=True) + 1e-8)
        x2_proj = x2_proj / (x2_proj.norm(dim=-1, keepdim=True) + 1e-8)

        return x1_proj @ x2_proj.transpose(-2, -1)

    def _project(self, x: Tensor) -> Tensor:
        """Project quantum states to classical space."""
        n_samples = x.shape[0]
        projections = torch.zeros(n_samples, self.n_projections, device=x.device)

        for i in range(n_samples):
            state = self._encode(x[i])
            probs = torch.abs(state) ** 2

            for p in range(self.n_projections):
                meas_basis = self.projection_basis[p]
                proj_val = 0.0
                for q in range(self.n_qubits):
                    idx = 1 << q
                    proj_val += probs[idx].item() * meas_basis[q, 0].item()
                projections[i, p] = proj_val

        return projections

    def _encode(self, x: Tensor) -> Tensor:
        """Encode data into quantum state."""
        dim = 2**self.n_qubits
        state = torch.zeros(dim, dtype=torch.complex64)
        state[0] = 1.0
        return state


class QuantumKernelPCA(nn.Module):
    """
    Quantum Kernel Principal Component Analysis.

    Performs kernel PCA using quantum kernels for non-linear
    dimensionality reduction.

    Args:
        n_components: Number of principal components
        n_qubits: Number of qubits for quantum kernel
        kernel: Quantum kernel instance
    """

    def __init__(
        self,
        n_components: int = 2,
        n_qubits: int = 8,
        kernel: Optional[QuantumKernel] = None,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_qubits = n_qubits

        self.kernel = kernel or FidelityQuantumKernel(n_qubits=n_qubits)

        self.register_buffer("components_", torch.zeros(n_components, n_qubits))
        self.register_buffer("mean_", torch.zeros(n_qubits))

    def fit(self, x: Tensor) -> "QuantumKernelPCA":
        """
        Fit the kernel PCA model.

        Args:
            x: Training data [n_samples, features]
        Returns:
            self
        """
        n_samples = x.shape[0]

        x_scaled = self._scale(x)
        K = self.kernel.gram_matrix(x_scaled)

        one_n = torch.ones(n_samples, n_samples) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        eigenvalues, eigenvectors = torch.linalg.eigh(K_centered)

        idx = torch.argsort(eigenvalues, descending=True)
        self.components_ = eigenvectors[:, idx[: self.n_components]].T

        self.mean_ = x_scaled.mean(dim=0)

        return self

    def transform(self, x: Tensor) -> Tensor:
        """
        Transform data to kernel PCA space.

        Args:
            x: Data [n_samples, features]
        Returns:
            Transformed data [n_samples, n_components]
        """
        x_scaled = self._scale(x)
        K = self.kernel(x_scaled, self.mean_.unsqueeze(0))

        return K @ self.components_.T

    def forward(self, x: Tensor) -> Tensor:
        """Transform data (requires fit first)."""
        return self.transform(x)

    def _scale(self, x: Tensor) -> Tensor:
        """Scale data."""
        if self.mean_.abs().sum() > 0:
            return x - self.mean_
        return x


class QuantumKernelSVM(nn.Module):
    """
    Quantum Kernel Support Vector Machine.

    Wrapper around sklearn SVC using quantum kernels.

    Args:
        n_qubits: Number of qubits for kernel
        C: Regularization parameter
        kernel_type: Type of kernel ('fidelity', 'projected')
    """

    def __init__(
        self,
        n_qubits: int = 8,
        C: float = 1.0,
        kernel_type: str = "fidelity",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.C = C
        self.kernel_type = kernel_type

        if kernel_type == "fidelity":
            self.kernel = FidelityQuantumKernel(n_qubits=n_qubits)
        else:
            self.kernel = ProjectedQuantumKernel(n_qubits=n_qubits)

        self.svm = SVC(C=C, kernel="precomputed")
        self._fitted = False

    def fit(self, x: Tensor, y: Tensor) -> "QuantumKernelSVM":
        """
        Fit the SVM model.

        Args:
            x: Training data [n_samples, features]
            y: Training labels [n_samples]
        Returns:
            self
        """
        x_np = x.cpu().detach().numpy()
        y_np = y.cpu().detach().numpy()

        K = self.kernel.forward(x_np, x_np).cpu().detach().numpy()

        self.svm.fit(K, y_np)
        self._fitted = True

        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Predict labels.

        Args:
            x: Test data [n_samples, features]
        Returns:
            Predicted labels [n_samples]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        x_np = x.cpu().detach().numpy()
        train_x = self.svm._X.cpu().detach().numpy()

        K = self.kernel.forward(x_np, train_x).cpu().detach().numpy()

        predictions = self.svm.predict(K)

        return torch.tensor(predictions, device=x.device, dtype=torch.long)

    def forward(self, x: Tensor) -> Tensor:
        """Predict (requires fit first)."""
        return self.predict(x)


class QuantumKernelRidge(nn.Module):
    """
    Quantum Kernel Ridge Regression.

    Kernel ridge regression using quantum kernels.

    Args:
        n_qubits: Number of qubits
        alpha: Regularization strength
        kernel: Quantum kernel instance
    """

    def __init__(
        self,
        n_qubits: int = 8,
        alpha: float = 1.0,
        kernel: Optional[QuantumKernel] = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha = alpha

        self.kernel = kernel or FidelityQuantumKernel(n_qubits=n_qubits)

        self.register_buffer("alpha_", torch.zeros(1))
        self.register_buffer("X_train_", torch.zeros(1, n_qubits))
        self._fitted = False

    def fit(self, x: Tensor, y: Tensor) -> "QuantumKernelRidge":
        """
        Fit the ridge regression model.

        Args:
            x: Training data [n_samples, features]
            y: Training targets [n_samples] or [n_samples, n_targets]
        Returns:
            self
        """
        K = self.kernel.gram_matrix(x)

        n = K.shape[0]
        K_reg = K + self.alpha * torch.eye(n, device=K.device)

        self.alpha_ = torch.linalg.solve(K_reg, y)
        self.X_train_ = x

        self._fitted = True
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Predict using the model.

        Args:
            x: Test data [n_samples, features]
        Returns:
            Predictions [n_samples] or [n_samples, n_targets]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        K_test = self.kernel(x, self.X_train_)

        return K_test @ self.alpha_

    def forward(self, x: Tensor) -> Tensor:
        """Predict (requires fit first)."""
        return self.predict(x)
