"""
Compressed Sensing Module

Implements compressed sensing reconstruction algorithms including:
- Orthogonal Matching Pursuit (OMP)
- Iterative Hard Thresholding (IHT)
- Compressive Sampling Matching Pursuit (CoSaMP)
- Total Variation minimization
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CompressedSensingReconstructor(nn.Module):
    """Base class for compressed sensing reconstruction.

    Provides common functionality for CS reconstruction
    algorithms that solve: min ||x||_0 s.t. ||Ax - y||_2 < epsilon
    """

    def __init__(
        self,
        sensing_matrix: nn.Module,
        signal_dim: int,
        sparsity: int = 10,
    ):
        super().__init__()
        self.sensing_matrix = sensing_matrix
        self.signal_dim = signal_dim
        self.sparsity = sparsity

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """Reconstruct signal from measurements.

        Args:
            measurements: Compressed measurements

        Returns:
            Reconstructed sparse signal
        """
        raise NotImplementedError


class OMP(CompressedSensingReconstructor):
    """Orthogonal Matching Pursuit implementation.

    Greedy algorithm for sparse recovery that iteratively
    selects atoms from the dictionary that best match the residual.

    Solves: min ||x||_0 s.t. ||Ax - y||_2 < epsilon

    Example:
        >>> sensing_matrix = SensingMatrix(num_measurements=50, signal_dim=256)
        >>> omp = OMP(sensing_matrix, signal_dim=256, sparsity=10)
        >>> measurements = sensing_matrix(signal)
        >>> reconstructed = omp(measurements)
    """

    def __init__(
        self,
        sensing_matrix: nn.Module,
        signal_dim: int,
        sparsity: int = 10,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
    ):
        super().__init__(sensing_matrix, signal_dim, sparsity)
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """Reconstruct using Orthogonal Matching Pursuit.

        Args:
            measurements: Compressed measurements

        Returns:
            Reconstructed sparse signal
        """
        if measurements.dim() == 1:
            measurements = measurements.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        A = self.sensing_matrix.matrix
        batch_size = measurements.shape[0]

        x_reconstructed = torch.zeros(
            batch_size, self.signal_dim, device=measurements.device
        )

        for b in range(batch_size):
            y = measurements[b]
            residual = y.clone()
            indices = []
            support = torch.zeros(
                self.signal_dim, dtype=torch.long, device=measurements.device
            )

            for _ in range(self.sparsity):
                correlations = torch.abs(A @ residual)
                correlations[support > 0] = -1

                _, idx = torch.max(correlations, dim=0)
                idx = idx.item()

                if idx in indices:
                    break

                indices.append(idx)
                support[idx] = 1

                A_support = A[:, indices]
                try:
                    x_support = torch.linalg.lstsq(A_support, y).solution
                except:
                    break

                residual = y - A_support @ x_support

                if torch.norm(residual) < self.tolerance:
                    break

            if indices:
                x_reconstructed[b, indices] = x_support

        if squeeze_output:
            x_reconstructed = x_reconstructed.squeeze(0)

        return x_reconstructed


class IHT(CompressedSensingReconstructor):
    """Iterative Hard Thresholding algorithm.

    Projected gradient descent with hard thresholding
    for sparse recovery.

    Solves: min 0.5*||Ax - y||_2^2 s.t. ||x||_0 <= k

    Example:
        >>> iht = IHT(sensing_matrix, signal_dim=256, sparsity=10)
        >>> iht(measurements)
    """

    def __init__(
        self,
        sensing_matrix: nn.Module,
        signal_dim: int,
        sparsity: int = 10,
        step_size: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        super().__init__(sensing_matrix, signal_dim, sparsity)
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """Reconstruct using Iterative Hard Thresholding.

        Args:
            measurements: Compressed measurements

        Returns:
            Reconstructed sparse signal
        """
        if measurements.dim() == 1:
            measurements = measurements.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        A = self.sensing_matrix.matrix
        batch_size = measurements.shape[0]
        x = torch.zeros(batch_size, self.signal_dim, device=measurements.device)

        for b in range(batch_size):
            y = measurements[b]

            for _ in range(self.max_iterations):
                residual = y - A @ x[b]
                gradient = A.T @ residual
                x_temp = x[b] + self.step_size * gradient
                x[b] = self._hard_threshold(x_temp, self.sparsity)

                if torch.norm(residual) < self.tolerance:
                    break

        if squeeze_output:
            x = x.squeeze(0)

        return x

    def _hard_threshold(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply hard thresholding - keep top-k values.

        Args:
            x: Input signal
            k: Number of non-zero entries

        Returns:
            Sparse signal with only top-k values
        """
        if k >= x.numel():
            return x

        values, indices = torch.topk(torch.abs(x), k, sorted=False)
        result = torch.zeros_like(x)
        result[indices] = x[indices]
        return result


class CoSaMP(CompressedSensingReconstructor):
    """Compressive Sampling Matching Pursuit.

    Greedy algorithm that identifies large signal components
    using signal proxy and then prunes the support.

    Example:
        >>> cosamp = CoSaMP(sensing_matrix, signal_dim=256, sparsity=10)
        >>> cosamp(measurements)
    """

    def __init__(
        self,
        sensing_matrix: nn.Module,
        signal_dim: int,
        sparsity: int = 10,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        super().__init__(sensing_matrix, signal_dim, sparsity)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """Reconstruct using CoSaMP.

        Args:
            measurements: Compressed measurements

        Returns:
            Reconstructed sparse signal
        """
        if measurements.dim() == 1:
            measurements = measurements.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        A = self.sensing_matrix.matrix
        batch_size = measurements.shape[0]
        x = torch.zeros(batch_size, self.signal_dim, device=measurements.device)
        support = torch.zeros(
            batch_size, self.signal_dim, dtype=torch.bool, device=measurements.device
        )

        for b in range(batch_size):
            y = measurements[b]
            residual = y.clone()

            for _ in range(self.max_iterations):
                proxy = A.T @ residual
                proxy[support[b]] = -torch.abs(proxy[support[b]])

                Omega = torch.topk(torch.abs(proxy), 2 * self.sparsity, sorted=False)[1]
                support_new = support[b].clone()
                support_new[Omega] = True

                A_support = A[:, support_new]
                try:
                    x_temp = torch.linalg.lstsq(A_support, y).solution
                except:
                    continue

                x_new = torch.zeros(self.signal_dim, device=measurements.device)
                x_new[support_new] = x_temp

                x_new = self._hard_threshold(x_new, self.sparsity)

                residual = y - A @ x_new

                if torch.norm(residual) < self.tolerance:
                    x[b] = x_new
                    support[b] = torch.zeros(
                        self.signal_dim, dtype=torch.bool, device=measurements.device
                    )
                    support[b][
                        torch.topk(torch.abs(x[b]), self.sparsity, sorted=False)[1]
                    ] = True
                    break

                x[b] = x_new
                support[b] = support_new

        if squeeze_output:
            x = x.squeeze(0)

        return x

    def _hard_threshold(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply hard thresholding."""
        if k >= x.numel():
            return x
        values, indices = torch.topk(torch.abs(x), k, sorted=False)
        result = torch.zeros_like(x)
        result[indices] = x[indices]
        return result


class TVMinimization(nn.Module):
    """Total Variation minimization for compresssed sensing.

    Uses primal-dual algorithm to solve:
    min ||x||_TV s.t. ||Ax - y||_2 < epsilon

    Particularly effective for piecewise constant signals
    and images.

    Example:
        >>> tv_min = TVMinimization(sensing_matrix, image_size=(32, 32))
        >>> reconstructed = tv_min(measurements)
    """

    def __init__(
        self,
        sensing_matrix: nn.Module,
        image_size: Tuple[int, int],
        num_iterations: int = 100,
        tau: float = 0.01,
        lambda_: float = 0.1,
    ):
        super().__init__()
        self.sensing_matrix = sensing_matrix
        self.image_size = image_size
        self.num_iterations = num_iterations
        self.tau = tau
        self.lambda_ = lambda_

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """Reconstruct using Total Variation minimization.

        Args:
            measurements: Compressed measurements

        Returns:
            Reconstructed image
        """
        if measurements.dim() == 1:
            measurements = measurements.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        A = self.sensing_matrix.matrix
        batch_size = measurements.shape[0]
        channels = 1

        x = torch.zeros(
            batch_size, channels, *self.image_size, device=measurements.device
        )
        x_bar = x.clone()
        y = torch.zeros_like(x)

        for _ in range(self.num_iterations):
            grad_x = self._gradient(x_bar)
            div_p = self._divergence(y)

            residual = x - self.tau * div_p
            x = self._prox_f(residual, measurements, A)

            x_bar = 2 * x - x_bar

            y = y + self.tau * grad_x
            y = y / (torch.max(torch.abs(y), dim=1, keepdim=True)[0] + 1e-8)

        if squeeze_output:
            x = x.squeeze(0)

        return x

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute image gradient."""
        grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        return torch.cat([grad_x, grad_y], dim=1)

    def _divergence(self, p: torch.Tensor) -> torch.Tensor:
        """Compute divergence."""
        p_x, p_y = p[:, :1], p[:, 1:]
        div_x = F.pad(p_x[:, :, :, 1:] - p_x[:, :, :, :-1], (1, 0, 0, 0))
        div_y = F.pad(p_y[:, :, 1:, :] - p_y[:, :-1, :, :], (0, 0, 1, 0))
        return div_x + div_y

    def _prox_f(
        self, x: torch.Tensor, measurements: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """Proximity operator for data fidelity term."""
        x_flat = x.reshape(x.shape[0], -1)
        Ax = x_flat @ A.T
        residual = Ax - measurements
        grad = residual @ A
        return x_flat - grad.reshape(x.shape)


class LearnedCompressedSensing(nn.Module):
    """Learned compressed sensing with neural network.

    Uses a neural network to learn the reconstruction
    mapping from measurements to signal.

    Example:
        >>> lcs = LearnedCompressedSensing(
        ...     num_measurements=50,
        ...     signal_dim=256,
        ...     hidden_dims=[128, 256]
        ... )
        >>> reconstructed = lcs(measurements)
    """

    def __init__(
        self,
        num_measurements: int,
        signal_dim: int,
        hidden_dims: list = [256, 512, 256],
    ):
        super().__init__()
        self.num_measurements = num_measurements
        self.signal_dim = signal_dim

        layers = []
        in_dim = num_measurements

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, signal_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """Reconstruct using learned network.

        Args:
            measurements: Compressed measurements

        Returns:
            Reconstructed signal
        """
        return self.network(measurements)


class AdaptiveCompressedSensing(nn.Module):
    """Adaptive compressed sensing with iterative refinement.

    Adaptively selects measurements based on current
    reconstruction estimate.

    Example:
        >>> acs = AdaptiveCompressedSensing(
        ...     max_measurements=100,
        ...     signal_dim=256,
        ...     sparsity=10
        ... )
        >>> reconstructed, selected_indices = acs(partial_measurements)
    """

    def __init__(
        self,
        max_measurements: int,
        signal_dim: int,
        sparsity: int = 10,
        num_adaptive_steps: int = 5,
    ):
        super().__init__()
        self.max_measurements = max_measurements
        self.signal_dim = signal_dim
        self.sparsity = sparsity
        self.num_adaptive_steps = num_adaptive_steps

        self.sensing_matrix = nn.Parameter(
            torch.randn(max_measurements, signal_dim) / np.sqrt(max_measurements),
            requires_grad=False,
        )

    def forward(
        self,
        measurements: torch.Tensor,
        num_measurements: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct with adaptive measurement selection.

        Args:
            measurements: Current measurements
            num_measurements: Number of measurements to use

        Returns:
            Tuple of (reconstructed signal, measurement indices)
        """
        if num_measurements is None:
            num_measurements = (
                measurements.shape[1]
                if measurements.dim() > 1
                else self.max_measurements
            )

        A = self.sensing_matrix[:num_measurements]

        x = A.T @ torch.linalg.solve(
            A @ A.T + 1e-6 * torch.eye(num_measurements, device=A.device), measurements
        )

        indices = torch.arange(num_measurements, device=measurements.device)

        return x, indices


def create_sensing_matrix(
    num_measurements: int,
    signal_dim: int,
    matrix_type: str = "gaussian",
) -> torch.Tensor:
    """Create a sensing matrix for compressed sensing.

    Args:
        num_measurements: Number of measurements
        signal_dim: Dimension of signal
        matrix_type: Type of matrix ('gaussian', 'bernoulli', 'rademacher')

    Returns:
        Sensing matrix
    """
    if matrix_type == "gaussian":
        return torch.randn(num_measurements, signal_dim) / np.sqrt(num_measurements)
    elif matrix_type == "bernoulli":
        matrix = torch.randint(0, 2, (num_measurements, signal_dim)).float() * 2 - 1
        return matrix / np.sqrt(num_measurements)
    elif matrix_type == "rademacher":
        return torch.randint(0, 2, (num_measurements, signal_dim)).float() * 2 - 1
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")


def compute_coherence(sensing_matrix: torch.Tensor) -> float:
    """Compute mutual coherence of sensing matrix.

    Args:
        sensing_matrix: The sensing matrix

    Returns:
        Mutual coherence value
    """
    A = sensing_matrix / torch.norm(sensing_matrix, dim=1, keepdim=True)
    G = A @ A.T
    G = G - torch.eye(G.shape[0], device=G.device)
    coherence = torch.max(torch.abs(G)).item()
    return coherence
