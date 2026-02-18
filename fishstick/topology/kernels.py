"""
Topological Kernels.

Provides kernel functions for persistence diagrams including
persistence heat kernels, weighted kernels, and slicing kernels
for kernel methods with topological features.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import torch
from torch import Tensor
import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class PersistenceKernel:
    """Base class for persistence kernels."""

    name: str

    def compute(
        self,
        diagram1: Tensor,
        diagram2: Tensor,
    ) -> float:
        """
        Compute kernel value between two diagrams.

        Args:
            diagram1: First persistence diagram [n1, 2] or [n1, 3]
            diagram2: Second persistence diagram [n2, 2] or [n2, 3]

        Returns:
            Kernel value
        """
        raise NotImplementedError


class PersistenceScaleSpaceKernel(PersistenceKernel):
    """
    Persistence Scale-Space Kernel.

    Defines a kernel based on the scale-space theory,
    treating birth-death pairs as Gaussian distributions.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        kernel_type: str = "gaussian",
    ):
        """
        Initialize scale-space kernel.

        Args:
            sigma: Gaussian kernel bandwidth
            kernel_type: Type of kernel ('gaussian', 'laplace')
        """
        super().__init__(name="scale_space")
        self.sigma = sigma
        self.kernel_type = kernel_type

    def compute(
        self,
        diagram1: Tensor,
        diagram2: Tensor,
    ) -> float:
        """Compute scale-space kernel."""
        if len(diagram1) == 0 or len(diagram2) == 0:
            return 0.0

        kernel_sum = 0.0

        for p1 in diagram1:
            for p2 in diagram2:
                if len(p1) >= 2 and len(p2) >= 2:
                    birth_diff = (p1[0] - p2[0]) ** 2
                    death_diff = (p1[1] - p2[1]) ** 2
                    mid_diff = birth_diff + death_diff

                    if self.kernel_type == "gaussian":
                        kernel_val = np.exp(-mid_diff / (2 * self.sigma**2))
                    else:
                        kernel_val = np.exp(-np.sqrt(mid_diff) / self.sigma)

                    kernel_sum += kernel_val

        return kernel_sum / (len(diagram1) * len(diagram2))


class PersistenceHeatKernel(PersistenceKernel):
    """
    Persistence Heat Kernel.

    Defines a kernel based on heat flow on the
    persistence diagram viewed as a metric space.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        t: float = 1.0,
    ):
        """
        Initialize heat kernel.

        Args:
            sigma: Kernel bandwidth
            t: Heat diffusion time
        """
        super().__init__(name="heat")
        self.sigma = sigma
        self.t = t

    def compute(
        self,
        diagram1: Tensor,
        diagram2: Tensor,
    ) -> float:
        """Compute heat kernel."""
        if len(diagram1) == 0 or len(diagram2) == 0:
            return 0.0

        kernel_sum = 0.0

        for p1 in diagram1:
            for p2 in diagram2:
                if len(p1) >= 2 and len(p2) >= 2:
                    squared_dist = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

                    kernel_val = np.exp(-squared_dist / (4 * self.t * self.sigma**2))

                    persistence_weight = 1.0
                    if len(p1) >= 3 and len(p2) >= 3:
                        persistence_weight = max(p1[1] - p1[0], 0) * max(
                            p2[1] - p2[0], 0
                        )
                        persistence_weight = np.sqrt(persistence_weight + 1e-10)

                    kernel_sum += kernel_val * persistence_weight

        return kernel_sum


class WeightedKernel(PersistenceKernel):
    """
    Weighted Persistence Kernel.

    Adds weighting based on persistence and dimension
    to emphasize significant topological features.
    """

    def __init__(
        self,
        weight_persistence: bool = True,
        weight_dimension: bool = True,
        kernel_type: str = "gaussian",
        sigma: float = 1.0,
    ):
        """
        Initialize weighted kernel.

        Args:
            weight_persistence: Weight by persistence
            weight_dimension: Weight by dimension
            kernel_type: Base kernel type
            sigma: Kernel bandwidth
        """
        super().__init__(name="weighted")
        self.weight_persistence = weight_persistence
        self.weight_dimension = weight_dimension
        self.kernel_type = kernel_type
        self.sigma = sigma

    def compute(
        self,
        diagram1: Tensor,
        diagram2: Tensor,
    ) -> float:
        """Compute weighted kernel."""
        if len(diagram1) == 0 or len(diagram2) == 0:
            return 0.0

        kernel_sum = 0.0

        for p1 in diagram1:
            for p2 in diagram2:
                if len(p1) < 2 or len(p2) < 2:
                    continue

                birth_diff = (p1[0] - p2[0]) ** 2
                death_diff = (p1[1] - p2[1]) ** 2

                if self.kernel_type == "gaussian":
                    kernel_val = np.exp(
                        -(birth_diff + death_diff) / (2 * self.sigma**2)
                    )
                else:
                    kernel_val = np.exp(-np.sqrt(birth_diff + death_diff) / self.sigma)

                weight = 1.0

                if self.weight_persistence:
                    pers1 = max(p1[1] - p1[0], 0)
                    pers2 = max(p2[1] - p2[0], 0)
                    weight *= np.sqrt(pers1 * pers2 + 1e-10)

                if self.weight_dimension and len(p1) >= 3 and len(p2) >= 3:
                    dim1, dim2 = int(p1[2]), int(p2[2])
                    weight *= np.exp(-abs(dim1 - dim2))

                kernel_sum += kernel_val * weight

        return kernel_sum


class SlicedWassersteinKernel(PersistenceKernel):
    """
    Sliced Wasserstein Kernel.

    Computes kernel based on sliced Wasserstein distance
    for efficient approximation of optimal transport.
    """

    def __init__(
        self,
        n_slices: int = 10,
        p: float = 2.0,
    ):
        """
        Initialize sliced Wasserstein kernel.

        Args:
            n_slices: Number of projection directions
            p: Order of Wasserstein distance
        """
        super().__init__(name="sliced_wasserstein")
        self.n_slices = n_slices
        self.p = p

    def compute(
        self,
        diagram1: Tensor,
        diagram2: Tensor,
    ) -> float:
        """Compute sliced Wasserstein kernel."""
        if len(diagram1) == 0:
            diagram1 = torch.zeros(1, 2)
        if len(diagram2) == 0:
            diagram2 = torch.zeros(1, 2)

        angles = torch.linspace(0, torch.pi, self.n_slices)

        kernel_sum = 0.0

        for angle in angles:
            dir_vector = torch.tensor([torch.cos(angle), torch.sin(angle)])

            proj1 = diagram1[:, :2] @ dir_vector
            proj2 = diagram2[:, :2] @ dir_vector

            sw_dist = self._wasserstein_1d(proj1, proj2, self.p)

            kernel_sum += np.exp(-sw_dist)

        return kernel_sum / self.n_slices

    def _wasserstein_1d(
        self,
        x: Tensor,
        y: Tensor,
        p: float,
    ) -> float:
        """Compute 1D Wasserstein distance."""
        x_sorted, _ = torch.sort(x)
        y_sorted, _ = torch.sort(y)

        if len(x_sorted) == 0:
            x_sorted = torch.zeros(1)
        if len(y_sorted) == 0:
            y_sorted = torch.zeros(1)

        n = max(len(x_sorted), len(y_sorted))

        if len(x_sorted) < n:
            x_sorted = torch.cat(
                [x_sorted, torch.full(n - len(x_sorted), x_sorted[-1])]
            )
        if len(y_sorted) < n:
            y_sorted = torch.cat(
                [y_sorted, torch.full(n - len(y_sorted), y_sorted[-1])]
            )

        dist = torch.sum(torch.abs(x_sorted - y_sorted) ** p)

        return (dist / n).item() ** (1 / p)


class PersistenceFisherKernel(PersistenceKernel):
    """
    Persistence Fisher Kernel.

    Computes kernel based on Fisher information geometry
    treating persistence diagrams as probability distributions.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        n_samples: int = 100,
    ):
        """
        Initialize Fisher kernel.

        Args:
            sigma: Bandwidth for Gaussian smoothing
            n_samples: Number of samples for distribution
        """
        super().__init__(name="fisher")
        self.sigma = sigma
        self.n_samples = n_samples

    def compute(
        self,
        diagram1: Tensor,
        diagram2: Tensor,
    ) -> float:
        """Compute Fisher kernel."""
        if len(diagram1) == 0 or len(diagram2) == 0:
            return 0.0

        samples1 = self._sample_diagram(diagram1)
        samples2 = self._sample_diagram(diagram2)

        pdf1 = self._gaussian_kde(samples1)
        pdf2 = self._gaussian_kde(samples2)

        kernel_sum = 0.0
        for s1 in samples1:
            for s2 in samples2:
                kernel_sum += pdf1(s1) * pdf2(s2)

        return kernel_sum / (self.n_samples**2)

    def _sample_diagram(self, diagram: Tensor) -> Tensor:
        """Sample points from persistence diagram."""
        samples = []

        for _ in range(self.n_samples):
            idx = np.random.randint(len(diagram))
            point = diagram[idx]

            if len(point) >= 2:
                birth = point[0].item()
                death = point[1].item()

                if death > birth:
                    sample = np.random.uniform(birth, death)
                else:
                    sample = birth
                samples.append(sample)

        return torch.tensor(samples if samples else [0.0])

    def _gaussian_kde(self, samples: Tensor) -> Callable:
        """Create Gaussian KDE from samples."""
        samples_np = samples.numpy()

        def kde(x: float) -> float:
            if len(samples_np) == 0:
                return 0.0
            return np.mean(
                [np.exp(-((x - s) ** 2) / (2 * self.sigma**2)) for s in samples_np]
            )

        return kde


class TopologicalKernelMatrix:
    """
    Topological Kernel Matrix.

    Computes kernel matrices for collections of persistence diagrams.
    """

    def __init__(
        self,
        kernel: Optional[PersistenceKernel] = None,
    ):
        """
        Initialize kernel matrix computer.

        Args:
            kernel: Kernel function to use
        """
        self.kernel = kernel or PersistenceScaleSpaceKernel()

    def compute_matrix(
        self,
        diagrams: List[Tensor],
    ) -> Tensor:
        """
        Compute kernel matrix for a list of diagrams.

        Args:
            diagrams: List of persistence diagrams

        Returns:
            Kernel matrix [n, n]
        """
        n = len(diagrams)
        matrix = torch.zeros(n, n)

        for i in range(n):
            for j in range(i, n):
                if isinstance(self.kernel, PersistenceScaleSpaceKernel):
                    k_val = self.kernel.compute(diagrams[i], diagrams[j])
                elif isinstance(self.kernel, PersistenceHeatKernel):
                    k_val = self.kernel.compute(diagrams[i], diagrams[j])
                elif isinstance(self.kernel, WeightedKernel):
                    k_val = self.kernel.compute(diagrams[i], diagrams[j])
                elif isinstance(self.kernel, SlicedWassersteinKernel):
                    k_val = self.kernel.compute(diagrams[i], diagrams[j])
                elif isinstance(self.kernel, PersistenceFisherKernel):
                    k_val = self.kernel.compute(diagrams[i], diagrams[j])
                else:
                    k_val = 0.0

                matrix[i, j] = k_val
                matrix[j, i] = k_val

        return matrix


def persistence_kernel_gradient(
    diagram: Tensor,
    kernel: PersistenceKernel,
    other_diagram: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute gradient of persistence kernel w.r.t. diagram points.

    Args:
        diagram: Input diagram [n, 2]
        kernel: Kernel function
        other_diagram: Second diagram for cross-kernel

    Returns:
        Gradient tensor
    """
    if other_diagram is None:
        other_diagram = diagram

    n = len(diagram)
    grad = torch.zeros(n, 2)

    sigma = 1.0

    for i in range(n):
        for p2 in other_diagram:
            if len(p2) >= 2:
                diff = diagram[i, :2] - p2[:2]
                kernel_val = torch.exp(-torch.sum(diff**2) / (2 * sigma**2))
                grad[i] += -diff / (sigma**2) * kernel_val

    return grad
