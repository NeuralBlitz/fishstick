"""
Topological Signatures.

Provides various topological signature representations
for machine learning including persistence images, silhouettes,
and combined vectorizations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from torch import Tensor
import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class TopologicalSignature:
    """Base class for topological signatures."""

    name: str
    dimension: int

    def to_vector(self) -> Tensor:
        """Convert signature to feature vector."""
        raise NotImplementedError


class PersistenceImage(TopologicalSignature):
    """
    Persistence Image Representation.

    Converts persistence diagram to a 2D grayscale image
    using Gaussian weighting based on persistence.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (50, 50),
        sigma: float = 1.0,
        weight_function: str = "linear",
    ):
        """
        Initialize persistence image.

        Args:
            resolution: Image resolution (birth, persistence)
            sigma: Gaussian kernel bandwidth
            weight_function: Weight function type
        """
        super().__init__(name="persistence_image", dimension=2)
        self.resolution = resolution
        self.sigma = sigma
        self.weight_function = weight_function

    def compute(
        self,
        diagram: Tensor,
        weight_power: float = 1.0,
    ) -> Tensor:
        """
        Compute persistence image from diagram.

        Args:
            diagram: Persistence diagram [n, 2] or [n, 3]
            weight_power: Power for persistence weighting

        Returns:
            Persistence image [resolution[0], resolution[1]]
        """
        if len(diagram) == 0:
            return torch.zeros(self.resolution)

        births = diagram[:, 0].numpy()
        persistences = (diagram[:, 1] - diagram[:, 0]).clamp(min=0).numpy()

        birth_min, birth_max = births.min(), births.max()
        pers_min, pers_max = persistences.min(), persistences.max()

        if birth_max == birth_min:
            birth_max = birth_min + 1
        if pers_max == pers_min:
            pers_max = pers_min + 1

        birth_grid = np.linspace(birth_min, birth_max, self.resolution[0])
        pers_grid = np.linspace(pers_min, pers_max, self.resolution[1])

        image = np.zeros(self.resolution)

        for b, p in zip(births, persistences):
            if p <= 0:
                continue

            weight = self._compute_weight(p, weight_power, pers_min, pers_max)

            for i, birth_val in enumerate(birth_grid):
                for j, pers_val in enumerate(pers_grid):
                    dist = (b - birth_val) ** 2 + (p - pers_val) ** 2
                    image[i, j] += weight * np.exp(-dist / (2 * self.sigma**2))

        return torch.tensor(image, dtype=torch.float32)

    def _compute_weight(
        self,
        persistence: float,
        power: float,
        min_pers: float,
        max_pers: float,
    ) -> float:
        """Compute persistence weight."""
        if self.weight_function == "linear":
            return persistence**power
        elif self.weight_function == "normalized":
            normalized = (persistence - min_pers) / (max_pers - min_pers + 1e-10)
            return normalized**power
        else:
            return 1.0


class TopologicalVectorization(TopologicalSignature):
    """
    Topological Vectorization.

    Provides various methods to vectorize persistence diagrams
    into fixed-length feature vectors.
    """

    def __init__(
        self,
        n_features: int = 100,
        feature_type: str = "landscape",
    ):
        """
        Initialize vectorization.

        Args:
            n_features: Number of features in vector
            feature_type: Type of vectorization
        """
        super().__init__(name=f"vectorization_{feature_type}", dimension=1)
        self.n_features = n_features
        self.feature_type = feature_type

    def compute(
        self,
        diagram: Tensor,
    ) -> Tensor:
        """
        Compute vectorization from diagram.

        Args:
            diagram: Persistence diagram [n, 2] or [n, 3]

        Returns:
            Feature vector [n_features]
        """
        if len(diagram) == 0:
            return torch.zeros(self.n_features)

        if self.feature_type == "landscape":
            return self._landscape_vector(diagram)
        elif self.feature_type == "barcode":
            return self._barcode_vector(diagram)
        elif self.feature_type == "curve":
            return self._curve_vector(diagram)
        else:
            return self._silhouette_vector(diagram)

    def _landscape_vector(self, diagram: Tensor) -> Tensor:
        """Compute persistence landscape features."""
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistences = (deaths - births).clamp(min=0)

        max_pers = persistences.max().item()
        if max_pers == 0:
            max_pers = 1.0

        t_values = torch.linspace(0, max_pers, self.n_features)

        features = []
        for t in t_values:
            count = torch.sum((births <= t) & (deaths > t)).float()
            features.append(count)

        return torch.tensor(features, dtype=torch.float32)

    def _barcode_vector(self, diagram: Tensor) -> Tensor:
        """Compute barcode histogram features."""
        births = diagram[:, 0].numpy()
        deaths = diagram[:, 1].numpy()
        persistences = np.maximum(deaths - births, 0)

        if len(persistences) == 0:
            return torch.zeros(self.n_features)

        max_pers = persistences.max()
        if max_pers == 0:
            max_pers = 1.0

        hist, _ = np.histogram(persistences, bins=self.n_features, range=(0, max_pers))

        return torch.tensor(hist, dtype=torch.float32) / (len(persistences) + 1e-10)

    def _curve_vector(self, diagram: Tensor) -> Tensor:
        """Compute Betti curve features."""
        births = diagram[:, 0]
        deaths = diagram[:, 1]

        max_filtration = deaths.max().item()
        if max_filtration == 0:
            max_filtration = 1.0

        t_values = torch.linspace(0, max_filtration, self.n_features)

        features = []
        for t in t_values:
            count = torch.sum((births <= t) & (deaths > t)).float()
            features.append(count)

        return torch.tensor(features, dtype=torch.float32)

    def _silhouette_vector(self, diagram: Tensor) -> Tensor:
        """Compute silhouette features."""
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistences = (deaths - births).clamp(min=0)

        total_persistence = persistences.sum().item()
        if total_persistence == 0:
            return torch.zeros(self.n_features)

        weights = persistences / total_persistence

        max_pers = persistences.max().item()
        if max_pers == 0:
            max_pers = 1.0

        t_values = torch.linspace(0, max_pers, self.n_features)

        features = []
        for t in t_values:
            weighted_count = torch.sum(((births <= t) & (deaths > t)).float() * weights)
            features.append(weighted_count)

        return torch.tensor(features, dtype=torch.float32)


class CombinedSignature(TopologicalSignature):
    """
    Combined Topological Signature.

    Combines multiple topological signatures into
    a single feature vector.
    """

    def __init__(
        self,
        include_images: bool = True,
        include_vectorization: bool = True,
        n_vectorization_features: int = 50,
        image_resolution: Tuple[int, int] = (20, 20),
    ):
        """
        Initialize combined signature.

        Args:
            include_images: Include persistence images
            include_vectorization: Include vectorization
            n_vectorization_features: Number of vectorization features
            image_resolution: Image resolution
        """
        super().__init__(name="combined", dimension=1)
        self.include_images = include_images
        self.include_vectorization = include_vectorization
        self.n_features = n_vectorization_features
        self.image_res = image_resolution

        self.image_extractor = PersistenceImage(resolution=image_resolution)
        self.vector_extractor = TopologicalVectorization(
            n_features=n_vectorization_features,
            feature_type="landscape",
        )

    def compute(
        self,
        diagram: Tensor,
    ) -> Tensor:
        """
        Compute combined signature.

        Args:
            diagram: Persistence diagram

        Returns:
            Combined feature vector
        """
        features = []

        if self.include_vectorization:
            vector = self.vector_extractor.compute(diagram)
            features.append(vector)

        if self.include_images:
            image = self.image_extractor.compute(diagram)
            image_flat = image.flatten()
            features.append(image_flat)

        return torch.cat(features)


class MultiScaleSignature(TopologicalSignature):
    """
    Multi-Scale Topological Signature.

    Computes topological features at multiple filtration scales
    to capture both local and global structure.
    """

    def __init__(
        self,
        n_scales: int = 5,
        scale_range: Tuple[float, float] = (0.1, 2.0),
        n_features_per_scale: int = 20,
    ):
        """
        Initialize multi-scale signature.

        Args:
            n_scales: Number of scales to sample
            scale_range: Range of filtration values
            n_features_per_scale: Features per scale
        """
        super().__init__(name="multiscale", dimension=1)
        self.n_scales = n_scales
        self.scale_range = scale_range
        self.n_features_per_scale = n_features_per_scale

    def compute(
        self,
        diagram: Tensor,
    ) -> Tensor:
        """
        Compute multi-scale signature.

        Args:
            diagram: Persistence diagram

        Returns:
            Multi-scale feature vector
        """
        scales = np.linspace(self.scale_range[0], self.scale_range[1], self.n_scales)

        features = []

        for scale in scales:
            filtered_diag = self._filter_by_scale(diagram, scale)

            vectorizer = TopologicalVectorization(
                n_features=self.n_features_per_scale,
                feature_type="landscape",
            )
            scale_features = vectorizer.compute(filtered_diag)

            features.append(scale_features)

        return torch.cat(features)

    def _filter_by_scale(
        self,
        diagram: Tensor,
        scale: float,
    ) -> Tensor:
        """Filter diagram by persistence scale."""
        if len(diagram) == 0:
            return diagram

        persistences = (diagram[:, 1] - diagram[:, 0]).clamp(min=0)

        threshold = scale * persistences.mean().item()

        mask = persistences >= threshold

        return diagram[mask]


class AdaptiveSignature(TopologicalSignature):
    """
    Adaptive Topological Signature.

    Computes signatures that adapt to the structure
    of the persistence diagram.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        n_features_per_cluster: int = 10,
    ):
        """
        Initialize adaptive signature.

        Args:
            n_clusters: Number of clusters for adaptive sampling
            n_features_per_cluster: Features per cluster
        """
        super().__init__(name="adaptive", dimension=1)
        self.n_clusters = n_clusters
        self.n_features_per_cluster = n_features_per_cluster

    def compute(
        self,
        diagram: Tensor,
    ) -> Tensor:
        """
        Compute adaptive signature.

        Args:
            diagram: Persistence diagram

        Returns:
            Adaptive feature vector
        """
        if len(diagram) < self.n_clusters:
            n_features = len(diagram) * self.n_features_per_cluster
            vectorizer = TopologicalVectorization(
                n_features=max(n_features, 1),
                feature_type="landscape",
            )
            return vectorizer.compute(diagram)

        persistences = (diagram[:, 1] - diagram[:, 0]).clamp(min=0).numpy()

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=min(self.n_clusters, len(diagram)))
            labels = kmeans.fit_predict(persistences.reshape(-1, 1))
        except:
            labels = np.zeros(len(persistences), dtype=int)

        features = []

        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_diag = diagram[cluster_mask]

            if len(cluster_diag) > 0:
                vectorizer = TopologicalVectorization(
                    n_features=self.n_features_per_cluster,
                    feature_type="landscape",
                )
                cluster_features = vectorizer.compute(cluster_diag)
                features.append(cluster_features)

        if len(features) == 0:
            return torch.zeros(self.n_features_per_cluster * self.n_clusters)

        return torch.cat(features)


def compute_all_signatures(
    diagram: Tensor,
    n_features: int = 100,
) -> dict:
    """
    Compute all available signatures.

    Args:
        diagram: Persistence diagram
        n_features: Number of features per signature

    Returns:
        Dictionary of signature name to feature vector
    """
    signatures = {}

    vectorizer = TopologicalVectorization(
        n_features=n_features,
        feature_type="landscape",
    )
    signatures["landscape"] = vectorizer.compute(diagram)

    vectorizer_barcode = TopologicalVectorization(
        n_features=n_features,
        feature_type="barcode",
    )
    signatures["barcode"] = vectorizer_barcode.compute(diagram)

    vectorizer_silhouette = TopologicalVectorization(
        n_features=n_features,
        feature_type="silhouette",
    )
    signatures["silhouette"] = vectorizer_silhouette.compute(diagram)

    img_extractor = PersistenceImage(resolution=(10, 10))
    image = img_extractor.compute(diagram)
    signatures["image"] = image.flatten()

    return signatures
