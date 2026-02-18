"""
Tabular Data Augmentation Module

Augmentation techniques for tabular data including:
- SMOTE (Synthetic Minority Over-sampling)
- Random noise injection
- Feature shuffling
- Row mixing
- ADASYN
"""

from typing import Optional, Tuple, List, Union, Dict, Any
import torch
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from fishstick.augmentation_ext.base import AugmentationBase


class SMOTE(AugmentationBase):
    """
    Synthetic Minority Over-sampling Technique (SMOTE).

    Reference: Chawla et al., "SMOTE", 2002

    Args:
        sampling_strategy: Strategy for minority class sampling
        k_neighbors: Number of neighbors for interpolation
        random_state: Random seed
    """

    def __init__(
        self,
        sampling_strategy: str = "minority",
        k_neighbors: int = 5,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.rng = np.random.RandomState(seed)

    def fit_resample(
        self, X: NDArray[np.floating], y: NDArray[np.integer]
    ) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """
        Apply SMOTE to balance the dataset.

        Args:
            X: Feature array (N, D)
            y: Label array (N,)

        Returns:
            Resampled features and labels
        """
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        minority_idx = np.where(y == minority_class)[0]
        minority_samples = X[minority_idx]

        n_minority = len(minority_samples)
        n_majority = np.max(counts)

        if n_majority <= n_minority:
            return X, y

        n_synthetic = n_majority - n_minority

        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, n_minority))
        nn.fit(minority_samples)

        synthetic_samples = []
        for _ in range(n_synthetic):
            idx = self.rng.randint(0, n_minority)
            sample = minority_samples[idx]

            distances, neighbors = nn.kneighbors(sample.reshape(1, -1))
            neighbor_idx = self.rng.choice(neighbors[0][1:], size=1)[0]
            neighbor = minority_samples[neighbor_idx]

            alpha = self.rng.random()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic)

        synthetic_samples = np.array(synthetic_samples)
        synthetic_labels = np.full(len(synthetic_samples), minority_class)

        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.concatenate([y, synthetic_labels])

        return X_resampled, y_resampled

    def __call__(
        self, data: Tuple[NDArray[np.floating], NDArray[np.integer]]
    ) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Apply SMOTE to tabular data."""
        X, y = data
        return self.fit_resample(X, y)


class RandomNoiseInjection(AugmentationBase):
    """Add random noise to tabular features."""

    def __init__(
        self,
        noise_level: float = 0.1,
        noise_type: str = "gaussian",
        feature_range: Optional[Tuple[float, float]] = None,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.feature_range = feature_range
        self.rng = np.random.RandomState(seed)

    def __call__(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Add noise to tabular features.

        Args:
            X: Feature array (N, D)

        Returns:
            Noisy features
        """
        if not self._should_apply():
            return X

        X_aug = X.copy()

        std = X.std(axis=0)
        std[std == 0] = 1.0

        if self.noise_type == "gaussian":
            noise = self.rng.normal(0, self.noise_level * std, X.shape)
        elif self.noise_type == "uniform":
            noise = self.rng.uniform(-self.noise_level, self.noise_level, X.shape) * std
        elif self.noise_type == "laplace":
            noise = self.rng.laplace(0, self.noise_level * std, X.shape)
        else:
            noise = self.rng.normal(0, self.noise_level * std, X.shape)

        X_aug = X_aug + noise

        if self.feature_range is not None:
            X_aug = np.clip(X_aug, self.feature_range[0], self.feature_range[1])

        return X_aug


class FeatureShuffle(AugmentationBase):
    """Shuffle features within similar samples."""

    def __init__(
        self,
        n_features: Optional[int] = None,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.n_features = n_features
        self.rng = np.random.RandomState(seed)

    def __call__(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Shuffle selected features.

        Args:
            X: Feature array (N, D)

        Returns:
            Features with some columns shuffled
        """
        if not self._should_apply():
            return X

        X_aug = X.copy()
        n_features = self.n_features or max(1, X.shape[1] // 4)
        n_features = min(n_features, X.shape[1])

        feature_indices = self.rng.choice(X.shape[1], size=n_features, replace=False)

        for idx in feature_indices:
            perm = self.rng.permutation(X.shape[0])
            X_aug[:, idx] = X_aug[perm, idx]

        return X_aug


class RowMixing(AugmentationBase):
    """Mix rows in tabular data."""

    def __init__(
        self,
        mix_ratio: float = 0.5,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.mix_ratio = mix_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Mix rows in tabular data.

        Args:
            X: Feature array (N, D)

        Returns:
            Mixed features
        """
        if not self._should_apply() or X.shape[0] < 2:
            return X

        X_aug = X.copy()
        batch_size = X.shape[0]
        indices = self.rng.permutation(batch_size)

        alpha = self.rng.beta(self.mix_ratio, self.mix_ratio, size=batch_size)
        alpha = alpha.reshape(-1, 1)

        X_aug = alpha * X_aug + (1 - alpha) * X_aug[indices]

        return X_aug


class SMOTETomek(AugmentationBase):
    """
    SMOTE combined with Tomek links for cleaning.

    Reference: Batista et al., "SMOTE + Tomek", 2004
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.k_neighbors = k_neighbors
        self.rng = np.random.RandomState(seed)
        self.smote = SMOTE(k_neighbors=k_neighbors, seed=seed)

    def __call__(
        self, data: Tuple[NDArray[np.floating], NDArray[np.integer]]
    ) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Apply SMOTE-Tomek cleaning."""
        X, y = data

        X_res, y_res = self.smote.fit_resample(X, y)

        X_clean, y_clean = self._remove_tomek_links(X_res, y_res)

        return X_clean, y_clean

    def _remove_tomek_links(
        self, X: NDArray[np.floating], y: NDArray[np.integer]
    ) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Remove Tomek links from dataset."""
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)

        to_remove = []
        for i in range(len(X)):
            _, neighbors = nn.kneighbors(X[i].reshape(1, -1))
            neighbor = neighbors[0][1]

            if y[i] != y[neighbor]:
                to_remove.append(i)

        mask = np.ones(len(X), dtype=bool)
        mask[to_remove] = False

        return X[mask], y[mask]


class ADASYN(AugmentationBase):
    """
    Adaptive Synthetic Sampling.

    Reference: He et al., "ADASYN", 2008
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        beta: float = 1.0,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.k_neighbors = k_neighbors
        self.beta = beta
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, data: Tuple[NDArray[np.floating], NDArray[np.integer]]
    ) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Apply ADASYN to dataset."""
        X, y = data

        classes, counts = np.unique(y, return_counts=True)
        majority_count = np.max(counts)
        minority_class = classes[np.argmin(counts)]
        minority_idx = np.where(y == minority_class)[0]
        minority_samples = X[minority_idx]

        d = counts / majority_count
        d_hat = d.sum()
        G = (majority_count - counts[minority_idx[0]]) * d_hat

        if len(classes) > 1:
            G = int((majority_count - counts.min()) * self.beta)
        else:
            return X, y

        if G <= 0:
            return X, y

        nn = NearestNeighbors(
            n_neighbors=min(self.k_neighbors + 1, len(minority_samples))
        )
        nn.fit(minority_samples)

        ratios = []
        for sample in minority_samples:
            _, neighbors = nn.kneighbors(sample.reshape(1, -1))
            k_neighbors_labels = y[neighbors[0][1:]]
            r = np.sum(k_neighbors_labels != minority_class) / self.k_neighbors
            ratios.append(r)

        ratios = np.array(ratios)
        ratios[ratios == 0] = 0.0001
        ratios = ratios / ratios.sum()

        n_synthetic_per_sample = (ratios * G).astype(int)
        n_synthetic_per_sample[n_synthetic_per_sample == 0] = 1

        synthetic_samples = []
        for i, (sample, n) in enumerate(zip(minority_samples, n_synthetic_per_sample)):
            for _ in range(n):
                neighbor_idx = self.rng.choice(minority_samples.shape[0])
                neighbor = minority_samples[neighbor_idx]
                alpha = self.rng.random()
                synthetic = sample + alpha * (neighbor - sample)
                synthetic_samples.append(synthetic)

        if synthetic_samples:
            synthetic_samples = np.array(synthetic_samples)
            synthetic_labels = np.full(len(synthetic_samples), minority_class)
            X_resampled = np.vstack([X, synthetic_samples])
            y_resampled = np.concatenate([y, synthetic_labels])
            return X_resampled, y_resampled

        return X, y


class FeatureNoiseMask(AugmentationBase):
    """Randomly mask features with noise."""

    def __init__(
        self,
        mask_prob: float = 0.15,
        noise_scale: float = 0.1,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.mask_prob = mask_prob
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)

    def __call__(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply feature masking with noise.

        Args:
            X: Feature array (N, D)

        Returns:
            Augmented features
        """
        if not self._should_apply():
            return X

        X_aug = X.copy()
        mask = self.rng.random(X.shape) < self.mask_prob

        noise = self.rng.normal(0, self.noise_scale, X.shape)
        X_aug[mask] = noise[mask]

        return X_aug


class CutoffAugmentation(AugmentationBase):
    """Randomly zero out continuous features."""

    def __init__(
        self,
        cutoff_prob: float = 0.25,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.cutoff_prob = cutoff_prob
        self.rng = np.random.RandomState(seed)

    def __call__(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply cutoff augmentation.

        Args:
            X: Feature array (N, D)

        Returns:
            Augmented features
        """
        if not self._should_apply():
            return X

        X_aug = X.copy()
        mask = self.rng.random(X.shape) < self.cutoff_prob
        X_aug[mask] = 0

        return X_aug


class SwapNoise(AugmentationBase):
    """Swap noise - randomly swap feature values between samples."""

    def __init__(
        self,
        swap_prob: float = 0.15,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.swap_prob = swap_prob
        self.rng = np.random.RandomState(seed)

    def __call__(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply swap noise augmentation.

        Args:
            X: Feature array (N, D)

        Returns:
            Augmented features
        """
        if not self._should_apply():
            return X

        X_aug = X.copy()
        n_samples, n_features = X.shape

        swap_mask = self.rng.random(X.shape) < self.swap_prob

        for j in range(n_features):
            swap_indices = np.where(swap_mask[:, j])[0]
            if len(swap_indices) > 0:
                swap_targets = self.rng.choice(
                    n_samples, size=len(swap_indices), replace=True
                )
                X_aug[swap_indices, j] = X[swap_targets, j]

        return X_aug


class cGANAugmentation(AugmentationBase):
    """Conditional GAN-based augmentation (placeholder for integration)."""

    def __init__(
        self,
        generator: Any = None,
        latent_dim: int = 100,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.generator = generator
        self.latent_dim = latent_dim
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        data: Tuple[NDArray[np.floating], NDArray[np.integer]],
        n_samples: int = 100,
    ) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Generate synthetic samples using cGAN."""
        X, y = data

        if self.generator is None:
            return X, y

        classes = np.unique(y)
        X_aug = []
        y_aug = []

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            cls_samples = X[cls_idx]

            for _ in range(n_samples):
                z = self.rng.normal(0, 1, (1, self.latent_dim))
                label = np.full((1, 1), cls)
                synthetic = self.generator.predict([z, label])
                X_aug.append(synthetic[0])
                y_aug.append(cls)

        if X_aug:
            X_aug = np.array(X_aug)
            y_aug = np.array(y_aug)
            return np.vstack([X, X_aug]), np.concatenate([y, y_aug])

        return X, y


def get_tabular_augmentation_pipeline(
    task: str = "classification",
    imbalance_ratio: float = 0.1,
    intensity: float = 1.0,
) -> List[Any]:
    """
    Get a pre-configured tabular augmentation pipeline.

    Args:
        task: Task type (classification, regression)
        imbalance_ratio: Ratio of minority to majority class
        intensity: Overall augmentation intensity

    Returns:
        List of augmentation operations
    """
    return [
        RandomNoiseInjection(noise_level=0.1 * intensity),
        FeatureShuffle(n_features=None),
        SwapNoise(swap_prob=0.15 * intensity),
        CutoffAugmentation(cutoff_prob=0.1 * intensity),
    ]
