"""
Augmentation Pipeline and Utilities

Advanced pipeline constructs for data augmentation including:
- Augmentation scheduling and caching
- Conditional augmentation
- Multi-modal augmentation
"""

from typing import Optional, Dict, Any, List, Union, Callable, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
import random
import hashlib
import pickle
from pathlib import Path

from fishstick.augmentation_ext.base import AugmentationBase


class AugmentationType(Enum):
    """Types of augmentation."""

    IMAGE = "image"
    VIDEO = "video"
    TABULAR = "tabular"
    GRAPH = "graph"
    AUDIO = "audio"


@dataclass
class AugmentationInfo:
    """Information about an augmentation operation."""

    name: str
    augmentation_type: AugmentationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    intensity: float = 1.0


class ConditionalAugmentation:
    """Apply augmentations conditionally based on metadata."""

    def __init__(
        self,
        augmentation: AugmentationBase,
        condition_fn: Callable[[Dict[str, Any]], bool],
    ):
        """
        Initialize conditional augmentation.

        Args:
            augmentation: The augmentation to apply
            condition_fn: Function that returns True if augmentation should be applied
        """
        self.augmentation = augmentation
        self.condition_fn = condition_fn

    def __call__(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Apply augmentation conditionally."""
        if metadata is None:
            metadata = {}

        if self.condition_fn(metadata):
            return self.augmentation(data)
        return data


class AugmentationCache:
    """Cache augmented data to avoid redundant computation."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cached data
            max_size: Maximum number of cached items
            seed: Random seed
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self.rng = np.random.RandomState(seed)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, data: Any, augmentation_params: Dict[str, Any]) -> str:
        """Generate cache key from data and parameters."""
        try:
            data_hash = hashlib.md5(pickle.dumps(data)).hexdigest()
        except:
            data_hash = str(hash(str(data)))

        params_hash = hashlib.md5(str(augmentation_params).encode()).hexdigest()
        return f"{data_hash}_{params_hash}"

    def get(self, data: Any, augmentation_params: Dict[str, Any]) -> Optional[Any]:
        """Get cached augmented data."""
        key = self._get_key(data, augmentation_params)

        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]

        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                self.cache[key] = cached_data
                self.access_count[key] = 1
                return cached_data

        return None

    def put(
        self, data: Any, augmentation_params: Dict[str, Any], augmented_data: Any
    ) -> None:
        """Store augmented data in cache."""
        key = self._get_key(data, augmentation_params)

        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]

        self.cache[key] = augmented_data
        self.access_count[key] = 1

        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(augmented_data, f)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()

        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


class AugmentationSequence:
    """Sequence of augmentations to apply in order."""

    def __init__(
        self,
        augmentations: List[AugmentationBase],
        probabilities: Optional[List[float]] = None,
        cache: Optional[AugmentationCache] = None,
    ):
        """
        Initialize the sequence.

        Args:
            augmentations: List of augmentations
            probabilities: Probability for each augmentation
            cache: Optional cache for augmented data
        """
        self.augmentations = augmentations
        self.probabilities = probabilities or [1.0] * len(augmentations)
        self.cache = cache

    def __call__(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Apply sequence of augmentations."""
        result = data

        for aug, prob in zip(self.augmentations, self.probabilities):
            if random.random() < prob:
                params = {"probability": prob}

                if self.cache:
                    cached = self.cache.get(result, params)
                    if cached is not None:
                        result = cached
                        continue

                result = aug(result)

                if self.cache:
                    self.cache.put(data, params, result)

        return result

    def add(self, augmentation: AugmentationBase, probability: float = 1.0) -> None:
        """Add an augmentation to the sequence."""
        self.augmentations.append(augmentation)
        self.probabilities.append(probability)


class AugmentationEnsemble:
    """Ensemble of augmentations - randomly select one to apply."""

    def __init__(
        self,
        augmentations: List[AugmentationBase],
        n_select: int = 1,
    ):
        """
        Initialize the ensemble.

        Args:
            augmentations: List of augmentations to choose from
            n_select: Number of augmentations to apply
        """
        self.augmentations = augmentations
        self.n_select = min(n_select, len(augmentations))

    def __call__(self, data: Any) -> Any:
        """Apply random selection of augmentations."""
        selected = random.sample(self.augmentations, self.n_select)
        result = data
        for aug in selected:
            result = aug(result)
        return result


class AugmentationMixer:
    """Mix multiple augmentations with learned weights."""

    def __init__(
        self,
        augmentations: List[AugmentationBase],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize the mixer.

        Args:
            augmentations: List of augmentations
            weights: Weight for each augmentation
        """
        self.augmentations = augmentations
        self.weights = weights or [1.0 / len(augmentations)] * len(augmentations)
        self.weights = [w / sum(self.weights) for w in self.weights]

    def __call__(self, data: Any) -> Any:
        """Apply mixed augmentations."""
        result = data
        for aug, weight in zip(self.augmentations, self.weights):
            if random.random() < weight:
                result = aug(result)
        return result

    def update_weights(self, weights: List[float]) -> None:
        """Update augmentation weights."""
        self.weights = weights
        self.weights = [w / sum(self.weights) for w in self.weights]


class MultiModalAugmentation:
    """Augmentation pipeline for multi-modal data."""

    def __init__(
        self,
        augmentation_map: Dict[AugmentationType, AugmentationBase],
    ):
        """
        Initialize multi-modal augmentation.

        Args:
            augmentation_map: Map of augmentation types to operations
        """
        self.augmentation_map = augmentation_map

    def __call__(
        self, data: Dict[AugmentationType, Any]
    ) -> Dict[AugmentationType, Any]:
        """Apply appropriate augmentation to each modality."""
        result = {}
        for modality, modality_data in data.items():
            if modality in self.augmentation_map:
                result[modality] = self.augmentation_map[modality](modality_data)
            else:
                result[modality] = modality_data
        return result


class ProgressiveAugmentation:
    """Progressively increase augmentation complexity during training."""

    def __init__(
        self,
        stages: List[List[AugmentationBase]],
        transition_steps: int = 1000,
    ):
        """
        Initialize progressive augmentation.

        Args:
            stages: List of augmentation stages
            transition_steps: Steps between stages
        """
        self.stages = stages
        self.transition_steps = transition_steps
        self.current_stage = 0
        self.step_count = 0

    def step(self) -> None:
        """Step to the next stage."""
        self.step_count += 1
        if self.step_count >= self.transition_steps:
            self.step_count = 0
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1

    def get_current_augmentations(self) -> List[AugmentationBase]:
        """Get current stage augmentations."""
        return self.stages[self.current_stage]

    def __call__(self, data: Any) -> Any:
        """Apply current stage augmentations."""
        augmentations = self.get_current_augmentations()
        result = data
        for aug in augmentations:
            result = aug(result)
        return result


class AugmentationFactory:
    """Factory for creating augmentation pipelines."""

    _registry: Dict[str, Type[AugmentationBase]] = {}

    @classmethod
    def register(cls, name: str, augmentation_cls: Type[AugmentationBase]) -> None:
        """Register an augmentation class."""
        cls._registry[name] = augmentation_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AugmentationBase:
        """Create an augmentation from registry."""
        if name not in cls._registry:
            raise ValueError(f"Unknown augmentation: {name}")
        return cls._registry[name](**kwargs)

    @classmethod
    def create_pipeline(cls, config: List[Dict[str, Any]]) -> AugmentationSequence:
        """Create a pipeline from configuration."""
        augmentations = []
        probabilities = []

        for item in config:
            name = item.get("name")
            params = item.get("parameters", {})
            prob = item.get("probability", 1.0)

            aug = cls.create(name, **params)
            augmentations.append(aug)
            probabilities.append(prob)

        return AugmentationSequence(augmentations, probabilities)


def create_standard_pipeline(
    modality: AugmentationType,
    intensity: float = 1.0,
) -> AugmentationSequence:
    """
    Create a standard augmentation pipeline for a modality.

    Args:
        modality: The data modality
        intensity: Overall augmentation intensity

    Returns:
        Configured AugmentationSequence
    """
    if modality == AugmentationType.IMAGE:
        from fishstick.augmentation_ext.image_augmentation import (
            RandomHorizontalFlip,
            RandomRotation,
            ColorJitter,
            RandomErasing,
            MixUp,
        )

        augmentations = [
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15 * intensity),
            ColorJitter(brightness=0.2 * intensity, contrast=0.2 * intensity),
            RandomErasing(p=0.3 * intensity),
        ]

    elif modality == AugmentationType.VIDEO:
        from fishstick.augmentation_ext.video_augmentation import (
            RandomHorizontalFlip,
            RandomCropResize,
            TemporalDropout,
        )

        augmentations = [
            RandomHorizontalFlip(p=0.5),
            RandomCropResize(crop_size=(224, 224)),
            TemporalDropout(drop_prob=0.1 * intensity),
        ]

    elif modality == AugmentationType.AUDIO:
        from fishstick.augmentation_ext.audio_augmentation import (
            AddBackgroundNoise,
            TimeShift,
            VolumePerturbation,
        )

        augmentations = [
            AddBackgroundNoise(noise_level=0.1 * intensity),
            TimeShift(shift_limit=0.2 * intensity),
            VolumePerturbation(db_range=6 * intensity),
        ]

    elif modality == AugmentationType.GRAPH:
        from fishstick.augmentation_ext.graph_augmentation import (
            NodeDrop,
            EdgeDrop,
            AttributeMasking,
        )

        augmentations = [
            NodeDrop(drop_ratio=0.1 * intensity),
            EdgeDrop(drop_ratio=0.1 * intensity),
            AttributeMasking(mask_ratio=0.15 * intensity),
        ]

    elif modality == AugmentationType.TABULAR:
        from fishstick.augmentation_ext.tabular_augmentation import (
            RandomNoiseInjection,
            FeatureShuffle,
        )

        augmentations = [
            RandomNoiseInjection(noise_level=0.1 * intensity),
            FeatureShuffle(p=0.3 * intensity),
        ]

    else:
        augmentations = []

    return AugmentationSequence(augmentations)
