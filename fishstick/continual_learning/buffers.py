"""
Advanced Memory Buffers for Continual Learning.

Specialized buffers for storing and sampling experiences.

Classes:
- RingBuffer: Simple ring buffer
- HerdingBuffer: Herding-based buffer for representative sampling
- MeanOfFeatures: Feature mean buffer
- FeatureBuffer: Feature storage buffer
"""

from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np
from collections import deque


class RingBuffer:
    """
    Simple Ring Buffer for Experience Replay.

    Fixed-size buffer that overwrites oldest samples.

    Args:
        capacity: Maximum buffer size
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: List[Tuple[Tensor, Tensor]] = []
        self.position = 0

    def add(self, x: Tensor, y: Tensor) -> None:
        """Add sample to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y))
        else:
            self.buffer[self.position] = (x, y)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from buffer."""
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        x = torch.stack([self.buffer[i][0] for i in indices])
        y = torch.stack([self.buffer[i][1] for i in indices])

        return x, y

    def __len__(self) -> int:
        return len(self.buffer)


class HerdingBuffer:
    """
    Herding-based Buffer for Representative Sampling.

    Selects samples that approximate the mean of the data distribution.

    Reference:
        Rebuffi et al., "iCaRL: Incremental Classifier and Representation Learning", CVPR 2017

    Args:
        buffer_size: Maximum buffer size
        feature_extractor: Feature extraction network
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        feature_extractor: Optional[nn.Module] = None,
    ):
        self.buffer_size = buffer_size
        self.feature_extractor = feature_extractor

        self.features: List[Tensor] = []
        self.labels: List[Tensor] = []

    def update(
        self,
        x: Tensor,
        y: Tensor,
        model: Optional[nn.Module] = None,
    ) -> None:
        """
        Update buffer with herding selection.

        Args:
            x: Input samples
            y: Labels
            model: Model for feature extraction
        """
        if model is not None:
            self.feature_extractor = model

        with torch.no_grad():
            if self.feature_extractor is not None:
                features = self.feature_extractor(x)
            else:
                features = x

        self.features.extend(features.cpu().split(1))
        self.labels.extend(y.cpu().split(1))

        if len(self.features) > self.buffer_size:
            self._select_herding()

    def _select_herding(self) -> None:
        """Select samples using herding algorithm."""
        if len(self.features) == 0:
            return

        all_features = torch.cat(self.features, dim=0)
        all_labels = torch.cat(self.labels, dim=0)

        mean_features = all_features.mean(dim=0, keepdim=True)

        selected_indices = []

        for _ in range(self.buffer_size):
            if len(selected_indices) >= len(all_features):
                break

            remaining = [
                i for i in range(len(all_features)) if i not in selected_indices
            ]

            if len(remaining) == 0:
                break

            remaining_features = all_features[remaining]

            distances = (remaining_features - mean_features).pow(2).sum(dim=1)

            closest_idx = distances.argmin().item()

            selected_indices.append(remaining[closest_idx])

            running_mean = all_features[selected_indices].mean(dim=0, keepdim=True)

        self.features = [all_features[i] for i in selected_indices]
        self.labels = [all_labels[i] for i in selected_indices]

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from buffer."""
        if len(self.features) == 0:
            raise ValueError("Buffer is empty")

        indices = np.random.choice(len(self.features), batch_size, replace=False)

        x = torch.cat([self.features[i] for i in indices])
        y = torch.cat([self.labels[i] for i in indices])

        return x, y

    def __len__(self) -> int:
        return len(self.features)


class MeanOfFeatures:
    """
    Mean of Features Buffer.

    Maintains running mean of features for each class.

    Args:
        num_classes: Number of classes
        feature_dim: Feature dimension
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 512,
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.class_means: Tensor = torch.zeros(num_classes, feature_dim)
        self.class_counts: Tensor = torch.zeros(num_classes)

    def update(
        self,
        features: Tensor,
        labels: Tensor,
    ) -> None:
        """Update class means."""
        for i in range(self.num_classes):
            mask = labels == i

            if mask.sum() > 0:
                class_features = features[mask]

                current_mean = self.class_means[i]
                current_count = self.class_counts[i]

                new_count = current_count + mask.sum()

                self.class_means[i] = (
                    current_mean * current_count + class_features.sum(dim=0)
                ) / new_count
                self.class_counts[i] = new_count

    def get_means(self) -> Tensor:
        """Get class means."""
        return self.class_means

    def get_prototype(self, class_id: int) -> Tensor:
        """Get prototype for a class."""
        return self.class_means[class_id]


class FeatureBuffer:
    """
    Feature Storage Buffer.

    Stores features and labels for replay.

    Args:
        buffer_size: Maximum buffer size
        feature_dim: Feature dimension
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        feature_dim: int = 512,
    ):
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim

        self.features: Tensor = torch.zeros(buffer_size, feature_dim)
        self.labels: Tensor = torch.zeros(buffer_size, dtype=torch.long)

        self.current_size = 0
        self.position = 0

    def add(
        self,
        features: Tensor,
        labels: Tensor,
    ) -> None:
        """Add features to buffer."""
        batch_size = features.size(0)

        for i in range(batch_size):
            self.features[self.position] = features[i]
            self.labels[self.position] = labels[i]

            self.position = (self.position + 1) % self.buffer_size

        self.current_size = min(self.current_size + batch_size, self.buffer_size)

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from buffer."""
        if self.current_size == 0:
            raise ValueError("Buffer is empty")

        indices = np.random.choice(self.current_size, batch_size, replace=False)

        return self.features[indices], self.labels[indices]

    def __len__(self) -> int:
        return self.current_size


class ClassBalancedBuffer:
    """
    Class-Balanced Replay Buffer.

    Maintains equal representation from each class.

    Args:
        buffer_size: Total buffer size
        num_classes: Number of classes
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        num_classes: int = 10,
    ):
        self.buffer_size = buffer_size
        self.num_classes = num_classes

        self.per_class = buffer_size // num_classes

        self.class_buffers: Dict[int, RingBuffer] = {
            i: RingBuffer(self.per_class) for i in range(num_classes)
        }

    def add(
        self,
        x: Tensor,
        y: Tensor,
    ) -> None:
        """Add samples maintaining class balance."""
        for i in range(self.num_classes):
            mask = y == i

            if mask.sum() > 0:
                class_x = x[mask]
                class_y = y[mask]

                for j in range(len(class_x)):
                    self.class_buffers[i].add(class_x[j], class_y[j])

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample balanced batch."""
        samples_per_class = batch_size // self.num_classes

        all_x = []
        all_y = []

        for i in range(self.num_classes):
            if len(self.class_buffers[i]) > 0:
                class_x, class_y = self.class_buffers[i].sample(samples_per_class)
                all_x.append(class_x)
                all_y.append(class_y)

        return torch.cat(all_x), torch.cat(all_y)

    def __len__(self) -> int:
        return sum(len(buf) for buf in self.class_buffers.values())
