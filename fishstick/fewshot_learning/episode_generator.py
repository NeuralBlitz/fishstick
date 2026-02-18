"""
Episode Generation utilities for few-shot learning.

Provides utilities for generating few-shot tasks/episodes from datasets,
including various sampling strategies.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from .types import FewShotTask, MetaBatch, EpisodeConfig


class EpisodeGenerator:
    """Base class for episode generation.

    Args:
        dataset: Source dataset with labeled examples
        config: Episode configuration
    """

    def __init__(
        self,
        dataset: Dataset,
        config: Optional[EpisodeConfig] = None,
    ):
        self.dataset = dataset
        self.config = config or EpisodeConfig()

        self._class_to_indices = self._build_class_index()
        self._classes = list(self._class_to_indices.keys())

    def _build_class_index(self) -> Dict[int, List[int]]:
        """Build mapping from class to example indices."""
        class_to_indices = {}

        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]

            if isinstance(label, torch.Tensor):
                label = label.item()

            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        return class_to_indices

    def sample_classes(self, num_classes: Optional[int] = None) -> List[int]:
        """Sample random classes for an episode.

        Args:
            num_classes: Number of classes to sample (defaults to config.n_way)

        Returns:
            List of class indices
        """
        num_classes = num_classes or self.config.n_way
        return random.sample(self._classes, min(num_classes, len(self._classes)))

    def sample_episode(self) -> FewShotTask:
        """Sample a single few-shot episode.

        Returns:
            FewShotTask with support and query sets
        """
        classes = self.sample_classes()

        support_x, support_y = [], []
        query_x, query_y = [], []

        for task_class_idx, cls in enumerate(classes):
            cls_indices = self._class_to_indices[cls]
            sampled = random.sample(
                cls_indices, self.config.n_shot + self.config.n_query
            )

            support_indices = sampled[: self.config.n_shot]
            query_indices = sampled[self.config.n_shot :]

            for idx in support_indices:
                x, _ = self.dataset[idx]
                support_x.append(x)
                support_y.append(task_class_idx)

            for idx in query_indices:
                x, _ = self.dataset[idx]
                query_x.append(x)
                query_y.append(task_class_idx)

        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y, dtype=torch.long)

        return FewShotTask(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            n_way=self.config.n_way,
            n_shot=self.config.n_shot,
            n_query=self.config.n_query,
            classes=classes,
        )

    def sample_meta_batch(self) -> MetaBatch:
        """Sample a meta-batch of episodes.

        Returns:
            MetaBatch containing multiple FewShotTasks
        """
        tasks = [self.sample_episode() for _ in range(self.config.meta_batch_size)]

        return MetaBatch(
            tasks=tasks,
            meta_batch_size=self.config.meta_batch_size,
        )

    def __iter__(self):
        """Iterate over episodes."""
        return self

    def __next__(self) -> FewShotTask:
        """Generate next episode."""
        return self.sample_episode()


class TaskSampler:
    """Custom sampler for generating few-shot tasks.

    Args:
        dataset: Source dataset
        n_way: Number of classes per episode
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        num_episodes: Number of episodes per epoch
        classes_per_episode: Number of classes to sample from
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
        num_episodes: int = 100,
        classes_per_episode: Optional[int] = None,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.num_episodes = num_episodes
        self.classes_per_episode = classes_per_episode or n_way

        self._class_to_indices = self._build_class_index()
        self._classes = list(self._class_to_indices.keys())

    def _build_class_index(self) -> Dict[int, List[int]]:
        class_to_indices = {}

        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]

            if isinstance(label, torch.Tensor):
                label = label.item()

            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        return class_to_indices

    def __iter__(self):
        """Generate episodes."""
        for _ in range(self.num_episodes):
            classes = random.sample(
                self._classes, min(self.classes_per_episode, len(self._classes))
            )

            episode_indices = []

            for class_idx, cls in enumerate(classes):
                cls_indices = self._class_to_indices[cls]

                if len(cls_indices) < self.n_shot + self.n_query:
                    sampled = cls_indices * (
                        (self.n_shot + self.n_query) // len(cls_indices) + 1
                    )
                    sampled = sampled[: self.n_shot + self.n_query]
                else:
                    sampled = random.sample(cls_indices, self.n_shot + self.n_query)

                episode_indices.extend(sampled)

            yield episode_indices

    def __len__(self):
        return self.num_episodes


class NWayKShotSampler:
    """Sampler for N-way K-shot episode generation.

    Provides more control over episode sampling with class remapping.

    Args:
        dataset: Source dataset
        n_way: Number of classes per episode
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        self._class_to_indices = self._build_class_index()
        self._classes = sorted(self._class_to_indices.keys())

    def _build_class_index(self) -> Dict[int, List[int]]:
        class_to_indices = {}

        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]

            if isinstance(label, torch.Tensor):
                label = label.item()

            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        return class_to_indices

    def sample_task(
        self,
        classes: Optional[List[int]] = None,
    ) -> Tuple[FewShotTask, Dict[int, int]]:
        """Sample a few-shot task with optional class specification.

        Args:
            classes: Optional list of classes to use

        Returns:
            Tuple of (FewShotTask, class_mapping)
        """
        if classes is None:
            classes = random.sample(self._classes, self.n_way)

        class_mapping = {orig: new for new, orig in enumerate(classes)}

        support_x, support_y = [], []
        query_x, query_y = [], []

        for orig_cls in classes:
            cls_indices = self._class_to_indices[orig_cls]

            if len(cls_indices) < self.n_shot + self.n_query:
                sampled = (cls_indices * 2)[: self.n_shot + self.n_query]
            else:
                sampled = random.sample(cls_indices, self.n_shot + self.n_query)

            for idx in sampled[: self.n_shot]:
                x, _ = self.dataset[idx]
                support_x.append(x)
                support_y.append(class_mapping[orig_cls])

            for idx in sampled[self.n_shot :]:
                x, _ = self.dataset[idx]
                query_x.append(x)
                query_y.append(class_mapping[orig_cls])

        return FewShotTask(
            support_x=torch.stack(support_x),
            support_y=torch.tensor(support_y, dtype=torch.long),
            query_x=torch.stack(query_x),
            query_y=torch.tensor(query_y, dtype=torch.long),
            n_way=self.n_way,
            n_shot=self.n_shot,
            n_query=self.n_query,
            classes=classes,
        ), class_mapping


class TransductiveTaskSampler(TaskSampler):
    """Sampler for transductive few-shot learning.

    Uses query examples to improve prototype estimation through
    label propagation or centroid updating.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
        num_episodes: int = 100,
    ):
        super().__init__(dataset, n_way, n_shot, n_query, num_episodes)

    def sample_task(self) -> FewShotTask:
        """Sample a task with balanced query set."""
        return self.sample_episode()


class DomainShiftSampler(TaskSampler):
    """Sampler for domain shift few-shot learning.

    Samples episodes where support and query come from different domains.
    """

    def __init__(
        self,
        source_dataset: Dataset,
        target_dataset: Dataset,
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
        num_episodes: int = 100,
    ):
        super().__init__(source_dataset, n_way, n_shot, n_query, num_episodes)
        self.target_dataset = target_dataset


def create_episode_loader(
    dataset: Dataset,
    n_way: int = 5,
    n_shot: int = 5,
    n_query: int = 15,
    num_episodes: int = 100,
    batch_size: int = 1,
) -> DataLoader:
    """Create a DataLoader for few-shot episodes.

    Args:
        dataset: Source dataset
        n_way: Number of classes per episode
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        num_episodes: Number of episodes per epoch
        batch_size: Number of episodes per batch

    Returns:
        DataLoader that yields episodes
    """
    sampler = TaskSampler(
        dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        num_episodes=num_episodes,
    )

    def collate_fn(indices_list):
        if isinstance(indices_list[0], list):
            indices = indices_list[0]
        else:
            indices = indices_list

        xs, ys = [], []
        for idx in indices:
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)

        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )
