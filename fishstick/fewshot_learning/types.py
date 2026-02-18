"""
Type definitions and data structures for few-shot learning.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from enum import Enum
import torch
from torch import Tensor


class DistanceMetric(str, Enum):
    """Supported distance metrics for few-shot learning."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    MAHALANOBIS = "mahalanobis"


class EpisodeType(str, Enum):
    """Types of episodes for few-shot learning."""

    STANDARD = "standard"
    TRANSDUCTIVE = "transductive"
    SEMI_SUPERVISED = "semi_supervised"
    CROSS_DOMAIN = "cross_domain"


@dataclass
class FewShotTask:
    """Represents a single few-shot learning task/episode.

    Attributes:
        support_x: Support set inputs [n_way * n_shot, ...]
        support_y: Support set labels [n_way * n_shot]
        query_x: Query set inputs [n_way * n_query, ...]
        query_y: Query set labels [n_way * n_query]
        n_way: Number of classes in the task
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        classes: List of class indices in this task
    """

    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor
    n_way: int
    n_shot: int
    n_query: int
    classes: List[int] = field(default_factory=list)

    @property
    def support_size(self) -> int:
        """Total support set size."""
        return self.n_way * self.n_shot

    @property
    def query_size(self) -> int:
        """Total query set size."""
        return self.n_way * self.n_query

    @property
    def n_classes(self) -> int:
        """Alias for n_way."""
        return self.n_way


@dataclass
class MetaBatch:
    """A meta-batch containing multiple few-shot tasks.

    Attributes:
        tasks: List of FewShotTask objects
        meta_batch_size: Number of tasks in the meta-batch
    """

    tasks: List[FewShotTask]
    meta_batch_size: int

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass
class AdaptationResult:
    """Result of task adaptation in meta-learning.

    Attributes:
        adapted_model: The adapted model parameters
        adapted_logits: Predictions on support set after adaptation
        query_logits: Predictions on query set
        adaptation_loss: Loss during adaptation
        query_loss: Loss on query set
    """

    adapted_model: Dict[str, Tensor]
    adapted_logits: Optional[Tensor] = None
    query_logits: Optional[Tensor] = None
    adaptation_loss: Optional[Tensor] = None
    query_loss: Optional[Tensor] = None


@dataclass
class EvaluationResult:
    """Results from few-shot evaluation.

    Attributes:
        accuracy: Classification accuracy
        confidence_interval: 95% confidence interval
        std: Standard deviation
        num_samples: Number of evaluation episodes
    """

    accuracy: float
    confidence_interval: Tuple[float, float]
    std: float
    num_samples: int

    def __str__(self) -> str:
        ci_low, ci_high = self.confidence_interval
        return (
            f"Accuracy: {self.accuracy:.2%} ± {self.std:.2%} "
            f"(95% CI: [{ci_low:.2%}, {ci_high:.2%}], n={self.num_samples})"
        )


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning algorithms.

    Attributes:
        n_way: Number of classes per task
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        meta_batch_size: Number of tasks per meta-update
        inner_lr: Learning rate for inner loop
        outer_lr: Learning rate for outer loop
        num_inner_steps: Number of gradient steps in inner loop
        first_order: Use first-order approximation (for MAML)
        distance_metric: Distance metric for metric-based methods
    """

    n_way: int = 5
    n_shot: int = 5
    n_query: int = 15
    meta_batch_size: int = 4
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    first_order: bool = False
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN


@dataclass
class TrainingState:
    """Tracks the state of few-shot meta-training.

    Attributes:
        epoch: Current epoch
        iteration: Current iteration
        train_loss: Training loss history
        val_accuracy: Validation accuracy history
        test_accuracy: Test accuracy history
        best_val_accuracy: Best validation accuracy seen
    """

    epoch: int = 0
    iteration: int = 0
    train_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)
    best_val_accuracy: float = 0.0


EncoderFactory = Callable[[], torch.nn.Module]
LossFunction = Callable[[Tensor, Tensor], Tensor]
DistanceFunction = Callable[[Tensor, Tensor], Tensor]
