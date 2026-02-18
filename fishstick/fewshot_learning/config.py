"""
Configuration classes and defaults for few-shot learning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union
import torch


@dataclass
class MAMLConfig:
    """Configuration for MAML algorithm.

    Attributes:
        inner_lr: Learning rate for inner loop adaptation
        num_inner_steps: Number of gradient steps in inner loop
        first_order: Use first-order approximation (FOMAML)
        inner_optimizer: Optimizer type for inner loop ('sgd', 'adam')
        inner_momentum: Momentum for inner loop SGD
        grad_clip: Maximum gradient norm for clipping (None to disable)
        learn_inner_lr: Learn per-parameter inner loop learning rates
        inner_lr_init: Initial learning rate if learn_inner_lr is True
    """

    inner_lr: float = 0.01
    num_inner_steps: int = 5
    first_order: bool = False
    inner_optimizer: str = "sgd"
    inner_momentum: float = 0.0
    grad_clip: Optional[float] = None
    learn_inner_lr: bool = False
    inner_lr_init: float = 0.1


@dataclass
class PrototypicalConfig:
    """Configuration for Prototypical Networks.

    Attributes:
        distance: Distance metric ('euclidean', 'cosine', 'manhattan')
        learn_distance: Learn the distance function parameters
        temperature: Temperature for softmax over distances
        normalized: Normalize prototypes and query embeddings
        squared: Use squared Euclidean distance
    """

    distance: str = "euclidean"
    learn_distance: bool = False
    temperature: float = 1.0
    normalized: bool = True
    squared: bool = False


@dataclass
class RelationNetworkConfig:
    """Configuration for Relation Networks.

    Attributes:
        relation_dim: Dimension of relation module hidden layer
        num_relation_layers: Number of layers in relation module
        activation: Activation function ('relu', 'leaky_relu', 'gelu')
        dropout: Dropout probability in relation module
    """

    relation_dim: int = 8
    num_relation_layers: int = 3
    activation: str = "relu"
    dropout: float = 0.0


@dataclass
class MatchingNetworkConfig:
    """Configuration for Matching Networks.

    Attributes:
        attention: Attention mechanism ('cosine', 'embedded', 'dot')
        full_context: Use full context embedding
        lstm_layers: Number of LSTM layers for context encoding
        lstm_dropout: Dropout in LSTM
    """

    attention: str = "cosine"
    full_context: bool = True
    lstm_layers: int = 1
    lstm_dropout: float = 0.0


@dataclass
class EpisodeConfig:
    """Configuration for episode generation.

    Attributes:
        n_way: Number of classes per episode
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        meta_batch_size: Number of episodes per meta-batch
        classes_per_episode: Number of unique episode class sets
        num_episodes: Total number of episodes per epoch
        include_unlabeled: Include unlabeled examples in episodes
        unlabeled_ratio: Ratio of unlabeled to labeled examples
        domain_shift: Apply domain shift augmentation
    """

    n_way: int = 5
    n_shot: int = 5
    n_query: int = 15
    meta_batch_size: int = 4
    classes_per_episode: int = 20
    num_episodes: int = 100
    include_unlabeled: bool = False
    unlabeled_ratio: float = 0.5
    domain_shift: bool = False


@dataclass
class TrainingConfig:
    """Configuration for few-shot training.

    Attributes:
        epochs: Number of meta-training epochs
        meta_lr: Learning rate for meta-learner
        inner_lr: Learning rate for inner loop (if applicable)
        weight_decay: Weight decay for optimizer
        grad_clip: Maximum gradient norm
        scheduler: Learning rate scheduler ('step', 'cosine', 'none')
        scheduler_step_size: Step size for step scheduler
        gamma: Multiplicative factor for step scheduler
        warmup_steps: Number of warmup steps
        eval_every: Evaluate every N episodes
        save_every: Save checkpoint every N epochs
        device: Device to train on ('cuda', 'cpu')
    """

    epochs: int = 100
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    scheduler: str = "cosine"
    scheduler_step_size: int = 30
    gamma: float = 0.1
    warmup_steps: int = 0
    eval_every: int = 500
    save_every: int = 10
    device: str = "cuda"


@dataclass
class EvaluatorConfig:
    """Configuration for few-shot evaluation.

    Attributes:
        n_way: Number of classes in evaluation episodes
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        num_episodes: Number of evaluation episodes
        compute_ci: Compute confidence intervals
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        transductive: Use transductive inference
        unlabeled_ratio: Ratio of unlabeled examples (for semi-supervised)
    """

    n_way: int = 5
    n_shot: int = 5
    n_query: int = 15
    num_episodes: int = 1000
    compute_ci: bool = True
    ci_level: float = 0.95
    transductive: bool = False
    unlabeled_ratio: float = 0.0


DEFAULT_MAML_CONFIG = MAMLConfig()
DEFAULT_PROTONET_CONFIG = PrototypicalConfig()
DEFAULT_RELATION_CONFIG = RelationNetworkConfig()
DEFAULT_MATCHING_CONFIG = MatchingNetworkConfig()
DEFAULT_EPISODE_CONFIG = EpisodeConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EVALUATOR_CONFIG = EvaluatorConfig()


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    optimizer_type: str = "adam",
) -> torch.optim.Optimizer:
    """Create an optimizer for meta-learning.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')

    Returns:
        Configured optimizer
    """
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    step_size: int = 30,
    gamma: float = 0.1,
    warmup_steps: int = 0,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        step_size: Step size for step scheduler
        gamma: Multiplicative factor
        warmup_steps: Number of warmup steps

    Returns:
        Scheduler or None
    """
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif scheduler_type == "none":
        return None
    else:
        return None
