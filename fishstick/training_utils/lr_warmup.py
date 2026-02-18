"""
Learning Rate Warmup Utilities

Comprehensive learning rate warmup implementations including linear, exponential,
cosine, polynomial, and constant warmup strategies with scheduler integration.
"""

from typing import Optional, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


class WarmupType(Enum):
    """Learning rate warmup types."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    SQUARE = "square"
    SQRT = "sqrt"


class LRWarmup(ABC):
    """
    Abstract base class for learning rate warmup strategies.

    Defines the interface for all warmup implementations.
    """

    def __init__(
        self,
        warmup_steps: int,
        warmup_ratio: Optional[float] = None,
        warmup_start_lr: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_start_lr = warmup_start_lr

    @abstractmethod
    def get_lr_multiplier(self, step: int) -> float:
        """
        Get the learning rate multiplier for a given step.

        Args:
            step: Current training step

        Returns:
            Learning rate multiplier in range [0, 1]
        """
        pass

    def __call__(self, step: int) -> float:
        """Allow warmup to be called as a function."""
        return self.get_lr_multiplier(step)


class LinearWarmup(LRWarmup):
    """
    Linear learning rate warmup.

    Linearly increases learning rate from warmup_start_lr to 1.0
    over the warmup period.

    Example:
        >>> warmup = LinearWarmup(warmup_steps=1000, warmup_start_lr=1e-7)
        >>> lr_mult = warmup.get_lr_multiplier(500)  # ~0.5
    """

    def get_lr_multiplier(self, step: int) -> float:
        if self.warmup_steps == 0:
            return 1.0

        if step >= self.warmup_steps:
            return 1.0

        if self.warmup_ratio is not None:
            return self.warmup_ratio + (1 - self.warmup_ratio) * (
                step / self.warmup_steps
            )

        progress = step / self.warmup_steps
        return self.warmup_start_lr + (1 - self.warmup_start_lr) * progress


class ExponentialWarmup(LRWarmup):
    """
    Exponential learning rate warmup.

    Exponentially increases learning rate from warmup_start_lr to 1.0
    providing a more gradual start than linear warmup.

    Example:
        >>> warmup = ExponentialWarmup(warmup_steps=1000, warmup_start_lr=1e-7)
        >>> lr_mult = warmup.get_lr_multiplier(500)
    """

    def get_lr_multiplier(self, step: int) -> float:
        if self.warmup_steps == 0:
            return 1.0

        if step >= self.warmup_steps:
            return 1.0

        if self.warmup_ratio is not None:
            return self.warmup_ratio ** (1 - step / self.warmup_steps)

        log_min = math.log(self.warmup_start_lr + 1e-10)
        log_max = 0.0
        log_current = log_min + (log_max - log_min) * (step / self.warmup_steps)

        return math.exp(log_current)


class CosineWarmup(LRWarmup):
    """
    Cosine learning rate warmup.

    Uses cosine curve for smooth learning rate increase from warmup_start_lr to 1.0.
    Provides a very gradual start that accelerates towards the end of warmup.

    Example:
        >>> warmup = CosineWarmup(warmup_steps=1000, warmup_start_lr=1e-7)
        >>> lr_mult = warmup.get_lr_multiplier(500)
    """

    def get_lr_multiplier(self, step: int) -> float:
        if self.warmup_steps == 0:
            return 1.0

        if step >= self.warmup_steps:
            return 1.0

        if self.warmup_ratio is not None:
            return (
                self.warmup_ratio
                + (1 - self.warmup_ratio)
                * (1 - math.cos(math.pi * step / self.warmup_steps))
                / 2
            )

        progress = step / self.warmup_steps
        cosine_factor = (1 - math.cos(math.pi * progress)) / 2
        return self.warmup_start_lr + (1 - self.warmup_start_lr) * cosine_factor


class PolynomialWarmup(LRWarmup):
    """
    Polynomial learning rate warmup.

    Uses polynomial function for learning rate increase.
    The power parameter controls the curve shape.

    Example:
        >>> warmup = PolynomialWarmup(warmup_steps=1000, power=2.0)
        >>> lr_mult = warmup.get_lr_multiplier(500)
    """

    def __init__(self, warmup_steps: int, power: float = 2.0, **kwargs):
        super().__init__(warmup_steps, **kwargs)
        self.power = power

    def get_lr_multiplier(self, step: int) -> float:
        if self.warmup_steps == 0:
            return 1.0

        if step >= self.warmup_steps:
            return 1.0

        if self.warmup_ratio is not None:
            return (
                self.warmup_ratio
                + (1 - self.warmup_ratio) * (step / self.warmup_steps) ** self.power
            )

        progress = step / self.warmup_steps
        poly_factor = progress**self.power
        return self.warmup_start_lr + (1 - self.warmup_start_lr) * poly_factor


class ConstantWarmup(LRWarmup):
    """
    Constant learning rate warmup.

    Keeps learning rate at initial value during warmup, then jumps to target.
    Useful for when you want to train with a small fixed LR before main training.
    """

    def get_lr_multiplier(self, step: int) -> float:
        if self.warmup_steps == 0:
            return 1.0

        if step >= self.warmup_steps:
            return 1.0

        return self.warmup_ratio or self.warmup_start_lr


class SquareWarmup(LRWarmup):
    """
    Square learning rate warmup.

    Uses square function for rapid increase at start, then slower increase.
    """

    def __init__(self, warmup_steps: int, **kwargs):
        super().__init__(warmup_steps, **kwargs)

    def get_lr_multiplier(self, step: int) -> float:
        return PolynomialWarmup(
            self.warmup_steps, power=2.0, warmup_start_lr=self.warmup_start_lr
        ).get_lr_multiplier(step)


class SqrtWarmup(LRWarmup):
    """
    Square root learning rate warmup.

    Uses square root function for slow start, then faster increase.
    """

    def __init__(self, warmup_steps: int, **kwargs):
        super().__init__(warmup_steps, **kwargs)

    def get_lr_multiplier(self, step: int) -> float:
        return PolynomialWarmup(
            self.warmup_steps, power=0.5, warmup_start_lr=self.warmup_start_lr
        ).get_lr_multiplier(step)


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup support.

    Combines a warmup phase with a main learning rate schedule.
    Supports various warmup types and main schedule strategies.

    Example:
        >>> warmup_scheduler = WarmupScheduler(
        ...     optimizer,
        ...     warmup_type="linear",
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     min_lr=1e-6,
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_type: Union[str, WarmupType] = "linear",
        warmup_steps: int = 1000,
        warmup_ratio: Optional[float] = None,
        warmup_start_lr: float = 0.0,
        total_steps: Optional[int] = None,
        schedule_type: str = "cosine",
        min_lr: float = 0.0,
        num_cycles: float = 0.5,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        if isinstance(warmup_type, str):
            warmup_type = WarmupType(warmup_type)

        self.warmup_type = warmup_type
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_start_lr = warmup_start_lr
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        self.power = power

        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.last_epoch < self.warmup_steps:
            warmup = self._get_warmup()
            return [
                base_lr * warmup.get_lr_multiplier(self.last_epoch)
                for base_lr in self.base_lrs
            ]

        if self.total_steps is None:
            return self.base_lrs

        post_warmup_step = self.last_epoch - self.warmup_steps
        post_warmup_total = self.total_steps - self.warmup_steps

        if post_warmup_total <= 0:
            return self.base_lrs

        progress = min(post_warmup_step / post_warmup_total, 1.0)

        if self.schedule_type == "cosine":
            lr_factor = self._cosine_decay(progress)
        elif self.schedule_type == "linear":
            lr_factor = self._linear_decay(progress)
        elif self.schedule_type == "polynomial":
            lr_factor = self._polynomial_decay(progress)
        elif self.schedule_type == "step":
            lr_factor = self._step_decay(progress)
        elif self.schedule_type == "exponential":
            lr_factor = self._exponential_decay(progress)
        else:
            lr_factor = 1.0

        return [
            self.min_lr + (base_lr - self.min_lr) * lr_factor
            for base_lr in self.base_lrs
        ]

    def _get_warmup(self) -> LRWarmup:
        """Get appropriate warmup instance."""
        warmup_map = {
            WarmupType.LINEAR: LinearWarmup,
            WarmupType.EXPONENTIAL: ExponentialWarmup,
            WarmupType.COSINE: CosineWarmup,
            WarmupType.POLYNOMIAL: PolynomialWarmup,
            WarmupType.CONSTANT: ConstantWarmup,
            WarmupType.SQUARE: SquareWarmup,
            WarmupType.SQRT: SqrtWarmup,
        }

        warmup_cls = warmup_map.get(self.warmup_type, LinearWarmup)
        return warmup_cls(
            warmup_steps=self.warmup_steps,
            warmup_ratio=self.warmup_ratio,
            warmup_start_lr=self.warmup_start_lr,
        )

    def _cosine_decay(self, progress: float) -> float:
        """Cosine decay function."""
        return self.num_cycles * (1 + math.cos(math.pi * progress)) / 2

    def _linear_decay(self, progress: float) -> float:
        """Linear decay function."""
        return 1 - progress

    def _polynomial_decay(self, progress: float) -> float:
        """Polynomial decay function."""
        return (1 - progress) ** self.power

    def _step_decay(self, progress: float) -> float:
        """Step decay function."""
        return 0.1 ** int(progress * 4)

    def _exponential_decay(self, progress: float) -> float:
        """Exponential decay function."""
        return math.exp(-progress * 5)


def create_warmup_scheduler(
    optimizer: Optimizer,
    warmup_type: Union[str, WarmupType] = "linear",
    warmup_steps: int = 1000,
    warmup_ratio: Optional[float] = None,
    warmup_start_lr: float = 0.0,
    total_steps: Optional[int] = None,
    schedule_type: str = "cosine",
    min_lr: float = 0.0,
    **kwargs,
) -> WarmupScheduler:
    """
    Create a learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule
        warmup_type: Type of warmup ('linear', 'exponential', 'cosine', 'polynomial', 'constant')
        warmup_steps: Number of warmup steps
        warmup_ratio: Target LR as ratio of base LR (alternative to warmup_start_lr)
        warmup_start_lr: Starting LR for warmup
        total_steps: Total training steps
        schedule_type: Main schedule type after warmup ('cosine', 'linear', 'polynomial', 'step')
        min_lr: Minimum learning rate
        **kwargs: Additional arguments

    Returns:
        Configured WarmupScheduler

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = create_warmup_scheduler(
        ...     optimizer,
        ...     warmup_type="cosine",
        ...     warmup_steps=1000,
        ...     total_steps=100000,
        ... )
    """
    return WarmupScheduler(
        optimizer=optimizer,
        warmup_type=warmup_type,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        warmup_start_lr=warmup_start_lr,
        total_steps=total_steps,
        schedule_type=schedule_type,
        min_lr=min_lr,
        **kwargs,
    )


@dataclass
class WarmupConfig:
    """Configuration for learning rate warmup."""

    warmup_type: str = "linear"
    warmup_steps: int = 1000
    warmup_ratio: Optional[float] = None
    warmup_start_lr: float = 0.0
    total_steps: Optional[int] = None
    schedule_type: str = "cosine"
    min_lr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "warmup_type": self.warmup_type,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "warmup_start_lr": self.warmup_start_lr,
            "total_steps": self.total_steps,
            "schedule_type": self.schedule_type,
            "min_lr": self.min_lr,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "WarmupConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


class WarmupCurve:
    """
    Utility for visualizing and analyzing warmup curves.

    Provides methods to compute and visualize learning rate curves
    for different warmup configurations.
    """

    def __init__(self, config: Union[WarmupConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = WarmupConfig.from_dict(config)
        self.config = config

    def get_curve(self, num_points: int = 100) -> List[float]:
        """
        Compute the learning rate curve.

        Args:
            num_points: Number of points to compute

        Returns:
            List of learning rate multipliers
        """
        warmup = self._create_warmup()
        total_steps = self.config.total_steps or self.config.warmup_steps * 2

        curve = []
        for step in range(0, total_steps, total_steps // num_points):
            if step < self.config.warmup_steps:
                lr_mult = warmup.get_lr_multiplier(step)
            else:
                post_warmup = step - self.config.warmup_steps
                post_total = total_steps - self.config.warmup_steps
                progress = min(post_warmup / post_total, 1.0)

                if self.config.schedule_type == "cosine":
                    lr_mult = self._cosine(progress)
                elif self.config.schedule_type == "linear":
                    lr_mult = 1 - progress
                else:
                    lr_mult = 1.0

            curve.append(lr_mult)

        return curve

    def _create_warmup(self) -> LRWarmup:
        """Create warmup instance."""
        warmup_map = {
            "linear": LinearWarmup,
            "exponential": ExponentialWarmup,
            "cosine": CosineWarmup,
            "polynomial": PolynomialWarmup,
            "constant": ConstantWarmup,
            "square": SquareWarmup,
            "sqrt": SqrtWarmup,
        }

        warmup_cls = warmup_map.get(self.config.warmup_type, LinearWarmup)
        return warmup_cls(
            warmup_steps=self.config.warmup_steps,
            warmup_ratio=self.config.warmup_ratio,
            warmup_start_lr=self.config.warmup_start_lr,
        )

    def _cosine(self, progress: float) -> float:
        """Cosine decay."""
        return (1 + math.cos(math.pi * progress)) / 2


class MultiStageWarmup:
    """
    Multi-stage warmup with different warmup phases.

    Allows for complex warmup schedules with different phases,
    such as slow warmup followed by fast warmup.
    """

    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        self._validate_stages()

    def _validate_stages(self):
        """Validate stage configurations."""
        cumulative_steps = 0
        for i, stage in enumerate(self.stages):
            steps = stage.get("steps", 0)
            cumulative_steps += steps
            stage["start_step"] = cumulative_steps - steps
            stage["end_step"] = cumulative_steps

    def get_lr_multiplier(self, step: int) -> float:
        """Get learning rate multiplier for a given step."""
        for stage in self.stages:
            if stage["start_step"] <= step < stage["end_step"]:
                warmup_type = stage.get("type", "linear")
                warmup_steps = stage["end_step"] - stage["start_step"]
                local_step = step - stage["start_step"]

                warmup = self._create_warmup(warmup_type, warmup_steps, stage)
                return warmup.get_lr_multiplier(local_step)

        return 1.0

    def _create_warmup(
        self,
        warmup_type: str,
        warmup_steps: int,
        stage_config: Dict[str, Any],
    ) -> LRWarmup:
        """Create warmup instance for a stage."""
        warmup_map = {
            "linear": LinearWarmup,
            "exponential": ExponentialWarmup,
            "cosine": CosineWarmup,
            "polynomial": PolynomialWarmup,
            "constant": ConstantWarmup,
        }

        warmup_cls = warmup_map.get(warmup_type, LinearWarmup)
        return warmup_cls(
            warmup_steps=warmup_steps,
            warmup_ratio=stage_config.get("warmup_ratio"),
            warmup_start_lr=stage_config.get("warmup_start_lr", 0.0),
        )
