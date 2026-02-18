"""Augmentation scheduling strategies."""

import torch
import numpy as np
from typing import List, Optional, Callable
from abc import ABC, abstractmethod


class AugmentSchedule(ABC):
    """Base class for augmentation scheduling."""

    @abstractmethod
    def get_probability(self, step: int) -> float:
        """Get augmentation probability at given step."""
        pass


class ConstantSchedule(AugmentSchedule):
    """Constant probability schedule."""

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def get_probability(self, step: int) -> float:
        return self.probability


class LinearSchedule(AugmentSchedule):
    """Linear interpolation schedule."""

    def __init__(
        self, start_value: float = 0.0, end_value: float = 1.0, total_steps: int = 10000
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_probability(self, step: int) -> float:
        fraction = min(step / self.total_steps, 1.0)
        return self.start_value + (self.end_value - self.start_value) * fraction


class ExponentialSchedule(AugmentSchedule):
    """Exponential decay/growth schedule."""

    def __init__(
        self,
        start_value: float = 1.0,
        end_value: float = 0.01,
        decay_rate: float = 0.9999,
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_rate = decay_rate

    def get_probability(self, step: int) -> float:
        value = self.start_value * (self.decay_rate**step)
        return max(self.end_value, value)


class CosineAnnealingSchedule(AugmentSchedule):
    """Cosine annealing schedule."""

    def __init__(
        self, start_value: float = 1.0, end_value: float = 0.0, total_steps: int = 10000
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_probability(self, step: int) -> float:
        fraction = min(step / self.total_steps, 1.0)
        cosine_factor = 0.5 * (1 + np.cos(np.pi * fraction))
        return self.end_value + (self.start_value - self.end_value) * cosine_factor


class PolicyScheduler:
    """Scheduler for augmentation policies."""

    def __init__(
        self,
        policies: List[Callable],
        schedule: Optional[AugmentSchedule] = None,
        default_probability: float = 0.5,
    ):
        self.policies = policies
        self.schedule = schedule or ConstantSchedule(default_probability)
        self.step_count = 0

    def __call__(self, data):
        """Apply random policy based on schedule."""
        prob = self.schedule.get_probability(self.step_count)
        self.step_count += 1

        if torch.rand(1).item() < prob:
            policy = torch.randint(0, len(self.policies), (1,)).item()
            return self.policies[policy](data)

        return data

    def step(self):
        """Increment scheduler step."""
        self.step_count += 1

    def reset(self):
        """Reset scheduler."""
        self.step_count = 0


class ScheduledAugment:
    """RandAugment-style scheduled augmentation."""

    def __init__(
        self,
        augment_fn: Callable,
        magnitude: int = 10,
        max_magnitude: int = 30,
        n_ops: int = 2,
        probability: float = 0.5,
    ):
        self.augment_fn = augment_fn
        self.magnitude = magnitude
        self.max_magnitude = max_magnitude
        self.n_ops = n_ops
        self.probability = probability

    def __call__(self, data):
        if torch.rand(1).item() > self.probability:
            return data

        magnitude = self.magnitude / self.max_magnitude
        for _ in range(self.n_ops):
            data = self.augment_fn(data, magnitude)

        return data


class AdaptiveProbabilityScheduler:
    """Adaptively adjust augmentation probability based on loss."""

    def __init__(
        self,
        initial_prob: float = 0.5,
        min_prob: float = 0.1,
        max_prob: float = 0.9,
        adaptation_factor: float = 0.01,
        window_size: int = 100,
    ):
        self.probability = initial_prob
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.adaptation_factor = adaptation_factor
        self.window_size = window_size
        self.loss_history = []

    def update(self, loss: float):
        """Update probability based on loss trend."""
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

        if len(self.loss_history) >= 2:
            loss_change = self.loss_history[-1] - self.loss_history[0]

            if loss_change > 0:
                self.probability = min(
                    self.max_prob, self.probability + self.adaptation_factor
                )
            else:
                self.probability = max(
                    self.min_prob, self.probability - self.adaptation_factor
                )

    def get_probability(self) -> float:
        return self.probability


class RandomApply:
    """Apply transformation with given probability."""

    def __init__(self, transform: Callable, probability: float = 0.5):
        self.transform = transform
        self.probability = probability

    def __call__(self, data):
        if torch.rand(1).item() < self.probability:
            return self.transform(data)
        return data


def create_schedule(schedule_type: str, **kwargs) -> AugmentSchedule:
    """Factory function to create schedules."""
    schedules = {
        "constant": ConstantSchedule,
        "linear": LinearSchedule,
        "exponential": ExponentialSchedule,
        "cosine": CosineAnnealingSchedule,
    }

    if schedule_type not in schedules:
        raise ValueError(f"Unknown schedule: {schedule_type}")

    return schedules[schedule_type](**kwargs)
