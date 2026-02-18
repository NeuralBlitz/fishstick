from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Scheduler(ABC):
    @abstractmethod
    def get_value(self, step: int) -> float:
        pass

    def __call__(self, step: int) -> float:
        return self.get_value(step)


class ConstantScheduler(Scheduler):
    def __init__(self, value: float):
        self.value = value

    def get_value(self, step: int) -> float:
        return self.value


class StepScheduler(Scheduler):
    def __init__(self, initial_value: float, step_size: int, gamma: float = 0.1):
        self.initial_value = initial_value
        self.step_size = step_size
        self.gamma = gamma

    def get_value(self, step: int) -> float:
        return self.initial_value * (self.gamma ** (step // self.step_size))


class ExponentialScheduler(Scheduler):
    def __init__(self, initial_value: float, decay_rate: float, decay_steps: int = 1):
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def get_value(self, step: int) -> float:
        return self.initial_value * math.exp(-self.decay_rate * step / self.decay_steps)


class CosineAnnealingScheduler(Scheduler):
    def __init__(self, initial_value: float, T_max: int, eta_min: float = 0.0):
        self.initial_value = initial_value
        self.T_max = T_max
        self.eta_min = eta_min

    def get_value(self, step: int) -> float:
        return (
            self.eta_min
            + (self.initial_value - self.eta_min)
            * (1 + math.cos(math.pi * step / self.T_max))
            / 2
        )


class CosineAnnealingWarmRestartsScheduler(Scheduler):
    def __init__(
        self, initial_value: float, T_0: int, T_mult: int = 1, eta_min: float = 0.0
    ):
        self.initial_value = initial_value
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

    def get_value(self, step: int) -> float:
        T_i = self.T_0
        while step >= T_i:
            step -= T_i
            T_i *= self.T_mult
        return (
            self.eta_min
            + (self.initial_value - self.eta_min)
            * (1 + math.cos(math.pi * step / T_i))
            / 2
        )


class CyclicScheduler(Scheduler):
    def __init__(
        self,
        initial_value: float,
        max_value: float,
        step_size: int,
        mode: str = "triangular",
        gamma: float = 1.0,
    ):
        self.initial_value = initial_value
        self.max_value = max_value
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

    def get_value(self, step: int) -> float:
        cycle = math.floor(step / self.step_size)
        x = step / self.step_size - cycle

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = self.gamma**cycle
        elif self.mode == "exp_range":
            scale = self.gamma ** (step / self.step_size)
        else:
            scale = 1.0

        return (
            self.initial_value
            + (self.max_value - self.initial_value) * max(0, 1 - x) * scale
        )


class PolynomialScheduler(Scheduler):
    def __init__(
        self,
        initial_value: float,
        end_value: float,
        max_steps: int,
        power: float = 1.0,
    ):
        self.initial_value = initial_value
        self.end_value = end_value
        self.max_steps = max_steps
        self.power = power

    def get_value(self, step: int) -> float:
        if step >= self.max_steps:
            return self.end_value
        return (self.initial_value - self.end_value) * (
            1 - step / self.max_steps
        ) ** self.power + self.end_value


class WarmupScheduler(Scheduler):
    def __init__(
        self, warmup_steps: int, warmup_value: float, target_scheduler: Scheduler
    ):
        self.warmup_steps = warmup_steps
        self.warmup_value = warmup_value
        self.target_scheduler = target_scheduler

    def get_value(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.warmup_value + (
                self.target_scheduler.get_value(0) - self.warmup_value
            ) * (step / self.warmup_steps)
        return self.target_scheduler.get_value(step - self.warmup_steps)


class LinearWarmupScheduler(Scheduler):
    def __init__(self, warmup_steps: int, initial_value: float, target_value: float):
        self.warmup_steps = warmup_steps
        self.initial_value = initial_value
        self.target_value = target_value

    def get_value(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.initial_value + (self.target_value - self.initial_value) * (
                step / self.warmup_steps
            )
        return self.target_value


class CooldownScheduler(Scheduler):
    def __init__(
        self,
        total_steps: int,
        cooldown_steps: int,
        initial_value: float,
        end_value: float,
    ):
        self.total_steps = total_steps
        self.cooldown_steps = cooldown_steps
        self.initial_value = initial_value
        self.end_value = end_value

    def get_value(self, step: int) -> float:
        if step >= self.total_steps - self.cooldown_steps:
            progress = (
                step - (self.total_steps - self.cooldown_steps)
            ) / self.cooldown_steps
            return self.initial_value - (self.initial_value - self.end_value) * progress
        return self.initial_value


class OneCycleScheduler(Scheduler):
    def __init__(self, max_value: float, total_steps: int, pct_start: float = 0.3):
        self.max_value = max_value
        self.total_steps = total_steps
        self.pct_start = pct_start

    def get_value(self, step: int) -> float:
        if step < self.total_steps * self.pct_start:
            progress = step / (self.total_steps * self.pct_start)
            return self.max_value * progress
        else:
            progress = (step - self.total_steps * self.pct_start) / (
                self.total_steps * (1 - self.pct_start)
            )
            return self.max_value * (1 + (1 - 2 * progress)) / 2


class LinearDecayScheduler(Scheduler):
    def __init__(self, initial_value: float, end_value: float, total_steps: int):
        self.initial_value = initial_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_value(self, step: int) -> float:
        progress = min(step / self.total_steps, 1.0)
        return self.initial_value + (self.end_value - self.initial_value) * progress


def get_scheduler(name: str, **kwargs: Any) -> Scheduler:
    name = name.lower()
    if name == "constant":
        return ConstantScheduler(**kwargs)
    elif name == "step":
        return StepScheduler(**kwargs)
    elif name == "exponential" or name == "exp":
        return ExponentialScheduler(**kwargs)
    elif name == "cosine" or name == "cosine_annealing":
        return CosineAnnealingScheduler(**kwargs)
    elif name == "cosine_warm_restarts":
        return CosineAnnealingWarmRestartsScheduler(**kwargs)
    elif name == "cyclic":
        return CyclicScheduler(**kwargs)
    elif name == "polynomial":
        return PolynomialScheduler(**kwargs)
    elif name == "warmup":
        return WarmupScheduler(**kwargs)
    elif name == "linear_warmup":
        return LinearWarmupScheduler(**kwargs)
    elif name == "cooldown":
        return CooldownScheduler(**kwargs)
    elif name == "onecycle":
        return OneCycleScheduler(**kwargs)
    elif name == "linear" or name == "linear_decay":
        return LinearDecayScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
