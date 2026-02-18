from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class EarlyStoppingCriterion(ABC):
    @abstractmethod
    def should_stop(self, history: list[float]) -> bool:
        pass


@dataclass
class PatienceCriterion(EarlyStoppingCriterion):
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"
    baseline: float | None = None

    def should_stop(self, history: list[float]) -> bool:
        if len(history) < self.patience:
            return False

        if self.baseline is not None:
            if self.mode == "min":
                return history[-1] < self.baseline
            else:
                return history[-1] > self.baseline

        recent = history[-self.patience :]
        if self.mode == "min":
            best = min(recent)
            return all(v >= best + self.min_delta for v in recent)
        else:
            best = max(recent)
            return all(v <= best - self.min_delta for v in recent)


@dataclass
class DeltaCriterion(EarlyStoppingCriterion):
    min_delta: float = 0.0001
    window_size: int = 10

    def should_stop(self, history: list[float]) -> bool:
        if len(history) < self.window_size * 2:
            return False

        recent = history[-self.window_size :]
        older = history[-2 * self.window_size : -self.window_size]

        if len(recent) < self.window_size or len(older) < self.window_size:
            return False

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)

        return abs(recent_mean - older_mean) < self.min_delta


@dataclass
class GradientNormCriterion(EarlyStoppingCriterion):
    threshold: float = 1e-6

    def should_stop(self, history: list[float]) -> bool:
        return False


@dataclass
class MetricThresholdCriterion(EarlyStoppingCriterion):
    threshold: float
    mode: str = "min"

    def should_stop(self, history: list[float]) -> bool:
        if not history:
            return False

        if self.mode == "min":
            return history[-1] <= self.threshold
        else:
            return history[-1] >= self.threshold


class EarlyStoppingMonitor:
    def __init__(
        self,
        criteria: list[EarlyStoppingCriterion] | None = None,
        mode: str = "min",
    ):
        self.criteria = criteria or [PatienceCriterion()]
        self.mode = mode
        self.history: list[float] = []
        self.best_value: float = float("inf") if mode == "min" else float("-inf")
        self.best_step: int = 0

    def update(self, value: float, step: int) -> None:
        self.history.append(value)

        if self.mode == "min":
            if value < self.best_value:
                self.best_value = value
                self.best_step = step
        else:
            if value > self.best_value:
                self.best_value = value
                self.best_step = step

    def should_stop(self) -> bool:
        if not self.history:
            return False

        for criterion in self.criteria:
            if criterion.should_stop(self.history):
                return True

        return False

    def get_best(self) -> tuple[float, int]:
        return self.best_value, self.best_step


class AdaptiveEarlyStopping:
    def __init__(
        self,
        initial_patience: int = 10,
        patience_multiplier: float = 1.5,
        min_patience: int = 5,
        max_patience: int = 100,
        mode: str = "min",
    ):
        self.initial_patience = initial_patience
        self.patience_multiplier = patience_multiplier
        self.min_patience = min_patience
        self.max_patience = max_patience
        self.mode = mode
        self.current_patience = initial_patience
        self.history: list[float] = []
        self.improvement_count = 0

    def update(self, value: float) -> bool:
        self.history.append(value)

        if len(self.history) < 2:
            return False

        if self.mode == "min":
            improved = value < self.history[-2] - 1e-5
        else:
            improved = value > self.history[-2] + 1e-5

        if improved:
            self.improvement_count += 1
            if self.improvement_count >= 3:
                self.current_patience = min(
                    int(self.current_patience * self.patience_multiplier),
                    self.max_patience,
                )
                self.improvement_count = 0
        else:
            self.improvement_count = 0

        if len(self.history) >= self.current_patience:
            recent = self.history[-self.current_patience :]
            if self.mode == "min":
                if min(recent) >= min(self.history[: -self.current_patience]) - 1e-5:
                    return True
            else:
                if max(recent) <= max(self.history[: -self.current_patience]) + 1e-5:
                    return True

        return False

    def reset(self) -> None:
        self.current_patience = self.initial_patience
        self.history = []
        self.improvement_count = 0


class MedianStoppingRule:
    def __init__(self, window_size: int = 5, threshold: float = 1.0):
        self.window_size = window_size
        self.threshold = threshold
        self.trial_history: dict[int, list[float]] = {}

    def update(self, trial_id: int, value: float) -> None:
        if trial_id not in self.trial_history:
            self.trial_history[trial_id] = []
        self.trial_history[trial_id].append(value)

    def should_stop(self, trial_id: int) -> bool:
        if trial_id not in self.trial_history:
            return False

        if len(self.trial_history[trial_id]) < self.window_size:
            return False

        trial_recent = self.trial_history[trial_id][-self.window_size :]

        all_medians = []
        for other_id, values in self.trial_history.items():
            if len(values) >= self.window_size:
                all_medians.append(np.median(values[-self.window_size :]))

        if not all_medians:
            return False

        trial_median = np.median(trial_recent)
        other_median = np.median(all_medians)

        if self.threshold > 0:
            return trial_median > other_median * self.threshold
        else:
            return trial_median < other_median / (-self.threshold)
