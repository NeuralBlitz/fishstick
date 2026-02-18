from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bayesian import Optimizer
from .search_space import ParameterType, SearchSpace
from .trial import ResultStorage, Trial


@dataclass
class SmartGridSearch(Optimizer):
    search_space: dict[str, SearchSpace]
    points_per_dimension: int = 10
    storage: ResultStorage | None = None
    seed: int | None = None

    def __post_init__(self):
        self.storage = self.storage or ResultStorage()
        self.rng = np.random.default_rng(self.seed)
        self.trial_counter = 0
        self.grid_points: list[dict[str, Any]] = []
        self.current_idx = 0
        self._generate_grid()

    def _generate_grid(self) -> None:
        param_names = list(self.search_space.keys())
        space_objects = [self.search_space[name] for name in param_names]

        def grid_recursive(idx: int, current: dict[str, Any]) -> None:
            if idx == len(param_names):
                self.grid_points.append(current.copy())
                return

            param = space_objects[idx]
            if param.param_type in (ParameterType.CHOICE, ParameterType.CATEGORICAL):
                for choice in param.choices:
                    current[param_names[idx]] = choice
                    grid_recursive(idx + 1, current)
            else:
                for i in range(self.points_per_dimension):
                    value = i / (self.points_per_dimension - 1)
                    current[param_names[idx]] = param.transform(value)
                    grid_recursive(idx + 1, current)

        grid_recursive(0, {})

    def suggest(self) -> dict[str, Any]:
        if self.current_idx >= len(self.grid_points):
            return {}
        params = self.grid_points[self.current_idx]
        self.current_idx += 1
        return params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)

    def get_best_trial(self) -> Trial | None:
        return self.storage.get_best()


@dataclass
class SmartRandomSearch(Optimizer):
    search_space: dict[str, SearchSpace]
    n_trials: int = 100
    storage: ResultStorage | None = None
    seed: int | None = None
    sampling_strategy: str = "uniform"

    def __post_init__(self):
        self.storage = self.storage or ResultStorage()
        self.seed = self.seed if self.seed is not None else 42
        self.rng = np.random.default_rng(self.seed)
        self.trial_counter = 0
        self.completed_trials: list[Trial] = []

    def _sample_adaptive(self) -> dict[str, Any]:
        params = {}
        for name, space in self.search_space.items():
            if self.sampling_strategy == "latin":
                params[name] = space.sample(self.rng)
            elif self.sampling_strategy == "sobol":
                params[name] = space.sample(self.rng)
            else:
                params[name] = space.sample(self.rng)
        return params

    def suggest(self) -> dict[str, Any]:
        if self.trial_counter >= self.n_trials:
            return {}
        params = self._sample_adaptive()
        self.trial_counter += 1
        return params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)
        self.completed_trials.append(trial)

    def get_best_trial(self) -> Trial | None:
        return self.storage.get_best()


@dataclass
class ConstrainedRandomSearch(SmartRandomSearch):
    constraints: list[callable[[dict[str, Any]], bool]] = field(default_factory=list)

    def _is_feasible(self, params: dict[str, Any]) -> bool:
        for constraint in self.constraints:
            try:
                if not constraint(params):
                    return False
            except Exception:
                return False
        return True

    def suggest(self) -> dict[str, Any]:
        if self.trial_counter >= self.n_trials:
            return {}

        max_attempts = 100
        for _ in range(max_attempts):
            params = super().suggest()
            if self._is_feasible(params):
                return params

        return {}
