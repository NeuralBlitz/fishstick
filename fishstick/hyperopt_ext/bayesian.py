from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    get_acquisition_function,
)
from .gaussian_process import GaussianProcess
from .search_space import ParameterType, SearchSpace
from .trial import ResultStorage, Trial


class Optimizer(ABC):
    @abstractmethod
    def suggest(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        pass

    def get_best_trial(self) -> Trial | None:
        return None

    def get_results(self) -> list:
        return []


@dataclass
class BayesianOptimizer(Optimizer):
    search_space: dict[str, SearchSpace]
    n_trials: int = 100
    storage: ResultStorage | None = None
    seed: int | None = None
    acquisition: str = "ei"
    random_fraction: float = 0.2
    n_candidates: int = 1000
    minimize: bool = True
    gp_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.storage = self.storage or ResultStorage()
        self.seed = seed if (seed := self.seed) is not None else 42
        self.rng = np.random.default_rng(self.seed)
        self.trial_counter = 0
        self.completed_trials: list[Trial] = []
        self.gp = GaussianProcess(**self.gp_params)
        self._initialized = False
        self._acquisition_fn = get_acquisition_function(
            self.acquisition, minimize=self.minimize
        )

    def _space_to_vector(self, params: dict[str, Any]) -> np.ndarray:
        vec = []
        for name in sorted(self.search_space.keys()):
            space = self.search_space[name]
            val = params[name]
            if space.param_type in (ParameterType.CHOICE, ParameterType.CATEGORICAL):
                idx = space.choices.index(val) if val in space.choices else 0
                vec.append(idx / max(len(space.choices) - 1, 1))
            else:
                if space.param_type == ParameterType.LOGUNIFORM:
                    log_low = (
                        math.log(space.low) if space.low and space.low > 0 else -7.0
                    )
                    log_high = (
                        math.log(space.high) if space.high and space.high > 0 else 7.0
                    )
                    val = (math.log(val) - log_low) / (log_high - log_low)
                elif space.param_type in (
                    ParameterType.INTEGER,
                    ParameterType.LOGINTEGER,
                ):
                    val = (
                        (val - space.low) / (space.high - space.low)
                        if space.high != space.low
                        else 0.5
                    )
                else:
                    val = (
                        (val - space.low) / (space.high - space.low)
                        if space.high != space.low
                        else 0.5
                    )
                vec.append(val)
        return np.array(vec)

    def _vector_to_space(self, vec: np.ndarray) -> dict[str, Any]:
        params = {}
        for i, name in enumerate(sorted(self.search_space.keys())):
            space = self.search_space[name]
            val = vec[i]
            params[name] = space.transform(val)
        return params

    def _generate_candidates(self, n: int) -> np.ndarray:
        candidates = []
        for _ in range(n):
            cand = self._space_to_vector(
                {
                    name: space.sample(self.rng)
                    for name, space in self.search_space.items()
                }
            )
            candidates.append(cand)
        return np.array(candidates)

    def _get_best_value(self) -> float:
        if not self.completed_trials:
            return float("inf") if self.minimize else float("-inf")
        if self.minimize:
            return min(
                t.metrics.get("loss", float("inf")) for t in self.completed_trials
            )
        return max(t.metrics.get("loss", float("-inf")) for t in self.completed_trials)

    def suggest(self) -> dict[str, Any]:
        if self.trial_counter >= self.n_trials:
            return {}

        if not self._initialized or self.trial_counter < int(
            self.n_trials * self.random_fraction
        ):
            params = {
                name: space.sample(self.rng)
                for name, space in self.search_space.items()
            }
        else:
            best_y = self._get_best_value()

            candidates = self._generate_candidates(self.n_candidates)

            mu, std = self.gp.predict(candidates, return_std=True)

            acq_values = self._acquisition_fn(mu, std, best_y, rng=self.rng)

            best_idx = np.argmax(acq_values)
            params = self._vector_to_space(candidates[best_idx])

        self.trial_counter += 1
        return params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)
        self.completed_trials.append(trial)

        if len(self.completed_trials) >= 2:
            X = np.array(
                [self._space_to_vector(t.params) for t in self.completed_trials]
            )
            y = np.array(
                [
                    t.metrics.get("loss", 0.0)
                    if self.minimize
                    else -t.metrics.get("loss", 0.0)
                    for t in self.completed_trials
                ]
            )
            self.gp.fit(X, y)
            self._initialized = True

    def get_best_trial(self) -> Trial | None:
        return self.storage.get_best(minimize=self.minimize)

    def get_results(self) -> list[Trial]:
        return self.completed_trials


class ConstrainedBayesianOptimizer(BayesianOptimizer):
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        **kwargs: Any,
    ):
        super().__init__(search_space, **kwargs)
        self.constraints = constraints or []

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


class ParallelBayesianOptimizer(BayesianOptimizer):
    def __init__(self, n_workers: int = 4, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_workers = n_workers

    def suggest_batch(self, n: int | None = None) -> list[dict[str, Any]]:
        n = n or self.n_workers
        suggestions = []
        for _ in range(n):
            suggestion = self.suggest()
            if suggestion:
                suggestions.append(suggestion)
        return suggestions
