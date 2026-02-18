from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .search_space import SearchSpace
from .trial import ResultStorage, Trial


@dataclass
class HyperbandBracket:
    bracket_id: int
    n: int
    r: int
    eta: float = 3.0

    def get_resource_allocation(self) -> list[tuple[int, int]]:
        s_max = int(math.log(self.n, self.eta))
        return [
            (self.n // (self.eta**i), self.r * (self.eta**i)) for i in range(s_max + 1)
        ]

    def get_promotion_rates(self) -> list[int]:
        allocation = self.get_resource_allocation()
        return [n for n, _ in allocation]


@dataclass
class HyperbandTrial:
    trial_id: int
    params: dict[str, Any]
    bracket_id: int
    round_id: int
    resource: int
    performance: float = float("inf")
    status: str = "pending"


class Hyperband:
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        objective_fn: Callable[[dict[str, Any], int], float],
        n_trials: int = 81,
        max_resource: int = 81,
        eta: float = 3.0,
        storage: ResultStorage | None = None,
        seed: int | None = None,
        minimize: bool = True,
    ):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.max_resource = max_resource
        self.eta = eta
        self.storage = storage or ResultStorage()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.minimize = minimize

        self.bracket = HyperbandBracket(0, n_trials, max_resource, eta)
        self.trial_counter = 0
        self.allocation = self.bracket.get_resource_allocation()

        self.active_trials: dict[int, HyperbandTrial] = {}
        self.completed_trials: list[HyperbandTrial] = []
        self.current_round = 0

    def suggest(self) -> dict[str, Any]:
        if self.current_round >= len(self.allocation):
            return {}

        n, r = self.allocation[self.current_round]

        n_trials = min(n, self.n_trials - len(self.completed_trials))
        if n_trials <= 0:
            return {}

        for _ in range(n_trials):
            params = {
                name: space.sample(self.rng)
                for name, space in self.search_space.items()
            }
            trial = HyperbandTrial(
                trial_id=self.trial_counter,
                params=params,
                bracket_id=0,
                round_id=self.current_round,
                resource=r,
            )
            self.active_trials[self.trial_counter] = trial
            self.trial_counter += 1
            return params

        return {}

    def observe(
        self, trial: Trial, metrics: dict[str, float]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)

        if trial.trial_id not in self.active_trials:
            return [], []

        hb_trial = self.active_trials[trial.trial_id]
        hb_trial.performance = metrics.get("loss", float("inf"))
        hb_trial.status = "completed"

        del self.active_trials[trial.trial_id]
        self.completed_trials.append(hb_trial)

        self._advance_round()

        return [], []

    def _advance_round(self) -> None:
        if len(self.completed_trials) >= self.allocation[self.current_round][0]:
            self.current_round += 1

    def get_best_trial(self) -> Trial | None:
        if not self.completed_trials:
            return None

        sorted_trials = sorted(
            self.completed_trials,
            key=lambda t: t.performance,
            reverse=not self.minimize,
        )
        best = sorted_trials[0]

        return Trial(
            trial_id=best.trial_id,
            params=best.params,
            metrics={"loss": best.performance},
        )


class SuccessiveHalving:
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        objective_fn: Callable[[dict[str, Any], int], float],
        n_trials: int = 81,
        min_resource: int = 1,
        max_resource: int = 81,
        reduction_factor: float = 3.0,
        storage: ResultStorage | None = None,
        seed: int | None = None,
        minimize: bool = True,
    ):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.storage = storage or ResultStorage()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.minimize = minimize

        self.s_max = int(math.log(max_resource / min_resource, reduction_factor))
        self.trial_counter = 0
        self.active_trials: dict[int, HyperbandTrial] = {}
        self.completed_trials: list[HyperbandTrial] = []
        self.current_round = 0

    def suggest(self) -> dict[str, Any]:
        if self.current_round > self.s_max:
            return {}

        n = self.n_trials // (self.reduction_factor**self.current_round)
        r = self.min_resource * (self.reduction_factor**self.current_round)

        if len(self.completed_trials) + len(self.active_trials) >= self.n_trials:
            return {}

        params = {
            name: space.sample(self.rng) for name, space in self.search_space.items()
        }
        trial = HyperbandTrial(
            trial_id=self.trial_counter,
            params=params,
            bracket_id=0,
            round_id=self.current_round,
            resource=r,
        )
        self.active_trials[self.trial_counter] = trial
        self.trial_counter += 1

        return params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> bool:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)

        if trial.trial_id not in self.active_trials:
            return False

        hb_trial = self.active_trials[trial.trial_id]
        hb_trial.performance = metrics.get("loss", float("inf"))
        hb_trial.status = "completed"

        del self.active_trials[trial.trial_id]
        self.completed_trials.append(hb_trial)

        n_current = self.n_trials // (self.reduction_factor**self.current_round)
        if len(self.completed_trials) >= n_current:
            self.current_round += 1

        return True

    def get_best_trial(self) -> Trial | None:
        if not self.completed_trials:
            return None

        sorted_trials = sorted(
            self.completed_trials,
            key=lambda t: t.performance,
            reverse=not self.minimize,
        )
        best = sorted_trials[0]

        return Trial(
            trial_id=best.trial_id,
            params=best.params,
            metrics={"loss": best.performance},
        )
