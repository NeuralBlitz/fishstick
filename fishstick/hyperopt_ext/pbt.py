from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .search_space import ParameterType, SearchSpace
from .trial import ResultStorage, Trial


@dataclass
class PopulationMember:
    trial_id: int
    params: dict[str, Any]
    performance: float = float("-inf")
    step: int = 0
    model_state: dict[str, Any] | None = None
    history: list[float] = field(default_factory=list)


class PopulationBasedTraining:
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        population_size: int = 20,
        n_generations: int = 50,
        objective_fn: Callable[[dict[str, Any]], float] | None = None,
        exploit_fraction: float = 0.25,
        explore_fraction: float = 0.25,
        mutation_scale: float = 0.2,
        storage: ResultStorage | None = None,
        seed: int | None = None,
        minimize: bool = True,
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.objective_fn = objective_fn
        self.exploit_fraction = exploit_fraction
        self.explore_fraction = explore_fraction
        self.mutation_scale = mutation_scale
        self.storage = storage or ResultStorage()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.minimize = minimize

        self.population: list[PopulationMember] = []
        self.generation = 0
        self.trial_counter = 0

    def _init_population(self) -> None:
        for i in range(self.population_size):
            params = {
                name: space.sample(self.rng)
                for name, space in self.search_space.items()
            }
            self.population.append(PopulationMember(i, params))

    def _mutate(self, member: PopulationMember) -> dict[str, Any]:
        new_params = member.params.copy()
        for name, space in self.search_space.items():
            if self.rng.random() < self.explore_fraction:
                if space.param_type in (
                    ParameterType.CHOICE,
                    ParameterType.CATEGORICAL,
                ):
                    new_params[name] = space.sample(self.rng)
                else:
                    current_val = new_params[name]
                    low = space.low if space.low else 1e-7
                    high = space.high if space.high else 1e7
                    new_val = current_val + self.rng.normal(
                        0, self.mutation_scale * (high - low)
                    )
                    new_val = max(low, min(high, new_val))
                    new_params[name] = new_val
        return new_params

    def _exploit_and_explore(self) -> None:
        sorted_pop = sorted(
            self.population, key=lambda m: m.performance, reverse=not self.minimize
        )
        n_exploit = int(self.population_size * self.exploit_fraction)

        top_performers = sorted_pop[:n_exploit]
        bottom_performers = sorted_pop[n_exploit:]

        for member in bottom_performers:
            donor = self.rng.choice(top_performers)
            new_params = self._mutate(
                PopulationMember(-1, donor.params, donor.performance)
            )
            member.params = new_params
            member.performance = float("-inf")
            member.history = []

    def suggest(self) -> dict[str, Any]:
        if not self.population:
            self._init_population()

        available = [m for m in self.population if m.performance == float("-inf")]
        if not available:
            if self.generation < self.n_generations:
                self._exploit_and_explore()
                self.generation += 1
                available = [
                    m for m in self.population if m.performance == float("-inf")
                ]
            else:
                return {}

        if not available:
            return {}

        member = available[0]
        return member.params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)

        performance = metrics.get("loss", 0.0)
        if not self.minimize:
            performance = -performance

        for member in self.population:
            if member.params == trial.params:
                member.performance = performance
                member.history.append(performance)
                break

    def get_best_member(self) -> PopulationMember | None:
        if not self.population:
            return None
        return (
            max(self.population, key=lambda m: m.performance)
            if not self.minimize
            else min(self.population, key=lambda m: m.performance)
        )

    def get_best_trial(self) -> Trial | None:
        best = self.get_best_member()
        if best is None:
            return None
        return Trial(
            trial_id=best.trial_id,
            params=best.params,
            metrics={"loss": best.performance},
        )


class PBTWithModelState(PopulationBasedTraining):
    def __init__(self, model_getter: Callable[[], Any] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_getter = model_getter

    def get_model_state(self, member: PopulationMember) -> dict[str, Any] | None:
        return member.model_state

    def set_model_state(self, member: PopulationMember, state: dict[str, Any]) -> None:
        member.model_state = state


class AsyncPBT(PopulationBasedTraining):
    def __init__(self, n_workers: int = 4, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_workers = n_workers

    def _exploit_and_explore(self) -> None:
        sorted_pop = sorted(
            self.population, key=lambda m: m.performance, reverse=not self.minimize
        )
        n_exploit = int(self.population_size * self.exploit_fraction)
        n_remain = self.population_size - n_exploit

        for i in range(n_remain):
            donor_idx = self.rng.integers(0, n_exploit)
            donor = sorted_pop[donor_idx]
            target = sorted_pop[n_exploit + i]

            if self.rng.random() < 0.5:
                target.params = donor.params.copy()
            else:
                target.params = self._mutate(donor)

            target.performance = float("-inf")
            target.history = []
