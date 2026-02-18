from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ParetoPoint:
    params: dict[str, Any]
    objectives: list[float]
    dominated: bool = False
    crowding_distance: float = 0.0


class ParetoFront:
    def __init__(
        self, minimize: list[bool] | None, reference_point: list[float] | None = None
    ):
        self.minimize = minimize or [True] * 2
        self.reference_point = reference_point
        self.points: list[ParetoPoint] = []

    def dominates(self, objectives1: list[float], objectives2: list[float]) -> bool:
        all_better_or_equal = True
        strictly_better = False

        for o1, o2, minimize in zip(objectives1, objectives2, self.minimize):
            if minimize:
                if o1 > o2:
                    all_better_or_equal = False
                    break
                if o1 < o2:
                    strictly_better = True
            else:
                if o1 < o2:
                    all_better_or_equal = False
                    break
                if o1 > o2:
                    strictly_better = True

        return all_better_or_equal and strictly_better

    def add(self, point: ParetoPoint) -> None:
        for existing in self.points[:]:
            if self.dominates(point.objectives, existing.objectives):
                existing.dominated = True
                self.points.remove(existing)

        if not any(self.dominates(p.objectives, point.objectives) for p in self.points):
            self.points.append(point)

    def get_pareto_front(self) -> list[ParetoPoint]:
        return [p for p in self.points if not p.dominated]

    def get_best_objective(self, objective_idx: int) -> ParetoPoint | None:
        front = self.get_pareto_front()
        if not front:
            return None

        if self.minimize[objective_idx]:
            return min(front, key=lambda p: p.objectives[objective_idx])
        return max(front, key=lambda p: p.objectives[objective_idx])

    def calculate_crowding_distance(self) -> None:
        front = self.get_pareto_front()
        if len(front) <= 2:
            for p in front:
                p.crowding_distance = float("inf")
            return

        for point in front:
            point.crowding_distance = 0.0

        for obj_idx in range(len(self.minimize)):
            sorted_front = sorted(front, key=lambda p: p.objectives[obj_idx])
            sorted_front[0].crowding_distance = float("inf")
            sorted_front[-1].crowding_distance = float("inf")

            obj_range = (
                sorted_front[-1].objectives[obj_idx]
                - sorted_front[0].objectives[obj_idx]
            )
            if obj_range == 0:
                continue

            for i in range(1, len(sorted_front) - 1):
                sorted_front[i].crowding_distance += (
                    sorted_front[i + 1].objectives[obj_idx]
                    - sorted_front[i - 1].objectives[obj_idx]
                ) / obj_range


@dataclass
class MultiObjectiveOptimizer:
    search_space: dict[str, Any]
    objectives: list[str]
    minimize: list[bool]
    n_trials: int = 100
    seed: int | None = None

    def __post_init__(self):
        self.seed = self.seed if self.seed is not None else 42
        self.rng = np.random.default_rng(self.seed)
        self.trial_counter = 0
        self.all_trials: list[ParetoPoint] = []
        self.pareto_front = ParetoFront(self.minimize)

    def suggest(self) -> dict[str, Any]:
        if self.trial_counter >= self.n_trials:
            return {}

        params = {
            name: space.sample(self.rng) for name, space in self.search_space.items()
        }
        self.trial_counter += 1
        return params

    def observe(self, params: dict[str, Any], objectives: list[float]) -> None:
        point = ParetoPoint(params=params, objectives=objectives)
        self.all_trials.append(point)
        self.pareto_front.add(point)

    def get_pareto_front(self) -> list[dict[str, Any]]:
        self.pareto_front.calculate_crowding_distance()
        front = self.pareto_front.get_pareto_front()
        return [
            {
                "params": p.params,
                "objectives": p.objectives,
                "crowding": p.crowding_distance,
            }
            for p in front
        ]


class NSGA2:
    def __init__(
        self,
        search_space: dict[str, Any],
        objectives: list[str],
        minimize: list[bool],
        population_size: int = 50,
        n_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.objectives = objectives
        self.minimize = minimize
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.population: list[ParetoPoint] = []
        self.generation = 0

    def _initialize_population(self) -> None:
        for _ in range(self.population_size):
            params = {
                name: space.sample(self.rng)
                for name, space in self.search_space.items()
            }
            self.population.append(ParetoPoint(params=params, objectives=[]))

    def _evaluate(self, params: dict[str, Any]) -> list[float]:
        raise NotImplementedError("Subclasses must implement _evaluate")

    def _fast_non_dominated_sort(self) -> list[list[ParetoPoint]]:
        fronts: list[list[ParetoPoint]] = [[]]
        domination_count: dict[int, int] = {}
        dominated_sets: dict[int, list[ParetoPoint]] = {}

        for i, p in enumerate(self.population):
            dominated_sets[i] = []
            domination_count[i] = 0

            for j, q in enumerate(self.population):
                if i == j:
                    continue
                if self._dominates(p.objectives, q.objectives):
                    dominated_sets[i].append(q)
                elif self._dominates(q.objectives, p.objectives):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                p.dominated = False
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front: list[ParetoPoint] = []
            for p in fronts[i]:
                for q in dominated_sets[self.population.index(p)]:
                    domination_count[self.population.index(q)] -= 1
                    if domination_count[self.population.index(q)] == 0:
                        q.dominated = False
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _dominates(self, objectives1: list[float], objectives2: list[float]) -> bool:
        all_better_or_equal = True
        strictly_better = False

        for o1, o2, minimize in zip(objectives1, objectives2, self.minimize):
            if minimize:
                if o1 > o2:
                    all_better_or_equal = False
                    break
                if o1 < o2:
                    strictly_better = True
            else:
                if o1 < o2:
                    all_better_or_equal = False
                    break
                if o1 > o2:
                    strictly_better = True

        return all_better_or_equal and strictly_better

    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        if self.rng.random() > self.crossover_prob:
            return parent1.copy()

        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if self.rng.random() < 0.5 else parent2[key]
        return child

    def _mutate(self, params: dict) -> dict:
        for name, space in self.search_space.items():
            if self.rng.random() < self.mutation_prob:
                params[name] = space.sample(self.rng)
        return params

    def run(self, evaluate_fn: callable) -> list[dict[str, Any]]:
        self._initialize_population()

        for point in self.population:
            point.objectives = evaluate_fn(point.params)

        for _ in range(self.n_generations):
            fronts = self._fast_non_dominated_sort()

            new_population: list[ParetoPoint] = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    remaining = self.population_size - len(new_population)
                    front.sort(key=lambda p: p.crowding_distance, reverse=True)
                    new_population.extend(front[:remaining])
                    break

            while len(new_population) < self.population_size:
                parent1, parent2 = self.rng.choice(new_population, 2, replace=False)
                child_params = self._crossover(parent1.params, parent2.params)
                child_params = self._mutate(child_params)
                child = ParetoPoint(
                    params=child_params, objectives=evaluate_fn(child_params)
                )
                new_population.append(child)

            self.population = new_population

        pareto = ParetoFront(self.minimize)
        for point in self.population:
            pareto.add(point)
        pareto.calculate_crowding_distance()

        return [
            {"params": p.params, "objectives": p.objectives}
            for p in pareto.get_pareto_front()
        ]
