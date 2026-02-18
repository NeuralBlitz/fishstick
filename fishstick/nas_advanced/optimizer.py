import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy
import random


class Architecture:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fitness: Optional[float] = None

    def mutate(self, mutation_rate: float = 0.1) -> "Architecture":
        new_config = deepcopy(self.config)
        for key in new_config:
            if isinstance(new_config[key], (int, float)):
                if random.random() < mutation_rate:
                    new_config[key] += random.uniform(-0.1, 0.1) * new_config[key]
            elif isinstance(new_config[key], list):
                if random.random() < mutation_rate:
                    idx = random.randint(0, len(new_config[key]) - 1)
                    new_config[key][idx] = random.choice(new_config[key])
        return Architecture(new_config)

    def crossover(self, other: "Architecture") -> Tuple["Architecture", "Architecture"]:
        config1 = {}
        config2 = {}
        for key in self.config:
            if random.random() < 0.5:
                config1[key] = self.config[key]
                config2[key] = other.config[key]
            else:
                config1[key] = other.config[key]
                config2[key] = self.config[key]
        return Architecture(config1), Architecture(config2)


class EvolutionarySearch:
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        fitness_fn: Callable[[Architecture], float],
        population_size: int = 50,
        generations: int = 100,
        elite_size: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
    ):
        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _create_random_architecture(self) -> Architecture:
        config = {}
        for key, values in self.search_space.items():
            config[key] = random.choice(values)
        return Architecture(config)

    def _initialize_population(self) -> List[Architecture]:
        return [self._create_random_architecture() for _ in range(self.population_size)]

    def _select_parents(self, population: List[Architecture]) -> List[Architecture]:
        population.sort(key=lambda x: x.fitness, reverse=True)
        return population[: self.elite_size]

    def _create_offspring(
        self,
        parents: List[Architecture],
        offspring_size: int,
    ) -> List[Architecture]:
        offspring = []
        while len(offspring) < offspring_size:
            if len(parents) >= 2 and random.random() < self.crossover_rate:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = parent1.crossover(parent2)
                offspring.append(child1.mutate(self.mutation_rate))
                if len(offspring) < offspring_size:
                    offspring.append(child2.mutate(self.mutation_rate))
            else:
                parent = random.choice(parents)
                offspring.append(parent.mutate(self.mutation_rate))
        return offspring

    def search(self) -> Tuple[Architecture, List[float]]:
        population = self._initialize_population()
        history = []

        for arch in population:
            arch.fitness = self.fitness_fn(arch)

        for gen in range(self.generations):
            population.sort(key=lambda x: x.fitness, reverse=True)
            history.append(population[0].fitness)

            print(f"Generation {gen + 1}, Best Fitness: {population[0].fitness:.4f}")

            parents = self._select_parents(population)
            offspring_size = self.population_size - self.elite_size
            offspring = self._create_offspring(parents, offspring_size)

            for arch in offspring:
                arch.fitness = self.fitness_fn(arch)

            population = parents + offspring[: self.population_size - len(parents)]

        population.sort(key=lambda x: x.fitness, reverse=True)
        return population[0], history


class RandomSearch:
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        fitness_fn: Callable[[Architecture], float],
        num_samples: int = 100,
    ):
        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.num_samples = num_samples

    def _create_random_architecture(self) -> Architecture:
        config = {}
        for key, values in self.search_space.items():
            config[key] = random.choice(values)
        return Architecture(config)

    def search(self) -> Tuple[Architecture, List[float]]:
        best_arch = None
        best_fitness = float("-inf")
        history = []

        for i in range(self.num_samples):
            arch = self._create_random_architecture()
            arch.fitness = self.fitness_fn(arch)

            history.append(arch.fitness)

            if arch.fitness > best_fitness:
                best_fitness = arch.fitness
                best_arch = arch

            if (i + 1) % 10 == 0:
                print(
                    f"Sample {i + 1}/{self.num_samples}, Best Fitness: {best_fitness:.4f}"
                )

        return best_arch, history


class BayesianOptimizer:
    def __init__(
        self,
        search_space: Dict[str, Any],
        fitness_fn: Callable[[Dict[str, Any]], float],
        n_initial_points: int = 5,
        acquisition: str = "ei",
    ):
        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.n_initial_points = n_initial_points
        self.acquisition = acquisition
        self.X_observed: List[List[float]] = []
        self.y_observed: List[float] = []
        self.dimensions = len(search_space)

    def _sample_random_config(self) -> Dict[str, Any]:
        config = {}
        for key, values in self.search_space.items():
            config[key] = random.choice(values)
        return config

    def _config_to_vector(self, config: Dict[str, Any]) -> List[float]:
        return [float(v) for v in config.values()]

    def _vector_to_config(self, vector: List[float]) -> Dict[str, Any]:
        config = {}
        keys = list(self.search_space.keys())
        for i, key in enumerate(keys):
            values = self.search_space[key]
            idx = int(vector[i] * (len(values) - 1))
            config[key] = values[idx]
        return config

    def _compute_acquisition(self, X_candidate: float, y_best: float) -> float:
        if len(self.y_observed) == 0:
            return 1.0

        mu = np.mean(self.y_observed)
        sigma = np.std(self.y_observed) + 1e-6

        if self.acquisition == "ei":
            z = (y_best - mu) / sigma
            ei = (y_best - mu) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
            return ei
        elif self.acquisition == "ucb":
            return mu + 2.0 * sigma
        else:
            return -mu

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    def _norm_pdf(self, x: float) -> float:
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def _get_best_candidate(self) -> Dict[str, Any]:
        candidates = []
        for _ in range(100):
            config = self._sample_random_config()
            candidates.append(config)

        if len(self.y_observed) > 0:
            y_best = max(self.y_observed)
        else:
            y_best = float("-inf")

        best_score = float("-inf")
        best_config = candidates[0]

        for config in candidates:
            vector = self._config_to_vector(config)
            score = self._compute_acquisition(0, y_best)
            if score > best_score:
                best_score = score
                best_config = config

        return best_config

    def _update_observations(self, config: Dict[str, Any]):
        vector = self._config_to_vector(config)
        self.X_observed.append(vector)
        fitness = self.fitness_fn(config)
        self.y_observed.append(fitness)

    def search(self, num_iterations: int = 50) -> Tuple[Dict[str, Any], List[float]]:
        history = []

        for _ in range(self.n_initial_points):
            config = self._sample_random_config()
            self._update_observations(config)
            print(f"Initial point, Fitness: {self.y_observed[-1]:.4f}")

        for i in range(num_iterations):
            candidate = self._get_best_candidate()
            self._update_observations(candidate)
            history.append(self.y_observed[-1])

            print(f"Iteration {i + 1}, Fitness: {self.y_observed[-1]:.4f}")

        best_idx = np.argmax(self.y_observed)
        best_config = self._vector_to_config(self.X_observed[best_idx])

        return best_config, history


class HyperbandSearch:
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        fitness_fn: Callable[[Architecture], float],
        max_iter: int = 27,
        eta: int = 3,
    ):
        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.max_iter = max_iter
        self.eta = eta

    def _get_configs(self, n: int) -> List[Architecture]:
        configs = []
        for _ in range(n):
            config = {}
            for key, values in self.search_space.items():
                config[key] = random.choice(values)
            configs.append(Architecture(config))
        return configs

    def _evaluate(self, configs: List[Architecture], budget: int) -> List[Architecture]:
        for config in configs:
            config.fitness = self.fitness_fn(config) / budget
        return configs

    def search(self) -> Tuple[Architecture, List[float]]:
        log_max_iter = int(np.log(self.max_iter) / np.log(self.eta))
        history = []
        best_arch = None
        best_fitness = float("-inf")

        for s in range(log_max_iter, -1, -1):
            n = int(np.ceil(self.max_iter / (self.eta**s)))
            r = self.max_iter * (self.eta ** (-s))

            configs = self._get_configs(n)

            for i in range(s + 1):
                n_configs = len(configs)
                r = int(r * self.eta)

                configs = self._evaluate(configs, r)

                configs.sort(key=lambda x: x.fitness, reverse=True)

                n_keep = int(n_configs / self.eta)
                configs = configs[:n_keep]

                for config in configs:
                    if config.fitness > best_fitness:
                        best_fitness = config.fitness
                        best_arch = config

                history.append(best_fitness)
                print(f"Bracket {s}, Round {i}, Best: {best_fitness:.4f}")

        return best_arch, history


class EfficientNAS:
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        fitness_fn: Callable[[Architecture], float],
        method: str = "evolutionary",
        **kwargs,
    ):
        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.method = method

        if method == "evolutionary":
            self.optimizer = EvolutionarySearch(search_space, fitness_fn, **kwargs)
        elif method == "random":
            self.optimizer = RandomSearch(search_space, fitness_fn, **kwargs)
        elif method == "bayesian":
            self.optimizer = BayesianOptimizer(search_space, fitness_fn, **kwargs)
        elif method == "hyperband":
            self.optimizer = HyperbandSearch(search_space, fitness_fn, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def search(self) -> Tuple[Architecture, List[float]]:
        return self.optimizer.search()
