from __future__ import annotations

import json
import math
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import numpy as np
import torch

T = TypeVar("T")
MetricType = TypeVar("MetricType", bound=float)


class ParameterType(Enum):
    UNIFORM = "uniform"
    LOGUNIFORM = "loguniform"
    CHOICE = "choice"
    CATEGORICAL = "categorical"


@dataclass
class SearchSpace:
    name: str
    param_type: ParameterType
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    default: Any = None

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        if rng is None:
            rng = np.random.default_rng()

        if self.param_type == ParameterType.UNIFORM:
            return rng.uniform(self.low, self.high)
        elif self.param_type == ParameterType.LOGUNIFORM:
            log_low = math.log(self.low) if self.low else -7.0
            log_high = math.log(self.high) if self.high else 7.0
            return math.exp(rng.uniform(log_low, log_high))
        elif self.param_type in (ParameterType.CHOICE, ParameterType.CATEGORICAL):
            return rng.choice(self.choices)
        raise ValueError(f"Unknown parameter type: {self.param_type}")

    def transform(self, value: float) -> Any:
        if self.param_type == ParameterType.UNIFORM:
            return value * (self.high - self.low) + self.low
        elif self.param_type == ParameterType.LOGUNIFORM:
            log_low = math.log(self.low) if self.low else -7.0
            log_high = math.log(self.high) if self.high else 7.0
            return math.exp(value * (log_high - log_low) + log_low)
        elif self.param_type in (ParameterType.CHOICE, ParameterType.CATEGORICAL):
            idx = int(value * (len(self.choices) - 1))
            return self.choices[min(idx, len(self.choices) - 1)]
        return value


def uniform(name: str, low: float, high: float, default: float | None = None) -> SearchSpace:
    return SearchSpace(name, ParameterType.UNIFORM, low, high, default=default or (low + high) / 2)


def loguniform(name: str, low: float, high: float, default: float | None = None) -> SearchSpace:
    return SearchSpace(name, ParameterType.LOGUNIFORM, low, high, default=default or math.sqrt(low * high))


def choice(name: str, choices: list[Any], default: Any | None = None) -> SearchSpace:
    return SearchSpace(name, ParameterType.CHOICE, choices=choices, default=default or choices[0])


def categorical(name: str, choices: list[str], default: str | None = None) -> SearchSpace:
    return SearchSpace(name, ParameterType.CATEGORICAL, choices=choices, default=default or choices[0])


@dataclass
class Trial:
    trial_id: int
    params: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    intermediate_results: list[dict[str, float]] = field(default_factory=list)

    def complete(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        self.status = "completed"
        self.end_time = time.time()

    def should_prune(self, grace_period: float = 60.0) -> bool:
        if self.status != "running":
            return False
        return time.time() - self.start_time < grace_period

    @property
    def duration(self) -> float | None:
        if self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class TrialResult:
    trial: Trial
    objective_value: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TrialStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"
    FAILED = "failed"


class Optimizer(ABC):
    @abstractmethod
    def suggest(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        pass

    def get_best_trial(self) -> Trial | None:
        return None

    def get_results(self) -> list[TrialResult]:
        return []


class ResultStorage:
    def __init__(self, storage_path: str | Path | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.trials: list[Trial] = []
        self.trial_results: list[TrialResult] = []
        self._objective_fn: Callable[[dict[str, Any]], float] | None = None

    def set_objective(self, fn: Callable[[dict[str, Any]], float]) -> None:
        self._objective_fn = fn

    def add_trial(self, trial: Trial) -> None:
        self.trials.append(trial)

    def complete_trial(self, trial: Trial, metrics: dict[str, float]) -> None:
        trial.complete(metrics)
        if self._objective_fn:
            obj_value = self._objective_fn(trial.params)
            self.trial_results.append(TrialResult(trial, obj_value, {"metrics": metrics}))

    def get_best(self, metric: str = "loss", minimize: bool = True) -> Trial | None:
        completed = [t for t in self.trials if t.status == "completed" and metric in t.metrics]
        if not completed:
            return None
        return min(completed, key=lambda t: t.metrics[metric]) if minimize else max(completed, key=lambda t: t.metrics[metric])

    def get_all_completed(self) -> list[Trial]:
        return [t for t in self.trials if t.status == "completed"]

    def save(self) -> None:
        if not self.storage_path:
            return
        data = {
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "params": t.params,
                    "metrics": t.metrics,
                    "status": t.status,
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                }
                for t in self.trials
            ]
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        for t_data in data.get("trials", []):
            trial = Trial(
                trial_id=t_data["trial_id"],
                params=t_data["params"],
                metrics=t_data.get("metrics", {}),
                status=t_data.get("status", "completed"),
                start_time=t_data.get("start_time", time.time()),
                end_time=t_data.get("end_time"),
            )
            self.trials.append(trial)


class GridSearch(Optimizer):
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        storage: ResultStorage | None = None,
    ):
        self.search_space = search_space
        self.storage = storage or ResultStorage()
        self.trial_counter = 0
        self.grid_points: list[dict[str, Any]] = []
        self.current_idx = 0
        self._generate_grid()

    def _generate_grid(self) -> None:
        param_names = list(self.search_space.keys())
        space_objects = [self.search_space[name] for name in param_names]

        def _grid_recursive(idx: int, current: dict[str, Any]) -> None:
            if idx == len(param_names):
                self.grid_points.append(current.copy())
                return

            param = space_objects[idx]
            if param.param_type in (ParameterType.CHOICE, ParameterType.CATEGORICAL):
                for choice in param.choices:
                    current[param_names[idx]] = choice
                    _grid_recursive(idx + 1, current)
            else:
                steps = 10
                for i in range(steps):
                    value = i / (steps - 1)
                    current[param_names[idx]] = param.transform(value)
                    _grid_recursive(idx + 1, current)

        _grid_recursive(0, {})

    def suggest(self) -> dict[str, Any]:
        if self.current_idx >= len(self.grid_points):
            return {}
        params = self.grid_points[self.current_idx]
        self.current_idx += 1
        return params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)


class RandomSearch(Optimizer):
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        n_trials: int = 100,
        storage: ResultStorage | None = None,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.storage = storage or ResultStorage()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trial_counter = 0
        self.completed_trials: list[Trial] = []

    def suggest(self) -> dict[str, Any]:
        if self.trial_counter >= self.n_trials:
            return {}
        params = {name: space.sample(self.rng) for name, space in self.search_space.items()}
        self.trial_counter += 1
        return params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)
        self.completed_trials.append(trial)

    def get_best_trial(self) -> Trial | None:
        return self.storage.get_best()


class GaussianProcess:
    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise_variance: float = 1e-5,
    ):
        self.length_scale = length_scale
        self.variance = variance
        self.noise_variance = noise_variance
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        return self.variance * np.exp(-0.5 / self.length_scale**2 * dists)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = np.array(X)
        self.y_train = np.array(y).reshape(-1, 1)

    def predict(self, X: np.ndarray, return_std: bool = False) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X)) * np.sqrt(self.variance)

        K = self._rbf_kernel(self.X_train, self.X_train) + self.noise_variance * np.eye(len(self.X_train))
        K_star = self._rbf_kernel(self.X_train, X)
        K_star_star = self._rbf_kernel(X, X)

        try:
            K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)

        mu = K_star.T @ K_inv @ self.y_train
        cov = K_star_star - K_star.T @ K_inv @ K_star

        mu = mu.flatten()
        if return_std:
            var = np.diag(cov)
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            return mu, std
        return mu, np.zeros(len(X))


class BayesianOptimization(Optimizer):
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        n_trials: int = 100,
        storage: ResultStorage | None = None,
        seed: int | None = None,
        acquisition: str = "ei",
        random_fraction: float = 0.2,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.storage = storage or ResultStorage()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.acquisition = acquisition
        self.random_fraction = random_fraction
        self.trial_counter = 0
        self.completed_trials: list[Trial] = []
        self.gp = GaussianProcess()
        self._initialized = False

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
                    log_low = math.log(space.low) if space.low else -7.0
                    log_high = math.log(space.high) if space.high else 7.0
                    val = (math.log(val) - log_low) / (log_high - log_low)
                else:
                    val = (val - space.low) / (space.high - space.low) if space.high != space.low else 0.5
                vec.append(val)
        return np.array(vec)

    def _vector_to_space(self, vec: np.ndarray) -> dict[str, Any]:
        params = {}
        for i, name in enumerate(sorted(self.search_space.keys())):
            space = self.search_space[name]
            val = vec[i]
            params[name] = space.transform(val)
        return params

    def _acquisition_ei(self, mu: np.ndarray, std: np.ndarray, best_y: float) -> np.ndarray:
        ei = np.zeros_like(mu)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best_y - 0.01) / (std + 1e-8)
            ei = (mu - best_y - 0.01) * self._norm_cdf(z) + std * self._norm_pdf(z)
            ei[std < 1e-8] = 0.0
        return ei

    def _acquisition_ucb(self, mu: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        return mu + kappa * std

    def _norm_cdf(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    def _norm_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def suggest(self) -> dict[str, Any]:
        if self.trial_counter >= self.n_trials:
            return {}

        if not self._initialized or self.trial_counter < int(self.n_trials * self.random_fraction):
            params = {name: space.sample(self.rng) for name, space in self.search_space.items()}
        else:
            best_y = min(t.metrics.get("loss", float("inf")) for t in self.completed_trials)

            candidates = []
            for _ in range(1000):
                cand = self._space_to_vector({name: space.sample(self.rng) for name, space in self.search_space.items()})
                candidates.append(cand)
            candidates = np.array(candidates)

            mu, std = self.gp.predict(candidates, return_std=True)

            if self.acquisition == "ei":
                acq = self._acquisition_ei(mu, std, best_y)
            else:
                acq = self._acquisition_ucb(mu, std)

            best_idx = np.argmax(acq)
            params = self._vector_to_space(candidates[best_idx])

        self.trial_counter += 1
        return params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)
        self.completed_trials.append(trial)

        if len(self.completed_trials) >= 2:
            X = np.array([self._space_to_vector(t.params) for t in self.completed_trials])
            y = np.array([t.metrics.get("loss", 0.0) for t in self.completed_trials])
            self.gp.fit(X, y)
            self._initialized = True

    def get_best_trial(self) -> Trial | None:
        return self.storage.get_best()


@dataclass
class HyperbandBracket:
    bracket_id: int
    n: int
    r: int
    eta: float = 3.0

    def get_resource分配(self) -> list[tuple[int, int]]:
        s_max = int(math.log(self.n, self.eta))
        return [(self.n // (self.eta**i), self.r * (self.eta**i)) for i in range(s_max + 1)]


class Hyperband(Optimizer):
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        objective_fn: Callable[[dict[str, Any], int], float],
        n_trials: int = 81,
        max_resource: int = 81,
        eta: float = 3.0,
        storage: ResultStorage | None = None,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.max_resource = max_resource
        self.eta = eta
        self.storage = storage or ResultStorage()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trial_counter = 0
        self.bracket = HyperbandBracket(0, n_trials, max_resource, eta)
        self.active_trials: dict[int, Trial] = {}
        self.current_round = 0
        self.round_trials: list[Tuple[Trial, int]] = []

    def suggest(self) -> dict[str, Any]:
        if self.trial_counter >= self.n_trials:
            return {}

        params = {name: space.sample(self.rng) for name, space in self.search_space.items()}
        self.trial_counter += 1
        return params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)

        if trial.trial_id in self.active_trials:
            del self.active_trials[trial.trial_id]


@dataclass
class PopulationMember:
    trial_id: int
    params: dict[str, Any]
    performance: float = float("-inf")
    step: int = 0


class PopulationBasedTraining(Optimizer):
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        population_size: int = 20,
        n_generations: int = 50,
        objective_fn: Callable[[dict[str, Any]], float],
        exploit_fraction: float = 0.25,
        explore_fraction: float = 0.25,
        storage: ResultStorage | None = None,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.objective_fn = objective_fn
        self.exploit_fraction = exploit_fraction
        self.explore_fraction = explore_fraction
        self.storage = storage or ResultStorage()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.population: list[PopulationMember] = []
        self.generation = 0
        self.trial_counter = 0

    def _init_population(self) -> None:
        for i in range(self.population_size):
            params = {name: space.sample(self.rng) for name, space in self.search_space.items()}
            self.population.append(PopulationMember(i, params))

    def _mutate(self, member: PopulationMember) -> dict[str, Any]:
        new_params = member.params.copy()
        for name, space in self.search_space.items():
            if self.rng.random() < self.explore_fraction:
                if space.param_type in (ParameterType.CHOICE, ParameterType.CATEGORICAL):
                    new_params[name] = space.sample(self.rng)
                else:
                    mutation_scale = 0.2
                    current_val = new_params[name]
                    low = space.low if space.low else 1e-7
                    high = space.high if space.high else 1e7
                    new_val = current_val + self.rng.normal(0, mutation_scale * (high - low))
                    new_val = max(low, min(high, new_val))
                    new_params[name] = new_val
        return new_params

    def _exploit_and_explore(self) -> None:
        sorted_pop = sorted(self.population, key=lambda m: m.performance, reverse=True)
        n_exploit = int(self.population_size * self.exploit_fraction)
        n_explore = int(self.population_size * self.explore_fraction)

        for i in range(n_exploit, self.population_size):
            if i < n_exploit + n_explore:
                parent = sorted_pop[self.rng.integers(0, n_exploit)]
                new_params = self._mutate(parent)
                sorted_pop[i].params = new_params
                sorted_pop[i].performance = float("-inf")
                sorted_pop[i].step = 0

        self.population = sorted_pop

    def suggest(self) -> dict[str, Any]:
        if not self.population:
            self._init_population()

        if self.generation >= self.n_generations:
            return {}

        member = self.population[self.trial_counter % self.population_size]
        self.trial_counter += 1

        if self.trial_counter % self.population_size == 0:
            self._exploit_and_explore()
            self.generation += 1

        return member.params

    def observe(self, trial: Trial, metrics: dict[str, float]) -> None:
        loss = metrics.get("loss", float("inf"))
        for member in self.population:
            if member.trial_id == trial.trial_id:
                member.performance = loss
                break
        else:
            member = PopulationMember(trial.trial_id, trial.params, loss)
            self.population.append(member)

        self.storage.add_trial(trial)
        self.storage.complete_trial(trial, metrics)

    def get_best_trial(self) -> Trial | None:
        return self.storage.get_best()


class ResultAnalyzer:
    def __init__(self, storage: ResultStorage):
        self.storage = storage

    def get_statistics(self) -> dict[str, Any]:
        completed = self.storage.get_all_completed()
        if not completed:
            return {}

        all_metrics = defaultdict(list)
        for trial in completed:
            for key, val in trial.metrics.items():
                all_metrics[key].append(val)

        return {
            "n_completed": len(completed),
            "n_total": len(self.storage.trials),
            "metrics": {k: {"mean": np.mean(v), "std": np.std(v), "min": np.min(v), "max": np.max(v)} for k, v in all_metrics.items()},
        }

    def get_param_importance(self, metric: str = "loss") -> dict[str, float]:
        completed = self.storage.get_all_completed()
        if len(completed) < 2:
            return {}

        param_values: dict[str, list[Any]] = {}
        metric_values = []

        for trial in completed:
            if metric in trial.metrics:
                for key, val in trial.params.items():
                    if key not in param_values:
                        param_values[key] = []
                    param_values[key].append(val)
                metric_values.append(trial.metrics[metric])

        if not metric_values:
            return {}

        importance = {}
        metric_arr = np.array(metric_values)
        for param, values in param_values.items():
            try:
                arr = np.array(values)
                if len(arr.shape) == 1:
                    corr = np.corrcoef(arr, metric_arr)[0, 1]
                    importance[param] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                importance[param] = 0.0

        return importance


class HyperoptRunner:
    def __init__(
        self,
        optimizer: Optimizer,
        objective_fn: Callable[[dict[str, Any]], dict[str, float]],
    ):
        self.optimizer = optimizer
        self.objective_fn = objective_fn

    def run(self, callbacks: list[Callable[[Trial], None]] | None = None) -> Trial:
        params = self.optimizer.suggest()
        if not params:
            return None

        trial = Trial(self.optimizer.trial_counter if hasattr(self.optimizer, "trial_counter") else 0, params)
        trial.status = "running"

        try:
            metrics = self.objective_fn(params)
            trial.complete(metrics)
        except Exception as e:
            trial.status = "failed"

        self.optimizer.observe(trial, trial.metrics)

        if callbacks:
            for cb in callbacks:
                cb(trial)

        return trial

    def run_all(self, n_trials: int | None = None, callbacks: list[Callable[[Trial], None]] | None = None) -> list[Trial]:
        trials = []
        max_trials = n_trials or float("inf")

        while len(trials) < max_trials:
            trial = self.run(callbacks)
            if trial is None:
                break
            trials.append(trial)

        return trials
