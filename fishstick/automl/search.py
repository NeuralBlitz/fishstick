"""
AutoML Search Algorithms
"""

from typing import Dict, Any, Callable, Optional, List
import random
import numpy as np
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Trial:
    """Represents a single trial in the search."""

    config: Dict[str, Any]
    score: Optional[float] = None
    metrics: Optional[Dict] = None
    status: str = "pending"
    train_time: float = 0.0


class SearchSpace:
    """Hyperparameter search space."""

    def __init__(self):
        self.params = {}

    def sample(self) -> Dict[str, Any]:
        config = {}
        for name, param in self.params.items():
            config[name] = param.sample()
        return config


class Choice:
    """Categorical choice parameter."""

    def __init__(self, choices: List):
        self.choices = choices

    def sample(self) -> Any:
        return random.choice(self.choices)


class Uniform:
    """Uniform continuous parameter."""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self) -> float:
        return random.uniform(self.low, self.high)


class LogUniform:
    """Log-uniform continuous parameter."""

    def __init__(self, low: float, high: float):
        self.low = np.log(low)
        self.high = np.log(high)

    def sample(self) -> float:
        return np.exp(random.uniform(self.low, self.high))


class Conditional:
    """Conditional parameter."""

    def __init__(self, condition: Callable, value: Any):
        self.condition = condition
        self.value = value

    def sample(self, config: Dict) -> Any:
        if self.condition(config):
            if callable(self.value):
                return self.value.sample()
            return self.value
        return None


class NASearch:
    """Neural Architecture Search base class."""

    def __init__(
        self,
        space: SearchSpace,
        objective_fn: Callable[[Dict], float],
        maximize: bool = True,
    ):
        self.space = space
        self.objective_fn = objective_fn
        self.maximize = maximize
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None

    def search(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict:
        raise NotImplementedError

    def _evaluate(self, config: Dict) -> Trial:
        trial = Trial(config=config)
        try:
            score = self.objective_fn(config)
            trial.score = score
            trial.status = "completed"
        except Exception as e:
            trial.status = "failed"
            trial.metrics = {"error": str(e)}
        return trial


class RandomSearch(NASearch):
    """Random search for hyperparameter optimization."""

    def __init__(
        self,
        space: SearchSpace,
        objective_fn: Callable[[Dict], float],
        maximize: bool = True,
    ):
        super().__init__(space, objective_fn, maximize)

    def search(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict:
        for i in range(n_trials):
            config = self.space.sample()
            trial = self._evaluate(config)
            self.trials.append(trial)

            if (
                self.best_trial is None
                or (self.maximize and trial.score > self.best_trial.score)
                or (not self.maximize and trial.score < self.best_trial.score)
            ):
                self.best_trial = trial

            print(f"Trial {i + 1}/{n_trials}: score={trial.score:.4f}")

        return {
            "best_config": self.best_trial.config,
            "best_score": self.best_trial.score,
            "trials": self.trials,
        }


class GridSearch(NASearch):
    """Grid search for hyperparameter optimization."""

    def __init__(
        self,
        space: SearchSpace,
        objective_fn: Callable[[Dict], float],
        maximize: bool = True,
        points_per_dim: int = 3,
    ):
        super().__init__(space, objective_fn, maximize)
        self.points_per_dim = points_per_dim

    def search(
        self, n_trials: Optional[int] = None, timeout: Optional[int] = None
    ) -> Dict:
        configs = self._grid_configs(self.space.params, [])

        for i, config in enumerate(configs):
            trial = self._evaluate(config)
            self.trials.append(trial)

            if (
                self.best_trial is None
                or (self.maximize and trial.score > self.best_trial.score)
                or (not self.maximize and trial.score < self.best_trial.score)
            ):
                self.best_trial = trial

            print(f"Trial {i + 1}/{len(configs)}: score={trial.score:.4f}")

        return {
            "best_config": self.best_trial.config,
            "best_score": self.best_trial.score,
            "trials": self.trials,
        }

    def _grid_configs(self, params: Dict, prefix: List) -> List[Dict]:
        if not params:
            return [dict(zip(prefix, values)) for values in [[]]]

        first_key = list(params.keys())[0]
        param = params[first_key]
        rest = {k: v for k, v in params.items() if k != first_key}

        configs = []
        if isinstance(param, Choice):
            for value in param.choices:
                new_prefix = prefix + [first_key]
                for rest_config in self._grid_configs(rest, new_prefix):
                    rest_config[first_key] = value
                    configs.append(rest_config)
        else:
            for rest_config in self._grid_configs(rest, prefix):
                configs.append(rest_config)

        return configs


class Hyperband(NASearch):
    """Hyperband early stopping for efficient search."""

    def __init__(
        self,
        space: SearchSpace,
        objective_fn: Callable[[Dict, int], float],
        maximize: bool = True,
        max_iter: int = 81,
        eta: int = 3,
    ):
        super().__init__(space, objective_fn, maximize)
        self.max_iter = max_iter
        self.eta = eta
        self.n_brackets = int(np.log(max_iter) / np.log(eta)) + 1

    def search(
        self, n_trials: Optional[int] = None, timeout: Optional[int] = None
    ) -> Dict:
        for bracket in range(self.n_brackets):
            n = int(np.ceil(self.max_iter / self.eta**bracket))
            r = self.max_iter * self.eta ** (-bracket)

            bracket_trials = []
            for _ in range(n):
                config = self.space.sample()
                trial = Trial(config=config)
                bracket_trials.append(trial)

            for i in range(int(r)):
                for trial in bracket_trials:
                    if trial.status == "pending":
                        budget = int(r * self.eta**i)
                        try:
                            score = self.objective_fn(trial.config, budget)
                            trial.score = score
                            trial.status = "completed"
                        except Exception as e:
                            trial.status = "failed"

                n_keep = int(n / self.eta)
                bracket_trials = sorted(
                    [t for t in bracket_trials if t.status == "completed"],
                    key=lambda x: x.score,
                    reverse=self.maximize,
                )[:n_keep]

                for trial in bracket_trials:
                    self.trials.append(trial)
                    if (
                        self.best_trial is None
                        or (self.maximize and trial.score > self.best_trial.score)
                        or (not self.maximize and trial.score < self.best_trial.score)
                    ):
                        self.best_trial = trial

        return {
            "best_config": self.best_trial.config,
            "best_score": self.best_trial.score,
            "trials": self.trials,
        }


class BayesianOptimization(NASearch):
    """Bayesian optimization using Gaussian processes."""

    def __init__(
        self,
        space: SearchSpace,
        objective_fn: Callable[[Dict], float],
        maximize: bool = True,
        n_initial_points: int = 5,
    ):
        super().__init__(space, objective_fn, maximize)
        self.n_initial_points = n_initial_points
        self.X_observed = []
        self.y_observed = []

    def search(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict:
        for i in range(min(self.n_initial_points, n_trials)):
            config = self.space.sample()
            trial = self._evaluate(config)
            self.trials.append(trial)
            self.X_observed.append(config)
            self.y_observed.append(trial.score if trial.score else 0)

            if (
                self.best_trial is None
                or (self.maximize and trial.score > self.best_trial.score)
                or (not self.maximize and trial.score < self.best_trial.score)
            ):
                self.best_trial = trial

        for i in range(n_trials - self.n_initial_points):
            config = self._sample_from_gp()
            trial = self._evaluate(config)
            self.trials.append(trial)
            self.X_observed.append(config)
            self.y_observed.append(trial.score if trial.score else 0)

            if (
                self.best_trial is None
                or (self.maximize and trial.score > self.best_trial.score)
                or (not self.maximize and trial.score < self.best_trial.score)
            ):
                self.best_trial = trial

        return {
            "best_config": self.best_trial.config,
            "best_score": self.best_trial.score,
            "trials": self.trials,
        }

    def _sample_from_gp(self) -> Dict:
        return self.space.sample()
