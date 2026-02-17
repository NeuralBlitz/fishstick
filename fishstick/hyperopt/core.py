"""
Hyperparameter Optimization

Automated hyperparameter tuning using grid search, random search, and Bayesian optimization.
"""

from typing import Dict, List, Any, Callable, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from itertools import product
import json
import os
from dataclasses import dataclass, asdict
import warnings


@dataclass
class SearchSpace:
    """
    Define search space for hyperparameters.

    Args:
        param_name: Name of the parameter
        param_type: Type of search ('choice', 'uniform', 'loguniform', 'int')
        values: For 'choice': list of values
        low: For 'uniform'/'loguniform'/'int': lower bound
        high: For 'uniform'/'loguniform'/'int': upper bound

    Example:
        >>> lr_space = SearchSpace('lr', 'loguniform', low=1e-5, high=1e-1)
        >>> optimizer_space = SearchSpace('optimizer', 'choice', values=['adam', 'sgd'])
    """

    param_name: str
    param_type: str  # 'choice', 'uniform', 'loguniform', 'int'
    values: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None

    def sample(self, rng: Optional[np.random.RandomState] = None) -> Any:
        """Sample a value from this search space."""
        if rng is None:
            rng = np.random.RandomState()

        if self.param_type == "choice":
            return rng.choice(self.values)
        elif self.param_type == "uniform":
            return rng.uniform(self.low, self.high)
        elif self.param_type == "loguniform":
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            return np.exp(rng.uniform(log_low, log_high))
        elif self.param_type == "int":
            return rng.randint(self.low, self.high + 1)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")

    def get_grid_values(self, num_points: int = 5) -> List[Any]:
        """Get evenly spaced values for grid search."""
        if self.param_type == "choice":
            return self.values
        elif self.param_type == "uniform":
            return list(np.linspace(self.low, self.high, num_points))
        elif self.param_type == "loguniform":
            return list(
                np.logspace(np.log10(self.low), np.log10(self.high), num_points)
            )
        elif self.param_type == "int":
            return list(range(int(self.low), int(self.high) + 1))
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


class GridSearch:
    """
    Exhaustive search over specified parameter values.

    Args:
        param_spaces: Dictionary of parameter names to SearchSpace objects
        model_fn: Function that creates model given hyperparameters
        trainer_fn: Function that trains model and returns metric

    Example:
        >>> param_spaces = {
        ...     'lr': SearchSpace('lr', 'loguniform', low=1e-4, high=1e-2),
        ...     'batch_size': SearchSpace('batch_size', 'choice', values=[32, 64, 128])
        ... }
        >>> gs = GridSearch(param_spaces, create_model, train_and_evaluate)
        >>> best_params, best_score = gs.search(num_points_per_param=3)
    """

    def __init__(
        self,
        param_spaces: Dict[str, SearchSpace],
        model_fn: Callable,
        trainer_fn: Callable,
        metric: str = "val_accuracy",
        mode: str = "max",
    ):
        self.param_spaces = param_spaces
        self.model_fn = model_fn
        self.trainer_fn = trainer_fn
        self.metric = metric
        self.mode = mode

        self.results = []

    def search(self, num_points_per_param: int = 5) -> Tuple[Dict, float]:
        """
        Perform grid search.

        Args:
            num_points_per_param: Number of points to try for continuous spaces

        Returns:
            Tuple of (best_params, best_score)
        """
        # Generate all combinations
        grid_values = {
            name: space.get_grid_values(num_points_per_param)
            for name, space in self.param_spaces.items()
        }

        param_names = list(grid_values.keys())
        combinations = list(product(*[grid_values[name] for name in param_names]))

        print(f"Grid Search: Trying {len(combinations)} combinations")

        best_score = float("-inf") if self.mode == "max" else float("inf")
        best_params = None

        for i, values in enumerate(combinations):
            params = dict(zip(param_names, values))
            print(f"\nTrial {i + 1}/{len(combinations)}: {params}")

            try:
                score = self._evaluate(params)
                self.results.append({"params": params, "score": score})

                if self._is_better(score, best_score):
                    best_score = score
                    best_params = params.copy()
                    print(f"  New best! {self.metric}: {score:.4f}")
                else:
                    print(f"  {self.metric}: {score:.4f}")

            except Exception as e:
                print(f"  Error: {str(e)}")
                continue

        return best_params, best_score

    def _evaluate(self, params: Dict) -> float:
        """Evaluate a single parameter configuration."""
        model = self.model_fn(**params)
        return self.trainer_fn(model, **params)

    def _is_better(self, score: float, best: float) -> bool:
        """Check if score is better than current best."""
        if self.mode == "max":
            return score > best
        else:
            return score < best

    def get_results_df(self):
        """Get results as pandas DataFrame."""
        try:
            import pandas as pd

            return pd.DataFrame(self.results)
        except ImportError:
            return self.results


class RandomSearch:
    """
    Random search over hyperparameter space.

    Often more efficient than grid search in high-dimensional spaces.

    Reference: Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization", 2012

    Args:
        param_spaces: Dictionary of parameter names to SearchSpace objects
        model_fn: Function that creates model given hyperparameters
        trainer_fn: Function that trains model and returns metric
        n_trials: Number of random trials

    Example:
        >>> rs = RandomSearch(param_spaces, create_model, train_and_evaluate, n_trials=50)
        >>> best_params, best_score = rs.search()
    """

    def __init__(
        self,
        param_spaces: Dict[str, SearchSpace],
        model_fn: Callable,
        trainer_fn: Callable,
        n_trials: int = 50,
        metric: str = "val_accuracy",
        mode: str = "max",
        seed: Optional[int] = None,
    ):
        self.param_spaces = param_spaces
        self.model_fn = model_fn
        self.trainer_fn = trainer_fn
        self.n_trials = n_trials
        self.metric = metric
        self.mode = mode
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.results = []

    def search(self) -> Tuple[Dict, float]:
        """Perform random search."""
        print(f"Random Search: Trying {self.n_trials} random configurations")

        best_score = float("-inf") if self.mode == "max" else float("inf")
        best_params = None

        for i in range(self.n_trials):
            # Sample parameters
            params = {
                name: space.sample(self.rng)
                for name, space in self.param_spaces.items()
            }

            print(f"\nTrial {i + 1}/{self.n_trials}: {params}")

            try:
                score = self._evaluate(params)
                self.results.append({"params": params, "score": score})

                if self._is_better(score, best_score):
                    best_score = score
                    best_params = params.copy()
                    print(f"  New best! {self.metric}: {score:.4f}")
                else:
                    print(f"  {self.metric}: {score:.4f}")

            except Exception as e:
                print(f"  Error: {str(e)}")
                continue

        return best_params, best_score

    def _evaluate(self, params: Dict) -> float:
        """Evaluate a single parameter configuration."""
        model = self.model_fn(**params)
        return self.trainer_fn(model, **params)

    def _is_better(self, score: float, best: float) -> bool:
        """Check if score is better than current best."""
        if self.mode == "max":
            return score > best
        else:
            return score < best

    def get_results_df(self):
        """Get results as pandas DataFrame."""
        try:
            import pandas as pd

            return pd.DataFrame(self.results)
        except ImportError:
            return self.results


class BayesianOptimization:
    """
    Bayesian Optimization for hyperparameter tuning.

    Uses Gaussian Process to model the objective function and acquisition function
    to select next points to evaluate.

    Requires scikit-optimize to be installed.

    Args:
        param_spaces: Dictionary of parameter names to SearchSpace objects
        model_fn: Function that creates model given hyperparameters
        trainer_fn: Function that trains model and returns metric
        n_initial_points: Number of random points to sample before fitting GP
        n_trials: Total number of trials

    Example:
        >>> bo = BayesianOptimization(param_spaces, create_model, train_and_evaluate, n_trials=30)
        >>> best_params, best_score = bo.search()
    """

    def __init__(
        self,
        param_spaces: Dict[str, SearchSpace],
        model_fn: Callable,
        trainer_fn: Callable,
        n_initial_points: int = 10,
        n_trials: int = 30,
        metric: str = "val_accuracy",
        mode: str = "max",
        seed: Optional[int] = None,
    ):
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical

            self._has_skopt = True
        except ImportError:
            warnings.warn(
                "scikit-optimize not installed. BayesianOptimization will fall back to RandomSearch."
            )
            self._has_skopt = False

        self.param_spaces = param_spaces
        self.model_fn = model_fn
        self.trainer_fn = trainer_fn
        self.n_initial_points = n_initial_points
        self.n_trials = n_trials
        self.metric = metric
        self.mode = mode
        self.seed = seed

        self.results = []
        self._skopt_spaces = None

    def _convert_to_skopt_space(self):
        """Convert SearchSpace objects to scikit-optimize spaces."""
        from skopt.space import Real, Integer, Categorical

        spaces = []
        for name, space in self.param_spaces.items():
            if space.param_type == "choice":
                spaces.append(Categorical(space.values, name=name))
            elif space.param_type == "uniform":
                spaces.append(Real(space.low, space.high, name=name))
            elif space.param_type == "loguniform":
                spaces.append(
                    Real(space.low, space.high, prior="log-uniform", name=name)
                )
            elif space.param_type == "int":
                spaces.append(Integer(int(space.low), int(space.high), name=name))

        return spaces

    def search(self) -> Tuple[Dict, float]:
        """Perform Bayesian optimization."""
        if not self._has_skopt:
            # Fallback to random search
            rs = RandomSearch(
                self.param_spaces,
                self.model_fn,
                self.trainer_fn,
                self.n_trials,
                self.metric,
                self.mode,
                self.seed,
            )
            return rs.search()

        from skopt import gp_minimize

        self._skopt_spaces = self._convert_to_skopt_space()
        param_names = list(self.param_spaces.keys())

        def objective(values):
            """Objective function for scikit-optimize."""
            params = dict(zip(param_names, values))
            try:
                score = self._evaluate(params)
                # Minimize negative score if maximizing
                return -score if self.mode == "max" else score
            except Exception as e:
                print(f"Error in evaluation: {e}")
                return float("inf") if self.mode == "min" else -float("inf")

        print(
            f"Bayesian Optimization: {self.n_trials} trials ({self.n_initial_points} random init)"
        )

        result = gp_minimize(
            objective,
            self._skopt_spaces,
            n_calls=self.n_trials,
            n_initial_points=self.n_initial_points,
            random_state=self.seed,
            verbose=True,
        )

        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun if self.mode == "max" else result.fun

        return best_params, best_score

    def _evaluate(self, params: Dict) -> float:
        """Evaluate a single parameter configuration."""
        model = self.model_fn(**params)
        return self.trainer_fn(model, **params)


class HyperBand:
    """
    HyperBand algorithm for hyperparameter optimization.

    Uses successive halving to allocate more resources to promising configurations.

    Reference: Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", 2018

    Args:
        param_spaces: Dictionary of parameter names to SearchSpace objects
        model_fn: Function that creates model given hyperparameters
        trainer_fn: Function that trains model for specified epochs and returns metric
        max_epochs: Maximum number of epochs for any configuration
        eta: Reduction factor for successive halving

    Example:
        >>> hb = HyperBand(param_spaces, create_model, partial_train, max_epochs=81)
        >>> best_params, best_score = hb.search()
    """

    def __init__(
        self,
        param_spaces: Dict[str, SearchSpace],
        model_fn: Callable,
        trainer_fn: Callable,  # Should accept 'epochs' parameter
        max_epochs: int = 81,
        eta: int = 3,
        metric: str = "val_accuracy",
        mode: str = "max",
        seed: Optional[int] = None,
    ):
        self.param_spaces = param_spaces
        self.model_fn = model_fn
        self.trainer_fn = trainer_fn
        self.max_epochs = max_epochs
        self.eta = eta
        self.metric = metric
        self.mode = mode
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.results = []

    def search(self) -> Tuple[Dict, float]:
        """Perform HyperBand search."""
        # Calculate number of brackets
        max_iter = int(np.log(self.max_epochs) / np.log(self.eta))
        B = (max_iter + 1) * self.max_epochs

        print(
            f"HyperBand: max_epochs={self.max_epochs}, eta={self.eta}, brackets={max_iter + 1}"
        )

        best_score = float("-inf") if self.mode == "max" else float("inf")
        best_params = None

        for bracket in range(max_iter + 1):
            n_configs = int(
                np.ceil(B / self.max_epochs / (bracket + 1) * self.eta**bracket)
            )
            n_epochs = int(self.max_epochs * self.eta ** (-bracket))

            print(
                f"\nBracket {bracket + 1}/{max_iter + 1}: {n_configs} configs, {n_epochs} epochs each"
            )

            # Sample configurations
            configs = []
            for _ in range(n_configs):
                params = {
                    name: space.sample(self.rng)
                    for name, space in self.param_spaces.items()
                }
                configs.append(params)

            # Successive halving
            for round_num in range(bracket + 1):
                n_to_keep = int(n_configs / self.eta**round_num)
                epochs = int(n_epochs * self.eta**round_num)

                print(
                    f"  Round {round_num + 1}: Evaluating {len(configs)} configs for {epochs} epochs"
                )

                # Evaluate all configurations
                scores = []
                for params in configs:
                    try:
                        model = self.model_fn(**params)
                        score = self.trainer_fn(model, epochs=epochs, **params)
                        scores.append(score)
                        self.results.append(
                            {"params": params, "score": score, "epochs": epochs}
                        )
                    except Exception as e:
                        print(f"    Error: {e}")
                        scores.append(
                            float("-inf") if self.mode == "max" else float("inf")
                        )

                # Keep top configurations
                if round_num < bracket:
                    if self.mode == "max":
                        top_indices = np.argsort(scores)[-n_to_keep:]
                    else:
                        top_indices = np.argsort(scores)[:n_to_keep]

                    configs = [configs[i] for i in top_indices]
                    print(f"    Kept top {len(configs)} configs")
                else:
                    # Last round - find best
                    if self.mode == "max":
                        best_idx = np.argmax(scores)
                    else:
                        best_idx = np.argmin(scores)

                    if self._is_better(scores[best_idx], best_score):
                        best_score = scores[best_idx]
                        best_params = configs[best_idx].copy()
                        print(f"    New best! {self.metric}: {best_score:.4f}")

        return best_params, best_score

    def _is_better(self, score: float, best: float) -> bool:
        """Check if score is better than current best."""
        if self.mode == "max":
            return score > best
        else:
            return score < best


def suggest_hyperparameters(
    model_type: str = "cnn",
    dataset_size: Optional[int] = None,
    task: str = "classification",
) -> Dict[str, Any]:
    """
    Suggest good starting hyperparameters based on heuristics.

    Args:
        model_type: Type of model ('cnn', 'transformer', 'mlp')
        dataset_size: Size of dataset
        task: Type of task

    Returns:
        Dictionary of suggested hyperparameters
    """
    suggestions = {
        "optimizer": "adamw",
        "lr": 1e-3,
        "batch_size": 32,
        "weight_decay": 0.01,
    }

    if model_type == "cnn":
        suggestions.update(
            {
                "lr": 1e-3,
                "batch_size": 64,
                "weight_decay": 5e-4,
            }
        )
    elif model_type == "transformer":
        suggestions.update(
            {
                "lr": 1e-4,
                "batch_size": 32,
                "weight_decay": 0.1,
                "warmup_steps": 1000,
            }
        )
    elif model_type == "mlp":
        suggestions.update(
            {
                "lr": 1e-3,
                "batch_size": 128,
                "weight_decay": 1e-5,
            }
        )

    # Adjust for dataset size
    if dataset_size is not None:
        if dataset_size < 10000:
            suggestions["batch_size"] = min(suggestions["batch_size"], 32)
            suggestions["weight_decay"] *= 2
        elif dataset_size > 100000:
            suggestions["batch_size"] = max(suggestions["batch_size"], 128)
            suggestions["lr"] *= 2

    return suggestions


class HyperparameterLogger:
    """Log and track hyperparameter search results."""

    def __init__(self, log_dir: str = "hyperopt_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.experiments = []

    def log_experiment(
        self, params: Dict, metrics: Dict, model_info: Optional[Dict] = None
    ):
        """Log a single experiment."""
        experiment = {
            "params": params,
            "metrics": metrics,
            "model_info": model_info or {},
            "timestamp": time.time(),
        }
        self.experiments.append(experiment)

    def save(self, filename: str = "hyperopt_results.json"):
        """Save all experiments to file."""
        filepath = self.log_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.experiments, f, indent=2)

    def load(self, filename: str = "hyperopt_results.json"):
        """Load experiments from file."""
        filepath = self.log_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                self.experiments = json.load(f)

    def get_best(self, metric: str = "val_accuracy", mode: str = "max") -> Dict:
        """Get best experiment based on metric."""
        if not self.experiments:
            return None

        if mode == "max":
            best_idx = max(
                range(len(self.experiments)),
                key=lambda i: self.experiments[i]["metrics"].get(metric, float("-inf")),
            )
        else:
            best_idx = min(
                range(len(self.experiments)),
                key=lambda i: self.experiments[i]["metrics"].get(metric, float("inf")),
            )

        return self.experiments[best_idx]


import time

__all__ = [
    "SearchSpace",
    "GridSearch",
    "RandomSearch",
    "BayesianOptimization",
    "HyperBand",
    "suggest_hyperparameters",
    "HyperparameterLogger",
]
