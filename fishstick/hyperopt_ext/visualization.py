from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    trials: list[dict[str, Any]]
    history: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class OptimizationVisualizer:
    def __init__(self, results: list[OptimizationResult] | None = None):
        self.results = results or []

    def add_result(self, result: OptimizationResult) -> None:
        self.results.append(result)

    def generate_report(self) -> str:
        lines = []
        lines.append("# Hyperparameter Optimization Report\n")

        for i, result in enumerate(self.results):
            lines.append(f"## Experiment {i + 1}\n")
            lines.append(f"- Trials: {result.n_trials}")
            lines.append(f"- Best value: {result.best_value:.6f}")
            lines.append(f"- Best params: {result.best_params}")
            lines.append("")

        return "\n".join(lines)

    def save_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.generate_report())

    def get_convergence_data(self) -> dict[str, list[float]]:
        data = {"values": [], "best_so_far": []}
        for result in self.results:
            data["values"].extend(result.history)
            best = float("inf")
            for v in result.history:
                best = min(best, v)
                data["best_so_far"].append(best)
        return data

    def get_param_importance(
        self, trials: list[dict[str, Any]], metric: str = "loss"
    ) -> dict[str, float]:
        if not trials:
            return {}

        param_values: dict[str, list[float]] = {}
        for trial in trials:
            if metric in trial.get("metrics", {}):
                for key, val in trial.get("params", {}).items():
                    if isinstance(val, (int, float)):
                        if key not in param_values:
                            param_values[key] = []
                        param_values[key].append(val)

        if not param_values:
            return {}

        metric_values = [t.get("metrics", {}).get(metric, 0) for t in trials]

        importance = {}
        for param, values in param_values.items():
            if len(values) > 1:
                corr = np.corrcoef(values, metric_values)[0, 1]
                importance[param] = abs(corr) if not np.isnan(corr) else 0.0

        return importance


class OptimizationHistory:
    def __init__(self):
        self.trials: list[dict[str, Any]] = []
        self.best_value: float | None = None
        self.best_params: dict[str, Any] = {}

    def add_trial(
        self,
        params: dict[str, Any],
        value: float,
        metrics: dict[str, float] | None = None,
    ) -> None:
        trial = {
            "params": params,
            "value": value,
            "metrics": metrics or {},
            "trial_id": len(self.trials),
        }
        self.trials.append(trial)

        if self.best_value is None or value < self.best_value:
            self.best_value = value
            self.best_params = params

    def get_history(self) -> list[float]:
        return [t["value"] for t in self.trials]

    def get_best_so_far(self) -> list[float]:
        history = self.get_history()
        best_so_far = []
        current_best = float("inf")
        for v in history:
            current_best = min(current_best, v)
            best_so_far.append(current_best)
        return best_so_far

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "trials": self.trials,
            "best_value": self.best_value,
            "best_params": self.best_params,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class ParameterGridAnalyzer:
    def __init__(self, trials: list[dict[str, Any]]):
        self.trials = trials

    def get_param_ranges(self) -> dict[str, tuple[float, float]]:
        ranges = {}
        for trial in self.trials:
            for key, val in trial.get("params", {}).items():
                if isinstance(val, (int, float)):
                    if key not in ranges:
                        ranges[key] = (val, val)
                    else:
                        low, high = ranges[key]
                        ranges[key] = (min(low, val), max(high, val))
        return ranges

    def get_param_distribution(self, param: str) -> dict[str, float]:
        values = []
        for trial in self.trials:
            if param in trial.get("params", {}):
                val = trial["params"][param]
                if isinstance(val, (int, float)):
                    values.append(val)

        if not values:
            return {}

        bins = 10
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return {f"{min_val}": 1.0}

        bin_edges = np.linspace(min_val, max_val, bins + 1)
        hist, _ = np.histogram(values, bins=bin_edges)
        hist = hist / sum(hist)

        distribution = {}
        for i in range(bins):
            key = f"[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f})"
            distribution[key] = hist[i]

        return distribution


def plot_optimization_history(
    trials: list[dict[str, Any]],
    metric: str = "loss",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    values = [t.get("metrics", {}).get(metric, 0) for t in trials]
    best_so_far = []
    current_best = float("inf")
    for v in values:
        current_best = min(current_best, v)
        best_so_far.append(current_best)

    data = {
        "iterations": list(range(len(trials))),
        "values": values,
        "best_so_far": best_so_far,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(data, f)

    return data
