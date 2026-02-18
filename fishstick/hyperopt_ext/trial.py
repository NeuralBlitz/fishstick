from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np


class TrialStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"
    FAILED = "failed"


@dataclass
class Trial:
    trial_id: int
    params: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    intermediate_results: list[dict[str, float]] = field(default_factory=list)
    training_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        self.status = TrialStatus.COMPLETED.value
        self.end_time = time.time()

    def prune(self) -> None:
        self.status = TrialStatus.PRUNED.value
        self.end_time = time.time()

    def fail(self, error: str | None = None) -> None:
        self.status = TrialStatus.FAILED.value
        self.end_time = time.time()
        if error:
            self.metadata["error"] = error

    def should_prune(self, grace_period: float = 60.0) -> bool:
        if self.status != TrialStatus.RUNNING.value:
            return False
        return time.time() - self.start_time < grace_period

    @property
    def duration(self) -> float | None:
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_running(self) -> bool:
        return self.status == TrialStatus.RUNNING.value

    @property
    def is_completed(self) -> bool:
        return self.status == TrialStatus.COMPLETED.value


@dataclass
class TrialResult:
    trial: Trial
    objective_value: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TrialCallback(ABC):
    @abstractmethod
    def on_trial_start(self, trial: Trial) -> None:
        pass

    @abstractmethod
    def on_trial_complete(self, trial: Trial, metrics: dict[str, float]) -> None:
        pass

    @abstractmethod
    def on_trial_prune(self, trial: Trial) -> None:
        pass


class TrialLogger(TrialCallback):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def on_trial_start(self, trial: Trial) -> None:
        if self.verbose:
            print(f"[Trial {trial.trial_id}] Started with params: {trial.params}")

    def on_trial_complete(self, trial: Trial, metrics: dict[str, float]) -> None:
        if self.verbose:
            print(f"[Trial {trial.trial_id}] Completed with metrics: {metrics}")

    def on_trial_prune(self, trial: Trial) -> None:
        if self.verbose:
            print(f"[Trial {trial.trial_id}] Pruned at step {trial.training_step}")


@dataclass
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
            self.trial_results.append(
                TrialResult(trial, obj_value, {"metrics": metrics})
            )

    def get_best(self, metric: str = "loss", minimize: bool = True) -> Trial | None:
        completed = [
            t
            for t in self.trials
            if t.status == TrialStatus.COMPLETED.value and metric in t.metrics
        ]
        if not completed:
            return None
        return (
            min(completed, key=lambda t: t.metrics[metric])
            if minimize
            else max(completed, key=lambda t: t.metrics[metric])
        )

    def get_all_completed(self) -> list[Trial]:
        return [t for t in self.trials if t.status == TrialStatus.COMPLETED.value]

    def get_all_running(self) -> list[Trial]:
        return [t for t in self.trials if t.status == TrialStatus.RUNNING.value]

    def get_all_pruned(self) -> list[Trial]:
        return [t for t in self.trials if t.status == TrialStatus.PRUNED.value]

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
                    "training_step": t.training_step,
                    "metadata": t.metadata,
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
                status=t_data.get("status", TrialStatus.COMPLETED.value),
                start_time=t_data.get("start_time", time.time()),
                end_time=t_data.get("end_time"),
                training_step=t_data.get("training_step", 0),
                metadata=t_data.get("metadata", {}),
            )
            self.trials.append(trial)

    def get_trial_by_id(self, trial_id: int) -> Trial | None:
        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial
        return None

    def get_statistics(self) -> dict[str, Any]:
        completed = self.get_all_completed()
        pruned = self.get_all_pruned()
        running = self.get_all_running()

        stats = {
            "total_trials": len(self.trials),
            "completed": len(completed),
            "pruned": len(pruned),
            "running": len(running),
        }

        if completed:
            metrics_keys = set()
            for t in completed:
                metrics_keys.update(t.metrics.keys())

            for key in metrics_keys:
                values = [t.metrics[key] for t in completed if key in t.metrics]
                if values:
                    stats[f"{key}_mean"] = float(np.mean(values))
                    stats[f"{key}_std"] = float(np.std(values))
                    stats[f"{key}_min"] = float(np.min(values))
                    stats[f"{key}_max"] = float(np.max(values))

        return stats
