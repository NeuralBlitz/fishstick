from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

import numpy as np

T = TypeVar("T")
MetricType = TypeVar("MetricType", bound=float)


class ParameterType(Enum):
    UNIFORM = "uniform"
    LOGUNIFORM = "loguniform"
    QUNIFORM = "quniform"
    LOGQUNIFORM = "logquniform"
    CHOICE = "choice"
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    LOGINTEGER = "loginteger"


@dataclass
class SearchSpace:
    name: str
    param_type: ParameterType
    low: float | None = None
    high: float | None = None
    steps: int | None = None
    choices: list[Any] | None = None
    default: Any = None
    quantize: float | None = None

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        if rng is None:
            rng = np.random.default_rng()

        if self.param_type == ParameterType.UNIFORM:
            value = rng.uniform(self.low, self.high)
            if self.quantize is not None:
                value = math.floor(value / self.quantize) * self.quantize
            return value
        elif self.param_type == ParameterType.LOGUNIFORM:
            log_low = math.log(self.low) if self.low and self.low > 0 else -7.0
            log_high = math.log(self.high) if self.high and self.high > 0 else 7.0
            value = math.exp(rng.uniform(log_low, log_high))
            if self.quantize is not None:
                value = math.floor(value / self.quantize) * self.quantize
            return value
        elif self.param_type == ParameterType.INTEGER:
            return rng.integers(int(self.low), int(self.high) + 1)
        elif self.param_type == ParameterType.LOGINTEGER:
            log_low = math.log(self.low) if self.low and self.low > 0 else -7.0
            log_high = math.log(self.high) if self.high and self.high > 0 else 7.0
            return int(math.exp(rng.uniform(log_low, log_high)))
        elif self.param_type == ParameterType.QUNIFORM:
            if self.steps:
                step_size = (self.high - self.low) / (self.steps - 1)
                idx = rng.integers(0, self.steps)
                return self.low + idx * step_size
            return rng.uniform(self.low, self.high)
        elif self.param_type in (ParameterType.CHOICE, ParameterType.CATEGORICAL):
            return rng.choice(self.choices)
        raise ValueError(f"Unknown parameter type: {self.param_type}")

    def transform(self, value: float) -> Any:
        if self.param_type == ParameterType.UNIFORM:
            return value * (self.high - self.low) + self.low
        elif self.param_type == ParameterType.LOGUNIFORM:
            log_low = math.log(self.low) if self.low and self.low > 0 else -7.0
            log_high = math.log(self.high) if self.high and self.high > 0 else 7.0
            return math.exp(value * (log_high - log_low) + log_low)
        elif self.param_type == ParameterType.INTEGER:
            return int(value * (self.high - self.low) + self.low)
        elif self.param_type == ParameterType.LOGINTEGER:
            log_low = math.log(self.low) if self.low and self.low > 0 else -7.0
            log_high = math.log(self.high) if self.high and self.high > 0 else 7.0
            return int(math.exp(value * (log_high - log_low) + log_low))
        elif self.param_type == ParameterType.QUNIFORM:
            if self.steps:
                step_size = (self.high - self.low) / (self.steps - 1)
                idx = int(value * (self.steps - 1))
                return self.low + idx * step_size
            return value * (self.high - self.low) + self.low
        elif self.param_type in (ParameterType.CHOICE, ParameterType.CATEGORICAL):
            idx = int(value * (len(self.choices) - 1))
            return self.choices[min(idx, len(self.choices) - 1)]
        return value

    def get_bounds(self) -> tuple[float, float]:
        if self.param_type == ParameterType.LOGUNIFORM:
            log_low = math.log(self.low) if self.low and self.low > 0 else -7.0
            log_high = math.log(self.high) if self.high and self.high > 0 else 7.0
            return (log_low, log_high)
        return (self.low or 0.0, self.high or 1.0)


def uniform(
    name: str,
    low: float,
    high: float,
    default: float | None = None,
    quantize: float | None = None,
) -> SearchSpace:
    return SearchSpace(
        name,
        ParameterType.UNIFORM,
        low,
        high,
        default=default or (low + high) / 2,
        quantize=quantize,
    )


def loguniform(
    name: str,
    low: float,
    high: float,
    default: float | None = None,
    quantize: float | None = None,
) -> SearchSpace:
    return SearchSpace(
        name,
        ParameterType.LOGUNIFORM,
        low,
        high,
        default=default or math.sqrt(low * high),
        quantize=quantize,
    )


def quniform(
    name: str,
    low: float,
    high: float,
    steps: int,
    default: float | None = None,
) -> SearchSpace:
    return SearchSpace(
        name,
        ParameterType.QUNIFORM,
        low,
        high,
        steps=steps,
        default=default or (low + high) / 2,
    )


def integer(
    name: str,
    low: int,
    high: int,
    default: int | None = None,
) -> SearchSpace:
    return SearchSpace(
        name,
        ParameterType.INTEGER,
        float(low),
        float(high),
        default=default or (low + high) // 2,
    )


def loginteger(
    name: str,
    low: int,
    high: int,
    default: int | None = None,
) -> SearchSpace:
    return SearchSpace(
        name,
        ParameterType.LOGINTEGER,
        float(low),
        float(high),
        default=default or int(math.sqrt(low * high)),
    )


def choice(name: str, choices: list[Any], default: Any | None = None) -> SearchSpace:
    return SearchSpace(
        name, ParameterType.CHOICE, choices=choices, default=default or choices[0]
    )


def categorical(
    name: str, choices: list[str], default: str | None = None
) -> SearchSpace:
    return SearchSpace(
        name, ParameterType.CATEGORICAL, choices=choices, default=default or choices[0]
    )


def grid(name: str, values: list[Any], default: Any | None = None) -> SearchSpace:
    return SearchSpace(
        name, ParameterType.CHOICE, choices=values, default=default or values[0]
    )


@dataclass
class HyperparameterSpace:
    params: dict[str, SearchSpace] = field(default_factory=dict)

    def add(self, space: SearchSpace) -> None:
        self.params[space.name] = space

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        return {name: space.sample(rng) for name, space in self.params.items()}

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, key: str) -> SearchSpace:
        return self.params[key]

    def keys(self) -> list[str]:
        return list(self.params.keys())


@dataclass
class CategoricalParameter:
    name: str
    choices: list[str]
    default: str | None = None

    def __post_init__(self):
        if self.default is None:
            self.default = self.choices[0]


@dataclass
class ContinuousParameter:
    name: str
    low: float
    high: float
    default: float | None = None
    scale: str = "linear"

    def __post_init__(self):
        if self.default is None:
            self.default = (self.low + self.high) / 2


@dataclass
class DiscreteParameter:
    name: str
    values: list[Any]
    default: Any | None = None

    def __post_init__(self):
        if self.default is None:
            self.default = self.values[0]


def create_search_space(**kwargs: SearchSpace) -> dict[str, SearchSpace]:
    return kwargs
