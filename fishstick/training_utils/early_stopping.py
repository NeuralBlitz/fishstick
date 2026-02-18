"""
Early Stopping Implementations

Advanced early stopping implementations including patience-based stopping,
delta-based stopping, composite stopping, and recovery-aware stopping.
"""

from typing import Optional, Dict, Any, List, Callable, Union, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import time


class EarlyStoppingMode(Enum):
    """Early stopping mode for monitoring metrics."""

    MIN = "min"
    MAX = "max"


class EarlyStopping(ABC):
    """
    Abstract base class for early stopping implementations.

    Defines the interface for all early stopping strategies.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: Union[str, EarlyStoppingMode] = "min",
        verbose: bool = True,
    ):
        if isinstance(mode, str):
            mode = EarlyStoppingMode(mode)

        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.should_stop = False

    @abstractmethod
    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.

        Args:
            epoch: Current epoch
            metrics: Dictionary of current metrics

        Returns:
            True if training should stop
        """
        pass

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.should_stop = False


class PatienceEarlyStopping(EarlyStopping):
    """
    Patience-based early stopping.

    Stops training when a monitored metric has not improved for a specified
    number of consecutive epochs (patience).

    Example:
        >>> early_stopping = PatienceEarlyStopping(
        ...     monitor="val_loss",
        ...     patience=10,
        ...     mode="min",
        ...     min_delta=0.001,
        ... )
        >>> for epoch in range(100):
        ...     metrics = train_one_epoch()
        ...     if early_stopping(epoch, metrics):
        ...         print(f"Stopped at epoch {epoch}")
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: Union[str, EarlyStoppingMode] = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
        verbose: bool = True,
    ):
        super().__init__(monitor=monitor, mode=mode, verbose=verbose)

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.best_state: Optional[Dict[str, Any]] = None

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop based on patience.

        Args:
            epoch: Current epoch
            metrics: Current metrics dictionary

        Returns:
            True if training should stop
        """
        current = metrics.get(self.monitor)

        if current is None:
            if self.verbose:
                print(f"Warning: Metric '{self.monitor}' not found in metrics")
            return False

        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
            return False

        if self.mode == EarlyStoppingMode.MIN:
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"Early stopping triggered at epoch {epoch}")
                print(
                    f"Best {self.monitor}: {self.best_value} at epoch {self.best_epoch}"
                )
            return True

        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        super().reset()
        self.best_value = None
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None


class DeltaEarlyStopping(EarlyStopping):
    """
    Delta-based early stopping.

    Stops training when improvement falls below a specified delta threshold,
    useful for detecting convergence.

    Example:
        >>> early_stopping = DeltaEarlyStopping(
        ...     monitor="val_loss",
        ...     delta=0.001,
        ...     window_size=5,
        ... )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        delta: float = 0.001,
        window_size: int = 5,
        mode: Union[str, EarlyStoppingMode] = "min",
        verbose: bool = True,
    ):
        super().__init__(monitor=monitor, mode=mode, verbose=verbose)

        self.delta = delta
        self.window_size = window_size

        self.history: List[float] = []
        self.best_value = None

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check if improvement is below delta threshold."""
        current = metrics.get(self.monitor)

        if current is None:
            return False

        self.history.append(current)

        if len(self.history) < self.window_size:
            return False

        if len(self.history) > self.window_size:
            self.history.pop(0)

        window_avg = sum(self.history) / len(self.history)

        if self.best_value is None:
            self.best_value = window_avg
            return False

        if self.mode == EarlyStoppingMode.MIN:
            improvement = self.best_value - window_avg
        else:
            improvement = window_avg - self.best_value

        if improvement < self.delta:
            self.should_stop = True
            if self.verbose:
                print(
                    f"Delta early stopping triggered: improvement {improvement} < delta {self.delta}"
                )
            return True

        if improvement > 0:
            self.best_value = window_avg

        return False


class BestEarlyStopping(EarlyStopping):
    """
    Early stopping that tracks best metrics and can restore best model.

    Tracks the best value of monitored metric and provides utilities
    for model restoration.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: Union[str, EarlyStoppingMode] = "min",
        verbose: bool = True,
    ):
        super().__init__(monitor=monitor, mode=mode, verbose=verbose)

        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.history: List[float] = []

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Track best value without stopping."""
        current = metrics.get(self.monitor)

        if current is None:
            return False

        self.history.append(current)

        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
        elif self.mode == EarlyStoppingMode.MIN:
            if current < self.best_value:
                self.best_value = current
                self.best_epoch = epoch
        else:
            if current > self.best_value:
                self.best_value = current
                self.best_epoch = epoch

        return False

    def get_best_epoch(self) -> int:
        """Get the epoch with best metric value."""
        return self.best_epoch

    def get_best_value(self) -> Optional[float]:
        """Get the best metric value."""
        return self.best_value


class EarlyStoppingWithRecovery(EarlyStopping):
    """
    Early stopping with recovery capability.

    Allows training to continue after a plateau if performance recovers,
    preventing premature stopping during temporary degradation.

    Example:
        >>> early_stopping = EarlyStoppingWithRecovery(
        ...     monitor="val_loss",
        ...     patience=10,
        ...     recovery_patience=5,
        ...     recovery_threshold=0.95,
        ... )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: Union[str, EarlyStoppingMode] = "min",
        min_delta: float = 0.0,
        recovery_patience: int = 5,
        recovery_threshold: float = 0.95,
        max_recoveries: int = 2,
        verbose: bool = True,
    ):
        super().__init__(monitor=monitor, mode=mode, verbose=verbose)

        self.patience = patience
        self.min_delta = min_delta
        self.recovery_patience = recovery_patience
        self.recovery_threshold = recovery_threshold
        self.max_recoveries = max_recoveries

        self.best_value: Optional[float] = None
        self.counter: int = 0
        self.recovery_counter: int = 0
        self.recovery_count: int = 0
        self.in_recovery: bool = False

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check for early stopping with recovery support."""
        current = metrics.get(self.monitor)

        if current is None:
            return False

        if self.best_value is None:
            self.best_value = current
            return False

        if self.mode == EarlyStoppingMode.MIN:
            improved = current < (self.best_value - self.min_delta)
            threshold_value = self.best_value * self.recovery_threshold
        else:
            improved = current > (self.best_value + self.min_delta)
            threshold_value = self.best_value / self.recovery_threshold

        if improved:
            if self.in_recovery:
                self.recovery_counter = 0

            self.best_value = current
            self.counter = 0
            self.in_recovery = False
        else:
            self.counter += 1

            if self.in_recovery:
                self.recovery_counter += 1
            elif self.counter >= self.patience:
                if self.recovery_count < self.max_recoveries:
                    self.in_recovery = True
                    self.recovery_counter = 0
                    self.recovery_count += 1
                    if self.verbose:
                        print(f"Entering recovery mode at epoch {epoch}")
                else:
                    self.should_stop = True
                    if self.verbose:
                        print(f"Early stopping with recovery: max recoveries reached")
                    return True

            if self.in_recovery and self.recovery_counter >= self.recovery_patience:
                if (
                    self.mode == EarlyStoppingMode.MIN and current < threshold_value
                ) or (self.mode == EarlyStoppingMode.MAX and current > threshold_value):
                    self.in_recovery = False
                    self.best_value = current
                    if self.verbose:
                        print(f"Recovery successful at epoch {epoch}")
                else:
                    self.should_stop = True
                    if self.verbose:
                        print(f"Recovery failed at epoch {epoch}")
                    return True

            if not self.in_recovery and self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                return True

        return False


class CompositeEarlyStopping(EarlyStopping):
    """
    Composite early stopping with multiple conditions.

    Combines multiple early stopping conditions with logical operators.

    Example:
        >>> es1 = PatienceEarlyStopping(monitor="val_loss", patience=10)
        >>> es2 = DeltaEarlyStopping(monitor="val_loss", delta=0.001)
        >>> early_stopping = CompositeEarlyStopping(
        ...     [es1, es2],
        ...     mode="any",  # Stop if any condition triggers
        ... )
    """

    def __init__(
        self,
        stoppers: List[EarlyStopping],
        mode: str = "any",
        verbose: bool = True,
    ):
        super().__init__(monitor="composite", mode="min", verbose=verbose)

        self.stoppers = stoppers
        self.mode = mode

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check if any or all conditions are met."""
        results = [stopper(epoch, metrics) for stopper in self.stoppers]

        if self.mode == "any":
            should_stop = any(results)
        else:
            should_stop = all(results)

        if should_stop and self.verbose:
            triggered = [i for i, r in enumerate(results) if r]
            print(
                f"Composite early stopping at epoch {epoch}: conditions {triggered} triggered"
            )

        self.should_stop = should_stop
        return should_stop

    def reset(self) -> None:
        """Reset all component stoppers."""
        super().reset()
        for stopper in self.stoppers:
            stopper.reset()


class EarlyStoppingCallback:
    """
    Callback-style early stopping for training loops.

    Provides a callback interface for integration with training loops.

    Example:
        >>> callback = EarlyStoppingCallback(
        ...     monitor="val_loss",
        ...     patience=10,
        ...     on_stop=lambda: print("Training stopped!"),
        ... )
        >>> for epoch in range(100):
        ...     train()
        ...     callback.on_epoch_end(epoch, {"val_loss": 0.5})
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: Union[str, EarlyStoppingMode] = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
        verbose: bool = True,
        on_stop: Optional[Callable] = None,
        on_improve: Optional[Callable] = None,
    ):
        self.stopper = PatienceEarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
            restore_best=restore_best,
            verbose=verbose,
        )
        self.on_stop = on_stop
        self.on_improve = on_improve
        self.last_value: Optional[float] = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check for early stopping at epoch end.

        Args:
            epoch: Current epoch
            metrics: Current metrics

        Returns:
            True if training should stop
        """
        current = metrics.get(self.stopper.monitor)

        if current is not None and self.last_value is not None:
            if self.on_improve is not None:
                if self.stopper.mode == EarlyStoppingMode.MIN:
                    improved = current < self.last_value - self.stopper.min_delta
                else:
                    improved = current > self.last_value + self.stopper.min_delta

                if improved:
                    self.on_improve(epoch, current)

        self.last_value = current

        if self.stopper(epoch, metrics):
            if self.on_stop is not None:
                self.on_stop()
            return True

        return False

    def reset(self) -> None:
        """Reset the early stopping callback."""
        self.stopper.reset()
        self.last_value = None


@dataclass
class EarlyStoppingHistory:
    """Track and analyze early stopping history."""

    epochs: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    is_improvement: List[bool] = field(default_factory=list)
    should_stop: List[bool] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def add(
        self,
        epoch: int,
        value: float,
        is_improvement: bool,
        should_stop: bool,
    ) -> None:
        """Add an entry to history."""
        self.epochs.append(epoch)
        self.values.append(value)
        self.is_improvement.append(is_improvement)
        self.should_stop.append(should_stop)
        self.timestamps.append(time.time())

    def get_improvement_rate(self, window: int = 10) -> float:
        """Get improvement rate over recent window."""
        if len(self.is_improvement) < window:
            window = len(self.is_improvement)

        if window == 0:
            return 0.0

        return sum(self.is_improvement[-window:]) / window

    def get_convergence_epoch(self, threshold: float = 0.01) -> Optional[int]:
        """Get epoch where convergence started."""
        if len(self.values) < 2:
            return None

        for i in range(1, len(self.values)):
            if abs(self.values[i] - self.values[i - 1]) < threshold:
                return self.epochs[i]

        return None


class AdaptivePatienceEarlyStopping(EarlyStopping):
    """
    Adaptive patience early stopping.

    Dynamically adjusts patience based on training dynamics,
    increasing patience during high variance periods.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_patience: int = 5,
        max_patience: int = 20,
        mode: Union[str, EarlyStoppingMode] = "min",
        min_delta: float = 0.0,
        variance_threshold: float = 0.1,
        verbose: bool = True,
    ):
        super().__init__(monitor=monitor, mode=mode, verbose=verbose)

        self.min_patience = min_patience
        self.max_patience = max_patience
        self.min_delta = min_delta
        self.variance_threshold = variance_threshold

        self.history: List[float] = []
        self.best_value: Optional[float] = None
        self.counter: int = 0
        self.current_patience: int = min_patience

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check with adaptive patience adjustment."""
        current = metrics.get(self.monitor)

        if current is None:
            return False

        self.history.append(current)

        if len(self.history) > 10:
            self.history.pop(0)

        if self.best_value is None:
            self.best_value = current
            return False

        if self.mode == EarlyStoppingMode.MIN:
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.counter = 0
            self.current_patience = self.min_patience
        else:
            self.counter += 1

            if len(self.history) >= 5:
                variance = np.var(self.history[-5:])
                if variance > self.variance_threshold:
                    self.current_patience = min(
                        self.current_patience + 2, self.max_patience
                    )

        if self.counter >= self.current_patience:
            self.should_stop = True
            if self.verbose:
                print(f"Adaptive early stopping at epoch {epoch}")
            return True

        return False
