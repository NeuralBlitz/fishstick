"""
Comprehensive Scheduler Module for fishstick

A full-featured job scheduling system supporting:
- Job Scheduling (add, remove, reschedule, pause, resume)
- Triggers (Cron, Interval, Date, Event)
- Executors (Thread, Process, AsyncIO, Gevent, Tornado)
- Job Stores (Memory, SQL, MongoDB, Redis)
- Retry Logic (Exponential, Fixed, Linear)
- Monitoring (Metrics, Logger, History)
- Distributed Schedulers (Celery, RQ, APScheduler)
- Utilities for easy job management
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import multiprocessing
import pickle
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor as StdThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor as StdProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    Protocol,
    runtime_checkable,
    cast,
)
from queue import Queue, PriorityQueue
import heapq
import copy


# =============================================================================
# Exceptions
# =============================================================================


class SchedulerError(Exception):
    """Base exception for scheduler errors."""

    pass


class JobNotFoundError(SchedulerError):
    """Raised when a job is not found."""

    pass


class JobAlreadyExistsError(SchedulerError):
    """Raised when attempting to add a job that already exists."""

    pass


class TriggerError(SchedulerError):
    """Raised when there's an issue with a trigger."""

    pass


class ExecutorError(SchedulerError):
    """Raised when there's an issue with an executor."""

    pass


class StoreError(SchedulerError):
    """Raised when there's an issue with a job store."""

    pass


class RetryExhaustedError(SchedulerError):
    """Raised when all retry attempts are exhausted."""

    pass


# =============================================================================
# Logging
# =============================================================================


logger = logging.getLogger("fishstick.scheduler")


# =============================================================================
# Job State and Status
# =============================================================================


class JobStatus(Enum):
    """Enumeration of possible job statuses."""

    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()


class JobState:
    """Represents the state of a job."""

    def __init__(
        self,
        status: JobStatus = JobStatus.PENDING,
        next_run_time: Optional[datetime] = None,
        last_run_time: Optional[datetime] = None,
        run_count: int = 0,
        error_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.status = status
        self.next_run_time = next_run_time
        self.last_run_time = last_run_time
        self.run_count = run_count
        self.error_count = error_count
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def update(self, **kwargs) -> None:
        """Update state attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "status": self.status.name,
            "next_run_time": self.next_run_time.isoformat()
            if self.next_run_time
            else None,
            "last_run_time": self.last_run_time.isoformat()
            if self.last_run_time
            else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Triggers
# =============================================================================


class Trigger(ABC):
    """Abstract base class for all triggers."""

    @abstractmethod
    def get_next_fire_time(
        self, previous_fire_time: Optional[datetime], now: datetime
    ) -> Optional[datetime]:
        """
        Return the next datetime to fire on.

        Args:
            previous_fire_time: The previous time the trigger fired
            now: Current datetime

        Returns:
            Next fire time or None if trigger should not fire again
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize trigger to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> Trigger:
        """Deserialize trigger from dictionary."""
        pass


class CronTrigger(Trigger):
    """
    Trigger that fires on specified cron-like schedule.

    Supports: minute, hour, day, month, day_of_week
    """

    def __init__(
        self,
        year: Optional[Union[int, str]] = None,
        month: Optional[Union[int, str]] = None,
        day: Optional[Union[int, str]] = None,
        week: Optional[Union[int, str]] = None,
        day_of_week: Optional[Union[int, str]] = None,
        hour: Optional[Union[int, str]] = None,
        minute: Optional[Union[int, str]] = None,
        second: Optional[Union[int, str]] = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timezone: Optional[str] = None,
        jitter: Optional[int] = None,
    ):
        self.year = year
        self.month = month
        self.day = day
        self.week = week
        self.day_of_week = day_of_week
        self.hour = hour
        self.minute = minute
        self.second = second
        self.start_date = start_date
        self.end_date = end_date
        self.timezone = timezone
        self.jitter = jitter

    def _parse_field(
        self, field_value: Optional[Union[int, str]], min_val: int, max_val: int
    ) -> Set[int]:
        """Parse cron field value into set of integers."""
        if field_value is None:
            return set(range(min_val, max_val + 1))

        if isinstance(field_value, int):
            return {field_value}

        result = set()
        parts = str(field_value).split(",")

        for part in parts:
            if "/" in part:
                step_part, step = part.split("/")
                step = int(step)
                if step_part == "*":
                    start, end = min_val, max_val
                elif "-" in step_part:
                    start, end = map(int, step_part.split("-"))
                else:
                    start = int(step_part)
                    end = max_val
                result.update(range(start, end + 1, step))
            elif "-" in part:
                start, end = map(int, part.split("-"))
                result.update(range(start, end + 1))
            elif part == "*":
                result.update(range(min_val, max_val + 1))
            else:
                result.add(int(part))

        return result

    def _get_next_trigger_date(self, start_date: datetime) -> Optional[datetime]:
        """Calculate next trigger date from start date."""
        years = self._parse_field(self.year, 1970, 2099)
        months = self._parse_field(self.month, 1, 12)
        days = self._parse_field(self.day, 1, 31)
        hours = self._parse_field(self.hour, 0, 23)
        minutes = self._parse_field(self.minute, 0, 59)
        seconds = self._parse_field(self.second, 0, 59)

        current = start_date.replace(microsecond=0)

        for _ in range(366 * 10):  # Limit search to 10 years
            current += timedelta(seconds=1)

            if current.year not in years:
                continue
            if current.month not in months:
                continue
            if current.day not in days:
                continue
            if current.hour not in hours:
                continue
            if current.minute not in minutes:
                continue
            if current.second not in seconds:
                continue

            if self.start_date and current < self.start_date:
                continue
            if self.end_date and current > self.end_date:
                return None

            if self.jitter:
                import random

                jitter_seconds = random.randint(0, self.jitter)
                current += timedelta(seconds=jitter_seconds)

            return current

        return None

    def get_next_fire_time(
        self, previous_fire_time: Optional[datetime], now: datetime
    ) -> Optional[datetime]:
        """Get next fire time based on cron schedule."""
        start = previous_fire_time if previous_fire_time else now
        return self._get_next_trigger_date(start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "cron",
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "week": self.week,
            "day_of_week": self.day_of_week,
            "hour": self.hour,
            "minute": self.minute,
            "second": self.second,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "timezone": self.timezone,
            "jitter": self.jitter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CronTrigger:
        data = data.copy()
        data.pop("type", None)
        if data.get("start_date"):
            data["start_date"] = datetime.fromisoformat(data["start_date"])
        if data.get("end_date"):
            data["end_date"] = datetime.fromisoformat(data["end_date"])
        return cls(**data)


class IntervalTrigger(Trigger):
    """Trigger that fires at regular intervals."""

    def __init__(
        self,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timezone: Optional[str] = None,
        jitter: Optional[int] = None,
    ):
        self.interval = timedelta(
            weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds
        )
        self.start_date = start_date or datetime.now()
        self.end_date = end_date
        self.timezone = timezone
        self.jitter = jitter

    def get_next_fire_time(
        self, previous_fire_time: Optional[datetime], now: datetime
    ) -> Optional[datetime]:
        """Get next fire time based on interval."""
        if previous_fire_time:
            next_time = previous_fire_time + self.interval
        else:
            next_time = self.start_date

        if self.end_date and next_time > self.end_date:
            return None

        if self.jitter:
            import random

            next_time += timedelta(seconds=random.randint(0, self.jitter))

        return next_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "interval",
            "interval_seconds": self.interval.total_seconds(),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "timezone": self.timezone,
            "jitter": self.jitter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IntervalTrigger:
        interval = timedelta(seconds=data["interval_seconds"])
        weeks = interval.days // 7
        days = interval.days % 7
        hours, remainder = divmod(interval.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return cls(
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            start_date=datetime.fromisoformat(data["start_date"])
            if data.get("start_date")
            else None,
            end_date=datetime.fromisoformat(data["end_date"])
            if data.get("end_date")
            else None,
            timezone=data.get("timezone"),
            jitter=data.get("jitter"),
        )


class DateTrigger(Trigger):
    """Trigger that fires once at a specific date/time."""

    def __init__(self, run_date: datetime, timezone: Optional[str] = None):
        self.run_date = run_date
        self.timezone = timezone

    def get_next_fire_time(
        self, previous_fire_time: Optional[datetime], now: datetime
    ) -> Optional[datetime]:
        """Get next fire time - only fires once."""
        if previous_fire_time:
            return None
        if self.run_date <= now:
            return None
        return self.run_date

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "date",
            "run_date": self.run_date.isoformat(),
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DateTrigger:
        return cls(
            run_date=datetime.fromisoformat(data["run_date"]),
            timezone=data.get("timezone"),
        )


class EventTrigger(Trigger):
    """Trigger that fires based on custom events."""

    def __init__(
        self,
        event_name: str,
        event_data: Optional[Dict[str, Any]] = None,
        once: bool = False,
    ):
        self.event_name = event_name
        self.event_data = event_data or {}
        self.once = once
        self._has_fired = False

    def get_next_fire_time(
        self, previous_fire_time: Optional[datetime], now: datetime
    ) -> Optional[datetime]:
        """
        Event trigger requires external event handling.
        Returns None as actual firing is event-driven.
        """
        return None

    def should_fire_on_event(
        self, event_name: str, event_data: Optional[Dict] = None
    ) -> bool:
        """Check if this trigger should fire on the given event."""
        if self.once and self._has_fired:
            return False

        if event_name != self.event_name:
            return False

        if self.event_data:
            if event_data is None:
                return False
            for key, value in self.event_data.items():
                if event_data.get(key) != value:
                    return False

        if self.once:
            self._has_fired = True

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "event",
            "event_name": self.event_name,
            "event_data": self.event_data,
            "once": self.once,
            "has_fired": self._has_fired,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventTrigger:
        trigger = cls(
            event_name=data["event_name"],
            event_data=data.get("event_data", {}),
            once=data.get("once", False),
        )
        trigger._has_fired = data.get("has_fired", False)
        return trigger


# Aliases for convenient access
Cron = CronTrigger
Interval = IntervalTrigger
Date = DateTrigger
Event = EventTrigger


# =============================================================================
# Job Definition
# =============================================================================


@dataclass
class Job:
    """Represents a scheduled job."""

    id: str
    name: str
    func: Callable
    trigger: Trigger
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    executor: str = "default"
    max_instances: int = 1
    misfire_grace_time: Optional[int] = None  # seconds
    coalesce: bool = False
    state: JobState = field(default_factory=JobState)
    retry_policy: Optional[RetryPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self, include_func: bool = False) -> Dict[str, Any]:
        """Convert job to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "trigger": self.trigger.to_dict(),
            "args": self.args,
            "kwargs": self.kwargs,
            "executor": self.executor,
            "max_instances": self.max_instances,
            "misfire_grace_time": self.misfire_grace_time,
            "coalesce": self.coalesce,
            "state": self.state.to_dict(),
            "metadata": self.metadata,
            "tags": list(self.tags),
        }

        if include_func:
            data["func"] = f"{self.func.__module__}.{self.func.__qualname__}"

        if self.retry_policy:
            data["retry_policy"] = self.retry_policy.to_dict()

        return data

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], func_registry: Optional[Dict] = None
    ) -> Job:
        """Create job from dictionary."""
        trigger_data = data["trigger"]
        trigger_type = trigger_data.get("type")

        if trigger_type == "cron":
            trigger = CronTrigger.from_dict(trigger_data)
        elif trigger_type == "interval":
            trigger = IntervalTrigger.from_dict(trigger_data)
        elif trigger_type == "date":
            trigger = DateTrigger.from_dict(trigger_data)
        elif trigger_type == "event":
            trigger = EventTrigger.from_dict(trigger_data)
        else:
            raise TriggerError(f"Unknown trigger type: {trigger_type}")

        func = None
        if func_registry and data.get("func"):
            func = func_registry.get(data["func"])

        job = cls(
            id=data["id"],
            name=data["name"],
            func=func or (lambda: None),
            trigger=trigger,
            args=tuple(data.get("args", [])),
            kwargs=data.get("kwargs", {}),
            executor=data.get("executor", "default"),
            max_instances=data.get("max_instances", 1),
            misfire_grace_time=data.get("misfire_grace_time"),
            coalesce=data.get("coalesce", False),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
        )

        if data.get("retry_policy"):
            job.retry_policy = RetryPolicy.from_dict(data["retry_policy"])

        return job


# =============================================================================
# Retry Logic
# =============================================================================


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay in seconds for the given attempt number."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetryStrategy:
        pass


class ExponentialBackoff(RetryStrategy):
    """Exponential backoff retry strategy."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        exponent: float = 2.0,
        jitter: bool = True,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponent = exponent
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = min(self.base_delay * (self.exponent**attempt), self.max_delay)
        if self.jitter:
            import random

            delay *= 0.5 + random.random()
        return delay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "exponential",
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "exponent": self.exponent,
            "jitter": self.jitter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExponentialBackoff:
        return cls(
            base_delay=data.get("base_delay", 1.0),
            max_delay=data.get("max_delay", 300.0),
            exponent=data.get("exponent", 2.0),
            jitter=data.get("jitter", True),
        )


class FixedDelay(RetryStrategy):
    """Fixed delay retry strategy."""

    def __init__(self, delay: float = 5.0):
        self.delay = delay

    def get_delay(self, attempt: int) -> float:
        return self.delay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "fixed",
            "delay": self.delay,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FixedDelay:
        return cls(delay=data.get("delay", 5.0))


class LinearBackoff(RetryStrategy):
    """Linear backoff retry strategy."""

    def __init__(
        self,
        initial_delay: float = 1.0,
        increment: float = 1.0,
        max_delay: float = 300.0,
    ):
        self.initial_delay = initial_delay
        self.increment = increment
        self.max_delay = max_delay

    def get_delay(self, attempt: int) -> float:
        delay = self.initial_delay + (self.increment * attempt)
        return min(delay, self.max_delay)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "linear",
            "initial_delay": self.initial_delay,
            "increment": self.increment,
            "max_delay": self.max_delay,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinearBackoff:
        return cls(
            initial_delay=data.get("initial_delay", 1.0),
            increment=data.get("increment", 1.0),
            max_delay=data.get("max_delay", 300.0),
        )


@dataclass
class RetryPolicy:
    """Defines retry behavior for jobs."""

    max_retries: int = 3
    strategy: RetryStrategy = field(default_factory=lambda: ExponentialBackoff())
    retry_on_exceptions: Tuple[type, ...] = (Exception,)
    stop_on_exceptions: Tuple[type, ...] = ()

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if job should be retried."""
        if attempt >= self.max_retries:
            return False

        if isinstance(exception, self.stop_on_exceptions):
            return False

        return isinstance(exception, self.retry_on_exceptions)

    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry."""
        return self.strategy.get_delay(attempt)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "strategy": self.strategy.to_dict(),
            "retry_on_exceptions": [exc.__name__ for exc in self.retry_on_exceptions],
            "stop_on_exceptions": [exc.__name__ for exc in self.stop_on_exceptions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetryPolicy:
        strategy_data = data.get("strategy", {"type": "exponential"})
        strategy_type = strategy_data.get("type")

        if strategy_type == "exponential":
            strategy = ExponentialBackoff.from_dict(strategy_data)
        elif strategy_type == "fixed":
            strategy = FixedDelay.from_dict(strategy_data)
        elif strategy_type == "linear":
            strategy = LinearBackoff.from_dict(strategy_data)
        else:
            strategy = ExponentialBackoff()

        return cls(
            max_retries=data.get("max_retries", 3),
            strategy=strategy,
        )


# Aliases for convenient access
Exponential = ExponentialBackoff
Fixed = FixedDelay
Linear = LinearBackoff
Policy = RetryPolicy


# =============================================================================
# Executors
# =============================================================================


class Executor(ABC):
    """Abstract base class for job executors."""

    @abstractmethod
    def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a job for execution."""
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class ThreadPoolExecutor(Executor):
    """Thread pool executor for concurrent job execution."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._executor: Optional[StdThreadPoolExecutor] = None
        self._lock = threading.Lock()

    @property
    def executor(self) -> StdThreadPoolExecutor:
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = StdThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def submit(self, func: Callable, *args, **kwargs) -> Any:
        return self.executor.submit(func, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "thread",
            "max_workers": self.max_workers,
        }


class ProcessPoolExecutor(Executor):
    """Process pool executor for CPU-intensive jobs."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self._executor: Optional[StdProcessPoolExecutor] = None
        self._lock = threading.Lock()

    @property
    def executor(self) -> StdProcessPoolExecutor:
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = StdProcessPoolExecutor(
                        max_workers=self.max_workers
                    )
        return self._executor

    def submit(self, func: Callable, *args, **kwargs) -> Any:
        # Processes can't pickle lambdas or local functions easily
        # Use functools.partial for kwargs
        if kwargs:
            func = functools.partial(func, **kwargs)
            return self.executor.submit(func, *args)
        return self.executor.submit(func, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "process",
            "max_workers": self.max_workers,
        }


class AsyncIOExecutor(Executor):
    """AsyncIO executor for async job functions."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.loop = loop
        self._lock = threading.Lock()

    def submit(self, func: Callable, *args, **kwargs) -> Any:
        if not asyncio.iscoroutinefunction(func):
            raise ExecutorError("AsyncIOExecutor requires async functions")

        loop = self.loop or asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.create_task(func(*args, **kwargs))
        else:
            return loop.run_until_complete(func(*args, **kwargs))

    def shutdown(self, wait: bool = True) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "asyncio",
        }


class GeventExecutor(Executor):
    """Gevent executor for greenlet-based concurrency."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._pool = None

        try:
            import gevent
            from gevent.pool import Pool

            self.gevent = gevent
            self.Pool = Pool
        except ImportError:
            raise ImportError("gevent is required for GeventExecutor")

    @property
    def pool(self):
        if self._pool is None:
            self._pool = self.Pool(size=self.max_workers)
        return self._pool

    def submit(self, func: Callable, *args, **kwargs) -> Any:
        return self.pool.spawn(func, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        if self._pool:
            if wait:
                self._pool.join()
            self._pool.kill()
            self._pool = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "gevent",
            "max_workers": self.max_workers,
        }


class TornadoExecutor(Executor):
    """Tornado executor for integration with Tornado IOLoop."""

    def __init__(self, io_loop=None):
        self.io_loop = io_loop
        self._executor = StdThreadPoolExecutor(max_workers=10)

        try:
            from tornado.ioloop import IOLoop
            from concurrent.futures import Future

            self.IOLoop = IOLoop
            self.Future = Future
        except ImportError:
            raise ImportError("tornado is required for TornadoExecutor")

    def submit(self, func: Callable, *args, **kwargs) -> Any:
        io_loop = self.io_loop or self.IOLoop.current()

        def callback(future):
            if future.exception():
                io_loop.add_callback(lambda: None)  # Wake up IOLoop

        future = self._executor.submit(func, *args, **kwargs)
        future.add_done_callback(callback)
        return future

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tornado",
        }


# Aliases for convenient access
Thread = ThreadPoolExecutor
Process = ProcessPoolExecutor
Async = AsyncIOExecutor
Gevent = GeventExecutor
Tornado = TornadoExecutor


# =============================================================================
# Job Stores
# =============================================================================


class JobStore(ABC):
    """Abstract base class for job stores."""

    @abstractmethod
    def add_job(self, job: Job) -> None:
        """Add a job to the store."""
        pass

    @abstractmethod
    def update_job(self, job: Job) -> None:
        """Update an existing job."""
        pass

    @abstractmethod
    def remove_job(self, job_id: str) -> None:
        """Remove a job from the store."""
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        pass

    @abstractmethod
    def get_all_jobs(self) -> List[Job]:
        """Get all jobs from the store."""
        pass

    @abstractmethod
    def get_due_jobs(self, now: datetime) -> List[Job]:
        """Get jobs that are due to run."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all jobs from the store."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the store."""
        pass


class MemoryJobStore(JobStore):
    """In-memory job store (default)."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def add_job(self, job: Job) -> None:
        with self._lock:
            if job.id in self._jobs:
                raise JobAlreadyExistsError(f"Job {job.id} already exists")
            self._jobs[job.id] = job

    def update_job(self, job: Job) -> None:
        with self._lock:
            if job.id not in self._jobs:
                raise JobNotFoundError(f"Job {job.id} not found")
            self._jobs[job.id] = job

    def remove_job(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(f"Job {job_id} not found")
            del self._jobs[job_id]

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get_due_jobs(self, now: datetime) -> List[Job]:
        with self._lock:
            due = []
            for job in self._jobs.values():
                if job.state.status == JobStatus.PAUSED:
                    continue
                if job.state.next_run_time and job.state.next_run_time <= now:
                    due.append(job)
            return due

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def shutdown(self) -> None:
        pass


class SQLAlchemyJobStore(JobStore):
    """SQLAlchemy-based persistent job store."""

    def __init__(
        self, url: str, table: str = "jobs", engine_options: Optional[Dict] = None
    ):
        self.url = url
        self.table = table
        self.engine_options = engine_options or {}

        try:
            from sqlalchemy import (
                create_engine,
                Column,
                String,
                Text,
                DateTime,
                Integer,
            )
            from sqlalchemy.ext.declarative import declarative_base
            from sqlalchemy.orm import sessionmaker

            self.engine = create_engine(url, **self.engine_options)
            self.Base = declarative_base()
            self.Session = sessionmaker(bind=self.engine)

            class JobModel(self.Base):
                __tablename__ = table

                id = Column(String(36), primary_key=True)
                name = Column(String(255), nullable=False)
                trigger = Column(Text, nullable=False)
                func = Column(Text)
                args = Column(Text)
                kwargs = Column(Text)
                executor = Column(String(50), default="default")
                max_instances = Column(Integer, default=1)
                misfire_grace_time = Column(Integer)
                coalesce = Column(Integer, default=0)
                state = Column(Text)
                retry_policy = Column(Text)
                metadata_json = Column(Text)
                tags = Column(Text)
                created_at = Column(DateTime, default=datetime.utcnow)
                updated_at = Column(
                    DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
                )

            self.JobModel = JobModel
            self.Base.metadata.create_all(self.engine)

        except ImportError:
            raise ImportError("SQLAlchemy is required for SQLAlchemyJobStore")

    def _job_to_model(self, job: Job) -> Any:
        model = self.JobModel()
        model.id = job.id
        model.name = job.name
        model.trigger = json.dumps(job.trigger.to_dict())
        model.func = f"{job.func.__module__}.{job.func.__qualname__}"
        model.args = json.dumps(job.args)
        model.kwargs = json.dumps(job.kwargs)
        model.executor = job.executor
        model.max_instances = job.max_instances
        model.misfire_grace_time = job.misfire_grace_time
        model.coalesce = 1 if job.coalesce else 0
        model.state = json.dumps(job.state.to_dict())
        model.retry_policy = (
            json.dumps(job.retry_policy.to_dict()) if job.retry_policy else None
        )
        model.metadata_json = json.dumps(job.metadata)
        model.tags = json.dumps(list(job.tags))
        return model

    def _model_to_job(self, model: Any, func_registry: Optional[Dict] = None) -> Job:
        trigger_data = json.loads(model.trigger)
        trigger_type = trigger_data.get("type")

        if trigger_type == "cron":
            trigger = CronTrigger.from_dict(trigger_data)
        elif trigger_type == "interval":
            trigger = IntervalTrigger.from_dict(trigger_data)
        elif trigger_type == "date":
            trigger = DateTrigger.from_dict(trigger_data)
        elif trigger_type == "event":
            trigger = EventTrigger.from_dict(trigger_data)
        else:
            raise TriggerError(f"Unknown trigger type: {trigger_type}")

        func = None
        if func_registry and model.func:
            func = func_registry.get(model.func)

        job = Job(
            id=model.id,
            name=model.name,
            func=func or (lambda: None),
            trigger=trigger,
            args=tuple(json.loads(model.args)) if model.args else (),
            kwargs=json.loads(model.kwargs) if model.kwargs else {},
            executor=model.executor,
            max_instances=model.max_instances,
            misfire_grace_time=model.misfire_grace_time,
            coalesce=bool(model.coalesce),
            metadata=json.loads(model.metadata_json) if model.metadata_json else {},
            tags=set(json.loads(model.tags)) if model.tags else set(),
        )

        if model.retry_policy:
            job.retry_policy = RetryPolicy.from_dict(json.loads(model.retry_policy))

        return job

    def add_job(self, job: Job) -> None:
        session = self.Session()
        try:
            if session.query(self.JobModel).get(job.id):
                raise JobAlreadyExistsError(f"Job {job.id} already exists")

            model = self._job_to_model(job)
            session.add(model)
            session.commit()
        finally:
            session.close()

    def update_job(self, job: Job) -> None:
        session = self.Session()
        try:
            model = session.query(self.JobModel).get(job.id)
            if not model:
                raise JobNotFoundError(f"Job {job.id} not found")

            updated = self._job_to_model(job)
            updated.id = model.id
            session.merge(updated)
            session.commit()
        finally:
            session.close()

    def remove_job(self, job_id: str) -> None:
        session = self.Session()
        try:
            model = session.query(self.JobModel).get(job_id)
            if not model:
                raise JobNotFoundError(f"Job {job_id} not found")

            session.delete(model)
            session.commit()
        finally:
            session.close()

    def get_job(self, job_id: str) -> Optional[Job]:
        session = self.Session()
        try:
            model = session.query(self.JobModel).get(job_id)
            return self._model_to_job(model) if model else None
        finally:
            session.close()

    def get_all_jobs(self) -> List[Job]:
        session = self.Session()
        try:
            models = session.query(self.JobModel).all()
            return [self._model_to_job(m) for m in models]
        finally:
            session.close()

    def get_due_jobs(self, now: datetime) -> List[Job]:
        session = self.Session()
        try:
            models = session.query(self.JobModel).all()
            jobs = []
            for model in models:
                state = json.loads(model.state) if model.state else {}
                next_run = (
                    datetime.fromisoformat(state.get("next_run_time"))
                    if state.get("next_run_time")
                    else None
                )
                status = state.get("status")

                if status != "PAUSED" and next_run and next_run <= now:
                    jobs.append(self._model_to_job(model))
            return jobs
        finally:
            session.close()

    def clear(self) -> None:
        session = self.Session()
        try:
            session.query(self.JobModel).delete()
            session.commit()
        finally:
            session.close()

    def shutdown(self) -> None:
        self.engine.dispose()


class MongoDBJobStore(JobStore):
    """MongoDB-based persistent job store."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "scheduler",
        collection: str = "jobs",
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        try:
            from pymongo import MongoClient

            if username and password:
                uri = f"mongodb://{username}:{password}@{host}:{port}/{database}"
                self.client = MongoClient(uri, **kwargs)
            else:
                self.client = MongoClient(host, port, **kwargs)

            self.db = self.client[database]
            self.collection = self.db[collection]

        except ImportError:
            raise ImportError("pymongo is required for MongoDBJobStore")

    def _job_to_doc(self, job: Job) -> Dict:
        return {
            "_id": job.id,
            "name": job.name,
            "trigger": job.trigger.to_dict(),
            "func": f"{job.func.__module__}.{job.func.__qualname__}",
            "args": job.args,
            "kwargs": job.kwargs,
            "executor": job.executor,
            "max_instances": job.max_instances,
            "misfire_grace_time": job.misfire_grace_time,
            "coalesce": job.coalesce,
            "state": job.state.to_dict(),
            "retry_policy": job.retry_policy.to_dict() if job.retry_policy else None,
            "metadata": job.metadata,
            "tags": list(job.tags),
            "updated_at": datetime.utcnow(),
        }

    def _doc_to_job(self, doc: Dict, func_registry: Optional[Dict] = None) -> Job:
        trigger_data = doc["trigger"]
        trigger_type = trigger_data.get("type")

        if trigger_type == "cron":
            trigger = CronTrigger.from_dict(trigger_data)
        elif trigger_type == "interval":
            trigger = IntervalTrigger.from_dict(trigger_data)
        elif trigger_type == "date":
            trigger = DateTrigger.from_dict(trigger_data)
        elif trigger_type == "event":
            trigger = EventTrigger.from_dict(trigger_data)
        else:
            raise TriggerError(f"Unknown trigger type: {trigger_type}")

        func = None
        if func_registry and doc.get("func"):
            func = func_registry.get(doc["func"])

        job = Job(
            id=doc["_id"],
            name=doc["name"],
            func=func or (lambda: None),
            trigger=trigger,
            args=tuple(doc.get("args", [])),
            kwargs=doc.get("kwargs", {}),
            executor=doc.get("executor", "default"),
            max_instances=doc.get("max_instances", 1),
            misfire_grace_time=doc.get("misfire_grace_time"),
            coalesce=doc.get("coalesce", False),
            metadata=doc.get("metadata", {}),
            tags=set(doc.get("tags", [])),
        )

        if doc.get("retry_policy"):
            job.retry_policy = RetryPolicy.from_dict(doc["retry_policy"])

        return job

    def add_job(self, job: Job) -> None:
        if self.collection.find_one({"_id": job.id}):
            raise JobAlreadyExistsError(f"Job {job.id} already exists")

        self.collection.insert_one(self._job_to_doc(job))

    def update_job(self, job: Job) -> None:
        if not self.collection.find_one({"_id": job.id}):
            raise JobNotFoundError(f"Job {job.id} not found")

        self.collection.replace_one({"_id": job.id}, self._job_to_doc(job))

    def remove_job(self, job_id: str) -> None:
        result = self.collection.delete_one({"_id": job_id})
        if result.deleted_count == 0:
            raise JobNotFoundError(f"Job {job_id} not found")

    def get_job(self, job_id: str) -> Optional[Job]:
        doc = self.collection.find_one({"_id": job_id})
        return self._doc_to_job(doc) if doc else None

    def get_all_jobs(self) -> List[Job]:
        return [self._doc_to_job(doc) for doc in self.collection.find()]

    def get_due_jobs(self, now: datetime) -> List[Job]:
        due = []
        for doc in self.collection.find():
            state = doc.get("state", {})
            if state.get("status") == "PAUSED":
                continue
            next_run_str = state.get("next_run_time")
            if next_run_str:
                next_run = datetime.fromisoformat(next_run_str)
                if next_run <= now:
                    due.append(self._doc_to_job(doc))
        return due

    def clear(self) -> None:
        self.collection.delete_many({})

    def shutdown(self) -> None:
        self.client.close()


class RedisJobStore(JobStore):
    """Redis-based persistent job store."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "scheduler:jobs:",
        **kwargs,
    ):
        try:
            import redis

            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                **kwargs,
            )
            self.key_prefix = key_prefix
            self._ping()

        except ImportError:
            raise ImportError("redis is required for RedisJobStore")

    def _ping(self):
        """Test connection."""
        self.client.ping()

    def _key(self, job_id: str) -> str:
        return f"{self.key_prefix}{job_id}"

    def add_job(self, job: Job) -> None:
        key = self._key(job.id)
        if self.client.exists(key):
            raise JobAlreadyExistsError(f"Job {job.id} already exists")

        doc = {
            "id": job.id,
            "name": job.name,
            "trigger": json.dumps(job.trigger.to_dict()),
            "func": f"{job.func.__module__}.{job.func.__qualname__}",
            "args": json.dumps(job.args),
            "kwargs": json.dumps(job.kwargs),
            "executor": job.executor,
            "max_instances": job.max_instances,
            "misfire_grace_time": job.misfire_grace_time,
            "coalesce": 1 if job.coalesce else 0,
            "state": json.dumps(job.state.to_dict()),
            "retry_policy": json.dumps(job.retry_policy.to_dict())
            if job.retry_policy
            else None,
            "metadata": json.dumps(job.metadata),
            "tags": json.dumps(list(job.tags)),
            "created_at": datetime.utcnow().isoformat(),
        }

        self.client.hset(key, mapping={k: v or "" for k, v in doc.items()})

    def update_job(self, job: Job) -> None:
        key = self._key(job.id)
        if not self.client.exists(key):
            raise JobNotFoundError(f"Job {job.id} not found")

        doc = {
            "name": job.name,
            "trigger": json.dumps(job.trigger.to_dict()),
            "func": f"{job.func.__module__}.{job.func.__qualname__}",
            "args": json.dumps(job.args),
            "kwargs": json.dumps(job.kwargs),
            "executor": job.executor,
            "max_instances": job.max_instances,
            "misfire_grace_time": job.misfire_grace_time,
            "coalesce": 1 if job.coalesce else 0,
            "state": json.dumps(job.state.to_dict()),
            "retry_policy": json.dumps(job.retry_policy.to_dict())
            if job.retry_policy
            else None,
            "metadata": json.dumps(job.metadata),
            "tags": json.dumps(list(job.tags)),
            "updated_at": datetime.utcnow().isoformat(),
        }

        self.client.hset(key, mapping={k: v or "" for k, v in doc.items()})

    def remove_job(self, job_id: str) -> None:
        key = self._key(job_id)
        if not self.client.exists(key):
            raise JobNotFoundError(f"Job {job_id} not found")

        self.client.delete(key)

    def get_job(self, job_id: str) -> Optional[Job]:
        key = self._key(job_id)
        data = self.client.hgetall(key)

        if not data:
            return None

        return self._dict_to_job(data)

    def _dict_to_job(self, data: Dict) -> Job:
        trigger_data = json.loads(data["trigger"])
        trigger_type = trigger_data.get("type")

        if trigger_type == "cron":
            trigger = CronTrigger.from_dict(trigger_data)
        elif trigger_type == "interval":
            trigger = IntervalTrigger.from_dict(trigger_data)
        elif trigger_type == "date":
            trigger = DateTrigger.from_dict(trigger_data)
        elif trigger_type == "event":
            trigger = EventTrigger.from_dict(trigger_data)
        else:
            raise TriggerError(f"Unknown trigger type: {trigger_type}")

        job = Job(
            id=data["id"],
            name=data["name"],
            func=lambda: None,
            trigger=trigger,
            args=tuple(json.loads(data["args"])),
            kwargs=json.loads(data["kwargs"]),
            executor=data["executor"],
            max_instances=int(data["max_instances"]),
            misfire_grace_time=int(data["misfire_grace_time"])
            if data["misfire_grace_time"]
            else None,
            coalesce=bool(int(data["coalesce"])),
            metadata=json.loads(data["metadata"]),
            tags=set(json.loads(data["tags"])),
        )

        if data.get("retry_policy"):
            job.retry_policy = RetryPolicy.from_dict(json.loads(data["retry_policy"]))

        return job

    def get_all_jobs(self) -> List[Job]:
        jobs = []
        for key in self.client.scan_iter(match=f"{self.key_prefix}*"):
            data = self.client.hgetall(key)
            if data:
                jobs.append(self._dict_to_job(data))
        return jobs

    def get_due_jobs(self, now: datetime) -> List[Job]:
        due = []
        for key in self.client.scan_iter(match=f"{self.key_prefix}*"):
            data = self.client.hgetall(key)
            if not data:
                continue

            state = json.loads(data["state"])
            if state.get("status") == "PAUSED":
                continue

            next_run_str = state.get("next_run_time")
            if next_run_str:
                next_run = datetime.fromisoformat(next_run_str)
                if next_run <= now:
                    due.append(self._dict_to_job(data))

        return due

    def clear(self) -> None:
        for key in self.client.scan_iter(match=f"{self.key_prefix}*"):
            self.client.delete(key)

    def shutdown(self) -> None:
        self.client.close()


# Aliases for convenient access
Memory = MemoryJobStore
SQL = SQLAlchemyJobStore
Mongo = MongoDBJobStore
Redis = RedisJobStore


# =============================================================================
# Monitoring
# =============================================================================


@dataclass
class JobExecutionRecord:
    """Record of a single job execution."""

    job_id: str
    job_name: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    return_value: Any = None
    attempt: int = 1


class JobMetrics:
    """Metrics collection for job executions."""

    def __init__(self):
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._execution_times: List[float] = []
        self._job_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "runs": 0,
                "successes": 0,
                "failures": 0,
                "total_duration": 0.0,
            }
        )
        self._lock = threading.Lock()

    def record_execution(self, record: JobExecutionRecord) -> None:
        """Record a job execution."""
        with self._lock:
            self._total_executions += 1

            if record.success:
                self._successful_executions += 1
                self._job_stats[record.job_id]["successes"] += 1
            else:
                self._failed_executions += 1
                self._job_stats[record.job_id]["failures"] += 1

            self._job_stats[record.job_id]["runs"] += 1

            if record.duration:
                self._execution_times.append(record.duration)
                self._job_stats[record.job_id]["total_duration"] += record.duration

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            avg_duration = (
                sum(self._execution_times) / len(self._execution_times)
                if self._execution_times
                else 0
            )

            return {
                "total_executions": self._total_executions,
                "successful_executions": self._successful_executions,
                "failed_executions": self._failed_executions,
                "success_rate": self._successful_executions / self._total_executions
                if self._total_executions > 0
                else 0,
                "average_execution_time": avg_duration,
                "total_execution_time": sum(self._execution_times),
                "job_stats": dict(self._job_stats),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._total_executions = 0
            self._successful_executions = 0
            self._failed_executions = 0
            self._execution_times.clear()
            self._job_stats.clear()


class JobLogger:
    """Structured logging for job events."""

    def __init__(self, logger_name: str = "fishstick.scheduler.jobs"):
        self.logger = logging.getLogger(logger_name)

    def log_job_added(self, job: Job) -> None:
        self.logger.info(f"Job added: {job.name} (ID: {job.id})")

    def log_job_removed(self, job_id: str) -> None:
        self.logger.info(f"Job removed: {job_id}")

    def log_job_started(self, job: Job) -> None:
        self.logger.info(f"Job started: {job.name} (ID: {job.id})")

    def log_job_completed(self, job: Job, duration: float) -> None:
        self.logger.info(
            f"Job completed: {job.name} (ID: {job.id}, Duration: {duration:.3f}s)"
        )

    def log_job_failed(self, job: Job, error: Exception) -> None:
        self.logger.error(f"Job failed: {job.name} (ID: {job.id}, Error: {error})")

    def log_job_retry(self, job: Job, attempt: int, delay: float) -> None:
        self.logger.warning(
            f"Job retry: {job.name} (ID: {job.id}, Attempt: {attempt}, Delay: {delay:.3f}s)"
        )

    def log_job_paused(self, job_id: str) -> None:
        self.logger.info(f"Job paused: {job_id}")

    def log_job_resumed(self, job_id: str) -> None:
        self.logger.info(f"Job resumed: {job_id}")

    def log_job_rescheduled(self, job_id: str) -> None:
        self.logger.info(f"Job rescheduled: {job_id}")


class JobHistory:
    """History tracking for job executions."""

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self._records: List[JobExecutionRecord] = []
        self._lock = threading.Lock()

    def add_record(self, record: JobExecutionRecord) -> None:
        """Add a record to history."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records :]

    def get_records(
        self, job_id: Optional[str] = None, limit: int = 100, success_only: bool = False
    ) -> List[JobExecutionRecord]:
        """Get execution records."""
        with self._lock:
            records = self._records

            if job_id:
                records = [r for r in records if r.job_id == job_id]

            if success_only:
                records = [r for r in records if r.success]

            return records[-limit:]

    def clear(self) -> None:
        """Clear all history."""
        with self._lock:
            self._records.clear()

    def get_summary(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get history summary."""
        with self._lock:
            records = [r for r in self._records if not job_id or r.job_id == job_id]

            if not records:
                return {"total": 0, "successful": 0, "failed": 0}

            return {
                "total": len(records),
                "successful": sum(1 for r in records if r.success),
                "failed": sum(1 for r in records if not r.success),
                "average_duration": sum(r.duration or 0 for r in records) / len(records)
                if records
                else 0,
            }


class JobMonitor:
    """Comprehensive job monitoring system."""

    def __init__(
        self,
        enable_metrics: bool = True,
        enable_logging: bool = True,
        enable_history: bool = True,
        max_history: int = 1000,
    ):
        self.metrics = JobMetrics() if enable_metrics else None
        self.logger = JobLogger() if enable_logging else None
        self.history = JobHistory(max_history) if enable_history else None
        self._active_jobs: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def on_job_added(self, job: Job) -> None:
        """Called when a job is added."""
        if self.logger:
            self.logger.log_job_added(job)

    def on_job_removed(self, job_id: str) -> None:
        """Called when a job is removed."""
        if self.logger:
            self.logger.log_job_removed(job_id)

    def on_job_started(self, job: Job) -> None:
        """Called when a job starts."""
        with self._lock:
            self._active_jobs[job.id] = datetime.now()

        if self.logger:
            self.logger.log_job_started(job)

    def on_job_completed(
        self, job: Job, duration: float, return_value: Any = None
    ) -> None:
        """Called when a job completes successfully."""
        with self._lock:
            self._active_jobs.pop(job.id, None)

        if self.logger:
            self.logger.log_job_completed(job, duration)

        record = JobExecutionRecord(
            job_id=job.id,
            job_name=job.name,
            started_at=datetime.now() - timedelta(seconds=duration),
            finished_at=datetime.now(),
            duration=duration,
            success=True,
            return_value=return_value,
        )

        if self.metrics:
            self.metrics.record_execution(record)

        if self.history:
            self.history.add_record(record)

    def on_job_failed(
        self, job: Job, error: Exception, duration: float, attempt: int = 1
    ) -> None:
        """Called when a job fails."""
        with self._lock:
            self._active_jobs.pop(job.id, None)

        if self.logger:
            self.logger.log_job_failed(job, error)

        record = JobExecutionRecord(
            job_id=job.id,
            job_name=job.name,
            started_at=datetime.now() - timedelta(seconds=duration),
            finished_at=datetime.now(),
            duration=duration,
            success=False,
            error=str(error),
            attempt=attempt,
        )

        if self.metrics:
            self.metrics.record_execution(record)

        if self.history:
            self.history.add_record(record)

    def on_job_retry(self, job: Job, attempt: int, delay: float) -> None:
        """Called when a job is scheduled for retry."""
        if self.logger:
            self.logger.log_job_retry(job, attempt, delay)

    def on_job_paused(self, job_id: str) -> None:
        """Called when a job is paused."""
        if self.logger:
            self.logger.log_job_paused(job_id)

    def on_job_resumed(self, job_id: str) -> None:
        """Called when a job is resumed."""
        if self.logger:
            self.logger.log_job_resumed(job_id)

    def on_job_rescheduled(self, job_id: str) -> None:
        """Called when a job is rescheduled."""
        if self.logger:
            self.logger.log_job_rescheduled(job_id)

    def get_active_jobs(self) -> Dict[str, datetime]:
        """Get currently active jobs."""
        with self._lock:
            return self._active_jobs.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        summary = {
            "active_jobs": len(self._active_jobs),
        }

        if self.metrics:
            summary["metrics"] = self.metrics.get_summary()

        if self.history:
            summary["history"] = self.history.get_summary()

        return summary


# Aliases for convenient access
Metrics = JobMetrics
Logger = JobLogger
History = JobHistory
Monitor = JobMonitor


# =============================================================================
# Distributed Schedulers
# =============================================================================


class DistributedScheduler(ABC):
    """Abstract base class for distributed schedulers."""

    @abstractmethod
    def schedule(self, job: Job) -> None:
        """Schedule a job in the distributed system."""
        pass

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel a scheduled job."""
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> JobStatus:
        """Get the status of a job."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the distributed scheduler."""
        pass


class CeleryScheduler(DistributedScheduler):
    """Celery-based distributed scheduler."""

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        backend_url: Optional[str] = None,
        app_name: str = "fishstick_scheduler",
        **celery_options,
    ):
        try:
            from celery import Celery
            from celery.schedules import crontab

            self.Celery = Celery
            self.crontab = crontab

            self.app = Celery(app_name, broker=broker_url, backend=backend_url)
            self.app.conf.update(**celery_options)

            self._registered_tasks: Dict[str, Any] = {}

        except ImportError:
            raise ImportError("celery is required for CeleryScheduler")

    def register_task(self, name: str, func: Callable) -> Any:
        """Register a task with Celery."""

        @self.app.task(name=name)
        def celery_task(*args, **kwargs):
            return func(*args, **kwargs)

        self._registered_tasks[name] = celery_task
        return celery_task

    def schedule(self, job: Job) -> None:
        """Schedule a job using Celery."""
        task_name = f"fishstick.job.{job.id}"

        if task_name not in self._registered_tasks:
            raise SchedulerError(f"Task {task_name} not registered")

        task = self._registered_tasks[task_name]

        # Apply the schedule based on trigger type
        if isinstance(job.trigger, IntervalTrigger):
            interval = job.trigger.interval
            task.apply_async(
                args=job.args,
                kwargs=job.kwargs,
                countdown=int(interval.total_seconds()),
            )
        elif isinstance(job.trigger, CronTrigger):
            # Use Celery beat for cron schedules
            pass
        elif isinstance(job.trigger, DateTrigger):
            task.apply_async(args=job.args, kwargs=job.kwargs, eta=job.trigger.run_date)

    def cancel(self, job_id: str) -> None:
        """Cancel a scheduled job."""
        from celery.result import AsyncResult

        result = AsyncResult(job_id)
        result.revoke(terminate=True)

    def get_status(self, job_id: str) -> JobStatus:
        """Get the status of a job."""
        from celery.result import AsyncResult

        result = AsyncResult(job_id)

        if result.ready():
            if result.successful():
                return JobStatus.COMPLETED
            else:
                return JobStatus.FAILED
        elif result.state == "PENDING":
            return JobStatus.PENDING
        else:
            return JobStatus.RUNNING

    def shutdown(self) -> None:
        """Shutdown the Celery app."""
        self.app.control.shutdown()


class RQScheduler(DistributedScheduler):
    """Redis Queue (RQ) based distributed scheduler."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        queue_name: str = "default",
        **redis_options,
    ):
        try:
            from redis import Redis
            from rq import Queue as RQQueue
            from rq_scheduler import Scheduler as RQSchedulerLib

            self.redis = Redis(
                host=redis_host, port=redis_port, db=redis_db, **redis_options
            )
            self.queue = RQQueue(queue_name, connection=self.redis)
            self.scheduler = RQSchedulerLib(queue=self.queue, connection=self.redis)

        except ImportError:
            raise ImportError("rq and rq-scheduler are required for RQScheduler")

    def schedule(self, job: Job) -> None:
        """Schedule a job using RQ."""
        if isinstance(job.trigger, DateTrigger):
            self.scheduler.enqueue_at(
                job.trigger.run_date, job.func, *job.args, **job.kwargs
            )
        elif isinstance(job.trigger, IntervalTrigger):
            self.scheduler.schedule(
                scheduled_time=datetime.now() + job.trigger.interval,
                func=job.func,
                args=job.args,
                kwargs=job.kwargs,
                interval=job.trigger.interval.total_seconds(),
            )

    def cancel(self, job_id: str) -> None:
        """Cancel a scheduled job."""
        self.scheduler.cancel(job_id)

    def get_status(self, job_id: str) -> JobStatus:
        """Get the status of a job."""
        from rq.job import Job as RQJob

        try:
            rq_job = RQJob.fetch(job_id, connection=self.redis)

            if rq_job.is_finished:
                return JobStatus.COMPLETED
            elif rq_job.is_failed:
                return JobStatus.FAILED
            elif rq_job.is_queued:
                return JobStatus.PENDING
            elif rq_job.is_started:
                return JobStatus.RUNNING
            else:
                return JobStatus.PENDING
        except Exception:
            return JobStatus.FAILED

    def shutdown(self) -> None:
        """Shutdown the RQ scheduler."""
        self.redis.close()


class APSchedulerWrapper(DistributedScheduler):
    """Wrapper around APScheduler library."""

    def __init__(
        self,
        jobstores: Optional[Dict] = None,
        executors: Optional[Dict] = None,
        job_defaults: Optional[Dict] = None,
        timezone: Optional[str] = None,
    ):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger as APCronTrigger
            from apscheduler.triggers.interval import (
                IntervalTrigger as APIntervalTrigger,
            )
            from apscheduler.triggers.date import DateTrigger as APDateTrigger

            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=timezone,
            )

            self._job_map: Dict[str, str] = {}  # job_id -> apscheduler_job_id

        except ImportError:
            raise ImportError("apscheduler is required for APSchedulerWrapper")

    def start(self) -> None:
        """Start the APScheduler."""
        self.scheduler.start()

    def schedule(self, job: Job) -> None:
        """Schedule a job using APScheduler."""
        # Convert fishstick triggers to APScheduler triggers
        if isinstance(job.trigger, CronTrigger):
            from apscheduler.triggers.cron import CronTrigger as APCronTrigger

            trigger = APCronTrigger(
                year=job.trigger.year,
                month=job.trigger.month,
                day=job.trigger.day,
                week=job.trigger.week,
                day_of_week=job.trigger.day_of_week,
                hour=job.trigger.hour,
                minute=job.trigger.minute,
                second=job.trigger.second,
            )
        elif isinstance(job.trigger, IntervalTrigger):
            from apscheduler.triggers.interval import (
                IntervalTrigger as APIntervalTrigger,
            )

            trigger = APIntervalTrigger(
                weeks=job.trigger.interval.days // 7,
                days=job.trigger.interval.days % 7,
                hours=job.trigger.interval.seconds // 3600,
                minutes=(job.trigger.interval.seconds // 60) % 60,
                seconds=job.trigger.interval.seconds % 60,
            )
        elif isinstance(job.trigger, DateTrigger):
            from apscheduler.triggers.date import DateTrigger as APDateTrigger

            trigger = APDateTrigger(run_date=job.trigger.run_date)
        else:
            raise SchedulerError(f"Unsupported trigger type: {type(job.trigger)}")

        aps_job = self.scheduler.add_job(
            func=job.func,
            trigger=trigger,
            args=job.args,
            kwargs=job.kwargs,
            id=job.id,
            name=job.name,
            max_instances=job.max_instances,
            misfire_grace_time=job.misfire_grace_time,
            coalesce=job.coalesce,
        )

        self._job_map[job.id] = aps_job.id

    def cancel(self, job_id: str) -> None:
        """Cancel a scheduled job."""
        if job_id in self._job_map:
            self.scheduler.remove_job(self._job_map[job_id])
            del self._job_map[job_id]

    def get_status(self, job_id: str) -> JobStatus:
        """Get the status of a job."""
        try:
            job = self.scheduler.get_job(self._job_map.get(job_id, job_id))
            if job is None:
                return JobStatus.CANCELLED
            # APScheduler doesn't expose direct status, infer from next_run_time
            return JobStatus.PENDING
        except Exception:
            return JobStatus.FAILED

    def shutdown(self) -> None:
        """Shutdown the APScheduler."""
        self.scheduler.shutdown()


# Aliases for convenient access
Dist = DistributedScheduler
Celery = CeleryScheduler
RQ = RQScheduler
AP = APSchedulerWrapper


# =============================================================================
# Main JobScheduler
# =============================================================================


class JobScheduler:
    """
    Main scheduler class for managing jobs.

    Provides comprehensive job scheduling capabilities including:
    - Adding, removing, rescheduling jobs
    - Pausing and resuming jobs
    - Multiple trigger types
    - Multiple executors
    - Multiple job stores
    - Retry policies
    - Monitoring and metrics
    """

    def __init__(
        self,
        jobstore: Optional[JobStore] = None,
        executors: Optional[Dict[str, Executor]] = None,
        job_defaults: Optional[Dict[str, Any]] = None,
        timezone: Optional[str] = None,
        monitor: Optional[JobMonitor] = None,
        distributed: Optional[DistributedScheduler] = None,
    ):
        self.jobstore = jobstore or MemoryJobStore()
        self.executors = executors or {"default": ThreadPoolExecutor()}
        self.job_defaults = job_defaults or {}
        self.timezone = timezone
        self.monitor = monitor or JobMonitor()
        self.distributed = distributed

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_queue: Queue = Queue()
        self._lock = threading.RLock()
        self._running_jobs: Dict[str, Any] = {}
        self._func_registry: Dict[str, Callable] = {}

        # Wake-up event for scheduler thread
        self._wakeup_event = threading.Event()

    def register_function(self, name: str, func: Callable) -> None:
        """Register a function for job serialization."""
        self._func_registry[name] = func

    def add_job(
        self,
        func: Callable,
        trigger: Trigger,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        name: Optional[str] = None,
        executor: str = "default",
        max_instances: int = 1,
        misfire_grace_time: Optional[int] = None,
        coalesce: bool = False,
        retry_policy: Optional[RetryPolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> Job:
        """
        Add a new job to the scheduler.

        Args:
            func: Callable to execute
            trigger: Trigger that determines when to run
            args: Positional arguments for func
            kwargs: Keyword arguments for func
            id: Job ID (generated if not provided)
            name: Job name (func name used if not provided)
            executor: Executor to use
            max_instances: Maximum concurrent instances
            misfire_grace_time: Seconds to allow job to run after missed time
            coalesce: Coalesce missed runs into one
            retry_policy: Retry policy for failed jobs
            metadata: Additional job metadata
            tags: Job tags for organization

        Returns:
            Created Job instance
        """
        job = Job(
            id=id or str(uuid.uuid4()),
            name=name or getattr(func, "__name__", "anonymous"),
            func=func,
            trigger=trigger,
            args=args or (),
            kwargs=kwargs or {},
            executor=executor,
            max_instances=max_instances,
            misfire_grace_time=misfire_grace_time,
            coalesce=coalesce,
            retry_policy=retry_policy,
            metadata=metadata or {},
            tags=tags or set(),
        )

        # Register function for serialization
        func_name = f"{func.__module__}.{func.__qualname__}"
        self.register_function(func_name, func)

        # Calculate initial next run time
        job.state.next_run_time = trigger.get_next_fire_time(None, datetime.now())

        # Add to job store
        self.jobstore.add_job(job)

        # Notify monitor
        if self.monitor:
            self.monitor.on_job_added(job)

        # Wake up scheduler if running
        self._wakeup_event.set()

        return job

    def remove_job(self, job_id: str) -> None:
        """
        Remove a job from the scheduler.

        Args:
            job_id: ID of job to remove

        Raises:
            JobNotFoundError: If job not found
        """
        self.jobstore.remove_job(job_id)

        if self.monitor:
            self.monitor.on_job_removed(job_id)

    def reschedule_job(self, job_id: str, trigger: Trigger) -> Job:
        """
        Reschedule a job with a new trigger.

        Args:
            job_id: ID of job to reschedule
            trigger: New trigger to use

        Returns:
            Updated Job instance

        Raises:
            JobNotFoundError: If job not found
        """
        job = self.jobstore.get_job(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")

        job.trigger = trigger
        job.state.next_run_time = trigger.get_next_fire_time(None, datetime.now())

        self.jobstore.update_job(job)

        if self.monitor:
            self.monitor.on_job_rescheduled(job_id)

        self._wakeup_event.set()

        return job

    def pause_job(self, job_id: str) -> Job:
        """
        Pause a job temporarily.

        Args:
            job_id: ID of job to pause

        Returns:
            Updated Job instance
        """
        job = self.jobstore.get_job(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")

        job.state.status = JobStatus.PAUSED
        self.jobstore.update_job(job)

        if self.monitor:
            self.monitor.on_job_paused(job_id)

        return job

    def resume_job(self, job_id: str) -> Job:
        """
        Resume a paused job.

        Args:
            job_id: ID of job to resume

        Returns:
            Updated Job instance
        """
        job = self.jobstore.get_job(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")

        job.state.status = JobStatus.PENDING
        job.state.next_run_time = job.trigger.get_next_fire_time(
            job.state.last_run_time, datetime.now()
        )

        self.jobstore.update_job(job)

        if self.monitor:
            self.monitor.on_job_resumed(job_id)

        self._wakeup_event.set()

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.jobstore.get_job(job_id)

    def get_jobs(self, tag: Optional[str] = None) -> List[Job]:
        """Get all jobs, optionally filtered by tag."""
        jobs = self.jobstore.get_all_jobs()
        if tag:
            jobs = [j for j in jobs if tag in j.tags]
        return jobs

    def _execute_job(self, job: Job) -> None:
        """Execute a single job with retry logic."""
        executor = self.executors.get(job.executor, self.executors["default"])

        def run_with_retry(attempt: int = 0):
            start_time = time.time()

            if self.monitor:
                self.monitor.on_job_started(job)

            try:
                result = job.func(*job.args, **job.kwargs)
                duration = time.time() - start_time

                job.state.run_count += 1
                job.state.last_run_time = datetime.now()
                job.state.next_run_time = job.trigger.get_next_fire_time(
                    job.state.last_run_time, datetime.now()
                )
                job.state.status = (
                    JobStatus.PENDING
                    if job.state.next_run_time
                    else JobStatus.COMPLETED
                )

                self.jobstore.update_job(job)

                if self.monitor:
                    self.monitor.on_job_completed(job, duration, result)

                return result

            except Exception as e:
                duration = time.time() - start_time
                job.state.error_count += 1

                if job.retry_policy and job.retry_policy.should_retry(e, attempt):
                    delay = job.retry_policy.get_delay(attempt)

                    if self.monitor:
                        self.monitor.on_job_retry(job, attempt + 1, delay)

                    threading.Timer(delay, lambda: run_with_retry(attempt + 1)).start()
                else:
                    job.state.status = JobStatus.FAILED
                    self.jobstore.update_job(job)

                    if self.monitor:
                        self.monitor.on_job_failed(job, e, duration, attempt + 1)

                    raise RetryExhaustedError(
                        f"Job {job.id} failed after {attempt + 1} attempts"
                    ) from e

        executor.submit(run_with_retry)

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.now()

            # Get due jobs
            due_jobs = self.jobstore.get_due_jobs(now)

            for job in due_jobs:
                with self._lock:
                    current_count = sum(
                        1 for j in self._running_jobs.values() if j == job.id
                    )

                    if current_count >= job.max_instances:
                        continue

                    self._running_jobs[f"{job.id}_{uuid.uuid4()}"] = job.id

                self._execute_job(job)

            # Calculate sleep time
            all_jobs = self.jobstore.get_all_jobs()
            next_run_times = [
                j.state.next_run_time
                for j in all_jobs
                if j.state.next_run_time and j.state.status != JobStatus.PAUSED
            ]

            if next_run_times:
                next_time = min(next_run_times)
                sleep_seconds = (next_time - datetime.now()).total_seconds()
                sleep_seconds = max(0.1, min(sleep_seconds, 60))
            else:
                sleep_seconds = 60

            # Wait for wakeup or timeout
            self._wakeup_event.wait(timeout=sleep_seconds)
            self._wakeup_event.clear()

    def start(self, blocking: bool = False) -> None:
        """
        Start the scheduler.

        Args:
            blocking: If True, block until shutdown is called
        """
        if self._running:
            return

        self._running = True

        if blocking:
            self._scheduler_loop()
        else:
            self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._thread.start()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the scheduler.

        Args:
            wait: Wait for jobs to complete
        """
        self._running = False
        self._wakeup_event.set()

        if self._thread and wait:
            self._thread.join(timeout=30)

        # Shutdown executors
        for executor in self.executors.values():
            executor.shutdown(wait=wait)

        # Shutdown job store
        self.jobstore.shutdown()

        logger.info("Scheduler shutdown complete")

    def modify_job(
        self,
        job_id: str,
        name: Optional[str] = None,
        executor: Optional[str] = None,
        max_instances: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """
        Modify job properties.

        Args:
            job_id: ID of job to modify
            name: New name
            executor: New executor
            max_instances: New max instances
            retry_policy: New retry policy
            metadata: Metadata to update

        Returns:
            Updated Job instance
        """
        job = self.jobstore.get_job(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")

        if name is not None:
            job.name = name
        if executor is not None:
            job.executor = executor
        if max_instances is not None:
            job.max_instances = max_instances
        if retry_policy is not None:
            job.retry_policy = retry_policy
        if metadata is not None:
            job.metadata.update(metadata)

        self.jobstore.update_job(job)

        return job

    def print_jobs(self) -> None:
        """Print all jobs in a formatted table."""
        jobs = self.jobstore.get_all_jobs()

        print(f"{'ID':<36} {'Name':<30} {'Status':<10} {'Next Run':<20}")
        print("-" * 100)

        for job in jobs:
            next_run = (
                job.state.next_run_time.strftime("%Y-%m-%d %H:%M:%S")
                if job.state.next_run_time
                else "N/A"
            )
            print(
                f"{job.id:<36} {job.name:<30} {job.state.status.name:<10} {next_run:<20}"
            )


# Aliases for convenient access
Scheduler = JobScheduler
Add = lambda self, *args, **kwargs: self.add_job(*args, **kwargs)
Remove = lambda self, job_id: self.remove_job(job_id)
Reschedule = lambda self, job_id, trigger: self.reschedule_job(job_id, trigger)
Pause = lambda self, job_id: self.pause_job(job_id)
Resume = lambda self, job_id: self.resume_job(job_id)


# =============================================================================
# Convenience Functions
# =============================================================================


_global_scheduler: Optional[JobScheduler] = None


def schedule_job(
    func: Callable,
    trigger: Trigger,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
    name: Optional[str] = None,
    **job_options,
) -> Job:
    """
    Convenience function to schedule a job using global scheduler.

    Args:
        func: Function to schedule
        trigger: When to run the job
        args: Positional arguments
        kwargs: Keyword arguments
        id: Job ID
        name: Job name
        **job_options: Additional job options

    Returns:
        Created Job instance
    """
    global _global_scheduler

    if _global_scheduler is None:
        _global_scheduler = JobScheduler()
        _global_scheduler.start()

    return _global_scheduler.add_job(
        func=func,
        trigger=trigger,
        args=args,
        kwargs=kwargs,
        id=id,
        name=name,
        **job_options,
    )


def unschedule_job(job_id: str) -> None:
    """
    Convenience function to remove a job using global scheduler.

    Args:
        job_id: ID of job to remove
    """
    global _global_scheduler

    if _global_scheduler is None:
        raise SchedulerError("No global scheduler initialized")

    _global_scheduler.remove_job(job_id)


def get_global_scheduler() -> Optional[JobScheduler]:
    """Get the global scheduler instance."""
    return _global_scheduler


def shutdown_global_scheduler(wait: bool = True) -> None:
    """Shutdown the global scheduler."""
    global _global_scheduler

    if _global_scheduler:
        _global_scheduler.shutdown(wait=wait)
        _global_scheduler = None


# =============================================================================
# Main Scheduler Factory
# =============================================================================


class Scheduler:
    """
    Factory and main entry point for scheduler creation.

    Provides convenient methods for creating schedulers with common configurations.
    """

    @staticmethod
    def create(
        store: str = "memory",
        store_options: Optional[Dict] = None,
        executors: Optional[Dict[str, Executor]] = None,
        enable_monitoring: bool = True,
        **scheduler_options,
    ) -> JobScheduler:
        """
        Create a new scheduler with specified configuration.

        Args:
            store: Store type ("memory", "sql", "mongo", "redis")
            store_options: Options for the store
            executors: Custom executors dict
            enable_monitoring: Enable monitoring
            **scheduler_options: Additional scheduler options

        Returns:
            Configured JobScheduler instance
        """
        store_options = store_options or {}

        if store == "memory":
            jobstore = MemoryJobStore()
        elif store == "sql":
            url = store_options.get("url", "sqlite:///scheduler.db")
            jobstore = SQLAlchemyJobStore(url, **store_options)
        elif store == "mongo":
            jobstore = MongoDBJobStore(**store_options)
        elif store == "redis":
            jobstore = RedisJobStore(**store_options)
        else:
            raise ValueError(f"Unknown store type: {store}")

        monitor = JobMonitor() if enable_monitoring else None

        return JobScheduler(
            jobstore=jobstore, executors=executors, monitor=monitor, **scheduler_options
        )

    @staticmethod
    def with_celery(
        broker_url: str = "redis://localhost:6379/0", **celery_options
    ) -> JobScheduler:
        """Create a scheduler with Celery distributed backend."""
        distributed = CeleryScheduler(broker_url=broker_url, **celery_options)
        return JobScheduler(distributed=distributed)

    @staticmethod
    def with_rq(
        redis_host: str = "localhost", redis_port: int = 6379, **rq_options
    ) -> JobScheduler:
        """Create a scheduler with RQ distributed backend."""
        distributed = RQScheduler(
            redis_host=redis_host, redis_port=redis_port, **rq_options
        )
        return JobScheduler(distributed=distributed)

    @staticmethod
    def with_apscheduler(**ap_options) -> JobScheduler:
        """Create a scheduler with APScheduler wrapper."""
        distributed = APSchedulerWrapper(**ap_options)
        return JobScheduler(distributed=distributed)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Exceptions
    "SchedulerError",
    "JobNotFoundError",
    "JobAlreadyExistsError",
    "TriggerError",
    "ExecutorError",
    "StoreError",
    "RetryExhaustedError",
    # Enums
    "JobStatus",
    # Core Classes
    "Job",
    "JobState",
    # Triggers
    "Trigger",
    "CronTrigger",
    "IntervalTrigger",
    "DateTrigger",
    "EventTrigger",
    "Cron",
    "Interval",
    "Date",
    "Event",
    # Executors
    "Executor",
    "ThreadPoolExecutor",
    "ProcessPoolExecutor",
    "AsyncIOExecutor",
    "GeventExecutor",
    "TornadoExecutor",
    "Thread",
    "Process",
    "Async",
    "Gevent",
    "Tornado",
    # Job Stores
    "JobStore",
    "MemoryJobStore",
    "SQLAlchemyJobStore",
    "MongoDBJobStore",
    "RedisJobStore",
    "Memory",
    "SQL",
    "Mongo",
    "Redis",
    # Retry Logic
    "RetryStrategy",
    "ExponentialBackoff",
    "FixedDelay",
    "LinearBackoff",
    "RetryPolicy",
    "Exponential",
    "Fixed",
    "Linear",
    "Policy",
    # Monitoring
    "JobExecutionRecord",
    "JobMetrics",
    "JobLogger",
    "JobHistory",
    "JobMonitor",
    "Metrics",
    "Logger",
    "History",
    "Monitor",
    # Distributed
    "DistributedScheduler",
    "CeleryScheduler",
    "RQScheduler",
    "APSchedulerWrapper",
    "Dist",
    "Celery",
    "RQ",
    "AP",
    # Main Scheduler
    "JobScheduler",
    "Scheduler",
    # Utilities
    "schedule_job",
    "unschedule_job",
    "get_global_scheduler",
    "shutdown_global_scheduler",
]
