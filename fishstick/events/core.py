"""
Fishstick Event System Core Module

Comprehensive event-driven architecture for ML workflows.
Provides event bus, handlers, storage, sourcing, CQRS, and streaming capabilities.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from queue import Queue, Empty
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    AsyncIterator,
    Iterator,
)

# =============================================================================
# Constants and Enums
# =============================================================================


class EventPriority(Enum):
    """Event priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class EventStatus(Enum):
    """Event processing status."""

    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()


class HandlerType(Enum):
    """Types of event handlers."""

    SYNC = auto()
    ASYNC = auto()
    CHAIN = auto()


# =============================================================================
# Event Base Class
# =============================================================================


@dataclass
class Event:
    """
    Base event class.

    Attributes:
        id: Unique event identifier
        type: Event type identifier
        timestamp: Event creation time
        source: Source component/module
        payload: Event data
        priority: Event priority
        correlation_id: ID for correlated events
        metadata: Additional event metadata
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "base"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "payload": self.payload,
            "priority": self.priority.name,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Create event from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data.get("type", "base"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", "unknown"),
            payload=data.get("payload", {}),
            priority=EventPriority[data.get("priority", "NORMAL")],
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"Event(type={self.type}, id={self.id[:8]}, source={self.source})"


# =============================================================================
# Event Types - ML Specific Events
# =============================================================================


@dataclass
class TrainingEvent(Event):
    """Training-related events."""

    type: str = "training"
    epoch: int = 0
    batch: int = 0
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    dataset_id: str = ""

    def __post_init__(self):
        if self.type == "base":
            self.type = "training"


@dataclass
class EvaluationEvent(Event):
    """Evaluation-related events."""

    type: str = "evaluation"
    model_id: str = ""
    dataset_id: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[Any] = None
    ground_truth: Optional[Any] = None

    def __post_init__(self):
        if self.type == "base":
            self.type = "evaluation"


@dataclass
class PredictionEvent(Event):
    """Prediction/inference events."""

    type: str = "prediction"
    model_id: str = ""
    input_data: Any = None
    output_data: Any = None
    inference_time_ms: float = 0.0
    confidence: Optional[float] = None

    def __post_init__(self):
        if self.type == "base":
            self.type = "prediction"


@dataclass
class ModelEvent(Event):
    """Model lifecycle events."""

    type: str = "model"
    action: str = ""  # create, update, delete, save, load, deploy
    model_id: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    version: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.type == "base":
            self.type = "model"


@dataclass
class DataEvent(Event):
    """Data processing events."""

    type: str = "data"
    action: str = ""  # load, transform, validate, split, preprocess
    dataset_id: str = ""
    record_count: int = 0
    schema: Dict[str, str] = field(default_factory=dict)
    data_quality_score: Optional[float] = None

    def __post_init__(self):
        if self.type == "base":
            self.type = "data"


@dataclass
class SystemEvent(Event):
    """System-level events."""

    type: str = "system"
    component: str = ""
    action: str = ""  # start, stop, error, warning, info
    severity: str = "info"  # info, warning, error, critical
    message: str = ""
    stack_trace: Optional[str] = None

    def __post_init__(self):
        if self.type == "base":
            self.type = "system"


# =============================================================================
# Event Bus
# =============================================================================


class EventBus:
    """
    Central event bus for publish-subscribe pattern.

    Supports:
    - Synchronous and asynchronous subscribers
    - Event filtering by type
    - Priority-based processing
    - Thread-safe operations
    """

    def __init__(self, max_workers: int = 10):
        self._subscribers: Dict[str, List[Callable[[Event], Any]]] = defaultdict(list)
        self._async_subscribers: Dict[
            str, List[Callable[[Event], Coroutine[Any, Any, Any]]]
        ] = defaultdict(list)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._event_queue: Queue = Queue()
        self._processing_thread: Optional[threading.Thread] = None
        self._event_history: List[Event] = []
        self._history_limit = 1000

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Any],
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Handler function
            priority: Handler priority
        """
        with self._lock:
            if asyncio.iscoroutinefunction(handler):
                self._async_subscribers[event_type].append(handler)
            else:
                self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[Event], Any]) -> bool:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: Type of event
            handler: Handler to remove

        Returns:
            True if handler was found and removed
        """
        with self._lock:
            if asyncio.iscoroutinefunction(handler):
                if handler in self._async_subscribers[event_type]:
                    self._async_subscribers[event_type].remove(handler)
                    return True
            else:
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)
                    return True
            return False

    def publish(self, event: Event, async_mode: bool = False) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
            async_mode: If True, process asynchronously
        """
        with self._lock:
            # Track event in history
            self._event_history.append(event)
            if len(self._event_history) > self._history_limit:
                self._event_history.pop(0)

            if async_mode:
                self._event_queue.put(event)
            else:
                self._notify_subscribers(event)

    def _notify_subscribers(self, event: Event) -> None:
        """Notify all subscribers of an event."""
        # Get subscribers for this event type and wildcards
        handlers = []
        handlers.extend(self._subscribers.get(event.type, []))
        handlers.extend(self._subscribers.get("*", []))

        # Execute synchronous handlers
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._handle_handler_error(event, handler, e)

    async def _notify_async_subscribers(self, event: Event) -> None:
        """Notify all async subscribers of an event."""
        handlers = []
        handlers.extend(self._async_subscribers.get(event.type, []))
        handlers.extend(self._async_subscribers.get("*", []))

        # Execute async handlers
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                self._handle_handler_error(event, handler, e)

    def _handle_handler_error(
        self, event: Event, handler: Callable, error: Exception
    ) -> None:
        """Handle errors from event handlers."""
        print(f"Error in handler {handler.__name__} for event {event}: {error}")

    def start(self) -> None:
        """Start the event processing loop."""
        if not self._running:
            self._running = True
            self._processing_thread = threading.Thread(
                target=self._process_events, daemon=True
            )
            self._processing_thread.start()

    def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

    def _process_events(self) -> None:
        """Background thread for processing async events."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.1)
                self._notify_subscribers(event)
            except Empty:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")

    def get_history(self, event_type: Optional[str] = None) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Optional filter by event type

        Returns:
            List of events
        """
        with self._lock:
            if event_type:
                return [e for e in self._event_history if e.type == event_type]
            return self._event_history.copy()

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()


# =============================================================================
# Event Handlers
# =============================================================================


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._metrics: Dict[str, Any] = {
            "processed": 0,
            "failed": 0,
            "avg_time_ms": 0.0,
        }

    @abstractmethod
    def handle(self, event: Event) -> Any:
        """Handle an event. Must be implemented by subclasses."""
        pass

    def __call__(self, event: Event) -> Any:
        """Make handler callable."""
        start_time = time.time()
        try:
            result = self.handle(event)
            self._update_metrics(
                success=True, elapsed_ms=(time.time() - start_time) * 1000
            )
            return result
        except Exception as e:
            self._update_metrics(
                success=False, elapsed_ms=(time.time() - start_time) * 1000
            )
            raise e

    def _update_metrics(self, success: bool, elapsed_ms: float) -> None:
        """Update handler metrics."""
        self._metrics["processed"] += 1
        if not success:
            self._metrics["failed"] += 1
        # Update running average
        n = self._metrics["processed"]
        self._metrics["avg_time_ms"] = (
            self._metrics["avg_time_ms"] * (n - 1) + elapsed_ms
        ) / n

    def get_metrics(self) -> Dict[str, Any]:
        """Get handler performance metrics."""
        return self._metrics.copy()


class SyncEventHandler(EventHandler):
    """Synchronous event handler."""

    def __init__(
        self,
        handler_func: Callable[[Event], Any],
        name: Optional[str] = None,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ):
        super().__init__(name)
        self._handler_func = handler_func
        self._filter_fn = filter_fn

    def handle(self, event: Event) -> Any:
        """Handle event synchronously."""
        if self._filter_fn and not self._filter_fn(event):
            return None
        return self._handler_func(event)


class AsyncEventHandler(EventHandler):
    """Asynchronous event handler."""

    def __init__(
        self,
        handler_func: Callable[[Event], Coroutine[Any, Any, Any]],
        name: Optional[str] = None,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ):
        super().__init__(name)
        self._handler_func = handler_func
        self._filter_fn = filter_fn

    async def handle(self, event: Event) -> Any:
        """Handle event asynchronously."""
        if self._filter_fn and not self._filter_fn(event):
            return None
        return await self._handler_func(event)

    def __call__(self, event: Event) -> Coroutine[Any, Any, Any]:
        """Make handler callable (returns coroutine)."""
        return self.handle(event)


class ChainEventHandler(EventHandler):
    """Chain of handlers that process events sequentially."""

    def __init__(
        self, handlers: Optional[List[EventHandler]] = None, name: Optional[str] = None
    ):
        super().__init__(name)
        self._handlers: List[EventHandler] = handlers or []

    def add_handler(self, handler: EventHandler) -> ChainEventHandler:
        """Add a handler to the chain."""
        self._handlers.append(handler)
        return self

    def remove_handler(self, handler: EventHandler) -> bool:
        """Remove a handler from the chain."""
        if handler in self._handlers:
            self._handlers.remove(handler)
            return True
        return False

    def handle(self, event: Event) -> List[Any]:
        """Process event through all handlers in chain."""
        results = []
        for handler in self._handlers:
            result = handler(event)
            results.append(result)
            # Stop chain if handler returns None
            if result is None:
                break
        return results


# =============================================================================
# Event Store
# =============================================================================


class EventStore(ABC):
    """Abstract event store for persisting events."""

    @abstractmethod
    def persist_event(self, event: Event) -> bool:
        """Persist a single event."""
        pass

    @abstractmethod
    def persist_events(self, events: List[Event]) -> int:
        """Persist multiple events. Returns count persisted."""
        pass

    @abstractmethod
    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Query events from store."""
        pass

    @abstractmethod
    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get a specific event by ID."""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store."""

    def __init__(self, max_events: int = 10000):
        self._events: List[Event] = []
        self._event_index: Dict[str, Event] = {}
        self._max_events = max_events
        self._lock = threading.RLock()

    def persist_event(self, event: Event) -> bool:
        """Persist event to memory."""
        with self._lock:
            self._events.append(event)
            self._event_index[event.id] = event

            # Enforce size limit
            if len(self._events) > self._max_events:
                removed = self._events.pop(0)
                del self._event_index[removed.id]

            return True

    def persist_events(self, events: List[Event]) -> int:
        """Persist multiple events."""
        count = 0
        for event in events:
            if self.persist_event(event):
                count += 1
        return count

    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Query events from memory."""
        with self._lock:
            events = self._events.copy()

            if event_type:
                events = [e for e in events if e.type == event_type]

            if start_time:
                events = [e for e in events if e.timestamp >= start_time]

            if end_time:
                events = [e for e in events if e.timestamp <= end_time]

            return events[-limit:]

    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        with self._lock:
            return self._event_index.get(event_id)

    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self._events.clear()
            self._event_index.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get store statistics."""
        with self._lock:
            return {
                "total_events": len(self._events),
                "max_events": self._max_events,
            }


class DatabaseEventStore(EventStore):
    """SQLite-based event store."""

    def __init__(self, db_path: str = "events.db"):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT,
                    payload TEXT,
                    priority TEXT,
                    correlation_id TEXT,
                    metadata TEXT
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(type)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_correlation ON events(correlation_id)"
            )

    def persist_event(self, event: Event) -> bool:
        """Persist event to database."""
        with self._lock:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO events 
                        (id, type, timestamp, source, payload, priority, correlation_id, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            event.id,
                            event.type,
                            event.timestamp.isoformat(),
                            event.source,
                            json.dumps(event.payload),
                            event.priority.name,
                            event.correlation_id,
                            json.dumps(event.metadata),
                        ),
                    )
                return True
            except Exception as e:
                print(f"Error persisting event: {e}")
                return False

    def persist_events(self, events: List[Event]) -> int:
        """Persist multiple events."""
        count = 0
        for event in events:
            if self.persist_event(event):
                count += 1
        return count

    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Query events from database."""
        with self._lock:
            query = "SELECT * FROM events WHERE 1=1"
            params = []

            if event_type:
                query += " AND type = ?"
                params.append(event_type)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

            events = []
            for row in rows:
                event = Event(
                    id=row[0],
                    type=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    source=row[3],
                    payload=json.loads(row[4]) if row[4] else {},
                    priority=EventPriority[row[5]] if row[5] else EventPriority.NORMAL,
                    correlation_id=row[6],
                    metadata=json.loads(row[7]) if row[7] else {},
                )
                events.append(event)

            return events

    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
                row = cursor.fetchone()

            if row:
                return Event(
                    id=row[0],
                    type=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    source=row[3],
                    payload=json.loads(row[4]) if row[4] else {},
                    priority=EventPriority[row[5]] if row[5] else EventPriority.NORMAL,
                    correlation_id=row[6],
                    metadata=json.loads(row[7]) if row[7] else {},
                )
            return None


class FileEventStore(EventStore):
    """File-based event store using JSON lines format."""

    def __init__(self, file_path: str = "events.jsonl"):
        self._file_path = Path(file_path)
        self._lock = threading.RLock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Ensure file exists."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._file_path.exists():
            self._file_path.touch()

    def persist_event(self, event: Event) -> bool:
        """Persist event to file."""
        with self._lock:
            try:
                with open(self._file_path, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
                return True
            except Exception as e:
                print(f"Error persisting event: {e}")
                return False

    def persist_events(self, events: List[Event]) -> int:
        """Persist multiple events."""
        with self._lock:
            count = 0
            try:
                with open(self._file_path, "a") as f:
                    for event in events:
                        f.write(json.dumps(event.to_dict()) + "\n")
                        count += 1
                return count
            except Exception as e:
                print(f"Error persisting events: {e}")
                return count

    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Query events from file."""
        events = []

        with self._lock:
            try:
                with open(self._file_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        event = Event.from_dict(data)

                        # Apply filters
                        if event_type and event.type != event_type:
                            continue
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue

                        events.append(event)
            except Exception as e:
                print(f"Error reading events: {e}")

        # Return last N events
        return events[-limit:]

    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        with self._lock:
            try:
                with open(self._file_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("id") == event_id:
                            return Event.from_dict(data)
            except Exception as e:
                print(f"Error reading event: {e}")
        return None


def persist_event(event: Event, store: EventStore, async_mode: bool = False) -> bool:
    """
    Persist an event to a store.

    Args:
        event: Event to persist
        store: Event store instance
        async_mode: Whether to persist asynchronously

    Returns:
        True if persisted successfully
    """
    if async_mode:
        # Run in thread pool
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(store.persist_event, event)
            return future.result()
    else:
        return store.persist_event(event)


# =============================================================================
# Event Sourcing
# =============================================================================


class AggregateRoot(ABC):
    """
    Aggregate root for event sourcing.

    Maintains state through application of events.
    """

    def __init__(self, aggregate_id: str):
        self._id = aggregate_id
        self._version = 0
        self._uncommitted_events: List[Event] = []
        self._is_replaying = False

    @property
    def id(self) -> str:
        """Get aggregate ID."""
        return self._id

    @property
    def version(self) -> int:
        """Get current version."""
        return self._version

    @abstractmethod
    def apply_event(self, event: Event) -> None:
        """Apply an event to update state."""
        pass

    def apply(self, event: Event) -> None:
        """Apply event and track if not replaying."""
        self.apply_event(event)
        self._version += 1

        if not self._is_replaying:
            self._uncommitted_events.append(event)

    def get_uncommitted_events(self) -> List[Event]:
        """Get uncommitted events."""
        return self._uncommitted_events.copy()

    def mark_committed(self) -> None:
        """Mark events as committed."""
        self._uncommitted_events.clear()

    def load_from_history(self, events: List[Event]) -> None:
        """Load aggregate from event history."""
        self._is_replaying = True
        try:
            for event in events:
                self.apply(event)
        finally:
            self._is_replaying = False


class EventSourcing:
    """
    Event sourcing manager.

    Manages aggregates and event persistence.
    """

    def __init__(self, event_store: EventStore):
        self._store = event_store
        self._aggregates: Dict[str, AggregateRoot] = {}
        self._lock = threading.RLock()

    def save_aggregate(self, aggregate: AggregateRoot) -> bool:
        """
        Save aggregate by persisting uncommitted events.

        Args:
            aggregate: Aggregate to save

        Returns:
            True if saved successfully
        """
        events = aggregate.get_uncommitted_events()
        if not events:
            return True

        count = self._store.persist_events(events)
        if count == len(events):
            aggregate.mark_committed()
            return True
        return False

    def load_aggregate(
        self, aggregate_id: str, aggregate_class: Type[AggregateRoot]
    ) -> Optional[AggregateRoot]:
        """
        Load aggregate from event history.

        Args:
            aggregate_id: Aggregate ID
            aggregate_class: Class to instantiate

        Returns:
            Loaded aggregate or None
        """
        # Query all events for this aggregate
        # In real implementation, would have a dedicated aggregate_id field
        events = self._store.get_events(limit=10000)
        aggregate_events = [
            e for e in events if e.payload.get("aggregate_id") == aggregate_id
        ]

        if not aggregate_events:
            return None

        aggregate = aggregate_class(aggregate_id)
        aggregate.load_from_history(sorted(aggregate_events, key=lambda e: e.timestamp))

        return aggregate

    def get_all_aggregate_ids(self, aggregate_type: str) -> List[str]:
        """Get all aggregate IDs of a given type."""
        events = self._store.get_events(event_type=aggregate_type, limit=10000)
        ids = set()
        for event in events:
            agg_id = event.payload.get("aggregate_id")
            if agg_id:
                ids.add(agg_id)
        return list(ids)


class EventProjection(ABC):
    """
    Event projection for read models.

    Projects events into a read-optimized view.
    """

    def __init__(self, name: str):
        self.name = name
        self._last_event_time: Optional[datetime] = None
        self._is_building = False

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Handle a single event."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset projection state."""
        pass

    def project(self, events: List[Event]) -> None:
        """Project multiple events."""
        self._is_building = True
        try:
            for event in sorted(events, key=lambda e: e.timestamp):
                self.handle_event(event)
                self._last_event_time = event.timestamp
        finally:
            self._is_building = False

    def rebuild(self, event_store: EventStore) -> None:
        """Rebuild projection from event store."""
        self.reset()
        events = event_store.get_events(limit=100000)
        self.project(events)


def replay_events(
    aggregate: AggregateRoot,
    events: List[Event],
    up_to_version: Optional[int] = None,
) -> None:
    """
    Replay events on an aggregate.

    Args:
        aggregate: Aggregate to replay on
        events: Events to replay
        up_to_version: Optional version limit
    """
    sorted_events = sorted(events, key=lambda e: e.timestamp)

    if up_to_version:
        sorted_events = sorted_events[:up_to_version]

    aggregate.load_from_history(sorted_events)


# =============================================================================
# CQRS (Command Query Responsibility Segregation)
# =============================================================================


T = TypeVar("T")


class Command(ABC, Generic[T]):
    """Base command class."""

    def __init__(self, command_id: Optional[str] = None):
        self.id = command_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def validate(self) -> bool:
        """Validate command."""
        pass


class CommandHandler(ABC, Generic[T]):
    """Base command handler."""

    @abstractmethod
    def handle(self, command: Command[T]) -> T:
        """Handle a command."""
        pass

    def __call__(self, command: Command[T]) -> T:
        if not command.validate():
            raise ValueError(f"Invalid command: {command}")
        return self.handle(command)


class Query(ABC, Generic[T]):
    """Base query class."""

    def __init__(self, query_id: Optional[str] = None):
        self.id = query_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()


class QueryHandler(ABC, Generic[T]):
    """Base query handler."""

    @abstractmethod
    def handle(self, query: Query[T]) -> T:
        """Handle a query."""
        pass

    def __call__(self, query: Query[T]) -> T:
        return self.handle(query)


class CommandBus:
    """Command bus for dispatching commands."""

    def __init__(self):
        self._handlers: Dict[Type[Command], CommandHandler] = {}
        self._middleware: List[Callable[[Command, Callable], Any]] = []
        self._lock = threading.RLock()

    def register(self, command_type: Type[Command], handler: CommandHandler) -> None:
        """Register a command handler."""
        with self._lock:
            self._handlers[command_type] = handler

    def unregister(self, command_type: Type[Command]) -> bool:
        """Unregister a command handler."""
        with self._lock:
            if command_type in self._handlers:
                del self._handlers[command_type]
                return True
            return False

    def add_middleware(self, middleware: Callable[[Command, Callable], Any]) -> None:
        """Add middleware to command pipeline."""
        self._middleware.append(middleware)

    def dispatch(self, command: Command[T]) -> T:
        """Dispatch a command to its handler."""
        command_type = type(command)

        with self._lock:
            handler = self._handlers.get(command_type)

        if not handler:
            raise ValueError(f"No handler registered for {command_type}")

        # Apply middleware
        execution = handler
        for mw in reversed(self._middleware):
            execution = lambda cmd, h=execution, m=mw: m(cmd, h)

        return execution(command)

    def dispatch_async(self, command: Command[T]) -> "asyncio.Future[T]":
        """Dispatch a command asynchronously."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, self.dispatch, command)


class QueryBus:
    """Query bus for dispatching queries."""

    def __init__(self):
        self._handlers: Dict[Type[Query], QueryHandler] = {}
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._cache_enabled = True

    def register(self, query_type: Type[Query], handler: QueryHandler) -> None:
        """Register a query handler."""
        with self._lock:
            self._handlers[query_type] = handler

    def unregister(self, query_type: Type[Query]) -> bool:
        """Unregister a query handler."""
        with self._lock:
            if query_type in self._handlers:
                del self._handlers[query_type]
                return True
            return False

    def dispatch(self, query: Query[T], use_cache: bool = False) -> T:
        """Dispatch a query to its handler."""
        query_type = type(query)

        # Check cache
        if use_cache and self._cache_enabled:
            cache_key = f"{query_type.__name__}:{query.id}"
            with self._lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]

        with self._lock:
            handler = self._handlers.get(query_type)

        if not handler:
            raise ValueError(f"No handler registered for {query_type}")

        result = handler(query)

        # Cache result
        if use_cache and self._cache_enabled:
            cache_key = f"{query_type.__name__}:{query.id}"
            with self._lock:
                self._cache[cache_key] = result

        return result

    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """Invalidate query cache."""
        with self._lock:
            if pattern:
                keys = [k for k in self._cache if pattern in k]
                for k in keys:
                    del self._cache[k]
            else:
                self._cache.clear()

    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable caching."""
        self._cache_enabled = enabled


# =============================================================================
# Streaming
# =============================================================================


class EventStream:
    """
    Event stream for real-time processing.

    Supports both push and pull models.
    """

    def __init__(self, name: str, buffer_size: int = 1000):
        self.name = name
        self._buffer_size = buffer_size
        self._events: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._subscribers: List[Callable[[Event], None]] = []
        self._running = False
        self._lock = threading.RLock()

    async def emit(self, event: Event) -> None:
        """Emit an event to the stream."""
        await self._events.put(event)

        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")

    async def consume(self) -> Event:
        """Consume an event from the stream."""
        return await self._events.get()

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to the stream."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> bool:
        """Unsubscribe from the stream."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
                return True
            return False

    async def __aiter__(self) -> AsyncIterator[Event]:
        """Async iterator support."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._events.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    def start(self) -> None:
        """Start the stream."""
        self._running = True

    def stop(self) -> None:
        """Stop the stream."""
        self._running = False


class StreamProcessor:
    """
    Processor for event streams.

    Applies transformations and filters to events.
    """

    def __init__(self, name: str):
        self.name = name
        self._filters: List[Callable[[Event], bool]] = []
        self._transforms: List[Callable[[Event], Event]] = []
        self._output_streams: List[EventStream] = []

    def add_filter(self, filter_fn: Callable[[Event], bool]) -> StreamProcessor:
        """Add a filter to the processor."""
        self._filters.append(filter_fn)
        return self

    def add_transform(self, transform_fn: Callable[[Event], Event]) -> StreamProcessor:
        """Add a transform to the processor."""
        self._transforms.append(transform_fn)
        return self

    def add_output_stream(self, stream: EventStream) -> StreamProcessor:
        """Add an output stream."""
        self._output_streams.append(stream)
        return self

    def process(self, event: Event) -> Optional[Event]:
        """
        Process a single event.

        Returns processed event or None if filtered out.
        """
        # Apply filters
        for filter_fn in self._filters:
            if not filter_fn(event):
                return None

        # Apply transforms
        processed_event = event
        for transform_fn in self._transforms:
            processed_event = transform_fn(processed_event)
            if processed_event is None:
                return None

        # Emit to output streams
        for stream in self._output_streams:
            asyncio.create_task(stream.emit(processed_event))

        return processed_event

    async def process_stream(self, stream: EventStream) -> None:
        """Process all events from a stream."""
        async for event in stream:
            self.process(event)


class EventFilter:
    """Event filter utilities."""

    @staticmethod
    def by_type(event_type: str) -> Callable[[Event], bool]:
        """Filter by event type."""
        return lambda e: e.type == event_type

    @staticmethod
    def by_source(source: str) -> Callable[[Event], bool]:
        """Filter by event source."""
        return lambda e: e.source == source

    @staticmethod
    def by_priority(min_priority: EventPriority) -> Callable[[Event], bool]:
        """Filter by minimum priority."""
        return lambda e: e.priority.value <= min_priority.value

    @staticmethod
    def by_time_range(
        start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> Callable[[Event], bool]:
        """Filter by time range."""

        def filter_fn(e: Event) -> bool:
            if start and e.timestamp < start:
                return False
            if end and e.timestamp > end:
                return False
            return True

        return filter_fn

    @staticmethod
    def compose(
        *filters: Callable[[Event], bool], mode: str = "and"
    ) -> Callable[[Event], bool]:
        """Compose multiple filters."""
        if mode == "and":
            return lambda e: all(f(e) for f in filters)
        elif mode == "or":
            return lambda e: any(f(e) for f in filters)
        else:
            raise ValueError(f"Invalid mode: {mode}")


class EventTransform:
    """Event transformation utilities."""

    @staticmethod
    def map_payload(
        mapping: Dict[str, Callable[[Any], Any]],
    ) -> Callable[[Event], Event]:
        """Transform event payload fields."""

        def transform(event: Event) -> Event:
            new_payload = dict(event.payload)
            for key, fn in mapping.items():
                if key in new_payload:
                    new_payload[key] = fn(new_payload[key])
            event.payload = new_payload
            return event

        return transform

    @staticmethod
    def enrich(**enrichments: Any) -> Callable[[Event], Event]:
        """Enrich event with additional data."""

        def transform(event: Event) -> Event:
            event.payload.update(enrichments)
            return event

        return transform

    @staticmethod
    def rename_payload_keys(**key_mapping: str) -> Callable[[Event], Event]:
        """Rename payload keys."""

        def transform(event: Event) -> Event:
            for old_key, new_key in key_mapping.items():
                if old_key in event.payload:
                    event.payload[new_key] = event.payload.pop(old_key)
            return event

        return transform

    @staticmethod
    def select_payload_keys(*keys: str) -> Callable[[Event], Event]:
        """Keep only specified payload keys."""

        def transform(event: Event) -> Event:
            event.payload = {k: v for k, v in event.payload.items() if k in keys}
            return event

        return transform

    @staticmethod
    def add_metadata(**metadata: Any) -> Callable[[Event], Event]:
        """Add metadata to event."""

        def transform(event: Event) -> Event:
            event.metadata.update(metadata)
            return event

        return transform


# =============================================================================
# Utilities
# =============================================================================


# Global event bus instance
_default_bus: Optional[EventBus] = None
_default_store: Optional[EventStore] = None


def get_default_bus() -> EventBus:
    """Get or create default event bus."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
        _default_bus.start()
    return _default_bus


def get_default_store() -> EventStore:
    """Get or create default event store."""
    global _default_store
    if _default_store is None:
        _default_store = InMemoryEventStore()
    return _default_store


def emit_event(
    event: Event,
    bus: Optional[EventBus] = None,
    store: Optional[EventStore] = None,
    persist: bool = False,
) -> None:
    """
    Emit an event to the bus and optionally persist it.

    Args:
        event: Event to emit
        bus: Event bus (uses default if None)
        store: Event store for persistence
        persist: Whether to persist the event
    """
    bus = bus or get_default_bus()
    bus.publish(event)

    if persist and store:
        store.persist_event(event)


def on_event(
    event_type: str,
    bus: Optional[EventBus] = None,
    priority: EventPriority = EventPriority.NORMAL,
) -> Callable:
    """
    Decorator for event handlers.

    Args:
        event_type: Type of event to handle
        bus: Event bus (uses default if None)
        priority: Handler priority

    Returns:
        Decorator function
    """

    def decorator(func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        bus_instance = bus or get_default_bus()
        bus_instance.subscribe(event_type, func, priority)

        @wraps(func)
        def wrapper(event: Event) -> Any:
            return func(event)

        return wrapper

    return decorator


class EventListener:
    """
    Event listener with context manager support.

    Automatically subscribes on enter and unsubscribes on exit.
    """

    def __init__(
        self,
        event_type: str,
        handler: Callable[[Event], Any],
        bus: Optional[EventBus] = None,
    ):
        self.event_type = event_type
        self.handler = handler
        self.bus = bus or get_default_bus()
        self._subscribed = False

    def __enter__(self) -> EventListener:
        """Subscribe to events."""
        self.bus.subscribe(self.event_type, self.handler)
        self._subscribed = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Unsubscribe from events."""
        if self._subscribed:
            self.bus.unsubscribe(self.event_type, self.handler)
            self._subscribed = False

    def start(self) -> None:
        """Start listening."""
        self.__enter__()

    def stop(self) -> None:
        """Stop listening."""
        self.__exit__(None, None, None)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_event_bus(max_workers: int = 10) -> EventBus:
    """Create a new event bus."""
    return EventBus(max_workers=max_workers)


def create_event_store(store_type: str = "memory", **kwargs) -> EventStore:
    """
    Create an event store.

    Args:
        store_type: "memory", "database", or "file"
        **kwargs: Store-specific arguments

    Returns:
        Event store instance
    """
    if store_type == "memory":
        return InMemoryEventStore(**kwargs)
    elif store_type == "database":
        return DatabaseEventStore(**kwargs)
    elif store_type == "file":
        return FileEventStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")


def create_stream(name: str, buffer_size: int = 1000) -> EventStream:
    """Create a new event stream."""
    return EventStream(name, buffer_size)


def create_processor(name: str) -> StreamProcessor:
    """Create a new stream processor."""
    return StreamProcessor(name)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Event Bus
    "EventBus",
    "publish",
    "subscribe",
    "unsubscribe",
    "Event",
    # Event Handlers
    "EventHandler",
    "AsyncEventHandler",
    "SyncEventHandler",
    "ChainEventHandler",
    # Event Types
    "TrainingEvent",
    "EvaluationEvent",
    "PredictionEvent",
    "ModelEvent",
    "DataEvent",
    "SystemEvent",
    # Event Store
    "EventStore",
    "InMemoryEventStore",
    "DatabaseEventStore",
    "FileEventStore",
    "persist_event",
    # Event Sourcing
    "EventSourcing",
    "AggregateRoot",
    "replay_events",
    "EventProjection",
    # CQRS
    "Command",
    "CommandHandler",
    "CommandBus",
    "Query",
    "QueryHandler",
    "QueryBus",
    # Streaming
    "EventStream",
    "StreamProcessor",
    "EventFilter",
    "EventTransform",
    # Utilities
    "emit_event",
    "on_event",
    "EventListener",
    "EventPriority",
    "EventStatus",
    "HandlerType",
    # Convenience functions
    "get_default_bus",
    "get_default_store",
    "create_event_bus",
    "create_event_store",
    "create_stream",
    "create_processor",
]
