"""
Comprehensive Error Handling Module for Fishstick.

This module provides a complete error handling infrastructure including:
- Exception hierarchies
- Error handling utilities
- Retry logic with backoff
- Circuit breaker pattern
- Fallback strategies
- Validation utilities
- Error logging and tracking
- Safe execution utilities
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Context variable for error context
_error_context: ContextVar[Optional["ErrorContext"]] = ContextVar(
    "error_context", default=None
)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class FishstickError(Exception):
    """Base exception for all Fishstick errors.

    Attributes:
        message: Error message
        code: Error code for categorization
        details: Additional error details
        context: Error context information
        timestamp: When the error occurred
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional["ErrorContext"] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        self.context = context or _error_context.get()
        self.timestamp = datetime.utcnow()
        self.cause = cause
        self.traceback_str = traceback.format_exc() if cause else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict() if self.context else None,
            "traceback": self.traceback_str,
        }

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.context:
            parts.append(f"Context: {self.context}")
        return " | ".join(parts)


class ModelError(FishstickError):
    """Error related to model operations.

    Raised when there are issues with:
    - Model loading/saving
    - Model architecture
    - Model state
    - Model compatibility
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details.update({"model_name": model_name})
        super().__init__(message, code="MODEL_ERROR", details=details, **kwargs)


class DataError(FishstickError):
    """Error related to data operations.

    Raised when there are issues with:
    - Data loading
    - Data preprocessing
    - Data format
    - Data corruption
    """

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details.update({"data_source": data_source})
        super().__init__(message, code="DATA_ERROR", details=details, **kwargs)


class TrainingError(FishstickError):
    """Error related to training operations.

    Raised when there are issues with:
    - Training loop
    - Loss computation
    - Optimization
    - Gradient issues
    """

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details.update({"epoch": epoch, "step": step})
        super().__init__(message, code="TRAINING_ERROR", details=details, **kwargs)


class InferenceError(FishstickError):
    """Error related to inference operations.

    Raised when there are issues with:
    - Model prediction
    - Batch processing
    - Output generation
    """

    def __init__(
        self,
        message: str,
        input_shape: Optional[tuple] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details.update({"input_shape": input_shape})
        super().__init__(message, code="INFERENCE_ERROR", details=details, **kwargs)


class ValidationError(FishstickError):
    """Error related to validation operations.

    Raised when there are issues with:
    - Input validation
    - Output validation
    - Schema validation
    - Type checking
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details.update({"field": field, "expected": expected, "actual": actual})
        super().__init__(message, code="VALIDATION_ERROR", details=details, **kwargs)


class ConfigurationError(FishstickError):
    """Error related to configuration operations.

    Raised when there are issues with:
    - Config loading
    - Config parsing
    - Missing configuration
    - Invalid configuration values
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details.update({"config_key": config_key, "config_file": config_file})
        super().__init__(message, code="CONFIG_ERROR", details=details, **kwargs)


class CircuitOpenError(FishstickError):
    """Error raised when circuit breaker is open."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        circuit_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details.update({"circuit_name": circuit_name})
        super().__init__(message, code="CIRCUIT_OPEN", details=details, **kwargs)


class RetryExhaustedError(FishstickError):
    """Error raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str = "All retry attempts exhausted",
        attempts: int = 0,
        last_error: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        details.update({"attempts": attempts, "last_error": str(last_error)})
        super().__init__(message, code="RETRY_EXHAUSTED", details=details, **kwargs)
        self.last_error = last_error


# =============================================================================
# ERROR HANDLING
# =============================================================================


@dataclass
class ErrorContext:
    """Context information for errors.

    Attributes:
        operation_id: Unique identifier for the operation
        operation_name: Name of the operation being performed
        component: Component where the error occurred
        user_id: Optional user identifier
        session_id: Optional session identifier
        metadata: Additional context metadata
        parent: Parent context for nested operations
    """

    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_name: Optional[str] = None
    component: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional["ErrorContext"] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "component": self.component,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent.operation_id if self.parent else None,
        }

    def child(
        self,
        operation_name: Optional[str] = None,
        component: Optional[str] = None,
        **metadata: Any,
    ) -> ErrorContext:
        """Create a child context."""
        child_metadata = {**self.metadata, **metadata}
        return ErrorContext(
            operation_name=operation_name or self.operation_name,
            component=component or self.component,
            user_id=self.user_id,
            session_id=self.session_id,
            metadata=child_metadata,
            parent=self,
        )

    def __enter__(self) -> ErrorContext:
        """Enter context manager."""
        self._token = _error_context.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        _error_context.reset(self._token)


class ErrorHandler:
    """Central error handling manager.

    Provides centralized error processing, logging, and recovery.
    """

    def __init__(self) -> None:
        self._error_callbacks: List[Callable[[FishstickError], None]] = []
        self._recovery_strategies: Dict[
            Type[Exception], Callable[[Exception], Any]
        ] = {}
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def register_callback(self, callback: Callable[[FishstickError], None]) -> None:
        """Register a callback for error handling."""
        self._error_callbacks.append(callback)

    def register_recovery(
        self,
        error_type: Type[Exception],
        strategy: Callable[[Exception], Any],
    ) -> None:
        """Register a recovery strategy for an error type."""
        self._recovery_strategies[error_type] = strategy

    def handle(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = True,
    ) -> Any:
        """Handle an error.

        Args:
            error: The exception to handle
            context: Optional error context
            reraise: Whether to reraise the error after handling

        Returns:
            Recovery result if a recovery strategy is available

        Raises:
            The original error if reraise is True
        """
        # Convert to FishstickError if needed
        if not isinstance(error, FishstickError):
            fishstick_error = FishstickError(
                message=str(error),
                cause=error,
                context=context,
            )
        else:
            fishstick_error = error
            if context:
                fishstick_error.context = context

        # Update error counts
        with self._lock:
            self._error_counts[fishstick_error.code] += 1

        # Log the error
        self._logger.error(
            f"Error [{fishstick_error.code}]: {fishstick_error.message}",
            extra={"error": fishstick_error.to_dict()},
        )

        # Execute callbacks
        for callback in self._error_callbacks:
            try:
                callback(fishstick_error)
            except Exception as e:
                self._logger.warning(f"Error callback failed: {e}")

        # Try recovery
        result = None
        for error_type, strategy in self._recovery_strategies.items():
            if isinstance(error, error_type):
                try:
                    result = strategy(error)
                    self._logger.info(f"Recovery successful for {error_type.__name__}")
                    break
                except Exception as e:
                    self._logger.error(f"Recovery failed: {e}")

        if reraise:
            raise fishstick_error

        return result

    def get_error_counts(self) -> Dict[str, int]:
        """Get error counts by code."""
        with self._lock:
            return dict(self._error_counts)

    def reset_counts(self) -> None:
        """Reset error counts."""
        with self._lock:
            self._error_counts.clear()


# Global error handler instance
_global_error_handler: ErrorHandler = ErrorHandler()


def handle_error(
    error: Exception,
    context: Optional[ErrorContext] = None,
    reraise: bool = True,
) -> Any:
    """Handle an error using the global error handler.

    Args:
        error: The exception to handle
        context: Optional error context
        reraise: Whether to reraise the error

    Returns:
        Recovery result if available
    """
    return _global_error_handler.handle(error, context, reraise)


def wrap_errors(
    *error_types: Type[Exception],
    wrapper: Optional[Type[FishstickError]] = None,
    message: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator to wrap exceptions in FishstickError.

    Args:
        error_types: Exception types to wrap
        wrapper: Custom wrapper exception class
        message: Custom error message

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except FishstickError:
                raise
            except error_types as e:
                error_class = wrapper or FishstickError
                error_msg = message or f"Error in {func.__name__}: {str(e)}"
                raise error_class(error_msg, cause=e) from e

        return cast(F, wrapper)

    return decorator


# =============================================================================
# RETRY LOGIC
# =============================================================================


class ExponentialBackoff:
    """Exponential backoff strategy for retries.

    Attributes:
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter
    """

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt."""
        import random

        delay = self.initial_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay *= 0.5 + random.random()

        return delay


@dataclass
class RetryPolicy:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        backoff: Backoff strategy to use
        retryable_errors: Exception types that trigger retry
        on_retry: Optional callback on each retry
        on_exhausted: Optional callback when retries exhausted
    """

    max_attempts: int = 3
    backoff: ExponentialBackoff = field(default_factory=ExponentialBackoff)
    retryable_errors: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )
    on_retry: Optional[Callable[[int, Exception], None]] = None
    on_exhausted: Optional[Callable[[Exception], None]] = None

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if error should trigger retry."""
        if attempt >= self.max_attempts - 1:
            return False
        return isinstance(error, self.retryable_errors)


# Fix for field default_factory
from typing import Tuple


def retry(
    max_attempts: int = 3,
    backoff: Optional[ExponentialBackoff] = None,
    retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_exhausted: Optional[Callable[[Exception], None]] = None,
) -> Callable[[F], F]:
    """Decorator to add retry logic to a function.

    Args:
        max_attempts: Maximum retry attempts
        backoff: Backoff strategy
        retryable_errors: Exception types to retry on
        on_retry: Callback on retry
        on_exhausted: Callback when exhausted

    Returns:
        Decorated function
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff=backoff or ExponentialBackoff(),
        retryable_errors=retryable_errors or (Exception,),
        on_retry=on_retry,
        on_exhausted=on_exhausted,
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if not policy.should_retry(e, attempt):
                        break

                    if on_retry:
                        try:
                            on_retry(attempt + 1, e)
                        except Exception:
                            pass

                    delay = policy.backoff.get_delay(attempt)
                    time.sleep(delay)

            if on_exhausted and last_error:
                try:
                    on_exhausted(last_error)
                except Exception:
                    pass

            raise RetryExhaustedError(
                attempts=max_attempts,
                last_error=last_error,
            ) from last_error

        return cast(F, wrapper)

    return decorator


@contextmanager
def with_retry(
    max_attempts: int = 3,
    backoff: Optional[ExponentialBackoff] = None,
    retryable_errors: Optional[Tuple[Type[Exception], ...]] = None,
) -> Iterator[None]:
    """Context manager for retry logic.

    Usage:
        with with_retry(max_attempts=3):
            risky_operation()
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff=backoff or ExponentialBackoff(),
        retryable_errors=retryable_errors or (Exception,),
    )

    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            yield
            return
        except Exception as e:
            last_error = e

            if not policy.should_retry(e, attempt):
                break

            delay = policy.backoff.get_delay(attempt)
            time.sleep(delay)

    if last_error:
        raise RetryExhaustedError(
            attempts=max_attempts,
            last_error=last_error,
        ) from last_error


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, reject requests
    HALF_OPEN = auto()  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by temporarily rejecting requests
    when failures exceed a threshold.

    Attributes:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening
        recovery_timeout: Time before attempting recovery
        half_open_max_calls: Max calls in half-open state
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        expected_error: Optional[Type[Exception]] = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.expected_error = expected_error

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    return True
                return False

            # HALF_OPEN
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.recovery_timeout

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._reset()
            else:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._logger.warning(
                    f"Circuit '{self.name}' opened after {self._failure_count} failures"
                )

    def _reset(self) -> None:
        """Reset circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._logger.info(f"Circuit '{self.name}' closed")

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with circuit breaker."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_execute():
                raise CircuitOpenError(circuit_name=self.name)

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                if self.expected_error is None or isinstance(e, self.expected_error):
                    self.record_failure()
                raise

        return cast(F, wrapper)


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_circuit_lock = threading.Lock()


def get_circuit_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    with _circuit_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, **kwargs)
        return _circuit_breakers[name]


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Decorator to apply circuit breaker pattern.

    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Failures before opening
        recovery_timeout: Time before recovery attempt
        **kwargs: Additional circuit breaker options

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        cb_name = name or func.__name__
        breaker = get_circuit_breaker(
            cb_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            **kwargs,
        )
        return breaker(func)

    return decorator


# =============================================================================
# FALLBACK STRATEGIES
# =============================================================================


@runtime_checkable
class FallbackStrategy(Protocol):
    """Protocol for fallback strategies."""

    def get_fallback(self, error: Exception, *args: Any, **kwargs: Any) -> Any:
        """Get fallback value for an error."""
        ...


class DefaultFallback:
    """Simple fallback that returns a default value."""

    def __init__(self, default_value: Any) -> None:
        self.default_value = default_value

    def get_fallback(self, error: Exception, *args: Any, **kwargs: Any) -> Any:
        """Return default value."""
        return self.default_value


class CacheFallback:
    """Fallback that returns cached value."""

    def __init__(self, cache: Optional[Dict[str, Any]] = None) -> None:
        self._cache: Dict[str, Any] = cache or {}
        self._lock = threading.Lock()

    def cache_result(self, key: str, value: Any) -> None:
        """Cache a result."""
        with self._lock:
            self._cache[key] = value

    def get_fallback(
        self,
        error: Exception,
        cache_key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Get cached fallback value."""
        if cache_key is None and args:
            cache_key = str(args[0])

        with self._lock:
            if cache_key and cache_key in self._cache:
                return self._cache[cache_key]

        raise FishstickError(
            message=f"No cached value for key: {cache_key}",
            cause=error,
        )

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


class FallbackChain:
    """Chain multiple fallback strategies."""

    def __init__(self, *strategies: FallbackStrategy) -> None:
        self.strategies = list(strategies)

    def get_fallback(self, error: Exception, *args: Any, **kwargs: Any) -> Any:
        """Try each fallback strategy in order."""
        last_error = error

        for strategy in self.strategies:
            try:
                return strategy.get_fallback(last_error, *args, **kwargs)
            except Exception as e:
                last_error = e

        raise FishstickError(
            message="All fallback strategies exhausted",
            cause=last_error,
        )


def fallback(
    strategy: FallbackStrategy,
    catch: Tuple[Type[Exception], ...] = (Exception,),
    reraise: bool = False,
) -> Callable[[F], F]:
    """Decorator to add fallback behavior.

    Args:
        strategy: Fallback strategy to use
        catch: Exception types to catch
        reraise: Whether to reraise after fallback

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except catch as e:
                result = strategy.get_fallback(e, *args, **kwargs)
                if reraise:
                    raise
                return result

        return cast(F, wrapper)

    return decorator


# =============================================================================
# VALIDATION
# =============================================================================


@runtime_checkable
class InputValidator(Protocol):
    """Protocol for input validators."""

    def validate(self, data: Any) -> bool:
        """Validate input data."""
        ...

    def get_errors(self) -> List[str]:
        """Get validation errors."""
        ...


@runtime_checkable
class OutputValidator(Protocol):
    """Protocol for output validators."""

    def validate(self, data: Any) -> bool:
        """Validate output data."""
        ...


class SchemaValidator:
    """JSON Schema-based validator.

    Uses jsonschema if available, otherwise basic validation.
    """

    def __init__(self, schema: Dict[str, Any]) -> None:
        self.schema = schema
        self._errors: List[str] = []

        # Try to import jsonschema
        try:
            import jsonschema

            self._validator = jsonschema.Draft7Validator(schema)
            self._use_jsonschema = True
        except ImportError:
            self._validator = None
            self._use_jsonschema = False

    def validate(self, data: Any) -> bool:
        """Validate data against schema."""
        self._errors = []

        if self._use_jsonschema and self._validator:
            errors = list(self._validator.iter_errors(data))
            self._errors = [str(e.message) for e in errors]
            return len(errors) == 0
        else:
            # Basic type validation
            return self._basic_validate(data)

    def _basic_validate(self, data: Any) -> bool:
        """Basic validation without jsonschema."""
        schema_type = self.schema.get("type")

        if schema_type == "object":
            if not isinstance(data, dict):
                self._errors.append(f"Expected object, got {type(data).__name__}")
                return False

            # Check required fields
            required = self.schema.get("required", [])
            for field in required:
                if field not in data:
                    self._errors.append(f"Missing required field: {field}")

            # Validate properties
            properties = self.schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in data:
                    prop_type = prop_schema.get("type")
                    value = data[prop]

                    if prop_type == "string" and not isinstance(value, str):
                        self._errors.append(f"Field '{prop}' should be string")
                    elif prop_type == "number" and not isinstance(value, (int, float)):
                        self._errors.append(f"Field '{prop}' should be number")
                    elif prop_type == "integer" and not isinstance(value, int):
                        self._errors.append(f"Field '{prop}' should be integer")
                    elif prop_type == "array" and not isinstance(value, list):
                        self._errors.append(f"Field '{prop}' should be array")
                    elif prop_type == "boolean" and not isinstance(value, bool):
                        self._errors.append(f"Field '{prop}' should be boolean")

        elif schema_type == "array":
            if not isinstance(data, list):
                self._errors.append(f"Expected array, got {type(data).__name__}")
                return False

        return len(self._errors) == 0

    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self._errors.copy()

    def validate_or_raise(self, data: Any) -> None:
        """Validate and raise ValidationError if invalid."""
        if not self.validate(data):
            raise ValidationError(
                message="Schema validation failed",
                details={"errors": self._errors},
            )


def validate_input(
    validator: InputValidator,
    data: Any,
    raise_on_error: bool = True,
) -> bool:
    """Validate input data.

    Args:
        validator: Validator to use
        data: Data to validate
        raise_on_error: Whether to raise exception on failure

    Returns:
        True if valid, False otherwise

    Raises:
        ValidationError: If validation fails and raise_on_error is True
    """
    is_valid = validator.validate(data)

    if not is_valid and raise_on_error:
        errors = (
            validator.get_errors()
            if hasattr(validator, "get_errors")
            else ["Validation failed"]
        )
        raise ValidationError(
            message="Input validation failed",
            details={"errors": errors},
        )

    return is_valid


# =============================================================================
# LOGGING
# =============================================================================


class ErrorLogger:
    """Structured error logging.

    Provides consistent error logging with context and metadata.
    """

    def __init__(
        self,
        name: str = "fishstick.errors",
        level: int = logging.ERROR,
    ) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._handlers: List[logging.Handler] = []

    def add_handler(self, handler: logging.Handler) -> None:
        """Add a log handler."""
        self.logger.addHandler(handler)
        self._handlers.append(handler)

    def add_file_handler(
        self,
        filepath: str,
        level: int = logging.ERROR,
    ) -> logging.Handler:
        """Add a file handler."""
        handler = logging.FileHandler(filepath)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.add_handler(handler)
        return handler

    def log(
        self,
        level: int,
        message: str,
        error: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an error with context."""
        log_data: Dict[str, Any] = {
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if error:
            log_data["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            if isinstance(error, FishstickError):
                log_data["error"].update(error.to_dict())

        if context:
            log_data["context"] = context.to_dict()

        if extra:
            log_data["extra"] = extra

        self.logger.log(level, json.dumps(log_data))

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an error."""
        self.log(logging.ERROR, message, error, context, extra)

    def warning(
        self,
        message: str,
        error: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a warning."""
        self.log(logging.WARNING, message, error, context, extra)

    def info(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log info."""
        self.log(logging.INFO, message, None, context, extra)


class ErrorTracker:
    """Track errors for monitoring and analysis.

    Maintains error statistics and history for debugging.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self._errors: List[Dict[str, Any]] = []
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._error_types: Dict[str, int] = defaultdict(int)

    def track(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
    ) -> str:
        """Track an error occurrence.

        Returns:
            Error tracking ID
        """
        error_id = str(uuid.uuid4())

        error_data: Dict[str, Any] = {
            "id": error_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
        }

        if isinstance(error, FishstickError):
            error_data["code"] = error.code
            error_data["details"] = error.details

        if context:
            error_data["context"] = context.to_dict()

        with self._lock:
            self._errors.append(error_data)

            # Trim history if needed
            if len(self._errors) > self.max_history:
                self._errors = self._errors[-self.max_history :]

            # Update counts
            error_type = type(error).__name__
            self._error_types[error_type] += 1

            if isinstance(error, FishstickError):
                self._error_counts[error.code] += 1

        return error_id

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            return {
                "total_errors": len(self._errors),
                "by_type": dict(self._error_types),
                "by_code": dict(self._error_counts),
                "recent_errors": self._errors[-10:],
            }

    def get_error_history(
        self,
        error_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get error history."""
        with self._lock:
            errors = self._errors

            if error_type:
                errors = [e for e in errors if e["type"] == error_type]

            return errors[-limit:]

    def clear(self) -> None:
        """Clear all tracked errors."""
        with self._lock:
            self._errors.clear()
            self._error_counts.clear()
            self._error_types.clear()


# Global error tracker
_global_error_tracker = ErrorTracker()


def log_error(
    error: Exception,
    message: Optional[str] = None,
    context: Optional[ErrorContext] = None,
    level: int = logging.ERROR,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Log an error and track it.

    Args:
        error: The error to log
        message: Optional custom message
        context: Optional error context
        level: Log level
        logger: Optional custom logger

    Returns:
        Error tracking ID
    """
    log = logger or logging.getLogger(__name__)

    msg = message or str(error)
    error_id = _global_error_tracker.track(error, context)

    log.log(
        level,
        f"[{error_id}] {msg}",
        extra={
            "error_id": error_id,
            "error_type": type(error).__name__,
        },
    )

    return error_id


def report_error(
    error: Exception,
    context: Optional[ErrorContext] = None,
    include_traceback: bool = True,
) -> Dict[str, Any]:
    """Generate a comprehensive error report.

    Args:
        error: The error to report
        context: Optional error context
        include_traceback: Whether to include traceback

    Returns:
        Error report dictionary
    """
    report: Dict[str, Any] = {
        "error_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }

    if isinstance(error, FishstickError):
        report["error"].update(error.to_dict())

    if context:
        report["context"] = context.to_dict()

    if include_traceback:
        report["traceback"] = traceback.format_exc()

    # Add system info
    import platform

    report["system"] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    return report


# =============================================================================
# UTILITIES
# =============================================================================


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    catch: Tuple[Type[Exception], ...] = (Exception,),
    on_error: Optional[Callable[[Exception], None]] = None,
    **kwargs: Any,
) -> Optional[T]:
    """Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        catch: Exception types to catch
        on_error: Optional error callback
        **kwargs: Keyword arguments

    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except catch as e:
        if on_error:
            try:
                on_error(e)
            except Exception:
                pass
        return default


@contextmanager
def ignore_errors(
    *error_types: Type[Exception],
    log: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Iterator[None]:
    """Context manager to ignore specified errors.

    Usage:
        with ignore_errors(ValueError, TypeError):
            risky_operation()
    """
    error_types = error_types or (Exception,)
    log_func = logger or logging.getLogger(__name__)

    try:
        yield
    except error_types as e:
        if log:
            log_func.debug(f"Ignored error: {e}")


class ErrorCollector:
    """Collect multiple errors without raising immediately.

    Useful for validation where you want to collect all errors
    before reporting them.
    """

    def __init__(self, raise_on_collect: bool = False) -> None:
        self.errors: List[Exception] = []
        self.raise_on_collect = raise_on_collect

    def collect(self, error: Exception) -> None:
        """Collect an error."""
        self.errors.append(error)
        if self.raise_on_collect:
            raise error

    def try_collect(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> Optional[T]:
        """Try to execute function and collect any errors."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.collect(e)
            return None

    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0

    def raise_if_errors(self) -> None:
        """Raise a combined error if any errors were collected."""
        if not self.errors:
            return

        if len(self.errors) == 1:
            raise self.errors[0]

        # Create combined error
        messages = [str(e) for e in self.errors]
        raise FishstickError(
            message=f"Multiple errors ({len(self.errors)}): {', '.join(messages[:3])}",
            details={
                "error_count": len(self.errors),
                "errors": [
                    {"type": type(e).__name__, "message": str(e)} for e in self.errors
                ],
            },
        )

    def clear(self) -> None:
        """Clear collected errors."""
        self.errors.clear()

    def __enter__(self) -> ErrorCollector:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and raise if errors exist."""
        self.raise_if_errors()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def ensure_error_context(
    operation_name: Optional[str] = None,
    component: Optional[str] = None,
    **metadata: Any,
) -> ErrorContext:
    """Get or create an error context.

    Args:
        operation_name: Name of the operation
        component: Component name
        **metadata: Additional metadata

    Returns:
        ErrorContext instance
    """
    existing = _error_context.get()

    if existing:
        return existing.child(operation_name, component, **metadata)

    return ErrorContext(
        operation_name=operation_name,
        component=component,
        metadata=metadata,
    )


def set_error_context(context: ErrorContext) -> Any:
    """Set the current error context.

    Returns:
        Token for resetting context
    """
    return _error_context.set(context)


def get_error_context() -> Optional[ErrorContext]:
    """Get the current error context."""
    return _error_context.get()


def reset_error_context(token: Any) -> None:
    """Reset error context using token."""
    _error_context.reset(token)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "FishstickError",
    "ModelError",
    "DataError",
    "TrainingError",
    "InferenceError",
    "ValidationError",
    "ConfigurationError",
    "CircuitOpenError",
    "RetryExhaustedError",
    # Error Handling
    "ErrorHandler",
    "handle_error",
    "wrap_errors",
    "ErrorContext",
    # Retry Logic
    "RetryPolicy",
    "retry",
    "with_retry",
    "ExponentialBackoff",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "circuit_breaker",
    "get_circuit_breaker",
    # Fallback
    "FallbackStrategy",
    "fallback",
    "DefaultFallback",
    "CacheFallback",
    "FallbackChain",
    # Validation
    "InputValidator",
    "OutputValidator",
    "SchemaValidator",
    "validate_input",
    # Logging
    "ErrorLogger",
    "log_error",
    "ErrorTracker",
    "report_error",
    # Utilities
    "safe_execute",
    "ignore_errors",
    "ErrorCollector",
    "ensure_error_context",
    "get_error_context",
    "set_error_context",
    "reset_error_context",
]
