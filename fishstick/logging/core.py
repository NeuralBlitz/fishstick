"""
fishstick Logging Module
========================
A comprehensive logging system providing flexible, extensible logging capabilities.

Features:
- Multiple loggers with hierarchical management
- Various handlers (stream, file, rotating, syslog, HTTP, queue)
- Multiple formatters (plain, JSON, colored, structured, custom)
- Configurable log levels with custom level support
- Context management for structured logging
- Logger adapters for extended functionality
- Utility functions for common logging patterns
"""

from __future__ import annotations

import atexit
import functools
import inspect
import json
import logging
import logging.config
import logging.handlers
import os
import queue
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union


# ============================================================================
# LEVELS
# ============================================================================

# Standard logging levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Level name mappings
_LEVEL_NAMES = {
    "DEBUG": DEBUG,
    "INFO": INFO,
    "WARNING": WARNING,
    "ERROR": ERROR,
    "CRITICAL": CRITICAL,
}

# Reverse mapping for lookup
_LEVEL_VALUES = {v: k for k, v in _LEVEL_NAMES.items()}


def set_level(logger: Union[str, logging.Logger], level: Union[int, str]) -> None:
    """Set the logging level for a logger.

    Args:
        logger: Logger name or logger instance
        level: Logging level (int or string name)
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    if isinstance(level, str):
        level = _LEVEL_NAMES.get(level.upper(), INFO)
    logger.setLevel(level)


def get_level(logger: Union[str, logging.Logger]) -> int:
    """Get the current logging level of a logger.

    Args:
        logger: Logger name or logger instance

    Returns:
        Current logging level
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    return logger.level


def add_level(name: str, level: int, method_name: Optional[str] = None) -> int:
    """Add a custom logging level.

    Args:
        name: Name of the new level
        level: Numeric value for the level
        method_name: Optional method name (defaults to lower case of name)

    Returns:
        The level value

    Example:
        >>> add_level("TRACE", 5)
        >>> logger = get_logger("myapp")
        >>> logger.trace("Very detailed debug message")
    """
    if method_name is None:
        method_name = name.lower()

    logging.addLevelName(level, name)

    def log_method(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            self._log(level, message, args, **kwargs)

    setattr(logging.Logger, method_name, log_method)
    _LEVEL_NAMES[name] = level
    _LEVEL_VALUES[level] = name

    return level


# ============================================================================
# FORMATTERS
# ============================================================================


class Formatter(logging.Formatter):
    """Standard formatter with optional color support detection."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
    ):
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        timestamp_field: str = "timestamp",
        indent: Optional[int] = None,
    ):
        super().__init__()
        self.fields = fields or [
            "name",
            "levelname",
            "message",
            "pathname",
            "lineno",
            "funcName",
            "created",
            "thread",
            "process",
        ]
        self.timestamp_field = timestamp_field
        self.indent = indent

    def format(self, record: logging.LogRecord) -> str:
        log_data = {}

        # Add standard fields
        for field in self.fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)

        # Add formatted timestamp
        log_data[self.timestamp_field] = datetime.fromtimestamp(
            record.created
        ).isoformat()

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith("_"):
                try:
                    json.dumps({key: value})  # Test if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, indent=self.indent, default=str)


class ColoredFormatter(logging.Formatter):
    """Formatter with ANSI color support."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: Optional[bool] = None,
    ):
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)

        if use_colors is None:
            use_colors = sys.stderr.isatty()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)

        if self.use_colors:
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            formatted = f"{color}{formatted}{reset}"

        return formatted


class StructuredFormatter(logging.Formatter):
    """Formatter for structured text output with key=value pairs."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        separator: str = " | ",
        kv_separator: str = "=",
    ):
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s"
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.separator = separator
        self.kv_separator = kv_separator

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)

        # Add message
        parts = [base, record.getMessage()]

        # Add structured data if present
        structured = getattr(record, "structured", None)
        if structured:
            kv_pairs = [f"{k}{self.kv_separator}{v}" for k, v in structured.items()]
            parts.extend(kv_pairs)

        # Add exception info
        if record.exc_info:
            parts.append(
                f"exception{self.kv_separator}{self.formatException(record.exc_info)}"
            )

        return self.separator.join(parts)


class CustomFormatter(logging.Formatter):
    """Custom formatter with per-level format strings."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        level_formats: Optional[Dict[int, str]] = None,
        default_fmt: Optional[str] = None,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.level_formats = level_formats or {}
        self.default_fmt = default_fmt or fmt
        self._formatters: Dict[int, logging.Formatter] = {}

    def format(self, record: logging.LogRecord) -> str:
        # Get format for this level
        fmt = self.level_formats.get(record.levelno, self.default_fmt)

        # Cache formatters
        if record.levelno not in self._formatters:
            self._formatters[record.levelno] = logging.Formatter(
                fmt=fmt, datefmt=self.datefmt
            )

        return self._formatters[record.levelno].format(record)


# ============================================================================
# HANDLERS
# ============================================================================


class StreamHandler(logging.StreamHandler):
    """Standard stream handler with formatter support."""

    def __init__(
        self,
        stream: Any = sys.stderr,
        formatter: Optional[logging.Formatter] = None,
        level: Union[int, str] = DEBUG,
    ):
        super().__init__(stream)
        if formatter:
            self.setFormatter(formatter)
        self.setLevel(level)


class FileHandler(logging.FileHandler):
    """File handler with automatic directory creation."""

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = False,
        formatter: Optional[logging.Formatter] = None,
        level: Union[int, str] = DEBUG,
    ):
        # Create directory if needed
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(str(filename), mode, encoding, delay)
        if formatter:
            self.setFormatter(formatter)
        self.setLevel(level)


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler based on file size."""

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        maxBytes: int = 10 * 1024 * 1024,  # 10 MB
        backupCount: int = 5,
        encoding: Optional[str] = None,
        delay: bool = False,
        formatter: Optional[logging.Formatter] = None,
        level: Union[int, str] = DEBUG,
    ):
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(str(filename), mode, maxBytes, backupCount, encoding, delay)
        if formatter:
            self.setFormatter(formatter)
        self.setLevel(level)


class TimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Rotating file handler based on time intervals."""

    def __init__(
        self,
        filename: Union[str, Path],
        when: str = "midnight",
        interval: int = 1,
        backupCount: int = 7,
        encoding: Optional[str] = None,
        delay: bool = False,
        utc: bool = False,
        atTime: Optional[datetime] = None,
        formatter: Optional[logging.Formatter] = None,
        level: Union[int, str] = DEBUG,
    ):
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            str(filename),
            when,
            interval,
            backupCount,
            encoding,
            delay,
            utc,
            atTime,
        )
        if formatter:
            self.setFormatter(formatter)
        self.setLevel(level)


class SysLogHandler(logging.handlers.SysLogHandler):
    """Syslog handler for system logging."""

    def __init__(
        self,
        address: Union[str, Tuple[str, int]] = "/dev/log",
        facility: int = logging.handlers.SysLogHandler.LOG_USER,
        socktype: Any = None,
        formatter: Optional[logging.Formatter] = None,
        level: Union[int, str] = DEBUG,
    ):
        super().__init__(address, facility, socktype)
        if formatter:
            self.setFormatter(formatter)
        self.setLevel(level)


class HTTPHandler(logging.handlers.HTTPHandler):
    """HTTP handler for remote logging."""

    def __init__(
        self,
        host: str,
        url: str,
        method: str = "POST",
        secure: bool = False,
        credentials: Optional[Tuple[str, str]] = None,
        context: Any = None,
        formatter: Optional[logging.Formatter] = None,
        level: Union[int, str] = DEBUG,
    ):
        super().__init__(host, url, method, secure, credentials, context)
        if formatter:
            self.setFormatter(formatter)
        self.setLevel(level)

    def mapLogRecord(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Map log record to dictionary for HTTP transmission."""
        return {
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "pathname": record.pathname,
            "lineno": record.lineno,
            "funcName": record.funcName,
        }


class QueueHandler(logging.handlers.QueueHandler):
    """Queue handler for asynchronous logging."""

    def __init__(
        self,
        queue: queue.Queue,
        respect_handler_level: bool = True,
        formatter: Optional[logging.Formatter] = None,
        level: Union[int, str] = DEBUG,
    ):
        super().__init__(queue)
        self.respect_handler_level = respect_handler_level
        if formatter:
            self.setFormatter(formatter)
        self.setLevel(level)


class QueueListener:
    """Listener for queue-based logging."""

    def __init__(
        self,
        log_queue: queue.Queue,
        handlers: List[logging.Handler],
        respect_handler_level: bool = True,
    ):
        self.queue = log_queue
        self.handlers = handlers
        self.respect_handler_level = respect_handler_level
        self._listener = logging.handlers.QueueListener(
            log_queue, *handlers, respect_handler_level=respect_handler_level
        )

    def start(self) -> None:
        """Start the queue listener."""
        self._listener.start()

    def stop(self) -> None:
        """Stop the queue listener."""
        self._listener.stop()

    def enqueue_sentinel(self) -> None:
        """Enqueue a sentinel to signal the listener to stop."""
        self.queue.put_nowait(None)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================================
# CONTEXT
# ============================================================================

# Context variable for storing logging context
_context_var: ContextVar[Dict[str, Any]] = ContextVar("logging_context", default={})


class ContextFilter(logging.Filter):
    """Filter that adds context information to log records."""

    def __init__(self, context_dict: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context_dict = context_dict or {}

    def filter(self, record: logging.LogRecord) -> bool:
        # Get context from context variable
        context = _context_var.get({})

        # Merge with filter's context
        merged = {**self.context_dict, **context}

        # Add to record
        for key, value in merged.items():
            if not hasattr(record, key):
                setattr(record, key, value)

        return True


def add_context(key: str, value: Any) -> None:
    """Add a context key-value pair.

    Args:
        key: Context key
        value: Context value
    """
    context = _context_var.get({})
    context[key] = value
    _context_var.set(context)


def get_context() -> Dict[str, Any]:
    """Get the current logging context.

    Returns:
        Current context dictionary
    """
    return _context_var.get({}).copy()


def clear_context() -> None:
    """Clear the current logging context."""
    _context_var.set({})


@contextmanager
def context(**kwargs):
    """Context manager for temporary logging context.

    Args:
        **kwargs: Context key-value pairs

    Example:
        >>> with context(request_id="123", user_id="456"):
        ...     logger.info("Processing request")
    """
    old_context = _context_var.get({})
    new_context = {**old_context, **kwargs}
    token = _context_var.set(new_context)
    try:
        yield
    finally:
        _context_var.reset(token)


# ============================================================================
# ADAPTERS
# ============================================================================


class LoggerAdapter(logging.LoggerAdapter):
    """Extended logger adapter with additional functionality."""

    def __init__(
        self,
        logger: logging.Logger,
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # Merge extra dict with kwargs
        kwargs["extra"] = {**(kwargs.get("extra") or {}), **self.extra}
        return msg, kwargs

    def with_context(self, **context) -> LoggerAdapter:
        """Create a new adapter with additional context."""
        new_extra = {**self.extra, **context}
        return LoggerAdapter(self.logger, new_extra)


class ExtraAdapter(LoggerAdapter):
    """Adapter that automatically adds extra fields to log records."""

    def __init__(
        self,
        logger: logging.Logger,
        extra: Optional[Dict[str, Any]] = None,
        prefix: str = "",
    ):
        super().__init__(logger, extra)
        self.prefix = prefix

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        extra = kwargs.get("extra", {})

        # Add prefixed extra fields
        for key, value in self.extra.items():
            extra[f"{self.prefix}{key}"] = value

        kwargs["extra"] = extra
        return msg, kwargs


class ContextAdapter(LoggerAdapter):
    """Adapter that integrates with logging context."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # Get context
        ctx = get_context()

        # Merge with extra
        extra = kwargs.get("extra", {})
        extra = {**ctx, **self.extra, **extra}
        kwargs["extra"] = extra

        return msg, kwargs


# ============================================================================
# LOGGERS
# ============================================================================


class Logger(logging.Logger):
    """Enhanced logger with additional functionality."""

    def __init__(self, name: str, level: int = DEBUG):
        super().__init__(name, level)

    def structured(self, level: int, message: str, **kwargs) -> None:
        """Log a structured message with key-value pairs."""
        extra = kwargs.pop("extra", {})
        extra["structured"] = kwargs
        self.log(level, message, extra=extra)

    def debug_structured(self, message: str, **kwargs) -> None:
        """Log a structured debug message."""
        self.structured(DEBUG, message, **kwargs)

    def info_structured(self, message: str, **kwargs) -> None:
        """Log a structured info message."""
        self.structured(INFO, message, **kwargs)

    def warning_structured(self, message: str, **kwargs) -> None:
        """Log a structured warning message."""
        self.structured(WARNING, message, **kwargs)

    def error_structured(self, message: str, **kwargs) -> None:
        """Log a structured error message."""
        self.structured(ERROR, message, **kwargs)

    def critical_structured(self, message: str, **kwargs) -> None:
        """Log a structured critical message."""
        self.structured(CRITICAL, message, **kwargs)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger by name.

    Args:
        name: Logger name (None returns root logger)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(name)
    return logging.getLogger()


def create_logger(
    name: str,
    level: Union[int, str] = DEBUG,
    handlers: Optional[List[logging.Handler]] = None,
    propagate: bool = False,
) -> Logger:
    """Create a new logger with specified configuration.

    Args:
        name: Logger name
        level: Logging level
        handlers: List of handlers to add
        propagate: Whether to propagate to parent loggers

    Returns:
        Configured logger instance
    """
    logger = Logger(name)

    if isinstance(level, str):
        level = _LEVEL_NAMES.get(level.upper(), DEBUG)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Add new handlers
    if handlers:
        for handler in handlers:
            logger.addHandler(handler)

    logger.propagate = propagate

    return logger


class LoggerManager:
    """Manager for logger configuration and lifecycle."""

    def __init__(self):
        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers: Dict[str, logging.Handler] = {}
        self._formatters: Dict[str, logging.Formatter] = {}
        self._lock = threading.Lock()

    def register_logger(self, name: str, logger: logging.Logger) -> None:
        """Register a logger instance."""
        with self._lock:
            self._loggers[name] = logger

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger."""
        with self._lock:
            if name not in self._loggers:
                self._loggers[name] = get_logger(name)
            return self._loggers[name]

    def register_handler(self, name: str, handler: logging.Handler) -> None:
        """Register a handler instance."""
        with self._lock:
            self._handlers[name] = handler

    def get_handler(self, name: str) -> Optional[logging.Handler]:
        """Get a registered handler."""
        with self._lock:
            return self._handlers.get(name)

    def register_formatter(self, name: str, formatter: logging.Formatter) -> None:
        """Register a formatter instance."""
        with self._lock:
            self._formatters[name] = formatter

    def get_formatter(self, name: str) -> Optional[logging.Formatter]:
        """Get a registered formatter."""
        with self._lock:
            return self._formatters.get(name)

    def configure_logger(
        self,
        name: str,
        handler_names: List[str],
        level: Union[int, str] = DEBUG,
    ) -> logging.Logger:
        """Configure a logger with registered handlers."""
        logger = self.get_logger(name)

        if isinstance(level, str):
            level = _LEVEL_NAMES.get(level.upper(), DEBUG)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers = []

        # Add handlers
        for handler_name in handler_names:
            handler = self.get_handler(handler_name)
            if handler:
                logger.addHandler(handler)

        return logger

    def shutdown(self) -> None:
        """Shutdown all registered loggers."""
        with self._lock:
            for logger in self._loggers.values():
                for handler in logger.handlers:
                    handler.flush()
                    handler.close()
            self._loggers.clear()
            self._handlers.clear()
            self._formatters.clear()


# Global logger manager instance
_manager = LoggerManager()


def get_manager() -> LoggerManager:
    """Get the global logger manager."""
    return _manager


# ============================================================================
# CONFIGURATION
# ============================================================================


def configure_logging(
    level: Union[int, str] = INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    handlers: Optional[List[logging.Handler]] = None,
    disable_existing_loggers: bool = False,
) -> None:
    """Configure basic logging setup.

    Args:
        level: Default logging level
        format_string: Log format string
        date_format: Date format string
        handlers: List of handlers to use
        disable_existing_loggers: Whether to disable existing loggers
    """
    if isinstance(level, str):
        level = _LEVEL_NAMES.get(level.upper(), INFO)

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create default console handler if none provided
    if handlers is None:
        console_handler = StreamHandler(
            stream=sys.stdout,
            formatter=Formatter(fmt=format_string, datefmt=date_format),
            level=level,
        )
        handlers = [console_handler]

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    # Configure existing loggers
    if disable_existing_loggers:
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).disabled = True


def load_config(
    config: Union[str, Path, Dict[str, Any]],
    disable_existing_loggers: bool = False,
) -> None:
    """Load logging configuration from dict or file.

    Args:
        config: Configuration dict, file path, or JSON/YAML string
        disable_existing_loggers: Whether to disable existing loggers
    """
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.exists():
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    try:
                        import yaml

                        config = yaml.safe_load(f)
                    except ImportError:
                        raise ImportError("PyYAML required for YAML config files")
                else:
                    config = json.load(f)
        else:
            # Try parsing as JSON string
            config = json.loads(config)

    dict_config(config, disable_existing_loggers)


def dict_config(
    config: Dict[str, Any],
    disable_existing_loggers: bool = False,
) -> None:
    """Configure logging from a dictionary.

    Args:
        config: Configuration dictionary
        disable_existing_loggers: Whether to disable existing loggers
    """
    config["disable_existing_loggers"] = disable_existing_loggers
    logging.config.dictConfig(config)


def file_config(
    filename: Union[str, Path],
    defaults: Optional[Dict[str, str]] = None,
    disable_existing_loggers: bool = False,
) -> None:
    """Configure logging from a configuration file.

    Supports .ini, .conf, .yaml, .yml, and .json files.

    Args:
        filename: Path to configuration file
        defaults: Default values for config file
        disable_existing_loggers: Whether to disable existing loggers
    """
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filename}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        load_config(path, disable_existing_loggers)
    elif suffix == ".json":
        load_config(path, disable_existing_loggers)
    elif suffix in (".ini", ".conf"):
        logging.config.fileConfig(
            str(filename),
            defaults=defaults,
            disable_existing_loggers=disable_existing_loggers,
        )
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


# ============================================================================
# UTILITIES
# ============================================================================


def log_exception(
    logger: Optional[logging.Logger] = None,
    level: int = ERROR,
    msg: str = "Exception occurred",
    exc_info: bool = True,
    stack_info: bool = False,
) -> None:
    """Log an exception with full traceback.

    Args:
        logger: Logger to use (default: root logger)
        level: Logging level
        msg: Message to log
        exc_info: Include exception info
        stack_info: Include stack info
    """
    if logger is None:
        logger = logging.getLogger()

    logger.log(level, msg, exc_info=exc_info, stack_info=stack_info)


@contextmanager
def log_time(
    logger: Optional[logging.Logger] = None,
    level: int = INFO,
    msg: str = "Operation",
    log_start: bool = False,
):
    """Context manager to log execution time.

    Args:
        logger: Logger to use (default: root logger)
        level: Logging level for completion message
        msg: Message prefix
        log_start: Whether to log when operation starts

    Example:
        >>> with log_time(logger, INFO, "Training"):
        ...     train_model()
    """
    if logger is None:
        logger = logging.getLogger()

    start_time = time.time()

    if log_start:
        logger.log(level, f"{msg} started")

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.log(level, f"{msg} completed in {elapsed:.4f}s")


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def log_function(
    logger: Optional[logging.Logger] = None,
    level: int = DEBUG,
    log_args: bool = True,
    log_result: bool = False,
    log_exception: bool = True,
) -> Callable[[F], F]:
    """Decorator to log function calls.

    Args:
        logger: Logger to use (default: module logger)
        level: Logging level
        log_args: Whether to log function arguments
        log_result: Whether to log return value
        log_exception: Whether to log exceptions

    Returns:
        Decorator function

    Example:
        >>> @log_function(log_args=True)
        ... def my_function(x, y):
        ...     return x + y
    """

    def decorator(func: F) -> F:
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__

            # Log entry
            if log_args:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                all_args = ", ".join(args_repr + kwargs_repr)
                logger.log(level, f"{func_name}({all_args}) called")
            else:
                logger.log(level, f"{func_name}() called")

            try:
                result = func(*args, **kwargs)

                # Log exit
                if log_result:
                    logger.log(level, f"{func_name}() returned {result!r}")
                else:
                    logger.log(level, f"{func_name}() completed")

                return result
            except Exception as e:
                if log_exception:
                    logger.exception(f"{func_name}() raised {type(e).__name__}: {e}")
                raise

        return wrapper

    return decorator


class LogCapture:
    """Context manager to capture log output."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        level: int = DEBUG,
    ):
        self.logger = logger or logging.getLogger()
        self.level = level
        self.records: List[logging.LogRecord] = []
        self._handler: Optional[logging.Handler] = None

    def emit(self, record: logging.LogRecord) -> None:
        """Handle a log record."""
        self.records.append(record)

    def __enter__(self):
        self._handler = logging.Handler()
        self._handler.emit = self.emit
        self._handler.setLevel(self.level)
        self.logger.addHandler(self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handler:
            self.logger.removeHandler(self._handler)

    def get_messages(self) -> List[str]:
        """Get list of log messages."""
        return [r.getMessage() for r in self.records]

    def contains(self, text: str) -> bool:
        """Check if any message contains text."""
        return any(text in msg for msg in self.get_messages())


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Levels
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "set_level",
    "get_level",
    "add_level",
    # Formatters
    "Formatter",
    "JSONFormatter",
    "ColoredFormatter",
    "StructuredFormatter",
    "CustomFormatter",
    # Handlers
    "StreamHandler",
    "FileHandler",
    "RotatingFileHandler",
    "TimedRotatingFileHandler",
    "SysLogHandler",
    "HTTPHandler",
    "QueueHandler",
    "QueueListener",
    # Context
    "ContextFilter",
    "add_context",
    "get_context",
    "clear_context",
    "context",
    # Adapters
    "LoggerAdapter",
    "ExtraAdapter",
    "ContextAdapter",
    # Loggers
    "Logger",
    "get_logger",
    "create_logger",
    "LoggerManager",
    "get_manager",
    # Configuration
    "configure_logging",
    "load_config",
    "dict_config",
    "file_config",
    # Utilities
    "log_exception",
    "log_time",
    "log_function",
    "LogCapture",
]
