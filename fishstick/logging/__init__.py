"""
fishstick Logging Module
========================
Comprehensive logging system for fishstick.

Example:
    >>> from fishstick.logging import get_logger, StreamHandler, ColoredFormatter
    >>>
    >>> # Create a logger
    >>> logger = get_logger("myapp")
    >>>
    >>> # Add a colored console handler
    >>> handler = StreamHandler(formatter=ColoredFormatter())
    >>> logger.addHandler(handler)
    >>>
    >>> # Log messages
    >>> logger.info("Application started")
    >>> logger.error("An error occurred", exc_info=True)
"""

from .core import (
    # Levels
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
    set_level,
    get_level,
    add_level,
    # Formatters
    Formatter,
    JSONFormatter,
    ColoredFormatter,
    StructuredFormatter,
    CustomFormatter,
    # Handlers
    StreamHandler,
    FileHandler,
    RotatingFileHandler,
    TimedRotatingFileHandler,
    SysLogHandler,
    HTTPHandler,
    QueueHandler,
    QueueListener,
    # Context
    ContextFilter,
    add_context,
    get_context,
    clear_context,
    context,
    # Adapters
    LoggerAdapter,
    ExtraAdapter,
    ContextAdapter,
    # Loggers
    Logger,
    get_logger,
    create_logger,
    LoggerManager,
    get_manager,
    # Configuration
    configure_logging,
    load_config,
    dict_config,
    file_config,
    # Utilities
    log_exception,
    log_time,
    log_function,
    LogCapture,
)

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
