"""
Fishstick Monitoring Core Module
================================
Comprehensive monitoring solution for ML systems with metrics, logging,
tracing, alerting, health checks, performance monitoring, and dashboards.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import platform
import psutil
import re
import smtplib
import sys
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
)

import boto3
from botocore.exceptions import ClientError
import requests
from datadog import initialize, api
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server,
    CollectorRegistry,
)
import opentelemetry.trace as trace
from opentelemetry import metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ============================================================================
# Constants and Types
# ============================================================================

T = TypeVar("T")
MetricValue = Union[int, float, bool]


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """Metric data structure."""

    name: str
    value: MetricValue
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class LogEntry:
    """Structured log entry."""

    level: str
    message: str
    timestamp: datetime
    service: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthResult:
    """Health check result."""

    status: HealthStatus
    component: str
    message: str
    timestamp: datetime
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert data structure."""

    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ============================================================================
# Metrics Classes
# ============================================================================


class BaseMetrics(ABC):
    """Abstract base class for metrics collectors."""

    def __init__(self, service_name: str = "fishstick"):
        self.service_name = service_name
        self._initialized = False

    @abstractmethod
    def record_metric(
        self, name: str, value: MetricValue, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value."""
        pass

    @abstractmethod
    def get_metrics(self, pattern: Optional[str] = None) -> List[Metric]:
        """Get recorded metrics."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the metrics collector."""
        pass


class PrometheusMetrics(BaseMetrics):
    """Prometheus metrics collector."""

    def __init__(
        self,
        service_name: str = "fishstick",
        port: int = 9090,
        endpoint: str = "/metrics",
    ):
        super().__init__(service_name)
        self.port = port
        self.endpoint = endpoint
        self.registry = CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._metric_values: List[Metric] = []
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize Prometheus server."""
        if not self._initialized:
            start_http_server(self.port, registry=self.registry)
            self._initialized = True
            logging.info(f"Prometheus metrics server started on port {self.port}")

    def record_metric(
        self, name: str, value: MetricValue, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric in Prometheus."""
        labels = labels or {}
        labels["service"] = self.service_name

        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(
                    name, f"Metric {name}", list(labels.keys()), registry=self.registry
                )

            self._metrics[name].labels(**labels).set(float(value))

            metric = Metric(
                name=name, value=value, timestamp=datetime.utcnow(), labels=labels
            )
            self._metric_values.append(metric)

    def get_metrics(self, pattern: Optional[str] = None) -> List[Metric]:
        """Get all recorded metrics."""
        with self._lock:
            if pattern:
                regex = re.compile(pattern)
                return [m for m in self._metric_values if regex.search(m.name)]
            return self._metric_values.copy()


class DatadogMetrics(BaseMetrics):
    """Datadog metrics collector."""

    def __init__(
        self,
        service_name: str = "fishstick",
        api_key: Optional[str] = None,
        app_key: Optional[str] = None,
    ):
        super().__init__(service_name)
        self.api_key = api_key or os.getenv("DATADOG_API_KEY")
        self.app_key = app_key or os.getenv("DATADOG_APP_KEY")
        self._metric_values: List[Metric] = []
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize Datadog client."""
        if not self._initialized:
            options = {
                "api_key": self.api_key,
                "app_key": self.app_key,
            }
            initialize(**options)
            self._initialized = True
            logging.info("Datadog metrics client initialized")

    def record_metric(
        self, name: str, value: MetricValue, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric in Datadog."""
        labels = labels or {}
        tags = [f"{k}:{v}" for k, v in labels.items()]
        tags.append(f"service:{self.service_name}")

        try:
            api.Metric.send(
                metric=name,
                points=[(time.time(), float(value))],
                tags=tags,
                type="gauge",
            )
        except Exception as e:
            logging.warning(f"Failed to send metric to Datadog: {e}")

        with self._lock:
            metric = Metric(
                name=name, value=value, timestamp=datetime.utcnow(), labels=labels
            )
            self._metric_values.append(metric)

    def get_metrics(self, pattern: Optional[str] = None) -> List[Metric]:
        """Get all recorded metrics."""
        with self._lock:
            if pattern:
                regex = re.compile(pattern)
                return [m for m in self._metric_values if regex.search(m.name)]
            return self._metric_values.copy()


class CloudWatchMetrics(BaseMetrics):
    """AWS CloudWatch metrics collector."""

    def __init__(
        self,
        service_name: str = "fishstick",
        namespace: str = "Fishstick",
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        super().__init__(service_name)
        self.namespace = namespace
        self.region = region
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self._client: Optional[Any] = None
        self._metric_values: List[Metric] = []
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize CloudWatch client."""
        if not self._initialized:
            self._client = boto3.client(
                "cloudwatch",
                region_name=self.region,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
            )
            self._initialized = True
            logging.info("CloudWatch metrics client initialized")

    def record_metric(
        self, name: str, value: MetricValue, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric in CloudWatch."""
        labels = labels or {}
        dimensions = [{"Name": k, "Value": str(v)} for k, v in labels.items()]
        dimensions.append({"Name": "Service", "Value": self.service_name})

        try:
            if self._client:
                self._client.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=[
                        {
                            "MetricName": name,
                            "Value": float(value),
                            "Unit": "None",
                            "Dimensions": dimensions,
                            "Timestamp": datetime.utcnow(),
                        }
                    ],
                )
        except ClientError as e:
            logging.warning(f"Failed to send metric to CloudWatch: {e}")

        with self._lock:
            metric = Metric(
                name=name, value=value, timestamp=datetime.utcnow(), labels=labels
            )
            self._metric_values.append(metric)

    def get_metrics(self, pattern: Optional[str] = None) -> List[Metric]:
        """Get all recorded metrics."""
        with self._lock:
            if pattern:
                regex = re.compile(pattern)
                return [m for m in self._metric_values if regex.search(m.name)]
            return self._metric_values.copy()


class CustomMetrics(BaseMetrics):
    """Custom in-memory metrics collector."""

    def __init__(self, service_name: str = "fishstick"):
        super().__init__(service_name)
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize custom metrics."""
        self._initialized = True

    def record_metric(
        self, name: str, value: MetricValue, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric."""
        labels = labels or {}
        labels["service"] = self.service_name

        with self._lock:
            metric = Metric(
                name=name, value=value, timestamp=datetime.utcnow(), labels=labels
            )
            self._metrics[name].append(metric)

    def get_metrics(self, pattern: Optional[str] = None) -> List[Metric]:
        """Get recorded metrics."""
        with self._lock:
            all_metrics = []
            for metrics in self._metrics.values():
                all_metrics.extend(metrics)

            if pattern:
                regex = re.compile(pattern)
                return [m for m in all_metrics if regex.search(m.name)]
            return all_metrics


class MetricsManager:
    """Manager for multiple metrics collectors."""

    def __init__(self):
        self._collectors: Dict[str, BaseMetrics] = {}
        self._lock = threading.Lock()

    def register_collector(self, name: str, collector: BaseMetrics) -> None:
        """Register a metrics collector."""
        with self._lock:
            self._collectors[name] = collector
            collector.initialize()

    def record_metric(
        self, name: str, value: MetricValue, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record metric to all collectors."""
        with self._lock:
            for collector in self._collectors.values():
                try:
                    collector.record_metric(name, value, labels)
                except Exception as e:
                    logging.warning(f"Failed to record metric: {e}")

    def get_metrics(
        self, collector: Optional[str] = None, pattern: Optional[str] = None
    ) -> List[Metric]:
        """Get metrics from collectors."""
        with self._lock:
            if collector and collector in self._collectors:
                return self._collectors[collector].get_metrics(pattern)

            all_metrics = []
            for col in self._collectors.values():
                all_metrics.extend(col.get_metrics(pattern))
            return all_metrics


# Global metrics manager
_metrics_manager = MetricsManager()


def record_metric(
    name: str, value: MetricValue, labels: Optional[Dict[str, str]] = None
) -> None:
    """Record a metric to all registered collectors."""
    _metrics_manager.record_metric(name, value, labels)


def get_metrics(
    collector: Optional[str] = None, pattern: Optional[str] = None
) -> List[Metric]:
    """Get metrics from collectors."""
    return _metrics_manager.get_metrics(collector, pattern)


# ============================================================================
# Logging Classes
# ============================================================================


class BaseLogging(ABC):
    """Abstract base class for logging systems."""

    def __init__(self, service_name: str = "fishstick"):
        self.service_name = service_name
        self._initialized = False

    @abstractmethod
    def log_event(self, entry: LogEntry) -> None:
        """Log an event."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the logging system."""
        pass


class StructuredLogging(BaseLogging):
    """Structured JSON logging."""

    def __init__(
        self,
        service_name: str = "fishstick",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_console: bool = True,
    ):
        super().__init__(service_name)
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.enable_console = enable_console
        self._logger: Optional[logging.Logger] = None

    def initialize(self) -> None:
        """Initialize structured logger."""
        if not self._initialized:
            self._logger = logging.getLogger(self.service_name)
            self._logger.setLevel(self.log_level)

            # Clear existing handlers
            self._logger.handlers.clear()

            # JSON formatter
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "level": record.levelname,
                        "message": record.getMessage(),
                        "service": record.name,
                        "source": {
                            "file": record.filename,
                            "line": record.lineno,
                            "function": record.funcName,
                        },
                        "metadata": getattr(record, "metadata", {}),
                        "trace_id": getattr(record, "trace_id", None),
                        "span_id": getattr(record, "span_id", None),
                    }
                    return json.dumps(log_data, default=str)

            # Console handler
            if self.enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(JSONFormatter())
                self._logger.addHandler(console_handler)

            # File handler
            if self.log_file:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setFormatter(JSONFormatter())
                self._logger.addHandler(file_handler)

            self._initialized = True

    def log_event(self, entry: LogEntry) -> None:
        """Log a structured event."""
        if not self._initialized:
            self.initialize()

        if self._logger:
            extra = {
                "metadata": entry.metadata,
                "trace_id": entry.trace_id,
                "span_id": entry.span_id,
            }

            level = getattr(logging, entry.level.upper(), logging.INFO)
            self._logger.log(level, entry.message, extra=extra)


class ELKStack(BaseLogging):
    """ELK Stack logging integration."""

    def __init__(
        self,
        service_name: str = "fishstick",
        elasticsearch_url: str = "http://localhost:9200",
        index_prefix: str = "fishstick-logs",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        super().__init__(service_name)
        self.elasticsearch_url = elasticsearch_url
        self.index_prefix = index_prefix
        self.username = username or os.getenv("ELASTICSEARCH_USERNAME")
        self.password = password or os.getenv("ELASTICSEARCH_PASSWORD")
        self._buffer: List[Dict] = []
        self._buffer_lock = threading.Lock()
        self._flush_interval = 5  # seconds
        self._flush_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        """Initialize ELK logging."""
        if not self._initialized:
            self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._flush_thread.start()
            self._initialized = True
            logging.info("ELK Stack logging initialized")

    def _flush_loop(self) -> None:
        """Background thread to flush logs."""
        while True:
            time.sleep(self._flush_interval)
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered logs to Elasticsearch."""
        with self._buffer_lock:
            if not self._buffer:
                return

            logs_to_send = self._buffer.copy()
            self._buffer.clear()

        try:
            # Bulk index to Elasticsearch
            index_name = f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"
            url = f"{self.elasticsearch_url}/{index_name}/_bulk"

            bulk_data = []
            for log in logs_to_send:
                bulk_data.append(json.dumps({"index": {}}))
                bulk_data.append(json.dumps(log))

            if bulk_data:
                auth = (self.username, self.password) if self.username else None
                response = requests.post(
                    url,
                    data="\n".join(bulk_data) + "\n",
                    headers={"Content-Type": "application/json"},
                    auth=auth,
                    timeout=30,
                )
                response.raise_for_status()
        except Exception as e:
            logging.warning(f"Failed to send logs to ELK: {e}")

    def log_event(self, entry: LogEntry) -> None:
        """Log an event to ELK."""
        log_data = {
            "@timestamp": entry.timestamp.isoformat(),
            "level": entry.level,
            "message": entry.message,
            "service": entry.service,
            "trace_id": entry.trace_id,
            "span_id": entry.span_id,
            "metadata": entry.metadata,
        }

        with self._buffer_lock:
            self._buffer.append(log_data)


class SplunkIntegration(BaseLogging):
    """Splunk logging integration."""

    def __init__(
        self,
        service_name: str = "fishstick",
        splunk_host: str = "localhost",
        splunk_port: int = 8088,
        token: Optional[str] = None,
        use_https: bool = True,
        verify_ssl: bool = True,
    ):
        super().__init__(service_name)
        self.splunk_host = splunk_host
        self.splunk_port = splunk_port
        self.token = token or os.getenv("SPLUNK_HEC_TOKEN")
        self.use_https = use_https
        self.verify_ssl = verify_ssl
        self._url = f"{'https' if use_https else 'http'}://{splunk_host}:{splunk_port}/services/collector"

    def initialize(self) -> None:
        """Initialize Splunk logging."""
        if not self._initialized:
            self._initialized = True
            logging.info("Splunk logging initialized")

    def log_event(self, entry: LogEntry) -> None:
        """Log an event to Splunk."""
        event_data = {
            "time": entry.timestamp.timestamp(),
            "source": entry.service,
            "sourcetype": "_json",
            "event": {
                "level": entry.level,
                "message": entry.message,
                "trace_id": entry.trace_id,
                "span_id": entry.span_id,
                "metadata": entry.metadata,
            },
        }

        try:
            headers = {
                "Authorization": f"Splunk {self.token}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                self._url,
                json=event_data,
                headers=headers,
                verify=self.verify_ssl,
                timeout=10,
            )
            response.raise_for_status()
        except Exception as e:
            logging.warning(f"Failed to send log to Splunk: {e}")


class LogAggregation(BaseLogging):
    """Log aggregation system."""

    def __init__(
        self,
        service_name: str = "fishstick",
        max_buffer_size: int = 1000,
        flush_interval: int = 10,
    ):
        super().__init__(service_name)
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval
        self._buffer: List[LogEntry] = []
        self._aggregated_stats: Dict[str, Dict] = defaultdict(
            lambda: {
                "count": 0,
                "levels": defaultdict(int),
                "first_seen": None,
                "last_seen": None,
            }
        )
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize log aggregation."""
        if not self._initialized:
            self._initialized = True
            threading.Thread(target=self._aggregation_loop, daemon=True).start()
            logging.info("Log aggregation initialized")

    def _aggregation_loop(self) -> None:
        """Background aggregation loop."""
        while True:
            time.sleep(self.flush_interval)
            self._process_aggregations()

    def _process_aggregations(self) -> None:
        """Process aggregated logs."""
        with self._lock:
            for entry in self._buffer:
                key = f"{entry.level}:{entry.message[:100]}"
                stats = self._aggregated_stats[key]
                stats["count"] += 1
                stats["levels"][entry.level] += 1
                stats["last_seen"] = entry.timestamp
                if stats["first_seen"] is None:
                    stats["first_seen"] = entry.timestamp

            self._buffer.clear()

    def log_event(self, entry: LogEntry) -> None:
        """Log an event for aggregation."""
        with self._lock:
            self._buffer.append(entry)

            if len(self._buffer) >= self.max_buffer_size:
                self._process_aggregations()

    def get_aggregations(self) -> Dict[str, Dict]:
        """Get aggregated log statistics."""
        with self._lock:
            return dict(self._aggregated_stats)


class LoggingManager:
    """Manager for multiple logging systems."""

    def __init__(self):
        self._loggers: Dict[str, BaseLogging] = {}
        self._lock = threading.Lock()

    def register_logger(self, name: str, logger: BaseLogging) -> None:
        """Register a logging system."""
        with self._lock:
            self._loggers[name] = logger
            logger.initialize()

    def log_event(self, entry: LogEntry) -> None:
        """Log an event to all systems."""
        with self._lock:
            for logger in self._loggers.values():
                try:
                    logger.log_event(entry)
                except Exception as e:
                    logging.warning(f"Failed to log event: {e}")


# Global logging manager
_logging_manager = LoggingManager()


def log_event(
    level: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> None:
    """Log an event to all registered logging systems."""
    entry = LogEntry(
        level=level,
        message=message,
        timestamp=datetime.utcnow(),
        service="fishstick",
        trace_id=trace_id,
        span_id=span_id,
        metadata=metadata or {},
    )
    _logging_manager.log_event(entry)


# ============================================================================
# Tracing Classes
# ============================================================================


class BaseTracing(ABC):
    """Abstract base class for tracing systems."""

    def __init__(self, service_name: str = "fishstick"):
        self.service_name = service_name
        self._initialized = False
        self._active_spans: Dict[str, Any] = {}

    @abstractmethod
    def start_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new span. Returns span ID."""
        pass

    @abstractmethod
    def end_span(
        self, span_id: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a span."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the tracing system."""
        pass


class OpenTelemetry(BaseTracing):
    """OpenTelemetry tracing implementation."""

    def __init__(
        self, service_name: str = "fishstick", exporter_endpoint: Optional[str] = None
    ):
        super().__init__(service_name)
        self.exporter_endpoint = exporter_endpoint or os.getenv(
            "OTEL_EXPORTER_ENDPOINT"
        )
        self._tracer: Optional[trace.Tracer] = None
        self._provider: Optional[TracerProvider] = None

    def initialize(self) -> None:
        """Initialize OpenTelemetry."""
        if not self._initialized:
            resource = Resource.create({"service.name": self.service_name})
            self._provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self._provider)
            self._tracer = trace.get_tracer(self.service_name)
            self._initialized = True
            logging.info("OpenTelemetry initialized")

    def start_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new OpenTelemetry span."""
        if not self._tracer:
            return str(uuid.uuid4())

        span_id = str(uuid.uuid4())

        # Create span context
        ctx = trace.get_current_span().get_span_context() if parent_id else None

        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            self._active_spans[span_id] = span
            return span_id

    def end_span(
        self, span_id: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """End an OpenTelemetry span."""
        if span_id in self._active_spans:
            span = self._active_spans.pop(span_id)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            span.end()


class JaegerTracing(BaseTracing):
    """Jaeger tracing implementation."""

    def __init__(
        self,
        service_name: str = "fishstick",
        agent_host: str = "localhost",
        agent_port: int = 6831,
    ):
        super().__init__(service_name)
        self.agent_host = agent_host
        self.agent_port = agent_port
        self._processor: Optional[Any] = None

    def initialize(self) -> None:
        """Initialize Jaeger tracing."""
        if not self._initialized:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.agent_host,
                agent_port=self.agent_port,
            )

            provider = TracerProvider()
            processor = BatchSpanProcessor(jaeger_exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

            self._processor = processor
            self._initialized = True
            logging.info(
                f"Jaeger tracing initialized on {self.agent_host}:{self.agent_port}"
            )

    def start_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new Jaeger span."""
        span_id = str(uuid.uuid4())

        tracer = trace.get_tracer(self.service_name)
        span = tracer.start_span(name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        self._active_spans[span_id] = span
        return span_id

    def end_span(
        self, span_id: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a Jaeger span."""
        if span_id in self._active_spans:
            span = self._active_spans.pop(span_id)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            span.end()


class ZipkinTracing(BaseTracing):
    """Zipkin tracing implementation."""

    def __init__(
        self,
        service_name: str = "fishstick",
        endpoint: str = "http://localhost:9411/api/v2/spans",
    ):
        super().__init__(service_name)
        self.endpoint = endpoint

    def initialize(self) -> None:
        """Initialize Zipkin tracing."""
        if not self._initialized:
            zipkin_exporter = ZipkinExporter(endpoint=self.endpoint)

            provider = TracerProvider()
            processor = BatchSpanProcessor(zipkin_exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

            self._initialized = True
            logging.info(f"Zipkin tracing initialized at {self.endpoint}")

    def start_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new Zipkin span."""
        span_id = str(uuid.uuid4())

        tracer = trace.get_tracer(self.service_name)
        span = tracer.start_span(name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        self._active_spans[span_id] = span
        return span_id

    def end_span(
        self, span_id: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a Zipkin span."""
        if span_id in self._active_spans:
            span = self._active_spans.pop(span_id)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            span.end()


class DistributedTracing(BaseTracing):
    """Distributed tracing implementation."""

    def __init__(self, service_name: str = "fishstick"):
        super().__init__(service_name)
        self._trace_context: Dict[str, Any] = {}
        self._spans: Dict[str, Dict] = {}

    def initialize(self) -> None:
        """Initialize distributed tracing."""
        if not self._initialized:
            self._initialized = True
            logging.info("Distributed tracing initialized")

    def start_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new distributed span."""
        span_id = str(uuid.uuid4())
        trace_id = (
            str(uuid.uuid4())
            if not parent_id
            else self._spans.get(parent_id, {}).get("trace_id")
        )

        span_data = {
            "id": span_id,
            "trace_id": trace_id,
            "parent_id": parent_id,
            "name": name,
            "start_time": datetime.utcnow(),
            "attributes": attributes or {},
            "service": self.service_name,
        }

        self._spans[span_id] = span_data
        self._active_spans[span_id] = span_data

        # Propagate trace context
        if trace_id:
            self._trace_context[trace_id] = {
                "trace_id": trace_id,
                "span_id": span_id,
                "service": self.service_name,
            }

        return span_id

    def end_span(
        self, span_id: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a distributed span."""
        if span_id in self._spans:
            span = self._spans[span_id]
            span["end_time"] = datetime.utcnow()
            span["duration_ms"] = (
                span["end_time"] - span["start_time"]
            ).total_seconds() * 1000

            if attributes:
                span["attributes"].update(attributes)

            if span_id in self._active_spans:
                del self._active_spans[span_id]

            # Log span completion
            log_event(
                "debug",
                f"Span completed: {span['name']}",
                {
                    "trace_id": span["trace_id"],
                    "span_id": span_id,
                    "duration_ms": span["duration_ms"],
                },
                trace_id=span["trace_id"],
                span_id=span_id,
            )


class TracingManager:
    """Manager for multiple tracing systems."""

    def __init__(self):
        self._tracers: Dict[str, BaseTracing] = {}
        self._lock = threading.Lock()

    def register_tracer(self, name: str, tracer: BaseTracing) -> None:
        """Register a tracing system."""
        with self._lock:
            self._tracers[name] = tracer
            tracer.initialize()

    def start_span(
        self,
        name: str,
        tracer: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a span in all or specific tracer."""
        with self._lock:
            if tracer and tracer in self._tracers:
                return self._tracers[tracer].start_span(name, parent_id, attributes)

            # Start in all tracers, return first span ID
            span_id = None
            for tr in self._tracers.values():
                sid = tr.start_span(name, parent_id, attributes)
                if span_id is None:
                    span_id = sid
            return span_id or str(uuid.uuid4())

    def end_span(
        self,
        span_id: str,
        tracer: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a span in all or specific tracer."""
        with self._lock:
            if tracer and tracer in self._tracers:
                self._tracers[tracer].end_span(span_id, attributes)
            else:
                for tr in self._tracers.values():
                    tr.end_span(span_id, attributes)


# Global tracing manager
_tracing_manager = TracingManager()


def start_span(
    name: str,
    tracer: Optional[str] = None,
    parent_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> str:
    """Start a span."""
    return _tracing_manager.start_span(name, tracer, parent_id, attributes)


def end_span(
    span_id: str,
    tracer: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """End a span."""
    _tracing_manager.end_span(span_id, tracer, attributes)


# ============================================================================
# Alerting Classes
# ============================================================================


class BaseAlerting(ABC):
    """Abstract base class for alerting systems."""

    def __init__(self):
        self._initialized = False

    @abstractmethod
    def trigger_alert(self, alert: Alert) -> bool:
        """Trigger an alert. Returns success status."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the alerting system."""
        pass


class PagerDutyAlert(BaseAlerting):
    """PagerDuty alerting integration."""

    def __init__(
        self, service_key: Optional[str] = None, api_token: Optional[str] = None
    ):
        super().__init__()
        self.service_key = service_key or os.getenv("PAGERDUTY_SERVICE_KEY")
        self.api_token = api_token or os.getenv("PAGERDUTY_API_TOKEN")
        self._url = "https://events.pagerduty.com/v2/enqueue"

    def initialize(self) -> None:
        """Initialize PagerDuty alerting."""
        if not self._initialized:
            self._initialized = True
            logging.info("PagerDuty alerting initialized")

    def _severity_to_pagerduty(self, severity: AlertSeverity) -> str:
        """Convert alert severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.HIGH: "error",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "info",
            AlertSeverity.INFO: "info",
        }
        return mapping.get(severity, "warning")

    def trigger_alert(self, alert: Alert) -> bool:
        """Trigger a PagerDuty alert."""
        payload = {
            "routing_key": self.service_key,
            "event_action": "trigger",
            "dedup_key": alert.id,
            "payload": {
                "summary": alert.title,
                "severity": self._severity_to_pagerduty(alert.severity),
                "source": alert.source,
                "custom_details": {"message": alert.message, **alert.metadata},
            },
        }

        try:
            response = requests.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Failed to send PagerDuty alert: {e}")
            return False


class OpsGenieAlert(BaseAlerting):
    """OpsGenie alerting integration."""

    def __init__(
        self, api_key: Optional[str] = None, api_url: str = "https://api.opsgenie.com"
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("OPSGENIE_API_KEY")
        self.api_url = api_url

    def initialize(self) -> None:
        """Initialize OpsGenie alerting."""
        if not self._initialized:
            self._initialized = True
            logging.info("OpsGenie alerting initialized")

    def _severity_to_opsgenie(self, severity: AlertSeverity) -> str:
        """Convert alert severity to OpsGenie priority."""
        mapping = {
            AlertSeverity.CRITICAL: "P1",
            AlertSeverity.HIGH: "P2",
            AlertSeverity.MEDIUM: "P3",
            AlertSeverity.LOW: "P4",
            AlertSeverity.INFO: "P5",
        }
        return mapping.get(severity, "P3")

    def trigger_alert(self, alert: Alert) -> bool:
        """Trigger an OpsGenie alert."""
        payload = {
            "message": alert.title,
            "description": alert.message,
            "priority": self._severity_to_opsgenie(alert.severity),
            "source": alert.source,
            "alias": alert.id,
            "details": alert.metadata,
        }

        try:
            response = requests.post(
                f"{self.api_url}/v2/alerts",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"GenieKey {self.api_key}",
                },
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Failed to send OpsGenie alert: {e}")
            return False


class SlackAlert(BaseAlerting):
    """Slack alerting integration."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None,
        username: str = "Fishstick Monitoring",
    ):
        super().__init__()
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.channel = channel or os.getenv("SLACK_CHANNEL")
        self.username = username

    def initialize(self) -> None:
        """Initialize Slack alerting."""
        if not self._initialized:
            self._initialized = True
            logging.info("Slack alerting initialized")

    def _severity_to_color(self, severity: AlertSeverity) -> str:
        """Convert alert severity to Slack color."""
        mapping = {
            AlertSeverity.CRITICAL: "#FF0000",
            AlertSeverity.HIGH: "#FF8800",
            AlertSeverity.MEDIUM: "#FFCC00",
            AlertSeverity.LOW: "#00CC00",
            AlertSeverity.INFO: "#0088FF",
        }
        return mapping.get(severity, "#808080")

    def trigger_alert(self, alert: Alert) -> bool:
        """Trigger a Slack alert."""
        payload = {
            "username": self.username,
            "channel": self.channel,
            "attachments": [
                {
                    "color": self._severity_to_color(alert.severity),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.name,
                            "short": True,
                        },
                        {"title": "Source", "value": alert.source, "short": True},
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True,
                        },
                        {"title": "Alert ID", "value": alert.id, "short": True},
                    ],
                    "footer": "Fishstick Monitoring",
                    "ts": alert.timestamp.timestamp(),
                }
            ],
        }

        # Add metadata fields
        if alert.metadata:
            for key, value in alert.metadata.items():
                payload["attachments"][0]["fields"].append(
                    {"title": key, "value": str(value), "short": True}
                )

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Failed to send Slack alert: {e}")
            return False


class EmailAlert(BaseAlerting):
    """Email alerting integration."""

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        use_tls: bool = True,
        from_address: str = "alerts@fishstick.local",
        to_addresses: Optional[List[str]] = None,
    ):
        super().__init__()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username or os.getenv("SMTP_USERNAME")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.use_tls = use_tls
        self.from_address = from_address
        self.to_addresses = to_addresses or []

    def initialize(self) -> None:
        """Initialize email alerting."""
        if not self._initialized:
            self._initialized = True
            logging.info("Email alerting initialized")

    def trigger_alert(self, alert: Alert) -> bool:
        """Trigger an email alert."""
        if not self.to_addresses:
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.name}] {alert.title}"
        msg["From"] = self.from_address
        msg["To"] = ", ".join(self.to_addresses)

        # Plain text version
        text_body = f"""
Alert: {alert.title}
Severity: {alert.severity.name}
Source: {alert.source}
Time: {alert.timestamp.isoformat()}
ID: {alert.id}

{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
        """

        # HTML version
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: {"#FF0000" if alert.severity == AlertSeverity.CRITICAL else "#FF8800"}">
    [{alert.severity.name}] {alert.title}
</h2>
<table>
    <tr><td><strong>Source:</strong></td><td>{alert.source}</td></tr>
    <tr><td><strong>Time:</strong></td><td>{alert.timestamp.isoformat()}</td></tr>
    <tr><td><strong>ID:</strong></td><td>{alert.id}</td></tr>
</table>
<p>{alert.message}</p>
<h3>Metadata:</h3>
<pre>{json.dumps(alert.metadata, indent=2)}</pre>
</body>
</html>
        """

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)

                server.sendmail(self.from_address, self.to_addresses, msg.as_string())
            return True
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
            return False


class AlertingManager:
    """Manager for multiple alerting systems."""

    def __init__(self):
        self._alerting_systems: Dict[str, BaseAlerting] = {}
        self._lock = threading.Lock()
        self._alert_history: List[Alert] = []
        self._max_history = 1000

    def register_alerting(self, name: str, system: BaseAlerting) -> None:
        """Register an alerting system."""
        with self._lock:
            self._alerting_systems[name] = system
            system.initialize()

    def trigger_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "fishstick",
        metadata: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Trigger an alert to specific or all systems."""
        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        with self._lock:
            self._alert_history.append(alert)
            if len(self._alert_history) > self._max_history:
                self._alert_history.pop(0)

            results = {}
            if system and system in self._alerting_systems:
                results[system] = self._alerting_systems[system].trigger_alert(alert)
            else:
                for name, sys in self._alerting_systems.items():
                    try:
                        results[name] = sys.trigger_alert(alert)
                    except Exception as e:
                        logging.error(f"Failed to trigger alert via {name}: {e}")
                        results[name] = False

            return results

    def get_alert_history(
        self, severity: Optional[AlertSeverity] = None, limit: int = 100
    ) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            alerts = self._alert_history
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts[-limit:]


# Global alerting manager
_alerting_manager = AlertingManager()


def trigger_alert(
    severity: AlertSeverity,
    title: str,
    message: str,
    source: str = "fishstick",
    metadata: Optional[Dict[str, Any]] = None,
    system: Optional[str] = None,
) -> Dict[str, bool]:
    """Trigger an alert."""
    return _alerting_manager.trigger_alert(
        severity, title, message, source, metadata, system
    )


# ============================================================================
# Health Checks
# ============================================================================


class HealthCheck:
    """Health check system."""

    def __init__(self, component: str, timeout: float = 5.0):
        self.component = component
        self.timeout = timeout
        self._check_function: Optional[Callable[[], HealthResult]] = None

    def set_check(self, func: Callable[[], HealthResult]) -> None:
        """Set the check function."""
        self._check_function = func

    def check(self) -> HealthResult:
        """Run the health check."""
        start_time = time.time()

        if not self._check_function:
            return HealthResult(
                status=HealthStatus.UNKNOWN,
                component=self.component,
                message="No check function defined",
                timestamp=datetime.utcnow(),
                latency_ms=0.0,
            )

        try:
            result = self._check_function()
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                component=self.component,
                message=str(e),
                timestamp=datetime.utcnow(),
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": traceback.format_exc()},
            )


class LivenessProbe(HealthCheck):
    """Liveness probe for Kubernetes/Docker."""

    def __init__(self, timeout: float = 5.0):
        super().__init__("liveness", timeout)

    def check(self) -> HealthResult:
        """Check if application is alive."""
        start_time = time.time()

        try:
            # Basic liveness check - is process running
            return HealthResult(
                status=HealthStatus.HEALTHY,
                component="liveness",
                message="Application is alive",
                timestamp=datetime.utcnow(),
                latency_ms=(time.time() - start_time) * 1000,
                details={
                    "pid": os.getpid(),
                    "uptime": time.time() - psutil.Process().create_time(),
                },
            )
        except Exception as e:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                component="liveness",
                message=str(e),
                timestamp=datetime.utcnow(),
                latency_ms=(time.time() - start_time) * 1000,
            )


class ReadinessProbe(HealthCheck):
    """Readiness probe for Kubernetes/Docker."""

    def __init__(self, timeout: float = 5.0):
        super().__init__("readiness", timeout)
        self._dependencies: Dict[str, Callable[[], bool]] = {}

    def add_dependency(self, name: str, check: Callable[[], bool]) -> None:
        """Add a dependency check."""
        self._dependencies[name] = check

    def check(self) -> HealthResult:
        """Check if application is ready."""
        start_time = time.time()

        failed_deps = []
        for name, check in self._dependencies.items():
            try:
                if not check():
                    failed_deps.append(name)
            except Exception as e:
                failed_deps.append(f"{name}:{e}")

        if failed_deps:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                component="readiness",
                message=f"Dependencies not ready: {', '.join(failed_deps)}",
                timestamp=datetime.utcnow(),
                latency_ms=(time.time() - start_time) * 1000,
                details={"failed_dependencies": failed_deps},
            )

        return HealthResult(
            status=HealthStatus.HEALTHY,
            component="readiness",
            message="Application is ready",
            timestamp=datetime.utcnow(),
            latency_ms=(time.time() - start_time) * 1000,
            details={"dependencies": list(self._dependencies.keys())},
        )


class HealthCheckManager:
    """Manager for health checks."""

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._liveness = LivenessProbe()
        self._readiness = ReadinessProbe()
        self._lock = threading.Lock()

    def register_check(self, name: str, check: HealthCheck) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[name] = check

    def check_health(self, check_type: Optional[str] = None) -> Dict[str, HealthResult]:
        """Run all or specific health checks."""
        results = {}

        with self._lock:
            if check_type == "liveness":
                results["liveness"] = self._liveness.check()
            elif check_type == "readiness":
                results["readiness"] = self._readiness.check()
            elif check_type and check_type in self._checks:
                results[check_type] = self._checks[check_type].check()
            else:
                results["liveness"] = self._liveness.check()
                results["readiness"] = self._readiness.check()
                for name, check in self._checks.items():
                    results[name] = check.check()

        return results

    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        results = self.check_health()
        return all(
            r.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            for r in results.values()
        )


# Global health check manager
_health_manager = HealthCheckManager()


def check_health(check_type: Optional[str] = None) -> Dict[str, HealthResult]:
    """Run health checks."""
    return _health_manager.check_health(check_type)


# ============================================================================
# Performance Monitoring
# ============================================================================


class PerformanceMonitor:
    """General performance monitoring."""

    def __init__(self, service_name: str = "fishstick"):
        self.service_name = service_name
        self._measurements: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_measurement(
        self,
        name: str,
        value: float,
        unit: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance measurement."""
        with self._lock:
            self._measurements.append(
                {
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "timestamp": datetime.utcnow(),
                    "metadata": metadata or {},
                }
            )

            # Record to metrics
            record_metric(f"performance.{name}", value, metadata)

    def get_measurements(
        self, name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recorded measurements."""
        with self._lock:
            measurements = self._measurements
            if name:
                measurements = [m for m in measurements if m["name"] == name]
            return measurements[-limit:]


class LatencyTracker:
    """Track operation latencies."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_latency(self, operation: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        with self._lock:
            self._latencies[operation].append(latency_ms)

            # Keep window size
            if len(self._latencies[operation]) > self.window_size:
                self._latencies[operation] = self._latencies[operation][
                    -self.window_size :
                ]

            # Record metric
            record_metric(f"latency.{operation}", latency_ms, {"unit": "ms"})

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get latency statistics."""
        with self._lock:
            if operation:
                latencies = self._latencies.get(operation, [])
                return {operation: self._calculate_stats(latencies)}

            return {
                op: self._calculate_stats(lats) for op, lats in self._latencies.items()
            }

    def _calculate_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of latencies."""
        if not latencies:
            return {}

        sorted_lats = sorted(latencies)
        n = len(sorted_lats)

        return {
            "count": float(n),
            "mean": sum(sorted_lats) / n,
            "min": sorted_lats[0],
            "max": sorted_lats[-1],
            "p50": sorted_lats[int(n * 0.5)],
            "p95": sorted_lats[int(n * 0.95)],
            "p99": sorted_lats[int(n * 0.99)],
        }

    @contextlib.contextmanager
    def track(self, operation: str):
        """Context manager to track operation latency."""
        start_time = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.record_latency(operation, latency_ms)


class ThroughputTracker:
    """Track operation throughput."""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._operations: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_operation(self, operation: str) -> None:
        """Record an operation occurrence."""
        with self._lock:
            now = datetime.utcnow()
            self._operations[operation].append(now)

            # Clean old entries
            cutoff = now - timedelta(seconds=self.window_seconds)
            self._operations[operation] = [
                t for t in self._operations[operation] if t > cutoff
            ]

    def get_throughput(self, operation: Optional[str] = None) -> Dict[str, float]:
        """Get throughput (ops/sec) for operations."""
        with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=self.window_seconds)

            if operation:
                count = sum(
                    1 for t in self._operations.get(operation, []) if t > cutoff
                )
                return {operation: count / self.window_seconds}

            return {
                op: sum(1 for t in times if t > cutoff) / self.window_seconds
                for op, times in self._operations.items()
            }


class ResourceMonitor:
    """Monitor system resources."""

    def __init__(self, sampling_interval: float = 5.0):
        self.sampling_interval = sampling_interval
        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start resource monitoring."""
        if not self._running:
            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self._monitor_thread.start()
            logging.info("Resource monitoring started")

    def stop(self) -> None:
        """Stop resource monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            self._sample()
            time.sleep(self.sampling_interval)

    def _sample(self) -> None:
        """Sample system resources."""
        try:
            sample = {
                "timestamp": datetime.utcnow(),
                "cpu": {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                    "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                },
                "memory": psutil.virtual_memory()._asdict(),
                "disk": psutil.disk_usage("/")._asdict(),
                "network": {
                    "bytes_sent": psutil.net_io_counters().bytes_sent,
                    "bytes_recv": psutil.net_io_counters().bytes_recv,
                },
                "process": {
                    "memory_percent": psutil.Process().memory_percent(),
                    "cpu_percent": psutil.Process().cpu_percent(),
                    "num_threads": psutil.Process().num_threads(),
                },
            }

            with self._lock:
                self._samples.append(sample)
                # Keep last hour of samples (approx)
                max_samples = int(3600 / self.sampling_interval)
                if len(self._samples) > max_samples:
                    self._samples = self._samples[-max_samples:]

            # Record metrics
            record_metric("resource.cpu_percent", sample["cpu"]["percent"])
            record_metric("resource.memory_percent", sample["memory"]["percent"])
            record_metric("resource.disk_percent", sample["disk"]["percent"])

        except Exception as e:
            logging.warning(f"Failed to sample resources: {e}")

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource stats."""
        with self._lock:
            if self._samples:
                return self._samples[-1]
            return {}

    def get_samples(
        self, duration_seconds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get resource samples."""
        with self._lock:
            if duration_seconds:
                cutoff = datetime.utcnow() - timedelta(seconds=duration_seconds)
                return [s for s in self._samples if s["timestamp"] > cutoff]
            return self._samples.copy()


# ============================================================================
# Dashboards
# ============================================================================


class BaseDashboard(ABC):
    """Abstract base class for dashboards."""

    def __init__(self, name: str):
        self.name = name
        self._panels: List[Dict[str, Any]] = []

    @abstractmethod
    def create_dashboard(self) -> str:
        """Create the dashboard. Returns dashboard ID/URL."""
        pass

    def add_panel(
        self,
        title: str,
        metric: str,
        panel_type: str = "graph",
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a panel to the dashboard."""
        self._panels.append(
            {
                "title": title,
                "metric": metric,
                "type": panel_type,
                "options": options or {},
            }
        )


class GrafanaDashboard(BaseDashboard):
    """Grafana dashboard integration."""

    def __init__(
        self,
        name: str,
        grafana_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        folder: str = "Fishstick",
    ):
        super().__init__(name)
        self.grafana_url = grafana_url
        self.api_key = api_key or os.getenv("GRAFANA_API_KEY")
        self.folder = folder

    def create_dashboard(self) -> str:
        """Create a Grafana dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "uid": None,
                "title": self.name,
                "tags": ["fishstick", "generated"],
                "timezone": "utc",
                "schemaVersion": 36,
                "refresh": "30s",
                "panels": self._build_panels(),
            },
            "folder": self.folder,
            "overwrite": True,
        }

        try:
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=dashboard,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("url", f"{self.grafana_url}/d/{result.get('uid', '')}")
        except Exception as e:
            logging.error(f"Failed to create Grafana dashboard: {e}")
            return ""

    def _build_panels(self) -> List[Dict[str, Any]]:
        """Build Grafana panels from dashboard definition."""
        panels = []
        y_pos = 0

        for i, panel in enumerate(self._panels):
            grafana_panel = {
                "id": i + 1,
                "title": panel["title"],
                "type": "timeseries" if panel["type"] == "graph" else panel["type"],
                "targets": [
                    {
                        "expr": panel["metric"],
                        "legendFormat": panel["title"],
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": y_pos},
            }

            if i % 2 == 1:
                y_pos += 8

            panels.append(grafana_panel)

        return panels


class KibanaDashboard(BaseDashboard):
    """Kibana dashboard integration."""

    def __init__(
        self,
        name: str,
        kibana_url: str = "http://localhost:5601",
        api_key: Optional[str] = None,
        index_pattern: str = "fishstick-logs-*",
    ):
        super().__init__(name)
        self.kibana_url = kibana_url
        self.api_key = api_key or os.getenv("KIBANA_API_KEY")
        self.index_pattern = index_pattern

    def create_dashboard(self) -> str:
        """Create a Kibana dashboard."""
        dashboard = {
            "attributes": {
                "title": self.name,
                "hits": 0,
                "description": "Fishstick monitoring dashboard",
                "panelsJSON": json.dumps(self._build_panels()),
                "optionsJSON": "{}",
                "version": 1,
                "timeRestore": False,
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps(
                        {"filter": [], "query": {"query": "", "language": "kuery"}}
                    )
                },
            }
        }

        try:
            response = requests.post(
                f"{self.kibana_url}/api/saved_objects/dashboard",
                json=dashboard,
                headers={
                    "kbn-xsrf": "true",
                    "Authorization": f"ApiKey {self.api_key}" if self.api_key else "",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            return f"{self.kibana_url}/app/dashboards#/view/{result.get('id', '')}"
        except Exception as e:
            logging.error(f"Failed to create Kibana dashboard: {e}")
            return ""

    def _build_panels(self) -> List[Dict[str, Any]]:
        """Build Kibana panels."""
        panels = []

        for i, panel in enumerate(self._panels):
            kibana_panel = {
                "id": str(uuid.uuid4()),
                "type": "visualization",
                "panelIndex": str(i),
                "gridData": {
                    "x": (i % 2) * 24,
                    "y": (i // 2) * 15,
                    "w": 24,
                    "h": 15,
                    "i": str(i),
                },
                "version": "8.0.0",
                "attributes": {
                    "title": panel["title"],
                    "visState": json.dumps(
                        {
                            "type": "metrics",
                            "params": {
                                "index_pattern": self.index_pattern,
                                "metric": panel["metric"],
                            },
                        }
                    ),
                },
            }
            panels.append(kibana_panel)

        return panels


class CustomDashboard(BaseDashboard):
    """Custom dashboard implementation."""

    def __init__(self, name: str, output_dir: str = "./dashboards"):
        super().__init__(name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_dashboard(self) -> str:
        """Create a custom HTML dashboard."""
        html_content = self._build_html()

        output_file = self.output_dir / f"{self.name.lower().replace(' ', '_')}.html"
        output_file.write_text(html_content)

        return str(output_file.absolute())

    def _build_html(self) -> str:
        """Build HTML dashboard."""
        panels_html = ""
        for panel in self._panels:
            panels_html += f"""
            <div class="panel">
                <h3>{panel["title"]}</h3>
                <div class="metric-value" data-metric="{panel["metric"]}">
                    Loading...
                </div>
            </div>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .panels {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .panel {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .panel h3 {{
            margin-top: 0;
            color: #666;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }}
    </style>
</head>
<body>
    <h1>{self.name}</h1>
    <div class="panels">
        {panels_html}
    </div>
    <script>
        // Auto-refresh every 30 seconds
        setInterval(() => {{
            location.reload();
        }}, 30000);
    </script>
</body>
</html>
"""


def create_dashboard(name: str, dashboard_type: str = "custom", **kwargs) -> str:
    """Create a dashboard."""
    if dashboard_type == "grafana":
        dashboard = GrafanaDashboard(name, **kwargs)
    elif dashboard_type == "kibana":
        dashboard = KibanaDashboard(name, **kwargs)
    else:
        dashboard = CustomDashboard(name, **kwargs)

    return dashboard.create_dashboard()


# ============================================================================
# Utility Functions
# ============================================================================


def monitor_model(
    model_name: str, model_version: str, metrics_to_track: Optional[List[str]] = None
) -> Callable:
    """Decorator to monitor model inference."""
    metrics_to_track = metrics_to_track or ["latency", "throughput", "errors"]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_id = None
            if "tracing" in metrics_to_track:
                span_id = start_span(
                    f"model.{model_name}",
                    attributes={
                        "model.name": model_name,
                        "model.version": model_version,
                    },
                )

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                latency_ms = (time.time() - start_time) * 1000

                # Record metrics
                if "latency" in metrics_to_track:
                    record_metric(
                        f"model.{model_name}.latency_ms",
                        latency_ms,
                        {"version": model_version},
                    )

                if "throughput" in metrics_to_track:
                    record_metric(
                        f"model.{model_name}.inference_count",
                        1,
                        {"version": model_version},
                    )

                # Log success
                log_event(
                    "info",
                    f"Model {model_name} v{model_version} inference completed",
                    {
                        "model_name": model_name,
                        "model_version": model_version,
                        "latency_ms": latency_ms,
                    },
                    span_id=span_id,
                )

                return result

            except Exception as e:
                if "errors" in metrics_to_track:
                    record_metric(
                        f"model.{model_name}.errors",
                        1,
                        {"version": model_version, "error_type": type(e).__name__},
                    )

                log_event(
                    "error",
                    f"Model {model_name} v{model_version} inference failed",
                    {
                        "model_name": model_name,
                        "model_version": model_version,
                        "error": str(e),
                    },
                    span_id=span_id,
                )

                raise
            finally:
                if span_id:
                    end_span(span_id)

        return wrapper

    return decorator


def track_performance(
    operation_name: str, track_memory: bool = True, track_cpu: bool = True
) -> Callable:
    """Decorator to track function performance."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()

            # Baseline
            start_time = time.time()
            start_memory = process.memory_info().rss if track_memory else 0
            start_cpu = process.cpu_percent() if track_cpu else 0

            span_id = start_span(f"operation.{operation_name}")

            try:
                result = func(*args, **kwargs)

                # Calculate metrics
                duration_ms = (time.time() - start_time) * 1000
                memory_delta = (
                    (process.memory_info().rss - start_memory) / (1024 * 1024)
                    if track_memory
                    else 0
                )
                cpu_delta = process.cpu_percent() - start_cpu if track_cpu else 0

                record_metric(f"perf.{operation_name}.duration_ms", duration_ms)
                if track_memory:
                    record_metric(
                        f"perf.{operation_name}.memory_delta_mb", memory_delta
                    )
                if track_cpu:
                    record_metric(f"perf.{operation_name}.cpu_delta", cpu_delta)

                return result
            finally:
                end_span(span_id)

        return wrapper

    return decorator


def setup_monitoring(
    service_name: str = "fishstick",
    enable_prometheus: bool = True,
    enable_datadog: bool = False,
    enable_cloudwatch: bool = False,
    enable_structured_logging: bool = True,
    enable_elk: bool = False,
    enable_splunk: bool = False,
    enable_opentelemetry: bool = True,
    enable_jaeger: bool = False,
    enable_zipkin: bool = False,
    enable_slack_alerts: bool = False,
    enable_pagerduty: bool = False,
    start_resource_monitor: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Setup complete monitoring stack."""
    results = {
        "initialized": [],
        "errors": [],
        "config": {
            "service_name": service_name,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    # Setup metrics
    try:
        if enable_prometheus:
            prometheus = PrometheusMetrics(service_name, **kwargs.get("prometheus", {}))
            _metrics_manager.register_collector("prometheus", prometheus)
            results["initialized"].append("prometheus")
    except Exception as e:
        results["errors"].append(f"prometheus: {e}")

    try:
        if enable_datadog:
            datadog = DatadogMetrics(service_name, **kwargs.get("datadog", {}))
            _metrics_manager.register_collector("datadog", datadog)
            results["initialized"].append("datadog")
    except Exception as e:
        results["errors"].append(f"datadog: {e}")

    try:
        if enable_cloudwatch:
            cloudwatch = CloudWatchMetrics(service_name, **kwargs.get("cloudwatch", {}))
            _metrics_manager.register_collector("cloudwatch", cloudwatch)
            results["initialized"].append("cloudwatch")
    except Exception as e:
        results["errors"].append(f"cloudwatch: {e}")

    # Always add custom metrics
    custom = CustomMetrics(service_name)
    _metrics_manager.register_collector("custom", custom)
    results["initialized"].append("custom")

    # Setup logging
    try:
        if enable_structured_logging:
            structured = StructuredLogging(
                service_name, **kwargs.get("structured_logging", {})
            )
            _logging_manager.register_logger("structured", structured)
            results["initialized"].append("structured_logging")
    except Exception as e:
        results["errors"].append(f"structured_logging: {e}")

    try:
        if enable_elk:
            elk = ELKStack(service_name, **kwargs.get("elk", {}))
            _logging_manager.register_logger("elk", elk)
            results["initialized"].append("elk")
    except Exception as e:
        results["errors"].append(f"elk: {e}")

    try:
        if enable_splunk:
            splunk = SplunkIntegration(service_name, **kwargs.get("splunk", {}))
            _logging_manager.register_logger("splunk", splunk)
            results["initialized"].append("splunk")
    except Exception as e:
        results["errors"].append(f"splunk: {e}")

    # Add log aggregation
    aggregation = LogAggregation(service_name)
    _logging_manager.register_logger("aggregation", aggregation)
    results["initialized"].append("log_aggregation")

    # Setup tracing
    try:
        if enable_opentelemetry:
            otel = OpenTelemetry(service_name, **kwargs.get("opentelemetry", {}))
            _tracing_manager.register_tracer("opentelemetry", otel)
            results["initialized"].append("opentelemetry")
    except Exception as e:
        results["errors"].append(f"opentelemetry: {e}")

    try:
        if enable_jaeger:
            jaeger = JaegerTracing(service_name, **kwargs.get("jaeger", {}))
            _tracing_manager.register_tracer("jaeger", jaeger)
            results["initialized"].append("jaeger")
    except Exception as e:
        results["errors"].append(f"jaeger: {e}")

    try:
        if enable_zipkin:
            zipkin = ZipkinTracing(service_name, **kwargs.get("zipkin", {}))
            _tracing_manager.register_tracer("zipkin", zipkin)
            results["initialized"].append("zipkin")
    except Exception as e:
        results["errors"].append(f"zipkin: {e}")

    # Always add distributed tracing
    dist = DistributedTracing(service_name)
    _tracing_manager.register_tracer("distributed", dist)
    results["initialized"].append("distributed_tracing")

    # Setup alerting
    try:
        if enable_slack_alerts:
            slack = SlackAlert(**kwargs.get("slack", {}))
            _alerting_manager.register_alerting("slack", slack)
            results["initialized"].append("slack_alerts")
    except Exception as e:
        results["errors"].append(f"slack: {e}")

    try:
        if enable_pagerduty:
            pagerduty = PagerDutyAlert(**kwargs.get("pagerduty", {}))
            _alerting_manager.register_alerting("pagerduty", pagerduty)
            results["initialized"].append("pagerduty_alerts")
    except Exception as e:
        results["errors"].append(f"pagerduty: {e}")

    # Setup health checks
    liveness = LivenessProbe()
    _health_manager.register_check("liveness", liveness)

    readiness = ReadinessProbe()
    _health_manager.register_check("readiness", readiness)
    results["initialized"].append("health_checks")

    # Start resource monitor
    if start_resource_monitor:
        resource_monitor = ResourceMonitor()
        resource_monitor.start()
        results["initialized"].append("resource_monitor")

    # Log initialization
    log_event(
        "info",
        f"Monitoring setup completed for {service_name}",
        {
            "initialized": results["initialized"],
            "errors": results["errors"],
            "config": results["config"],
        },
    )

    return results


# ============================================================================
# Convenience Exports
# ============================================================================

__all__ = [
    # Metrics
    "PrometheusMetrics",
    "DatadogMetrics",
    "CloudWatchMetrics",
    "CustomMetrics",
    "MetricsManager",
    "record_metric",
    "get_metrics",
    # Logging
    "StructuredLogging",
    "ELKStack",
    "SplunkIntegration",
    "LogAggregation",
    "LoggingManager",
    "log_event",
    "LogEntry",
    # Tracing
    "OpenTelemetry",
    "JaegerTracing",
    "ZipkinTracing",
    "DistributedTracing",
    "TracingManager",
    "start_span",
    "end_span",
    # Alerting
    "PagerDutyAlert",
    "OpsGenieAlert",
    "SlackAlert",
    "EmailAlert",
    "AlertingManager",
    "trigger_alert",
    "Alert",
    "AlertSeverity",
    # Health Checks
    "HealthCheck",
    "LivenessProbe",
    "ReadinessProbe",
    "HealthCheckManager",
    "check_health",
    "HealthResult",
    "HealthStatus",
    # Performance
    "PerformanceMonitor",
    "LatencyTracker",
    "ThroughputTracker",
    "ResourceMonitor",
    # Dashboards
    "GrafanaDashboard",
    "KibanaDashboard",
    "CustomDashboard",
    "create_dashboard",
    # Utilities
    "monitor_model",
    "track_performance",
    "setup_monitoring",
    # Types
    "Metric",
    "MetricValue",
]
