"""
Fishstick Workflow Module
=========================

A comprehensive workflow orchestration system for machine learning pipelines.
Provides workflow definition, execution, scheduling, state management, error handling,
monitoring, and integration with popular workflow orchestrators.

Author: Fishstick Team
Version: 1.0.0
"""

from __future__ import annotations

import abc
import asyncio
import concurrent.futures
import contextlib
import copy
import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import logging
import os
import pickle
import queue
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import typing
import uuid
import warnings
from collections import defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
    runtime_checkable,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fishstick.workflow")


# =============================================================================
# Type Variables and Protocols
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")
TaskInput = TypeVar("TaskInput")
TaskOutput = TypeVar("TaskOutput")


@runtime_checkable
class Executable(Protocol):
    """Protocol for executable objects."""

    def execute(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class AsyncExecutable(Protocol):
    """Protocol for async executable objects."""

    async def execute(self, *args: Any, **kwargs: Any) -> Any: ...


# =============================================================================
# Exceptions and Error Types
# =============================================================================


class WorkflowError(Exception):
    """Base exception for workflow errors."""

    pass


class TaskError(WorkflowError):
    """Exception raised when a task fails."""

    def __init__(
        self,
        message: str,
        task_id: str,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.task_id = task_id
        self.cause = cause
        self.context = context or {}


class DependencyError(WorkflowError):
    """Exception raised when a dependency cannot be satisfied."""

    pass


class ExecutionError(WorkflowError):
    """Exception raised during workflow execution."""

    pass


class SchedulingError(WorkflowError):
    """Exception raised during workflow scheduling."""

    pass


class StateError(WorkflowError):
    """Exception raised for state-related errors."""

    pass


class CheckpointError(WorkflowError):
    """Exception raised for checkpoint-related errors."""

    pass


# =============================================================================
# Enums and Constants
# =============================================================================


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class WorkflowStatus(Enum):
    """Workflow execution status."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESUMING = "resuming"


class DependencyType(Enum):
    """Types of task dependencies."""

    SEQUENTIAL = "sequential"  # Task B runs after Task A
    DATA = "data"  # Task B needs data from Task A
    RESOURCE = "resource"  # Tasks share a resource
    CONDITIONAL = "conditional"  # Conditional dependency


class ExecutionMode(Enum):
    """Workflow execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


class RetryStrategy(Enum):
    """Retry strategies for failed tasks."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


# =============================================================================
# Workflow Definition
# =============================================================================


@dataclass
class TaskContext:
    """Context passed to task execution."""

    workflow_id: str
    task_id: str
    attempt: int = 1
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
        }


@dataclass
class TaskResult:
    """Result of task execution."""

    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    attempt: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == TaskStatus.COMPLETED and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": str(self.error) if self.error else None,
            "execution_time": self.execution_time,
            "attempt": self.attempt,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


class Task(abc.ABC, Generic[TaskInput, TaskOutput]):
    """Base class for workflow tasks."""

    def __init__(
        self,
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        timeout: Optional[float] = None,
        retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.description = description
        self.timeout = timeout
        self.retries = retries
        self.metadata = metadata or {}
        self._status = TaskStatus.PENDING
        self._result: Optional[TaskResult] = None
        self._context: Optional[TaskContext] = None

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def result(self) -> Optional[TaskResult]:
        return self._result

    @abc.abstractmethod
    def execute(self, input_data: TaskInput, context: TaskContext) -> TaskOutput:
        """Execute the task. Must be implemented by subclasses."""
        pass

    def run(self, input_data: TaskInput, workflow_id: str) -> TaskResult:
        """Run the task with full context and error handling."""
        self._context = TaskContext(
            workflow_id=workflow_id, task_id=self.task_id, metadata=self.metadata.copy()
        )

        start_time = time.time()
        self._context.start_time = datetime.now()
        self._status = TaskStatus.RUNNING

        try:
            if self.timeout:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.execute, input_data, self._context)
                    output = future.result(timeout=self.timeout)
            else:
                output = self.execute(input_data, self._context)

            execution_time = time.time() - start_time
            self._context.end_time = datetime.now()

            self._result = TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                timestamp=datetime.now(),
            )
            self._status = TaskStatus.COMPLETED

        except Exception as e:
            execution_time = time.time() - start_time
            self._context.end_time = datetime.now()

            self._result = TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=e,
                execution_time=execution_time,
                timestamp=datetime.now(),
            )
            self._status = TaskStatus.FAILED

            raise TaskError(
                f"Task {self.name} ({self.task_id}) failed: {e}",
                task_id=self.task_id,
                cause=e,
                context={"input": input_data},
            ) from e

        return self._result

    async def run_async(self, input_data: TaskInput, workflow_id: str) -> TaskResult:
        """Async version of run."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.run, input_data, workflow_id
        )

    def validate(self) -> bool:
        """Validate task configuration."""
        return True

    def reset(self) -> None:
        """Reset task to initial state."""
        self._status = TaskStatus.PENDING
        self._result = None
        self._context = None

    def clone(self) -> Task:
        """Create a deep copy of the task."""
        new_task = copy.deepcopy(self)
        new_task.task_id = str(uuid.uuid4())
        new_task.reset()
        return new_task

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "status": self._status.value,
            "timeout": self.timeout,
            "retries": self.retries,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.task_id}, name={self.name}, status={self._status.value})"


class FunctionTask(Task):
    """Task that wraps a function."""

    def __init__(
        self,
        func: Callable[..., Any],
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        timeout: Optional[float] = None,
        retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            task_id, name or func.__name__, description, timeout, retries, metadata
        )
        self._func = func
        self._signature = inspect.signature(func)

    def execute(self, input_data: Any, context: TaskContext) -> Any:
        """Execute the wrapped function."""
        if isinstance(input_data, dict):
            return self._func(**input_data)
        elif isinstance(input_data, (list, tuple)):
            return self._func(*input_data)
        else:
            return self._func(input_data)


@dataclass
class Dependency:
    """Represents a dependency between tasks."""

    source_task_id: str
    target_task_id: str
    dependency_type: DependencyType = DependencyType.SEQUENTIAL
    condition: Optional[Callable[[TaskResult], bool]] = None
    data_mapping: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.data_mapping is None:
            self.data_mapping = {}

    def evaluate_condition(self, result: TaskResult) -> bool:
        """Evaluate the dependency condition."""
        if self.condition is None:
            return True
        return self.condition(result)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_task_id": self.source_task_id,
            "target_task_id": self.target_task_id,
            "dependency_type": self.dependency_type.value,
            "has_condition": self.condition is not None,
            "data_mapping": self.data_mapping,
        }


class ConditionalTask(Task):
    """Task that conditionally executes based on a predicate."""

    def __init__(
        self,
        predicate: Callable[[Any], bool],
        true_task: Task,
        false_task: Optional[Task] = None,
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
    ):
        super().__init__(task_id, name, description)
        self.predicate = predicate
        self.true_task = true_task
        self.false_task = false_task
        self._selected_task: Optional[Task] = None

    def execute(self, input_data: Any, context: TaskContext) -> Any:
        """Execute the conditional task."""
        if self.predicate(input_data):
            self._selected_task = self.true_task
        elif self.false_task:
            self._selected_task = self.false_task
        else:
            return None

        return self._selected_task.execute(input_data, context)

    def get_selected_task(self) -> Optional[Task]:
        """Get the task that was selected during execution."""
        return self._selected_task


class ParallelTask(Task):
    """Task that executes multiple subtasks in parallel."""

    def __init__(
        self,
        tasks: List[Task],
        max_workers: Optional[int] = None,
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
    ):
        super().__init__(task_id, name, description)
        self.tasks = tasks
        self.max_workers = max_workers or len(tasks)
        self._results: Dict[str, TaskResult] = {}

    def execute(self, input_data: Any, context: TaskContext) -> Dict[str, Any]:
        """Execute all subtasks in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(task.run, input_data, context.workflow_id): task
                for task in self.tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.task_id] = result.output
                    self._results[task.task_id] = result
                except Exception as e:
                    results[task.task_id] = e
                    self._results[task.task_id] = TaskResult(
                        task_id=task.task_id, status=TaskStatus.FAILED, error=e
                    )

        return results

    def get_results(self) -> Dict[str, TaskResult]:
        """Get results from all subtasks."""
        return self._results.copy()


class WorkflowDAG:
    """Directed Acyclic Graph for workflow tasks."""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._dependencies: Dict[str, List[Dependency]] = defaultdict(list)
        self._reverse_deps: Dict[str, List[str]] = defaultdict(list)
        self._validated = False

    def add_task(self, task: Task) -> WorkflowDAG:
        """Add a task to the DAG."""
        if task.task_id in self._tasks:
            raise ValueError(f"Task with ID {task.task_id} already exists")

        self._tasks[task.task_id] = task
        self._validated = False
        return self

    def add_dependency(self, dependency: Dependency) -> WorkflowDAG:
        """Add a dependency to the DAG."""
        if dependency.source_task_id not in self._tasks:
            raise ValueError(f"Source task {dependency.source_task_id} not found")
        if dependency.target_task_id not in self._tasks:
            raise ValueError(f"Target task {dependency.target_task_id} not found")

        self._dependencies[dependency.target_task_id].append(dependency)
        self._reverse_deps[dependency.source_task_id].append(dependency.target_task_id)
        self._validated = False
        return self

    def connect(self, source: Task, target: Task, **kwargs) -> WorkflowDAG:
        """Connect two tasks with a dependency."""
        self.add_dependency(
            Dependency(
                source_task_id=source.task_id, target_task_id=target.task_id, **kwargs
            )
        )
        return self

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self._tasks.values())

    def get_dependencies(self, task_id: str) -> List[Dependency]:
        """Get dependencies for a task."""
        return self._dependencies.get(task_id, [])

    def get_dependents(self, task_id: str) -> List[str]:
        """Get tasks that depend on the given task."""
        return self._reverse_deps.get(task_id, [])

    def get_predecessors(self, task_id: str) -> List[str]:
        """Get predecessor task IDs."""
        return [dep.source_task_id for dep in self._dependencies.get(task_id, [])]

    def topological_sort(self) -> List[str]:
        """Return tasks in topological order."""
        in_degree = {tid: 0 for tid in self._tasks}
        for deps in self._dependencies.values():
            for dep in deps:
                in_degree[dep.target_task_id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for dependent in self._reverse_deps.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._tasks):
            raise WorkflowError("Cycle detected in workflow DAG")

        return result

    def validate(self) -> bool:
        """Validate the DAG (no cycles, all dependencies valid)."""
        try:
            self.topological_sort()
            self._validated = True
            return True
        except WorkflowError:
            return False

    def is_validated(self) -> bool:
        """Check if DAG has been validated."""
        return self._validated

    def has_task(self, task_id: str) -> bool:
        """Check if task exists in DAG."""
        return task_id in self._tasks

    def remove_task(self, task_id: str) -> None:
        """Remove a task and its dependencies."""
        if task_id not in self._tasks:
            return

        del self._tasks[task_id]

        # Remove dependencies pointing to this task
        if task_id in self._dependencies:
            del self._dependencies[task_id]

        # Remove from other tasks' dependencies
        for deps in self._dependencies.values():
            deps[:] = [d for d in deps if d.source_task_id != task_id]

        # Remove from reverse dependencies
        if task_id in self._reverse_deps:
            del self._reverse_deps[task_id]

        for deps in self._reverse_deps.values():
            if task_id in deps:
                deps.remove(task_id)

        self._validated = False

    def clone(self) -> WorkflowDAG:
        """Create a deep copy of the DAG."""
        new_dag = WorkflowDAG()

        # Clone tasks
        task_id_map = {}
        for task_id, task in self._tasks.items():
            cloned_task = task.clone()
            task_id_map[task_id] = cloned_task.task_id
            new_dag.add_task(cloned_task)

        # Clone dependencies with new task IDs
        for deps in self._dependencies.values():
            for dep in deps:
                new_dep = Dependency(
                    source_task_id=task_id_map[dep.source_task_id],
                    target_task_id=task_id_map[dep.target_task_id],
                    dependency_type=dep.dependency_type,
                    condition=dep.condition,
                    data_mapping=dep.data_mapping.copy() if dep.data_mapping else None,
                )
                new_dag.add_dependency(new_dep)

        return new_dag

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dictionary."""
        return {
            "tasks": [task.to_dict() for task in self._tasks.values()],
            "dependencies": [
                dep.to_dict() for deps in self._dependencies.values() for dep in deps
            ],
            "validated": self._validated,
        }

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks.values())


@dataclass
class Workflow:
    """A workflow composed of tasks and their dependencies."""

    dag: WorkflowDAG = field(default_factory=WorkflowDAG)
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed Workflow"
    description: str = ""
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL

    def __post_init__(self):
        if not self.dag:
            self.dag = WorkflowDAG()

    def add_task(self, task: Task) -> Workflow:
        """Add a task to the workflow."""
        self.dag.add_task(task)
        return self

    def add_dependency(self, dependency: Dependency) -> Workflow:
        """Add a dependency to the workflow."""
        self.dag.add_dependency(dependency)
        return self

    def connect(self, source: Task, target: Task, **kwargs) -> Workflow:
        """Connect two tasks."""
        self.dag.connect(source, target, **kwargs)
        return self

    def chain(self, *tasks: Task) -> Workflow:
        """Chain multiple tasks sequentially."""
        for i in range(len(tasks) - 1):
            self.connect(tasks[i], tasks[i + 1])
        return self

    def validate(self) -> bool:
        """Validate the workflow."""
        return self.dag.validate()

    def get_tasks(self) -> List[Task]:
        """Get all tasks in the workflow."""
        return self.dag.get_tasks()

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.dag.get_task(task_id)

    def clone(self) -> Workflow:
        """Create a deep copy of the workflow."""
        return Workflow(
            dag=self.dag.clone(),
            workflow_id=str(uuid.uuid4()),
            name=f"{self.name}_copy",
            description=self.description,
            version=self.version,
            metadata=self.metadata.copy(),
            execution_mode=self.execution_mode,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workflow to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "execution_mode": self.execution_mode.value,
            "dag": self.dag.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Workflow:
        """Deserialize workflow from dictionary."""
        # This is a simplified version - full deserialization would need task registry
        workflow = cls(
            workflow_id=data.get("workflow_id", str(uuid.uuid4())),
            name=data.get("name", "Unnamed Workflow"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
            execution_mode=ExecutionMode(data.get("execution_mode", "sequential")),
        )
        return workflow

    def __repr__(self) -> str:
        return (
            f"Workflow(id={self.workflow_id}, name={self.name}, tasks={len(self.dag)})"
        )


# =============================================================================
# Execution
# =============================================================================


@dataclass
class ExecutionResult:
    """Result of workflow execution."""

    workflow_id: str
    status: WorkflowStatus
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[Exception] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED and self.error is None

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def get_task_output(self, task_id: str) -> Any:
        """Get output from a specific task."""
        result = self.task_results.get(task_id)
        return result.output if result else None

    def get_failed_tasks(self) -> List[str]:
        """Get IDs of failed tasks."""
        return [
            tid
            for tid, result in self.task_results.items()
            if result.status == TaskStatus.FAILED
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "task_results": {
                tid: tr.to_dict() for tid, tr in self.task_results.items()
            },
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": str(self.error) if self.error else None,
            "metrics": self.metrics,
            "duration": self.duration,
        }


class WorkflowExecutor(abc.ABC):
    """Abstract base class for workflow executors."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        error_handler: Optional[ErrorHandler] = None,
        monitor: Optional[WorkflowMonitor] = None,
    ):
        self.max_workers = max_workers
        self.error_handler = error_handler
        self.monitor = monitor
        self._execution_history: List[ExecutionResult] = []

    @abc.abstractmethod
    def execute(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a workflow. Must be implemented by subclasses."""
        pass

    @abc.abstractmethod
    async def execute_async(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a workflow asynchronously."""
        pass

    def _prepare_inputs(
        self, workflow: Workflow, inputs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare inputs for workflow execution."""
        return inputs or {}

    def _handle_task_error(
        self, task: Task, error: Exception, context: Dict[str, Any]
    ) -> Optional[TaskResult]:
        """Handle task error."""
        if self.error_handler:
            return self.error_handler.handle(error, task, context)
        raise error

    def get_execution_history(self) -> List[ExecutionResult]:
        """Get execution history."""
        return self._execution_history.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()


class SequentialExecutor(WorkflowExecutor):
    """Execute workflow tasks sequentially."""

    def execute(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute workflow sequentially."""
        if not workflow.validate():
            raise WorkflowError("Workflow validation failed")

        execution_context = context or {}
        task_inputs = self._prepare_inputs(workflow, inputs)

        result = ExecutionResult(
            workflow_id=workflow.workflow_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now(),
        )

        if self.monitor:
            self.monitor.on_workflow_start(workflow)

        try:
            task_order = workflow.dag.topological_sort()

            for task_id in task_order:
                task = workflow.dag.get_task(task_id)
                if task is None:
                    continue

                if self.monitor:
                    self.monitor.on_task_start(task)

                # Prepare input for this task
                task_input = task_inputs.get(task_id)

                # Get outputs from dependencies
                deps = workflow.dag.get_dependencies(task_id)
                if deps:
                    dep_outputs = {}
                    for dep in deps:
                        dep_result = result.task_results.get(dep.source_task_id)
                        if dep_result and dep_result.is_success:
                            dep_outputs[dep.source_task_id] = dep_result.output

                    if dep_outputs and task_input is None:
                        task_input = dep_outputs

                try:
                    task_result = task.run(task_input, workflow.workflow_id)
                    result.task_results[task_id] = task_result

                    if self.monitor:
                        self.monitor.on_task_complete(task, task_result)

                except Exception as e:
                    handled_result = self._handle_task_error(task, e, execution_context)

                    if handled_result:
                        result.task_results[task_id] = handled_result
                    else:
                        result.status = WorkflowStatus.FAILED
                        result.error = e
                        result.end_time = datetime.now()

                        if self.monitor:
                            self.monitor.on_workflow_error(workflow, e)

                        return result

            result.status = WorkflowStatus.COMPLETED
            result.end_time = datetime.now()

            if self.monitor:
                self.monitor.on_workflow_complete(workflow, result)

        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = e
            result.end_time = datetime.now()

            if self.monitor:
                self.monitor.on_workflow_error(workflow, e)

        self._execution_history.append(result)
        return result

    async def execute_async(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute workflow asynchronously (still sequential)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.execute, workflow, inputs, context
        )


class ParallelExecutor(WorkflowExecutor):
    """Execute independent workflow tasks in parallel."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        error_handler: Optional[ErrorHandler] = None,
        monitor: Optional[WorkflowMonitor] = None,
        use_processes: bool = False,
    ):
        super().__init__(max_workers, error_handler, monitor)
        self.use_processes = use_processes

    def execute(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute workflow with parallel task execution."""
        if not workflow.validate():
            raise WorkflowError("Workflow validation failed")

        execution_context = context or {}
        task_inputs = self._prepare_inputs(workflow, inputs)

        result = ExecutionResult(
            workflow_id=workflow.workflow_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now(),
        )

        if self.monitor:
            self.monitor.on_workflow_start(workflow)

        try:
            task_order = workflow.dag.topological_sort()
            completed_tasks: Set[str] = set()

            ExecutorClass = (
                ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            )
            max_workers = self.max_workers or (os.cpu_count() or 4)

            with ExecutorClass(max_workers=max_workers) as executor:
                pending_tasks: Dict[str, Task] = {
                    tid: workflow.dag.get_task(tid) for tid in task_order
                }

                futures: Dict[Future, str] = {}

                while pending_tasks or futures:
                    # Submit ready tasks
                    ready_tasks = [
                        (tid, task)
                        for tid, task in pending_tasks.items()
                        if all(
                            dep.source_task_id in completed_tasks
                            for dep in workflow.dag.get_dependencies(tid)
                        )
                    ]

                    for tid, task in ready_tasks:
                        if task is None:
                            continue

                        task_input = task_inputs.get(tid)

                        # Collect dependency outputs
                        deps = workflow.dag.get_dependencies(tid)
                        if deps:
                            dep_outputs = {}
                            for dep in deps:
                                dep_result = result.task_results.get(dep.source_task_id)
                                if dep_result and dep_result.is_success:
                                    dep_outputs[dep.source_task_id] = dep_result.output

                            if dep_outputs and task_input is None:
                                task_input = dep_outputs

                        if self.monitor:
                            self.monitor.on_task_start(task)

                        future = executor.submit(
                            task.run, task_input, workflow.workflow_id
                        )
                        futures[future] = tid
                        del pending_tasks[tid]

                    # Wait for at least one task to complete
                    if futures:
                        done_futures, _ = concurrent.futures.wait(
                            futures.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )

                        for future in done_futures:
                            tid = futures.pop(future)
                            completed_tasks.add(tid)

                            try:
                                task_result = future.result()
                                result.task_results[tid] = task_result

                                task = workflow.dag.get_task(tid)
                                if task and self.monitor:
                                    self.monitor.on_task_complete(task, task_result)

                            except Exception as e:
                                task = workflow.dag.get_task(tid)
                                handled_result = (
                                    self._handle_task_error(task, e, execution_context)
                                    if task
                                    else None
                                )

                                if handled_result:
                                    result.task_results[tid] = handled_result
                                else:
                                    result.status = WorkflowStatus.FAILED
                                    result.error = e
                                    result.end_time = datetime.now()

                                    if self.monitor:
                                        self.monitor.on_workflow_error(workflow, e)

                                    return result

            result.status = WorkflowStatus.COMPLETED
            result.end_time = datetime.now()

            if self.monitor:
                self.monitor.on_workflow_complete(workflow, result)

        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = e
            result.end_time = datetime.now()

            if self.monitor:
                self.monitor.on_workflow_error(workflow, e)

        self._execution_history.append(result)
        return result

    async def execute_async(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute workflow asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.execute, workflow, inputs, context
        )


class DistributedExecutor(WorkflowExecutor):
    """Execute workflow tasks in a distributed manner.

    This is a placeholder for distributed execution. In a real implementation,
    this would integrate with distributed computing frameworks like Ray, Dask,
    or a custom distributed task queue.
    """

    def __init__(
        self,
        cluster_address: Optional[str] = None,
        max_workers: Optional[int] = None,
        error_handler: Optional[ErrorHandler] = None,
        monitor: Optional[WorkflowMonitor] = None,
    ):
        super().__init__(max_workers, error_handler, monitor)
        self.cluster_address = cluster_address
        self._task_queue: queue.Queue = queue.Queue()
        self._result_store: Dict[str, TaskResult] = {}

    def execute(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute workflow in a distributed manner.

        Note: This is a simplified implementation that uses ThreadPoolExecutor
        as a stand-in for actual distributed execution.
        """
        warnings.warn(
            "DistributedExecutor is using local thread pool. "
            "Connect to a cluster for true distributed execution.",
            RuntimeWarning,
        )

        # Fall back to parallel execution
        parallel_executor = ParallelExecutor(
            max_workers=self.max_workers,
            error_handler=self.error_handler,
            monitor=self.monitor,
        )

        return parallel_executor.execute(workflow, inputs, context)

    async def execute_async(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute workflow asynchronously in a distributed manner."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.execute, workflow, inputs, context
        )

    def connect_cluster(self, address: str) -> None:
        """Connect to a distributed cluster."""
        self.cluster_address = address
        logger.info(f"Connected to distributed cluster at {address}")

    def disconnect(self) -> None:
        """Disconnect from the cluster."""
        self.cluster_address = None
        logger.info("Disconnected from distributed cluster")


def execute_workflow(
    workflow: Workflow,
    inputs: Optional[Dict[str, Any]] = None,
    executor: Optional[WorkflowExecutor] = None,
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
) -> ExecutionResult:
    """Execute a workflow with the specified executor.

    Args:
        workflow: The workflow to execute
        inputs: Input data for the workflow
        executor: Custom executor (optional)
        mode: Execution mode if no executor provided

    Returns:
        ExecutionResult with workflow results
    """
    if executor is None:
        if mode == ExecutionMode.SEQUENTIAL:
            executor = SequentialExecutor()
        elif mode == ExecutionMode.PARALLEL:
            executor = ParallelExecutor()
        elif mode == ExecutionMode.DISTRIBUTED:
            executor = DistributedExecutor()
        else:
            raise ValueError(f"Unknown execution mode: {mode}")

    return executor.execute(workflow, inputs)


# =============================================================================
# Scheduling
# =============================================================================


@dataclass
class Schedule:
    """Schedule configuration for a workflow."""

    schedule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    name: str = ""
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowScheduler(abc.ABC):
    """Abstract base class for workflow schedulers."""

    def __init__(self, executor: Optional[WorkflowExecutor] = None):
        self.executor = executor or SequentialExecutor()
        self._schedules: Dict[str, Schedule] = {}
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

    @abc.abstractmethod
    def calculate_next_run(self, schedule: Schedule) -> Optional[datetime]:
        """Calculate the next run time for a schedule."""
        pass

    def add_schedule(
        self,
        workflow: Workflow,
        name: str = "",
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Schedule:
        """Add a workflow schedule."""
        schedule = Schedule(
            workflow_id=workflow.workflow_id,
            name=name or f"Schedule for {workflow.name}",
            enabled=enabled,
            metadata=metadata or {},
        )

        schedule.next_run = self.calculate_next_run(schedule)
        self._schedules[schedule.schedule_id] = schedule

        logger.info(f"Added schedule: {schedule.name} (ID: {schedule.schedule_id})")
        return schedule

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            return True
        return False

    def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get a schedule by ID."""
        return self._schedules.get(schedule_id)

    def list_schedules(self) -> List[Schedule]:
        """List all schedules."""
        return list(self._schedules.values())

    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule."""
        schedule = self._schedules.get(schedule_id)
        if schedule:
            schedule.enabled = True
            return True
        return False

    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule."""
        schedule = self._schedules.get(schedule_id)
        if schedule:
            schedule.enabled = False
            return True
        return False

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self._scheduler_thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.now()

            for schedule in self._schedules.values():
                if not schedule.enabled:
                    continue

                if schedule.next_run and now >= schedule.next_run:
                    self._execute_schedule(schedule)
                    schedule.last_run = now
                    schedule.run_count += 1
                    schedule.next_run = self.calculate_next_run(schedule)

            time.sleep(1.0)  # Check every second

    def _execute_schedule(self, schedule: Schedule) -> None:
        """Execute a scheduled workflow."""
        logger.info(f"Executing scheduled workflow: {schedule.name}")
        # In a real implementation, this would retrieve the workflow and execute it
        # For now, we just log

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


class CronScheduler(WorkflowScheduler):
    """Schedule workflows using cron-like expressions."""

    def __init__(
        self,
        executor: Optional[WorkflowExecutor] = None,
        timezone: Optional[str] = None,
    ):
        super().__init__(executor)
        self.timezone = timezone
        self._cron_expressions: Dict[str, str] = {}

    def add_schedule(
        self,
        workflow: Workflow,
        cron_expression: str,
        name: str = "",
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Schedule:
        """Add a schedule with a cron expression.

        Cron expression format:
        minute hour day_of_month month day_of_week

        Example: "0 9 * * 1" - Every Monday at 9:00 AM
        """
        schedule = super().add_schedule(workflow, name, enabled, metadata)
        self._cron_expressions[schedule.schedule_id] = cron_expression
        return schedule

    def calculate_next_run(self, schedule: Schedule) -> Optional[datetime]:
        """Calculate next run time from cron expression."""
        cron_expr = self._cron_expressions.get(schedule.schedule_id)
        if not cron_expr:
            return None

        try:
            # Simple cron parser (simplified implementation)
            # In production, use a library like 'croniter' or 'schedule'
            parts = cron_expr.split()
            if len(parts) != 5:
                raise ValueError("Invalid cron expression")

            minute, hour, day, month, weekday = parts

            now = datetime.now()
            next_run = now + timedelta(minutes=1)

            # Very simplified calculation - real implementation would be more robust
            if minute != "*":
                next_run = next_run.replace(minute=int(minute), second=0, microsecond=0)

            if hour != "*":
                next_run = next_run.replace(hour=int(hour))

            if next_run <= now:
                next_run += timedelta(days=1)

            return next_run

        except Exception as e:
            logger.error(f"Error parsing cron expression: {e}")
            return None

    def get_cron_expression(self, schedule_id: str) -> Optional[str]:
        """Get the cron expression for a schedule."""
        return self._cron_expressions.get(schedule_id)


class IntervalScheduler(WorkflowScheduler):
    """Schedule workflows at fixed intervals."""

    def __init__(self, executor: Optional[WorkflowExecutor] = None):
        super().__init__(executor)
        self._intervals: Dict[str, timedelta] = {}

    def add_schedule(
        self,
        workflow: Workflow,
        interval: timedelta,
        name: str = "",
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Schedule:
        """Add a schedule with a fixed interval."""
        schedule = super().add_schedule(workflow, name, enabled, metadata)
        self._intervals[schedule.schedule_id] = interval
        schedule.next_run = datetime.now() + interval
        return schedule

    def calculate_next_run(self, schedule: Schedule) -> Optional[datetime]:
        """Calculate next run time based on interval."""
        interval = self._intervals.get(schedule.schedule_id)
        if not interval:
            return None

        if schedule.last_run:
            return schedule.last_run + interval
        return datetime.now() + interval

    def get_interval(self, schedule_id: str) -> Optional[timedelta]:
        """Get the interval for a schedule."""
        return self._intervals.get(schedule_id)


class EventScheduler(WorkflowScheduler):
    """Schedule workflows based on events."""

    def __init__(self, executor: Optional[WorkflowExecutor] = None):
        super().__init__(executor)
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_queue: queue.Queue = queue.Queue()
        self._event_thread: Optional[threading.Thread] = None

    def add_schedule(
        self,
        workflow: Workflow,
        event_types: List[str],
        name: str = "",
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Schedule:
        """Add a schedule triggered by events."""
        schedule = super().add_schedule(workflow, name, enabled, metadata)

        for event_type in event_types:
            self._event_handlers[event_type].append(
                lambda event, sid=schedule.schedule_id: self._handle_event(sid, event)
            )

        schedule.metadata["event_types"] = event_types
        return schedule

    def calculate_next_run(self, schedule: Schedule) -> Optional[datetime]:
        """Event-based schedules don't have a predictable next run."""
        return None

    def emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Emit an event to trigger schedules."""
        self._event_queue.put(
            {"type": event_type, "data": event_data, "time": datetime.now()}
        )

    def _handle_event(self, schedule_id: str, event: Dict[str, Any]) -> None:
        """Handle an event for a schedule."""
        schedule = self._schedules.get(schedule_id)
        if schedule and schedule.enabled:
            self._execute_schedule(schedule)
            schedule.last_run = datetime.now()
            schedule.run_count += 1

    def start(self) -> None:
        """Start the event scheduler."""
        super().start()
        self._event_thread = threading.Thread(target=self._process_events, daemon=True)
        self._event_thread.start()

    def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                event_type = event["type"]

                for handler in self._event_handlers.get(event_type, []):
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Error handling event: {e}")

            except queue.Empty:
                continue


def schedule_workflow(
    workflow: Workflow, scheduler: WorkflowScheduler, **schedule_kwargs
) -> Schedule:
    """Schedule a workflow with the given scheduler.

    Args:
        workflow: The workflow to schedule
        scheduler: The scheduler to use
        **schedule_kwargs: Additional arguments for the scheduler

    Returns:
        Schedule object
    """
    return scheduler.add_schedule(workflow, **schedule_kwargs)


# =============================================================================
# State Management
# =============================================================================


class WorkflowState(Enum):
    """Possible workflow states."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLEANING_UP = "cleaning_up"


class StateMachine:
    """Finite state machine for workflow state management."""

    VALID_TRANSITIONS = {
        WorkflowState.IDLE: [WorkflowState.INITIALIZING],
        WorkflowState.INITIALIZING: [WorkflowState.RUNNING, WorkflowState.FAILED],
        WorkflowState.RUNNING: [
            WorkflowState.PAUSED,
            WorkflowState.COMPLETED,
            WorkflowState.FAILED,
            WorkflowState.CANCELLED,
        ],
        WorkflowState.PAUSED: [WorkflowState.RUNNING, WorkflowState.CANCELLED],
        WorkflowState.COMPLETED: [WorkflowState.IDLE],
        WorkflowState.FAILED: [WorkflowState.IDLE, WorkflowState.INITIALIZING],
        WorkflowState.CANCELLED: [WorkflowState.IDLE],
        WorkflowState.CLEANING_UP: [WorkflowState.IDLE, WorkflowState.FAILED],
    }

    def __init__(self, initial_state: WorkflowState = WorkflowState.IDLE):
        self._state = initial_state
        self._state_history: List[Tuple[datetime, WorkflowState]] = [
            (datetime.now(), initial_state)
        ]
        self._lock = threading.RLock()
        self._listeners: List[Callable[[WorkflowState, WorkflowState], None]] = []

    @property
    def current_state(self) -> WorkflowState:
        with self._lock:
            return self._state

    def can_transition_to(self, new_state: WorkflowState) -> bool:
        """Check if transition to new_state is valid."""
        with self._lock:
            return new_state in self.VALID_TRANSITIONS.get(self._state, [])

    def transition_to(self, new_state: WorkflowState) -> bool:
        """Transition to a new state."""
        with self._lock:
            if not self.can_transition_to(new_state):
                raise StateError(
                    f"Invalid transition from {self._state.value} to {new_state.value}"
                )

            old_state = self._state
            self._state = new_state
            self._state_history.append((datetime.now(), new_state))

            for listener in self._listeners:
                try:
                    listener(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state transition listener: {e}")

            return True

    def add_listener(
        self, listener: Callable[[WorkflowState, WorkflowState], None]
    ) -> None:
        """Add a state transition listener."""
        self._listeners.append(listener)

    def remove_listener(
        self, listener: Callable[[WorkflowState, WorkflowState], None]
    ) -> None:
        """Remove a state transition listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def get_history(self) -> List[Tuple[datetime, WorkflowState]]:
        """Get state transition history."""
        with self._lock:
            return self._state_history.copy()

    def reset(self) -> None:
        """Reset state machine to IDLE."""
        with self._lock:
            self._state = WorkflowState.IDLE
            self._state_history = [(datetime.now(), WorkflowState.IDLE)]


@dataclass
class StateSnapshot:
    """Snapshot of workflow state for persistence."""

    workflow_id: str
    state: WorkflowState
    task_states: Dict[str, TaskStatus]
    task_results: Dict[str, TaskResult]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "task_states": {k: v.value for k, v in self.task_states.items()},
            "task_results": {k: v.to_dict() for k, v in self.task_results.items()},
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateSnapshot:
        return cls(
            workflow_id=data["workflow_id"],
            state=WorkflowState(data["state"]),
            task_states={k: TaskStatus(v) for k, v in data["task_states"].items()},
            task_results={k: TaskResult(**v) for k, v in data["task_results"].items()},
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", "1.0"),
        )


class StatePersistence:
    """Handle persistence of workflow state."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = (
            Path(storage_path)
            if storage_path
            else Path(tempfile.gettempdir()) / "fishstick_workflows"
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: StateSnapshot) -> str:
        """Save a state snapshot.

        Returns:
            Path to the saved snapshot
        """
        filename = f"{snapshot.workflow_id}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.storage_path / filename

        with open(filepath, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved state snapshot to {filepath}")
        return str(filepath)

    def load(self, filepath: str) -> StateSnapshot:
        """Load a state snapshot."""
        with open(filepath, "r") as f:
            data = json.load(f)

        return StateSnapshot.from_dict(data)

    def list_snapshots(self, workflow_id: Optional[str] = None) -> List[str]:
        """List available snapshots."""
        snapshots = []
        for f in self.storage_path.glob("*.json"):
            if workflow_id is None or f.name.startswith(workflow_id):
                snapshots.append(str(f))
        return sorted(snapshots, reverse=True)

    def delete_snapshot(self, filepath: str) -> bool:
        """Delete a snapshot."""
        path = Path(filepath)
        if path.exists():
            path.unlink()
            return True
        return False

    def get_latest_snapshot(self, workflow_id: str) -> Optional[StateSnapshot]:
        """Get the latest snapshot for a workflow."""
        snapshots = self.list_snapshots(workflow_id)
        if snapshots:
            return self.load(snapshots[0])
        return None


def resume_workflow(
    workflow: Workflow,
    snapshot: StateSnapshot,
    executor: Optional[WorkflowExecutor] = None,
) -> ExecutionResult:
    """Resume a workflow from a state snapshot.

    Args:
        workflow: The workflow to resume
        snapshot: State snapshot to resume from
        executor: Executor to use

    Returns:
        ExecutionResult
    """
    if executor is None:
        executor = SequentialExecutor()

    # Mark incomplete tasks as pending
    for task_id, status in snapshot.task_states.items():
        if status in [TaskStatus.RUNNING, TaskStatus.PENDING]:
            snapshot.task_states[task_id] = TaskStatus.PENDING

    # Create execution context with resumed state
    context = {
        "resumed": True,
        "original_snapshot": snapshot,
        "completed_tasks": {
            tid
            for tid, status in snapshot.task_states.items()
            if status == TaskStatus.COMPLETED
        },
    }

    # Build input map from completed task results
    inputs = {}
    for task_id, result in snapshot.task_results.items():
        if result.is_success:
            inputs[task_id] = result.output

    logger.info(f"Resuming workflow {workflow.workflow_id} from snapshot")

    return executor.execute(workflow, inputs, context)


# =============================================================================
# Error Handling
# =============================================================================


@dataclass
class RetryPolicy:
    """Policy for retrying failed tasks."""

    max_retries: int = 3
    retry_delay: float = 1.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_delay: float = 60.0
    exceptions: Tuple[Type[Exception], ...] = (Exception,)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry."""
        if self.retry_strategy == RetryStrategy.FIXED:
            return self.retry_delay
        elif self.retry_strategy == RetryStrategy.LINEAR:
            return min(self.retry_delay * attempt, self.max_delay)
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL:
            return min(self.retry_delay * (2 ** (attempt - 1)), self.max_delay)
        return self.retry_delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if task should be retried."""
        if attempt >= self.max_retries:
            return False
        return isinstance(exception, self.exceptions)


class ErrorHandler(abc.ABC):
    """Abstract base class for error handlers."""

    @abc.abstractmethod
    def handle(
        self, error: Exception, task: Task, context: Dict[str, Any]
    ) -> Optional[TaskResult]:
        """Handle an error. Returns TaskResult if handled, None if should propagate."""
        pass


class RetryHandler(ErrorHandler):
    """Error handler that retries failed tasks."""

    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicy()

    def handle(
        self, error: Exception, task: Task, context: Dict[str, Any]
    ) -> Optional[TaskResult]:
        """Handle error by retrying."""
        attempt = context.get("attempt", 1)

        if self.policy.should_retry(error, attempt):
            delay = self.policy.get_delay(attempt)
            logger.info(
                f"Retrying task {task.task_id} after {delay}s (attempt {attempt + 1})"
            )
            time.sleep(delay)

            try:
                # Retry the task
                result = task.run(context.get("input"), context.get("workflow_id", ""))
                result.attempt = attempt + 1
                return result
            except Exception as retry_error:
                # Recursively retry
                context["attempt"] = attempt + 1
                return self.handle(retry_error, task, context)

        return None  # Propagate error


class FallbackHandler(ErrorHandler):
    """Error handler that falls back to a fallback task."""

    def __init__(self, fallback_task: Optional[Task] = None):
        self.fallback_task = fallback_task

    def handle(
        self, error: Exception, task: Task, context: Dict[str, Any]
    ) -> Optional[TaskResult]:
        """Handle error by executing fallback task."""
        if self.fallback_task:
            logger.info(f"Executing fallback task for {task.task_id}")
            try:
                result = self.fallback_task.run(
                    context.get("input"), context.get("workflow_id", "")
                )
                result.metrics["fallback"] = True
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback task also failed: {fallback_error}")

        return None


class CompositeHandler(ErrorHandler):
    """Error handler that combines multiple handlers."""

    def __init__(self, handlers: List[ErrorHandler]):
        self.handlers = handlers

    def handle(
        self, error: Exception, task: Task, context: Dict[str, Any]
    ) -> Optional[TaskResult]:
        """Try each handler in order."""
        for handler in self.handlers:
            result = handler.handle(error, task, context)
            if result is not None:
                return result
        return None


class FallbackTask(Task):
    """Task that executes as a fallback when another task fails."""

    def __init__(
        self,
        primary_task: Task,
        fallback_task: Task,
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
    ):
        super().__init__(task_id, name, description)
        self.primary_task = primary_task
        self.fallback_task = fallback_task
        self._used_fallback = False

    def execute(self, input_data: Any, context: TaskContext) -> Any:
        """Execute primary task, falling back on failure."""
        try:
            return self.primary_task.execute(input_data, context)
        except Exception as e:
            logger.warning(f"Primary task failed, using fallback: {e}")
            self._used_fallback = True
            return self.fallback_task.execute(input_data, context)

    def used_fallback(self) -> bool:
        """Check if fallback was used."""
        return self._used_fallback


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow recovery."""

    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    task_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def save(self, path: str) -> None:
        """Save checkpoint to disk."""
        filepath = Path(path) / f"checkpoint_{self.checkpoint_id}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> WorkflowCheckpoint:
        """Load checkpoint from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class CheckpointManager:
    """Manager for workflow checkpoints."""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir
            else Path(tempfile.gettempdir()) / "checkpoints"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: Dict[str, WorkflowCheckpoint] = {}

    def create_checkpoint(
        self, workflow_id: str, task_id: str, data: Dict[str, Any]
    ) -> WorkflowCheckpoint:
        """Create a new checkpoint."""
        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id, task_id=task_id, data=data
        )

        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        checkpoint.save(self.checkpoint_dir)

        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Get a checkpoint by ID."""
        if checkpoint_id in self._checkpoints:
            return self._checkpoints[checkpoint_id]

        # Try to load from disk
        filepath = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        if filepath.exists():
            return WorkflowCheckpoint.load(str(filepath))

        return None

    def list_checkpoints(
        self, workflow_id: Optional[str] = None
    ) -> List[WorkflowCheckpoint]:
        """List checkpoints."""
        checkpoints = []

        for checkpoint in self._checkpoints.values():
            if workflow_id is None or checkpoint.workflow_id == workflow_id:
                checkpoints.append(checkpoint)

        # Also check disk
        for f in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                checkpoint = WorkflowCheckpoint.load(str(f))
                if workflow_id is None or checkpoint.workflow_id == workflow_id:
                    if checkpoint.checkpoint_id not in self._checkpoints:
                        checkpoints.append(checkpoint)
            except Exception:
                continue

        return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]

        filepath = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        if filepath.exists():
            filepath.unlink()
            return True

        return False


# =============================================================================
# Monitoring
# =============================================================================


@dataclass
class TaskMetrics:
    """Metrics for task execution."""

    task_id: str
    execution_time: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "execution_time": self.execution_time,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "custom_metrics": self.custom_metrics,
        }


class WorkflowMonitor(abc.ABC):
    """Abstract base class for workflow monitors."""

    @abc.abstractmethod
    def on_workflow_start(self, workflow: Workflow) -> None:
        """Called when workflow starts."""
        pass

    @abc.abstractmethod
    def on_workflow_complete(self, workflow: Workflow, result: ExecutionResult) -> None:
        """Called when workflow completes."""
        pass

    @abc.abstractmethod
    def on_workflow_error(self, workflow: Workflow, error: Exception) -> None:
        """Called when workflow errors."""
        pass

    @abc.abstractmethod
    def on_task_start(self, task: Task) -> None:
        """Called when task starts."""
        pass

    @abc.abstractmethod
    def on_task_complete(self, task: Task, result: TaskResult) -> None:
        """Called when task completes."""
        pass


class LoggingMonitor(WorkflowMonitor):
    """Monitor that logs workflow events."""

    def __init__(self, logger_name: str = "fishstick.workflow.monitor"):
        self.logger = logging.getLogger(logger_name)

    def on_workflow_start(self, workflow: Workflow) -> None:
        self.logger.info(f"Workflow {workflow.name} ({workflow.workflow_id}) started")

    def on_workflow_complete(self, workflow: Workflow, result: ExecutionResult) -> None:
        self.logger.info(
            f"Workflow {workflow.name} completed in {result.duration:.2f}s "
            f"with status {result.status.value}"
        )

    def on_workflow_error(self, workflow: Workflow, error: Exception) -> None:
        self.logger.error(f"Workflow {workflow.name} failed: {error}")

    def on_task_start(self, task: Task) -> None:
        self.logger.debug(f"Task {task.name} ({task.task_id}) started")

    def on_task_complete(self, task: Task, result: TaskResult) -> None:
        self.logger.debug(
            f"Task {task.name} completed in {result.execution_time:.2f}s "
            f"with status {result.status.value}"
        )


class MetricsMonitor(WorkflowMonitor):
    """Monitor that collects metrics."""

    def __init__(self):
        self.workflow_metrics: Dict[str, Dict[str, Any]] = {}
        self.task_metrics: Dict[str, TaskMetrics] = {}

    def on_workflow_start(self, workflow: Workflow) -> None:
        self.workflow_metrics[workflow.workflow_id] = {
            "start_time": datetime.now(),
            "tasks": [],
        }

    def on_workflow_complete(self, workflow: Workflow, result: ExecutionResult) -> None:
        metrics = self.workflow_metrics.get(workflow.workflow_id, {})
        metrics["end_time"] = datetime.now()
        metrics["duration"] = result.duration
        metrics["status"] = result.status.value
        metrics["failed_tasks"] = result.get_failed_tasks()

    def on_workflow_error(self, workflow: Workflow, error: Exception) -> None:
        metrics = self.workflow_metrics.get(workflow.workflow_id, {})
        metrics["error"] = str(error)

    def on_task_start(self, task: Task) -> None:
        self.task_metrics[task.task_id] = TaskMetrics(task_id=task.task_id)

    def on_task_complete(self, task: Task, result: TaskResult) -> None:
        metrics = self.task_metrics.get(task.task_id)
        if metrics:
            metrics.execution_time = result.execution_time
            if task.task_id not in self.workflow_metrics.get(result.task_id, {}).get(
                "tasks", []
            ):
                if result.task_id in self.workflow_metrics:
                    self.workflow_metrics[result.task_id]["tasks"].append(task.task_id)

    def get_metrics(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get collected metrics."""
        if workflow_id:
            return self.workflow_metrics.get(workflow_id, {})
        return self.workflow_metrics.copy()


class WorkflowLogger:
    """Structured logging for workflows."""

    def __init__(self, name: str = "fishstick.workflow"):
        self.logger = logging.getLogger(name)

    def log_task_start(self, task: Task, workflow_id: str) -> None:
        """Log task start."""
        self.logger.info(f"[Workflow {workflow_id}] Task {task.name} started")

    def log_task_complete(
        self, task: Task, result: TaskResult, workflow_id: str
    ) -> None:
        """Log task completion."""
        self.logger.info(
            f"[Workflow {workflow_id}] Task {task.name} completed "
            f"({result.execution_time:.2f}s)"
        )

    def log_task_error(self, task: Task, error: Exception, workflow_id: str) -> None:
        """Log task error."""
        self.logger.error(
            f"[Workflow {workflow_id}] Task {task.name} failed: {error}", exc_info=True
        )

    def log_workflow_start(self, workflow: Workflow) -> None:
        """Log workflow start."""
        self.logger.info(f"Starting workflow: {workflow.name} ({workflow.workflow_id})")

    def log_workflow_complete(
        self, workflow: Workflow, result: ExecutionResult
    ) -> None:
        """Log workflow completion."""
        self.logger.info(
            f"Workflow {workflow.name} completed: {result.status.value} "
            f"in {result.duration:.2f}s"
        )


class WorkflowVisualization:
    """Visualization utilities for workflows."""

    @staticmethod
    def to_dot(workflow: Workflow) -> str:
        """Generate DOT format for graphviz visualization."""
        lines = ["digraph workflow {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box];")

        # Add nodes
        for task in workflow.dag.get_tasks():
            label = f"{task.name}\\n({task.task_id[:8]})"
            lines.append(f'  "{task.task_id}" [label="{label}"];')

        # Add edges
        for task_id in workflow.dag.topological_sort():
            deps = workflow.dag.get_dependencies(task_id)
            for dep in deps:
                lines.append(f'  "{dep.source_task_id}" -> "{task_id}";')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def to_mermaid(workflow: Workflow) -> str:
        """Generate Mermaid diagram syntax."""
        lines = ["graph TD"]

        for task in workflow.dag.get_tasks():
            lines.append(f'  {task.task_id[:8]}["{task.name}"]')

        for task_id in workflow.dag.topological_sort():
            deps = workflow.dag.get_dependencies(task_id)
            for dep in deps:
                lines.append(f"  {dep.source_task_id[:8]} --> {task_id[:8]}")

        return "\n".join(lines)

    @staticmethod
    def to_dict(workflow: Workflow) -> Dict[str, Any]:
        """Generate dictionary representation."""
        return {
            "nodes": [
                {
                    "id": task.task_id,
                    "name": task.name,
                    "type": task.__class__.__name__,
                    "status": task.status.value,
                }
                for task in workflow.dag.get_tasks()
            ],
            "edges": [
                {
                    "source": dep.source_task_id,
                    "target": dep.target_task_id,
                    "type": dep.dependency_type.value,
                }
                for deps in workflow.dag._dependencies.values()
                for dep in deps
            ],
        }


# =============================================================================
# Integration
# =============================================================================


class AirflowIntegration:
    """Integration with Apache Airflow."""

    def __init__(self, dag_folder: Optional[str] = None):
        self.dag_folder = dag_folder

    def to_airflow_dag(self, workflow: Workflow) -> str:
        """Convert workflow to Airflow DAG Python code."""
        lines = [
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "from datetime import datetime, timedelta",
            "",
            f"default_args = {{",
            f"    'owner': 'airflow',",
            f"    'depends_on_past': False,",
            f"    'email_on_failure': False,",
            f"    'email_on_retry': False,",
            f"    'retries': 1,",
            f"    'retry_delay': timedelta(minutes=5),",
            f"}}",
            "",
            f"dag = DAG(",
            f"    '{workflow.name.lower().replace(' ', '_')}',",
            f"    default_args=default_args,",
            f"    description='{workflow.description}',",
            f"    schedule_interval=timedelta(days=1),",
            f"    start_date=datetime(2024, 1, 1),",
            f"    catchup=False,",
            f"    tags=['fishstick'],",
            f")",
            "",
        ]

        # Add tasks
        task_var_names = {}
        for i, task in enumerate(workflow.dag.get_tasks()):
            var_name = f"task_{i}"
            task_var_names[task.task_id] = var_name

            lines.extend(
                [
                    f"def _execute_{i}(**context):",
                    f"    # Task: {task.name}",
                    f"    pass  # TODO: Implement task logic",
                    "",
                    f"{var_name} = PythonOperator(",
                    f"    task_id='{task.name.lower().replace(' ', '_')}',",
                    f"    python_callable=_execute_{i},",
                    f"    dag=dag,",
                    f")",
                    "",
                ]
            )

        # Add dependencies
        for task_id in workflow.dag.topological_sort():
            deps = workflow.dag.get_dependencies(task_id)
            if deps:
                dep_vars = [task_var_names[d.source_task_id] for d in deps]
                target_var = task_var_names[task_id]
                lines.append(f"{' >> '.join(dep_vars)} >> {target_var}")

        return "\n".join(lines)

    def export(self, workflow: Workflow, filepath: str) -> None:
        """Export workflow as Airflow DAG file."""
        dag_code = self.to_airflow_dag(workflow)
        with open(filepath, "w") as f:
            f.write(dag_code)


class PrefectIntegration:
    """Integration with Prefect."""

    def __init__(self):
        pass

    def to_prefect_flow(self, workflow: Workflow) -> str:
        """Convert workflow to Prefect flow code."""
        lines = [
            "from prefect import flow, task",
            "from typing import Any",
            "",
            f"# Workflow: {workflow.name}",
            f"# Description: {workflow.description}",
            "",
        ]

        # Generate task functions
        for i, task in enumerate(workflow.dag.get_tasks()):
            lines.extend(
                [
                    f"@task(name='{task.name}')",
                    f"def task_{i}(input_data: Any = None) -> Any:",
                    f"    '''{task.description}'''",
                    f"    # TODO: Implement task logic",
                    f"    return input_data",
                    "",
                ]
            )

        # Generate flow function
        lines.extend(
            [
                f"@flow(name='{workflow.name}')",
                "def main_flow():",
                "    # Execute tasks in dependency order",
            ]
        )

        task_order = workflow.dag.topological_sort()
        task_results = {}

        for i, task_id in enumerate(task_order):
            task = workflow.dag.get_task(task_id)
            deps = workflow.dag.get_dependencies(task_id)

            if deps:
                dep_results = [task_results[d.source_task_id] for d in deps]
                lines.append(
                    f"    result_{i} = task_{i}.submit({', '.join(dep_results)})"
                )
            else:
                lines.append(f"    result_{i} = task_{i}.submit()")

            task_results[task_id] = f"result_{i}"

        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    main_flow()")

        return "\n".join(lines)

    def export(self, workflow: Workflow, filepath: str) -> None:
        """Export workflow as Prefect flow file."""
        flow_code = self.to_prefect_flow(workflow)
        with open(filepath, "w") as f:
            f.write(flow_code)


class DagsterIntegration:
    """Integration with Dagster."""

    def __init__(self):
        pass

    def to_dagster_job(self, workflow: Workflow) -> str:
        """Convert workflow to Dagster job code."""
        lines = [
            "from dagster import job, op, In, Out, Nothing",
            "from typing import Any",
            "",
            f"# Workflow: {workflow.name}",
            "",
        ]

        # Generate op functions
        for i, task in enumerate(workflow.dag.get_tasks()):
            lines.extend(
                [
                    f"@op(name='{task.name}')",
                    f"def op_{i}(context, input_data: Any = None) -> Any:",
                    f"    '''{task.description}'''",
                    f"    context.log.info('Executing {task.name}')",
                    f"    # TODO: Implement task logic",
                    f"    return input_data",
                    "",
                ]
            )

        # Generate job
        lines.extend(
            [
                f"@job(name='{workflow.name.lower().replace(' ', '_')}')",
                "def workflow_job():",
            ]
        )

        # Build dependency graph
        task_order = workflow.dag.topological_sort()
        task_vars = {}

        for i, task_id in enumerate(task_order):
            task_vars[task_id] = f"step_{i}"

        for i, task_id in enumerate(task_order):
            deps = workflow.dag.get_dependencies(task_id)

            if deps:
                dep_vars = [task_vars[d.source_task_id] for d in deps]
                lines.append(f"    {task_vars[task_id]} = op_{i}({'('.join(dep_vars)})")
            else:
                lines.append(f"    {task_vars[task_id]} = op_{i}()")

        return "\n".join(lines)

    def export(self, workflow: Workflow, filepath: str) -> None:
        """Export workflow as Dagster job file."""
        job_code = self.to_dagster_job(workflow)
        with open(filepath, "w") as f:
            f.write(job_code)


class KubeflowIntegration:
    """Integration with Kubeflow Pipelines."""

    def __init__(self):
        pass

    def to_kfp_pipeline(self, workflow: Workflow) -> str:
        """Convert workflow to Kubeflow Pipeline code."""
        lines = [
            "import kfp",
            "from kfp import dsl",
            "from kfp.dsl import component, PipelineTask",
            "",
            f"# Workflow: {workflow.name}",
            "",
        ]

        # Generate components
        for i, task in enumerate(workflow.dag.get_tasks()):
            lines.extend(
                [
                    f"@component(base_image='python:3.9')",
                    f"def component_{i}(input_path: str, output_path: str):",
                    f"    '''{task.description}'''",
                    f"    import json",
                    f"    # TODO: Implement task logic",
                    f"    with open(input_path, 'r') as f:",
                    f"        data = json.load(f)",
                    f"    # Process data...",
                    f"    with open(output_path, 'w') as f:",
                    f"        json.dump(data, f)",
                    "",
                ]
            )

        # Generate pipeline
        lines.extend(
            [
                f"@dsl.pipeline(name='{workflow.name}', description='{workflow.description}')",
                "def workflow_pipeline():",
            ]
        )

        task_order = workflow.dag.topological_sort()
        task_vars = {}

        for i, task_id in enumerate(task_order):
            task_vars[task_id] = f"task_{i}"

        for i, task_id in enumerate(task_order):
            deps = workflow.dag.get_dependencies(task_id)

            if deps:
                dep_task = task_vars[deps[0].source_task_id]
                lines.append(
                    f"    {task_vars[task_id]} = component_{i}("
                    f"input_path={dep_task}.outputs['output_path'])"
                )
            else:
                lines.append(
                    f"    {task_vars[task_id]} = component_{i}(input_path='input.json')"
                )

        lines.extend(
            [
                "",
                "if __name__ == '__main__':",
                "    kfp.compiler.Compiler().compile(",
                "        pipeline_func=workflow_pipeline,",
                f"        package_path='{workflow.name.lower().replace(' ', '_')}.yaml'",
                "    )",
            ]
        )

        return "\n".join(lines)

    def export(self, workflow: Workflow, filepath: str) -> None:
        """Export workflow as Kubeflow pipeline file."""
        pipeline_code = self.to_kfp_pipeline(workflow)
        with open(filepath, "w") as f:
            f.write(pipeline_code)


# =============================================================================
# Utilities
# =============================================================================


class WorkflowBuilder:
    """Builder pattern for constructing workflows."""

    def __init__(self, name: str = "", description: str = ""):
        self.workflow = Workflow(name=name, description=description)
        self._last_task: Optional[Task] = None

    def add_task(self, task: Task) -> WorkflowBuilder:
        """Add a task to the workflow."""
        self.workflow.add_task(task)
        self._last_task = task
        return self

    def then(self, task: Task) -> WorkflowBuilder:
        """Add a task after the last task."""
        if self._last_task is None:
            raise ValueError("No previous task to connect to")

        self.workflow.add_task(task)
        self.workflow.connect(self._last_task, task)
        self._last_task = task
        return self

    def also(self, *tasks: Task) -> WorkflowBuilder:
        """Add tasks that run in parallel after the last task."""
        if self._last_task is None:
            raise ValueError("No previous task to connect to")

        for task in tasks:
            self.workflow.add_task(task)
            self.workflow.connect(self._last_task, task)

        return self

    def merge(self, *tasks: Task) -> WorkflowBuilder:
        """Add tasks that all previous parallel tasks must complete before."""
        if self._last_task is None:
            raise ValueError("No tasks to merge from")

        for task in tasks:
            self.workflow.add_task(task)
            # Connect all leaf tasks to the new task
            for t in self.workflow.dag.get_tasks():
                if not self.workflow.dag.get_dependents(t.task_id):
                    self.workflow.connect(t, task)

        self._last_task = task
        return self

    def when(self, condition: Callable[[Any], bool]) -> WorkflowBuilder:
        """Add a conditional branch."""
        # Store condition for later use
        self._pending_condition = condition
        return self

    def then_do(self, task: Task) -> WorkflowBuilder:
        """Add task for when condition is true."""
        if not hasattr(self, "_pending_condition"):
            raise ValueError("No condition set. Use .when() first")

        conditional_task = ConditionalTask(
            predicate=self._pending_condition,
            true_task=task,
            name=f"conditional_{task.name}",
        )

        if self._last_task:
            self.workflow.connect(self._last_task, conditional_task)

        self.workflow.add_task(conditional_task)
        self._last_task = conditional_task
        del self._pending_condition
        return self

    def with_input(self, task_id: str, data: Any) -> WorkflowBuilder:
        """Set input data for a specific task."""
        if "inputs" not in self.workflow.metadata:
            self.workflow.metadata["inputs"] = {}
        self.workflow.metadata["inputs"][task_id] = data
        return self

    def with_metadata(self, key: str, value: Any) -> WorkflowBuilder:
        """Add metadata to the workflow."""
        self.workflow.metadata[key] = value
        return self

    def build(self) -> Workflow:
        """Build and return the workflow."""
        if not self.workflow.validate():
            raise WorkflowError("Workflow validation failed")
        return self.workflow


def create_workflow(name: str = "", description: str = "", *tasks: Task) -> Workflow:
    """Create a simple workflow from tasks.

    Args:
        name: Workflow name
        description: Workflow description
        *tasks: Tasks to chain sequentially

    Returns:
        Workflow
    """
    workflow = Workflow(name=name, description=description)

    for task in tasks:
        workflow.add_task(task)

    # Chain tasks sequentially
    for i in range(len(tasks) - 1):
        workflow.connect(tasks[i], tasks[i + 1])

    return workflow


def run_workflow(
    workflow: Workflow,
    inputs: Optional[Dict[str, Any]] = None,
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    **executor_kwargs,
) -> ExecutionResult:
    """Run a workflow with simplified interface.

    Args:
        workflow: Workflow to run
        inputs: Input data
        mode: Execution mode
        **executor_kwargs: Additional executor configuration

    Returns:
        ExecutionResult
    """
    if mode == ExecutionMode.SEQUENTIAL:
        executor = SequentialExecutor(**executor_kwargs)
    elif mode == ExecutionMode.PARALLEL:
        executor = ParallelExecutor(**executor_kwargs)
    elif mode == ExecutionMode.DISTRIBUTED:
        executor = DistributedExecutor(**executor_kwargs)
    else:
        raise ValueError(f"Unknown execution mode: {mode}")

    return executor.execute(workflow, inputs)


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    # Workflow Definition
    "Workflow",
    "Task",
    "FunctionTask",
    "Dependency",
    "WorkflowDAG",
    "ConditionalTask",
    "ParallelTask",
    "TaskContext",
    "TaskResult",
    # Execution
    "WorkflowExecutor",
    "SequentialExecutor",
    "ParallelExecutor",
    "DistributedExecutor",
    "execute_workflow",
    "ExecutionResult",
    # Scheduling
    "WorkflowScheduler",
    "CronScheduler",
    "IntervalScheduler",
    "EventScheduler",
    "Schedule",
    "schedule_workflow",
    # State Management
    "WorkflowState",
    "StateMachine",
    "StatePersistence",
    "StateSnapshot",
    "resume_workflow",
    # Error Handling
    "RetryPolicy",
    "ErrorHandler",
    "RetryHandler",
    "FallbackHandler",
    "CompositeHandler",
    "FallbackTask",
    "WorkflowCheckpoint",
    "CheckpointManager",
    # Monitoring
    "WorkflowMonitor",
    "LoggingMonitor",
    "MetricsMonitor",
    "TaskMetrics",
    "WorkflowLogger",
    "WorkflowVisualization",
    # Integration
    "AirflowIntegration",
    "PrefectIntegration",
    "DagsterIntegration",
    "KubeflowIntegration",
    # Utilities
    "WorkflowBuilder",
    "create_workflow",
    "run_workflow",
    # Enums
    "TaskStatus",
    "WorkflowStatus",
    "DependencyType",
    "ExecutionMode",
    "RetryStrategy",
    # Exceptions
    "WorkflowError",
    "TaskError",
    "DependencyError",
    "ExecutionError",
    "SchedulingError",
    "StateError",
    "CheckpointError",
]
