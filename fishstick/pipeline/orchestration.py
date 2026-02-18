"""
Comprehensive Pipeline Orchestration Module for fishstick

This module provides a complete pipeline orchestration framework with support for:
- Pipeline definition with DAG-based execution
- Multiple execution strategies (sequential, parallel, distributed)
- Flexible scheduling (cron, interval, event-based)
- Intelligent caching and memoization
- Robust error handling with retries and fallbacks
- Comprehensive monitoring and visualization
- Integration with MLflow, Kubeflow, and Airflow

Example:
    >>> from fishstick.pipeline.orchestration import Pipeline, Step, PipelineBuilder
    >>>
    >>> # Build a simple pipeline
    >>> builder = PipelineBuilder()
    >>> pipeline = (builder
    ...     .add_step("load", load_data)
    ...     .add_step("preprocess", preprocess_data, depends_on=["load"])
    ...     .add_step("train", train_model, depends_on=["preprocess"])
    ...     .build())
    >>>
    >>> # Execute the pipeline
    >>> result = run_pipeline(pipeline)
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import pickle
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import networkx as nx
import numpy as np
from croniter import croniter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Section 1: Pipeline Definition
# =============================================================================


class StepStatus(Enum):
    """Status of a pipeline step."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CACHED = auto()


@dataclass
class StepResult:
    """Result of executing a pipeline step."""

    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False

    @property
    def is_success(self) -> bool:
        """Check if step execution was successful."""
        return self.status in (StepStatus.COMPLETED, StepStatus.CACHED)


@dataclass
class StepContext:
    """Context passed to each step during execution."""

    pipeline_id: str
    step_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: Optional[Path] = None

    def get_input(self, key: str, default: Any = None) -> Any:
        """Get input value by key."""
        return self.inputs.get(key, default)


StepFunction = Callable[[StepContext], Any]


class Step:
    """
    A single step in a pipeline.

    Args:
        id: Unique identifier for the step
        func: Function to execute
        depends_on: List of step IDs this step depends on
        cache_key: Optional cache key function
        retry_policy: Retry policy for this step
        fallback: Fallback step to execute on failure

    Example:
        >>> def preprocess(ctx: StepContext) -> np.ndarray:
        ...     data = ctx.get_input("data")
        ...     return normalize(data)
        >>>
        >>> step = Step(
        ...     id="preprocess",
        ...     func=preprocess,
        ...     depends_on=["load_data"]
        ... )
    """

    def __init__(
        self,
        id: str,
        func: StepFunction,
        depends_on: Optional[List[str]] = None,
        cache_key: Optional[Callable[[StepContext], str]] = None,
        retry_policy: Optional["RetryPolicy"] = None,
        fallback: Optional["FallbackStep"] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.func = func
        self.depends_on = depends_on or []
        self.cache_key = cache_key
        self.retry_policy = retry_policy
        self.fallback = fallback
        self.metadata = metadata or {}
        self._result: Optional[StepResult] = None

    def execute(self, context: StepContext) -> StepResult:
        """
        Execute the step with the given context.

        Args:
            context: Execution context

        Returns:
            StepResult with execution details
        """
        start_time = datetime.now()

        try:
            output = self.func(context)

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self._result = StepResult(
                step_id=self.id,
                status=StepStatus.COMPLETED,
                output=output,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self._result = StepResult(
                step_id=self.id,
                status=StepStatus.FAILED,
                error=e,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
            )

        return self._result

    def get_cache_key(self, context: StepContext) -> Optional[str]:
        """Generate cache key for this step execution."""
        if self.cache_key:
            return self.cache_key(context)
        return None

    @property
    def result(self) -> Optional[StepResult]:
        """Get the result of the last execution."""
        return self._result


class ConditionalStep(Step):
    """
    A step that executes conditionally based on a predicate.

    Args:
        id: Unique identifier for the step
        func: Function to execute
        condition: Predicate function that determines if step should run

    Example:
        >>> def should_train(ctx: StepContext) -> bool:
        ...     return ctx.get_input("epoch", 0) % 10 == 0
        >>>
        >>> step = ConditionalStep(
        ...     id="train",
        ...     func=train_model,
        ...     condition=should_train
        ... )
    """

    def __init__(
        self,
        id: str,
        func: StepFunction,
        condition: Callable[[StepContext], bool],
        **kwargs,
    ):
        super().__init__(id, func, **kwargs)
        self.condition = condition

    def should_execute(self, context: StepContext) -> bool:
        """Check if step should execute based on condition."""
        return self.condition(context)

    def execute(self, context: StepContext) -> StepResult:
        """Execute only if condition is met."""
        if not self.should_execute(context):
            return StepResult(
                step_id=self.id,
                status=StepStatus.SKIPPED,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        return super().execute(context)


class LoopStep(Step):
    """
    A step that executes in a loop until a condition is met.

    Args:
        id: Unique identifier for the step
        func: Function to execute each iteration
        stop_condition: Function that determines when to stop looping
        max_iterations: Maximum number of iterations

    Example:
        >>> def train_iteration(ctx: StepContext) -> dict:
        ...     model = ctx.get_input("model")
        ...     loss = train_step(model)
        ...     return {"loss": loss, "model": model}
        >>>
        >>> def should_stop(ctx: StepContext) -> bool:
        ...     return ctx.get_input("loss", float('inf')) < 0.01
        >>>
        >>> step = LoopStep(
        ...     id="training_loop",
        ...     func=train_iteration,
        ...     stop_condition=should_stop,
        ...     max_iterations=1000
        ... )
    """

    def __init__(
        self,
        id: str,
        func: StepFunction,
        stop_condition: Callable[[StepContext], bool],
        max_iterations: int = 1000,
        **kwargs,
    ):
        super().__init__(id, func, **kwargs)
        self.stop_condition = stop_condition
        self.max_iterations = max_iterations

    def execute(self, context: StepContext) -> StepResult:
        """Execute in a loop until condition is met."""
        start_time = datetime.now()
        iteration = 0
        last_output = None

        try:
            while iteration < self.max_iterations:
                result = super().execute(context)

                if not result.is_success:
                    return result

                last_output = result.output
                iteration += 1

                # Update context with iteration info
                if isinstance(last_output, dict):
                    context.inputs.update(last_output)

                if self.stop_condition(context):
                    break

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            return StepResult(
                step_id=self.id,
                status=StepStatus.COMPLETED,
                output={"iterations": iteration, "result": last_output},
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                metrics={"iterations": iteration},
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            return StepResult(
                step_id=self.id,
                status=StepStatus.FAILED,
                error=e,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
            )


class DAG:
    """
    Directed Acyclic Graph for pipeline step dependencies.

    Manages dependencies between steps and provides topological ordering
    for execution.

    Example:
        >>> dag = DAG()
        >>> dag.add_node("load")
        >>> dag.add_node("preprocess")
        >>> dag.add_edge("load", "preprocess")
        >>> order = dag.topological_sort()
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str) -> None:
        """Add a node to the graph."""
        self.graph.add_node(node_id)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge between nodes."""
        self.graph.add_edge(from_node, to_node)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph."""
        self.graph.remove_node(node_id)

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove an edge from the graph."""
        self.graph.remove_edge(from_node, to_node)

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get all dependencies for a node."""
        return set(nx.ancestors(self.graph, node_id))

    def get_dependents(self, node_id: str) -> Set[str]:
        """Get all nodes that depend on a node."""
        return set(nx.descendants(self.graph, node_id))

    def topological_sort(self) -> List[str]:
        """
        Get nodes in topological order.

        Returns:
            List of node IDs in execution order

        Raises:
            ValueError: If graph contains cycles
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Pipeline contains cycles")

    def has_cycles(self) -> bool:
        """Check if graph contains cycles."""
        try:
            list(nx.topological_sort(self.graph))
            return False
        except nx.NetworkXUnfeasible:
            return True

    def get_parallel_groups(self) -> List[Set[str]]:
        """
        Get groups of nodes that can be executed in parallel.

        Returns:
            List of sets, where each set contains node IDs that can
            execute simultaneously
        """
        levels = {}
        for node in nx.topological_sort(self.graph):
            deps = self.get_dependencies(node)
            if deps:
                levels[node] = max(levels[d] for d in deps) + 1
            else:
                levels[node] = 0

        groups = defaultdict(set)
        for node, level in levels.items():
            groups[level].add(node)

        return [groups[i] for i in range(max(groups.keys()) + 1)]

    def visualize(self, output_path: Optional[str] = None) -> None:
        """Visualize the DAG (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.graph, k=2)
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=3000,
                font_size=10,
                font_weight="bold",
                arrows=True,
                arrowsize=20,
            )

            if output_path:
                plt.savefig(output_path, bbox_inches="tight", dpi=150)
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib required for visualization")


class Pipeline:
    """
    A complete pipeline with multiple steps.

    Args:
        id: Unique identifier for the pipeline
        steps: Dictionary mapping step IDs to Step objects
        description: Optional description

    Example:
        >>> pipeline = Pipeline(
        ...     id="training_pipeline",
        ...     steps={
        ...         "load": Step("load", load_data),
        ...         "preprocess": Step("preprocess", preprocess, depends_on=["load"]),
        ...         "train": Step("train", train_model, depends_on=["preprocess"]),
        ...     }
        ... )
    """

    def __init__(
        self,
        id: str,
        steps: Dict[str, Step],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.steps = steps
        self.description = description
        self.metadata = metadata or {}
        self.dag = self._build_dag()

    def _build_dag(self) -> DAG:
        """Build DAG from steps."""
        dag = DAG()

        for step_id, step in self.steps.items():
            dag.add_node(step_id)

        for step_id, step in self.steps.items():
            for dep in step.depends_on:
                if dep in self.steps:
                    dag.add_edge(dep, step_id)
                else:
                    raise ValueError(
                        f"Step '{step_id}' depends on unknown step '{dep}'"
                    )

        if dag.has_cycles():
            raise ValueError("Pipeline contains circular dependencies")

        return dag

    def get_execution_order(self) -> List[str]:
        """Get steps in execution order."""
        return self.dag.topological_sort()

    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a step by ID."""
        return self.steps.get(step_id)

    def add_step(self, step: Step) -> None:
        """Add a step to the pipeline."""
        self.steps[step.id] = step
        self.dag = self._build_dag()

    def remove_step(self, step_id: str) -> None:
        """Remove a step from the pipeline."""
        if step_id in self.steps:
            del self.steps[step_id]
            self.dag = self._build_dag()

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate pipeline structure.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check for cycles
        if self.dag.has_cycles():
            errors.append("Pipeline contains circular dependencies")

        # Check for orphaned steps
        all_deps = set()
        for step in self.steps.values():
            all_deps.update(step.depends_on)

        for dep in all_deps:
            if dep not in self.steps:
                errors.append(f"Unknown dependency: {dep}")

        # Validate each step
        for step_id, step in self.steps.items():
            if not callable(step.func):
                errors.append(f"Step '{step_id}' has non-callable function")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "metadata": self.metadata,
            "steps": {
                step_id: {
                    "id": step.id,
                    "depends_on": step.depends_on,
                    "metadata": step.metadata,
                }
                for step_id, step in self.steps.items()
            },
            "execution_order": self.get_execution_order(),
        }


# =============================================================================
# Section 2: Execution
# =============================================================================


class PipelineExecutor(ABC):
    """
    Abstract base class for pipeline execution strategies.

    Example:
        >>> executor = SequentialExecutor()
        >>> results = executor.execute(pipeline, inputs={"data": my_data})
    """

    @abstractmethod
    def execute(
        self,
        pipeline: Pipeline,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, StepResult]:
        """
        Execute a pipeline.

        Args:
            pipeline: Pipeline to execute
            inputs: Initial inputs to the pipeline
            **kwargs: Additional execution options

        Returns:
            Dictionary mapping step IDs to their results
        """
        pass

    def _create_context(
        self,
        pipeline: Pipeline,
        step: Step,
        inputs: Dict[str, Any],
        results: Dict[str, StepResult],
    ) -> StepContext:
        """Create execution context for a step."""
        step_inputs = dict(inputs)

        # Add outputs from dependencies
        for dep_id in step.depends_on:
            if dep_id in results and results[dep_id].is_success:
                step_inputs[f"{dep_id}_output"] = results[dep_id].output

        return StepContext(
            pipeline_id=pipeline.id,
            step_id=step.id,
            inputs=step_inputs,
            metadata={
                "pipeline_id": pipeline.id,
                "step_id": step.id,
            },
        )


class SequentialExecutor(PipelineExecutor):
    """
    Execute pipeline steps sequentially.

    Executes steps one at a time in topological order.

    Example:
        >>> executor = SequentialExecutor()
        >>> results = executor.execute(pipeline)
        >>> for step_id, result in results.items():
        ...     print(f"{step_id}: {result.status}")
    """

    def __init__(self, monitor: Optional["PipelineMonitor"] = None):
        self.monitor = monitor

    def execute(
        self,
        pipeline: Pipeline,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, StepResult]:
        """Execute pipeline sequentially."""
        inputs = inputs or {}
        results = {}
        execution_order = pipeline.get_execution_order()

        if self.monitor:
            self.monitor.on_pipeline_start(pipeline)

        try:
            for step_id in execution_order:
                step = pipeline.get_step(step_id)
                if step is None:
                    continue

                if self.monitor:
                    self.monitor.on_step_start(pipeline, step)

                context = self._create_context(pipeline, step, inputs, results)
                result = step.execute(context)
                results[step_id] = result

                if self.monitor:
                    self.monitor.on_step_end(pipeline, step, result)

                # Stop on failure if not handled
                if not result.is_success and not step.fallback:
                    break

        finally:
            if self.monitor:
                self.monitor.on_pipeline_end(pipeline, results)

        return results


class ParallelExecutor(PipelineExecutor):
    """
    Execute pipeline steps in parallel where possible.

    Executes independent steps concurrently using thread/process pools.

    Args:
        max_workers: Maximum number of parallel workers
        use_processes: Use process pool instead of thread pool

    Example:
        >>> executor = ParallelExecutor(max_workers=4)
        >>> results = executor.execute(pipeline)
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
        monitor: Optional["PipelineMonitor"] = None,
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.monitor = monitor

    def execute(
        self,
        pipeline: Pipeline,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, StepResult]:
        """Execute pipeline with parallelization."""
        inputs = inputs or {}
        results = {}
        parallel_groups = pipeline.dag.get_parallel_groups()

        if self.monitor:
            self.monitor.on_pipeline_start(pipeline)

        try:
            ExecutorClass = (
                ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            )

            for group in parallel_groups:
                with ExecutorClass(max_workers=self.max_workers) as executor:
                    futures = {}

                    for step_id in group:
                        step = pipeline.get_step(step_id)
                        if step is None:
                            continue

                        context = self._create_context(pipeline, step, inputs, results)

                        if self.monitor:
                            self.monitor.on_step_start(pipeline, step)

                        future = executor.submit(step.execute, context)
                        futures[future] = step

                    for future in as_completed(futures):
                        step = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            result = StepResult(
                                step_id=step.id,
                                status=StepStatus.FAILED,
                                error=e,
                            )

                        results[step.id] = result

                        if self.monitor:
                            self.monitor.on_step_end(pipeline, step, result)

        finally:
            if self.monitor:
                self.monitor.on_pipeline_end(pipeline, results)

        return results


class DistributedExecutor(PipelineExecutor):
    """
    Execute pipeline steps in a distributed environment.

    Supports Ray, Dask, or custom distributed backends.

    Args:
        backend: Distributed backend ('ray', 'dask', 'custom')
        backend_url: URL for connecting to distributed cluster

    Example:
        >>> executor = DistributedExecutor(backend='ray')
        >>> results = executor.execute(pipeline)
    """

    def __init__(
        self,
        backend: str = "ray",
        backend_url: Optional[str] = None,
        monitor: Optional["PipelineMonitor"] = None,
    ):
        self.backend = backend
        self.backend_url = backend_url
        self.monitor = monitor
        self._backend = None

    def _initialize_backend(self):
        """Initialize the distributed backend."""
        if self.backend == "ray":
            try:
                import ray

                if not ray.is_initialized():
                    ray.init(address=self.backend_url or "auto")
                self._backend = ray
            except ImportError:
                raise ImportError("Ray is required for distributed execution")

        elif self.backend == "dask":
            try:
                from dask.distributed import Client

                self._backend = Client(self.backend_url)
            except ImportError:
                raise ImportError("Dask is required for distributed execution")

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def execute(
        self,
        pipeline: Pipeline,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, StepResult]:
        """Execute pipeline in distributed mode."""
        if self._backend is None:
            self._initialize_backend()

        inputs = inputs or {}
        results = {}

        if self.monitor:
            self.monitor.on_pipeline_start(pipeline)

        try:
            if self.backend == "ray":
                results = self._execute_ray(pipeline, inputs)
            elif self.backend == "dask":
                results = self._execute_dask(pipeline, inputs)

        finally:
            if self.monitor:
                self.monitor.on_pipeline_end(pipeline, results)

        return results

    def _execute_ray(
        self,
        pipeline: Pipeline,
        inputs: Dict[str, Any],
    ) -> Dict[str, StepResult]:
        """Execute using Ray."""
        import ray

        @ray.remote
        def execute_step_remote(step_func, context_dict):
            """Remote function for step execution."""
            context = StepContext(**context_dict)
            return step_func(context)

        execution_order = pipeline.get_execution_order()

        for step_id in execution_order:
            step = pipeline.get_step(step_id)
            if step is None:
                continue

            context = self._create_context(pipeline, step, inputs, results)

            if self.monitor:
                self.monitor.on_step_start(pipeline, step)

            # Submit to Ray
            future = execute_step_remote.remote(
                step.func,
                {
                    "pipeline_id": context.pipeline_id,
                    "step_id": context.step_id,
                    "inputs": context.inputs,
                    "metadata": context.metadata,
                },
            )

            result = ray.get(future)
            results[step_id] = result

            if self.monitor:
                self.monitor.on_step_end(pipeline, step, result)

        return results

    def _execute_dask(
        self,
        pipeline: Pipeline,
        inputs: Dict[str, Any],
    ) -> Dict[str, StepResult]:
        """Execute using Dask."""
        from dask import delayed

        execution_order = pipeline.get_execution_order()
        delayed_results = {}

        for step_id in execution_order:
            step = pipeline.get_step(step_id)
            if step is None:
                continue

            context = self._create_context(pipeline, step, inputs, results)

            @delayed
            def execute_delayed(ctx):
                return step.execute(ctx)

            delayed_result = execute_delayed(context)
            delayed_results[step_id] = delayed_result

        # Compute all results
        computed = self._backend.compute(*delayed_results.values())

        for step_id, result in zip(delayed_results.keys(), computed):
            results[step_id] = result

        return results


# =============================================================================
# Section 3: Scheduling
# =============================================================================


class Scheduler(ABC):
    """
    Abstract base class for pipeline schedulers.

    Schedulers determine when and how often pipelines should run.

    Example:
        >>> scheduler = CronScheduler("0 0 * * *")  # Daily at midnight
        >>> scheduler.schedule(pipeline, executor)
    """

    @abstractmethod
    def should_run(self, last_run: Optional[datetime] = None) -> bool:
        """Check if pipeline should run now."""
        pass

    @abstractmethod
    def get_next_run(self, last_run: Optional[datetime] = None) -> Optional[datetime]:
        """Get the next scheduled run time."""
        pass

    def schedule(
        self,
        pipeline: Pipeline,
        executor: PipelineExecutor,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, StepResult]]:
        """
        Execute pipeline if it should run now.

        Args:
            pipeline: Pipeline to execute
            executor: Executor to use
            inputs: Pipeline inputs

        Returns:
            Results if executed, None otherwise
        """
        if self.should_run():
            return executor.execute(pipeline, inputs)
        return None


class CronScheduler(Scheduler):
    """
    Schedule pipeline execution using cron expressions.

    Args:
        cron_expression: Cron expression (e.g., "0 0 * * *" for daily)
        timezone: Timezone for scheduling

    Example:
        >>> # Run every day at 2:30 AM
        >>> scheduler = CronScheduler("30 2 * * *")
        >>>
        >>> # Run every Monday at 9:00 AM
        >>> scheduler = CronScheduler("0 9 * * 1")
    """

    def __init__(self, cron_expression: str, timezone: str = "UTC"):
        self.cron_expression = cron_expression
        self.timezone = timezone
        self._cron = croniter(cron_expression)

    def should_run(self, last_run: Optional[datetime] = None) -> bool:
        """Check if pipeline should run based on cron schedule."""
        if last_run is None:
            return True

        next_run = self._cron.get_next(datetime, last_run)
        return datetime.now() >= next_run

    def get_next_run(self, last_run: Optional[datetime] = None) -> Optional[datetime]:
        """Get next scheduled run time."""
        base_time = last_run or datetime.now()
        return self._cron.get_next(datetime, base_time)


class IntervalScheduler(Scheduler):
    """
    Schedule pipeline execution at fixed intervals.

    Args:
        interval: Time interval between runs
        unit: Unit of time ('seconds', 'minutes', 'hours', 'days')

    Example:
        >>> # Run every 5 minutes
        >>> scheduler = IntervalScheduler(5, 'minutes')
        >>>
        >>> # Run every hour
        >>> scheduler = IntervalScheduler(1, 'hours')
    """

    def __init__(self, interval: int, unit: str = "minutes"):
        self.interval = interval
        self.unit = unit

        # Convert to seconds
        unit_multipliers = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
        }

        if unit not in unit_multipliers:
            raise ValueError(f"Unknown unit: {unit}")

        self.interval_seconds = interval * unit_multipliers[unit]

    def should_run(self, last_run: Optional[datetime] = None) -> bool:
        """Check if enough time has passed since last run."""
        if last_run is None:
            return True

        elapsed = (datetime.now() - last_run).total_seconds()
        return elapsed >= self.interval_seconds

    def get_next_run(self, last_run: Optional[datetime] = None) -> Optional[datetime]:
        """Get next scheduled run time."""
        if last_run is None:
            return datetime.now()
        return last_run + timedelta(seconds=self.interval_seconds)


class EventScheduler(Scheduler):
    """
    Schedule pipeline execution based on events.

    Args:
        event_source: Source to listen for events
        event_filter: Optional filter for specific events

    Example:
        >>> def on_file_arrival(event):
        ...     return event['type'] == 'file_created' and event['path'].endswith('.csv')
        >>>
        >>> scheduler = EventScheduler(
        ...     event_source=file_watcher,
        ...     event_filter=on_file_arrival
        ... )
    """

    def __init__(
        self,
        event_source: Callable[[], Iterator[Dict[str, Any]]],
        event_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        self.event_source = event_source
        self.event_filter = event_filter or (lambda e: True)
        self._pending_events: List[Dict[str, Any]] = []

    def should_run(self, last_run: Optional[datetime] = None) -> bool:
        """Check if any relevant events have occurred."""
        # Check for new events
        try:
            for event in self.event_source():
                if self.event_filter(event):
                    self._pending_events.append(event)
        except StopIteration:
            pass

        return len(self._pending_events) > 0

    def get_next_run(self, last_run: Optional[datetime] = None) -> Optional[datetime]:
        """Event-based scheduling doesn't have a fixed next run time."""
        return None

    def get_pending_events(self) -> List[Dict[str, Any]]:
        """Get and clear pending events."""
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    def schedule(
        self,
        pipeline: Pipeline,
        executor: PipelineExecutor,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, StepResult]]:
        """Execute pipeline with event data as inputs."""
        if not self.should_run():
            return None

        events = self.get_pending_events()
        inputs = inputs or {}
        inputs["events"] = events

        return executor.execute(pipeline, inputs)


# =============================================================================
# Section 4: Caching
# =============================================================================


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Implementations can use memory, disk, Redis, etc.

    Example:
        >>> backend = DiskCacheBackend("./cache")
        >>> cache = StepCache(backend)
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class MemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend.

    Args:
        max_size: Maximum number of items to store
        default_ttl: Default TTL in seconds

    Example:
        >>> backend = MemoryCacheBackend(max_size=1000)
        >>> backend.set("key", value)
        >>> value = backend.get("key")
    """

    def __init__(self, max_size: int = 10000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, Optional[datetime]]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, respecting TTL."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        if expiry and datetime.now() > expiry:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl

        if ttl:
            expiry = datetime.now() + timedelta(seconds=ttl)
        else:
            expiry = None

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k][1] or datetime.min
            )
            del self._cache[oldest_key]

        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None


class DiskCacheBackend(CacheBackend):
    """
    Disk-based cache backend using pickle.

    Args:
        cache_dir: Directory to store cache files
        default_ttl: Default TTL in seconds

    Example:
        >>> backend = DiskCacheBackend("./pipeline_cache")
        >>> backend.set("step1", result)
    """

    def __init__(self, cache_dir: str = "./cache", default_ttl: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

    def _get_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash key to create safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _get_meta_path(self, key: str) -> Path:
        """Get metadata file path."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        cache_path = self._get_path(key)
        meta_path = self._get_meta_path(key)

        if not cache_path.exists():
            return None

        # Check TTL
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("expiry"):
                expiry = datetime.fromisoformat(meta["expiry"])
                if datetime.now() > expiry:
                    self.delete(key)
                    return None

        # Load value
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in disk cache."""
        cache_path = self._get_path(key)
        meta_path = self._get_meta_path(key)

        # Save value
        with open(cache_path, "wb") as f:
            pickle.dump(value, f)

        # Save metadata
        ttl = ttl or self.default_ttl
        meta = {"created": datetime.now().isoformat()}
        if ttl:
            meta["expiry"] = (datetime.now() + timedelta(seconds=ttl)).isoformat()

        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def delete(self, key: str) -> None:
        """Delete value from disk cache."""
        cache_path = self._get_path(key)
        meta_path = self._get_meta_path(key)

        if cache_path.exists():
            cache_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    def clear(self) -> None:
        """Clear all cached values."""
        for file_path in self.cache_dir.glob("*.pkl"):
            file_path.unlink()
        for file_path in self.cache_dir.glob("*.json"):
            file_path.unlink()

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None


class StepCache:
    """
    Cache for pipeline step results.

    Args:
        backend: Cache backend to use

    Example:
        >>> cache = StepCache(MemoryCacheBackend())
        >>>
        >>> def expensive_operation(ctx):
        ...     return cache.get_or_compute(
        ...         key="expensive_op",
        ...         compute_fn=lambda: heavy_computation(),
        ...         step_id=ctx.step_id
        ...     )
    """

    def __init__(self, backend: CacheBackend):
        self.backend = backend

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        self.backend.set(key, value, ttl)

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        step_id: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> Tuple[Any, bool]:
        """
        Get from cache or compute and store.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            step_id: Optional step ID for logging
            ttl: Optional TTL

        Returns:
            Tuple of (value, was_cached)
        """
        cached = self.get(key)
        if cached is not None:
            if step_id:
                logger.info(f"Cache hit for step '{step_id}'")
            return cached, True

        value = compute_fn()
        self.set(key, value, ttl)

        if step_id:
            logger.info(f"Cache miss for step '{step_id}', computed and cached")

        return value, False

    def invalidate(self, key: str) -> None:
        """Invalidate a cached entry."""
        self.backend.delete(key)

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Returns:
            Number of entries invalidated
        """
        # This is a simplified implementation
        # A full implementation would need to track all keys
        count = 0
        # Implementation depends on backend capabilities
        return count

    def clear(self) -> None:
        """Clear all cached entries."""
        self.backend.clear()


def memoize(
    cache: Optional[StepCache] = None,
    key_func: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = None,
):
    """
    Decorator for memoizing function results.

    Args:
        cache: StepCache instance (creates default if None)
        key_func: Function to generate cache key from arguments
        ttl: Time-to-live in seconds

    Example:
        >>> cache = StepCache(MemoryCacheBackend())
        >>>
        >>> @memoize(cache=cache)
        ... def expensive_computation(x, y):
        ...     return x ** y
        >>>
        >>> result = expensive_computation(2, 10)  # Computed
        >>> result = expensive_computation(2, 10)  # Cached
    """
    if cache is None:
        cache = StepCache(MemoryCacheBackend())

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                key = hashlib.md5(key.encode()).hexdigest()

            # Try to get from cache
            value, was_cached = cache.get_or_compute(
                key=key,
                compute_fn=lambda: func(*args, **kwargs),
                ttl=ttl,
            )

            return value

        # Attach cache to function for manual invalidation
        wrapper.cache = cache
        wrapper.invalidate = lambda key=None: cache.invalidate(key or func.__name__)

        return wrapper

    return decorator


def invalidate_cache(cache: StepCache, pattern: Optional[str] = None) -> int:
    """
    Invalidate cache entries.

    Args:
        cache: Cache to invalidate
        pattern: Optional pattern to match keys (invalidates all if None)

    Returns:
        Number of entries invalidated

    Example:
        >>> cache = StepCache(MemoryCacheBackend())
        >>> invalidate_cache(cache)  # Clear all
        >>> invalidate_cache(cache, pattern="step1_*")  # Clear pattern
    """
    if pattern is None:
        cache.clear()
        logger.info("Cache cleared")
        return -1  # Unknown count
    else:
        count = cache.invalidate_pattern(pattern)
        logger.info(f"Invalidated {count} cache entries matching '{pattern}'")
        return count


# =============================================================================
# Section 5: Error Handling
# =============================================================================


@dataclass
class RetryPolicy:
    """
    Policy for retrying failed steps.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to retry on

    Example:
        >>> policy = RetryPolicy(
        ...     max_retries=3,
        ...     delay=1.0,
        ...     backoff_factor=2.0,
        ...     exceptions=(ConnectionError, TimeoutError)
        ... )
    """

    max_retries: int = 3
    delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    exceptions: Tuple[type, ...] = (Exception,)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if should retry based on exception and attempt number."""
        if attempt >= self.max_retries:
            return False
        return isinstance(exception, self.exceptions)

    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry."""
        delay = self.delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


class FallbackStep:
    """
    Fallback step to execute when a step fails.

    Args:
        func: Fallback function
        use_partial_results: Whether to use partial results from failed step

    Example:
        >>> def fallback_load(ctx):
        ...     logger.warning("Using fallback data source")
        ...     return load_from_backup()
        >>>
        >>> fallback = FallbackStep(fallback_load)
        >>> step = Step("load", load_data, fallback=fallback)
    """

    def __init__(
        self,
        func: StepFunction,
        use_partial_results: bool = False,
    ):
        self.func = func
        self.use_partial_results = use_partial_results

    def execute(
        self,
        context: StepContext,
        failed_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute fallback function."""
        try:
            if self.use_partial_results and failed_result:
                context.inputs["partial_result"] = failed_result.output

            output = self.func(context)

            return StepResult(
                step_id=context.step_id,
                status=StepStatus.COMPLETED,
                output=output,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        except Exception as e:
            return StepResult(
                step_id=context.step_id,
                status=StepStatus.FAILED,
                error=e,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )


class ErrorHandler:
    """
    Centralized error handler for pipeline execution.

    Args:
        on_error: Callback for error handling
        continue_on_error: Whether to continue pipeline on error

    Example:
        >>> def handle_error(step_id, error, context):
        ...     logger.error(f"Step {step_id} failed: {error}")
        ...     send_alert(f"Pipeline error in {step_id}")
        >>>
        >>> handler = ErrorHandler(on_error=handle_error)
    """

    def __init__(
        self,
        on_error: Optional[Callable[[str, Exception, StepContext], None]] = None,
        continue_on_error: bool = False,
    ):
        self.on_error = on_error
        self.continue_on_error = continue_on_error
        self._errors: List[Tuple[str, Exception, StepContext]] = []

    def handle(
        self,
        step_id: str,
        error: Exception,
        context: StepContext,
    ) -> bool:
        """
        Handle an error.

        Returns:
            Whether to continue pipeline execution
        """
        self._errors.append((step_id, error, context))

        if self.on_error:
            try:
                self.on_error(step_id, error, context)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")

        return self.continue_on_error

    def get_errors(self) -> List[Tuple[str, Exception, StepContext]]:
        """Get all recorded errors."""
        return self._errors.copy()

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self._errors) > 0


class Checkpoint:
    """
    Checkpoint manager for pipeline state persistence.

    Args:
        checkpoint_dir: Directory to store checkpoints
        save_frequency: Save checkpoint every N steps

    Example:
        >>> checkpoint = Checkpoint("./checkpoints")
        >>> checkpoint.save(pipeline, results, step_id="step5")
        >>>
        >>> # Later, resume from checkpoint
        >>> state = checkpoint.load("pipeline_id")
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_frequency: int = 1,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency

    def save(
        self,
        pipeline: Pipeline,
        results: Dict[str, StepResult],
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save pipeline checkpoint.

        Returns:
            Path to checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{pipeline.id}_{timestamp}"
        if step_id:
            checkpoint_name += f"_{step_id}"

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"

        checkpoint_data = {
            "pipeline_id": pipeline.id,
            "timestamp": datetime.now().isoformat(),
            "step_id": step_id,
            "results": {
                k: {
                    "step_id": v.step_id,
                    "status": v.status.name,
                    "output": v.output,
                    "execution_time": v.execution_time,
                }
                for k, v in results.items()
            },
            "metadata": metadata or {},
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load checkpoint from file."""
        checkpoint_path = Path(checkpoint_path)

        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[Path]:
        """List available checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("*.pkl"))

        if pipeline_id:
            checkpoints = [c for c in checkpoints if c.name.startswith(pipeline_id)]

        return sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)

    def get_latest(self, pipeline_id: str) -> Optional[Path]:
        """Get latest checkpoint for a pipeline."""
        checkpoints = self.list_checkpoints(pipeline_id)
        return checkpoints[0] if checkpoints else None

    def resume_from_checkpoint(
        self,
        pipeline: Pipeline,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Resume pipeline from checkpoint.

        Returns:
            Tuple of (checkpoint_data, completed_step_ids)
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest(pipeline.id)

        if checkpoint_path is None:
            return {}, []

        checkpoint_data = self.load(checkpoint_path)
        completed_steps = [
            step_id
            for step_id, result in checkpoint_data["results"].items()
            if result["status"] in ("COMPLETED", "CACHED")
        ]

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        logger.info(f"Completed steps: {completed_steps}")

        return checkpoint_data, completed_steps


# =============================================================================
# Section 6: Monitoring
# =============================================================================


class StepMetrics:
    """
    Metrics collector for individual steps.

    Tracks execution time, memory usage, and custom metrics.

    Example:
        >>> metrics = StepMetrics("training_step")
        >>> metrics.record_execution_time(10.5)
        >>> metrics.record_memory_usage(1024**3)
        >>> metrics.add_custom_metric("accuracy", 0.95)
    """

    def __init__(self, step_id: str):
        self.step_id = step_id
        self.execution_time: float = 0.0
        self.memory_usage: Optional[int] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.custom_metrics: Dict[str, Any] = {}

    def record_execution_time(self, seconds: float) -> None:
        """Record step execution time."""
        self.execution_time = seconds

    def record_memory_usage(self, bytes_used: int) -> None:
        """Record memory usage in bytes."""
        self.memory_usage = bytes_used

    def add_custom_metric(self, name: str, value: Any) -> None:
        """Add a custom metric."""
        self.custom_metrics[name] = value

    def start(self) -> None:
        """Mark step start time."""
        self.start_time = datetime.now()

    def end(self) -> None:
        """Mark step end time."""
        self.end_time = datetime.now()
        if self.start_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "step_id": self.step_id,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "custom_metrics": self.custom_metrics,
        }


class PipelineLogger:
    """
    Structured logger for pipeline execution.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to

    Example:
        >>> pl_logger = PipelineLogger("my_pipeline", log_file="pipeline.log")
        >>> pl_logger.log_step_start("training")
        >>> pl_logger.log_step_end("training", success=True, duration=10.5)
    """

    def __init__(
        self,
        name: str = "pipeline",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_step_start(self, step_id: str, inputs: Optional[Dict] = None) -> None:
        """Log step start."""
        msg = f"Step '{step_id}' started"
        if inputs:
            msg += f" with inputs: {list(inputs.keys())}"
        self.logger.info(msg)

    def log_step_end(
        self,
        step_id: str,
        success: bool,
        duration: float,
        output_type: Optional[str] = None,
    ) -> None:
        """Log step completion."""
        status = "completed" if success else "failed"
        msg = f"Step '{step_id}' {status} in {duration:.2f}s"
        if output_type:
            msg += f" (output: {output_type})"

        if success:
            self.logger.info(msg)
        else:
            self.logger.error(msg)

    def log_pipeline_start(self, pipeline_id: str, steps: List[str]) -> None:
        """Log pipeline start."""
        self.logger.info(f"Pipeline '{pipeline_id}' started with {len(steps)} steps")

    def log_pipeline_end(
        self,
        pipeline_id: str,
        success: bool,
        completed_steps: int,
        failed_steps: int,
        total_duration: float,
    ) -> None:
        """Log pipeline completion."""
        status = "completed" if success else "failed"
        self.logger.info(
            f"Pipeline '{pipeline_id}' {status}: "
            f"{completed_steps} completed, {failed_steps} failed, "
            f"total time: {total_duration:.2f}s"
        )

    def log_error(self, step_id: str, error: Exception) -> None:
        """Log an error."""
        self.logger.error(f"Step '{step_id}' error: {str(error)}")
        self.logger.debug(traceback.format_exc())

    def log_metric(self, step_id: str, metric_name: str, value: Any) -> None:
        """Log a metric."""
        self.logger.info(f"Step '{step_id}' metric - {metric_name}: {value}")


class PipelineMonitor:
    """
    Monitor for pipeline execution.

    Tracks step execution, collects metrics, and provides callbacks.

    Args:
        logger: PipelineLogger instance
        metrics_collector: Optional metrics collector

    Example:
        >>> monitor = PipelineMonitor(PipelineLogger())
        >>> executor = SequentialExecutor(monitor=monitor)
        >>> results = executor.execute(pipeline)
    """

    def __init__(
        self,
        logger: Optional[PipelineLogger] = None,
        metrics_collector: Optional[Dict[str, StepMetrics]] = None,
    ):
        self.logger = logger or PipelineLogger()
        self.metrics = metrics_collector or {}
        self.step_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.pipeline_callbacks: List[Callable] = []

    def on_step_start(self, pipeline: Pipeline, step: Step) -> None:
        """Called when a step starts."""
        self.logger.log_step_start(step.id)

        metrics = StepMetrics(step.id)
        metrics.start()
        self.metrics[step.id] = metrics

        # Run callbacks
        for callback in self.step_callbacks[step.id]:
            try:
                callback("start", step, None)
            except Exception as e:
                logger.warning(f"Step callback error: {e}")

    def on_step_end(
        self,
        pipeline: Pipeline,
        step: Step,
        result: StepResult,
    ) -> None:
        """Called when a step ends."""
        metrics = self.metrics.get(step.id)
        if metrics:
            metrics.end()
            if result.execution_time:
                metrics.record_execution_time(result.execution_time)

        self.logger.log_step_end(
            step.id,
            success=result.is_success,
            duration=result.execution_time or 0.0,
            output_type=type(result.output).__name__ if result.output else None,
        )

        if result.error:
            self.logger.log_error(step.id, result.error)

        # Run callbacks
        for callback in self.step_callbacks[step.id]:
            try:
                callback("end", step, result)
            except Exception as e:
                logger.warning(f"Step callback error: {e}")

    def on_pipeline_start(self, pipeline: Pipeline) -> None:
        """Called when pipeline starts."""
        self.logger.log_pipeline_start(pipeline.id, list(pipeline.steps.keys()))

    def on_pipeline_end(
        self,
        pipeline: Pipeline,
        results: Dict[str, StepResult],
    ) -> None:
        """Called when pipeline ends."""
        completed = sum(1 for r in results.values() if r.is_success)
        failed = len(results) - completed
        total_time = sum(r.execution_time or 0 for r in results.values())

        success = failed == 0
        self.logger.log_pipeline_end(
            pipeline.id,
            success=success,
            completed_steps=completed,
            failed_steps=failed,
            total_duration=total_time,
        )

        # Run callbacks
        for callback in self.pipeline_callbacks:
            try:
                callback(pipeline, results)
            except Exception as e:
                logger.warning(f"Pipeline callback error: {e}")

    def register_step_callback(
        self,
        step_id: str,
        callback: Callable[[str, Step, Optional[StepResult]], None],
    ) -> None:
        """Register a callback for step events."""
        self.step_callbacks[step_id].append(callback)

    def register_pipeline_callback(
        self,
        callback: Callable[[Pipeline, Dict[str, StepResult]], None],
    ) -> None:
        """Register a callback for pipeline events."""
        self.pipeline_callbacks.append(callback)

    def get_metrics(
        self, step_id: Optional[str] = None
    ) -> Union[StepMetrics, Dict[str, StepMetrics]]:
        """Get metrics for a step or all steps."""
        if step_id:
            return self.metrics.get(step_id)
        return self.metrics


class PipelineVisualization:
    """
    Visualization utilities for pipelines.

    Creates visual representations of pipeline structure and execution.

    Example:
        >>> viz = PipelineVisualization()
        >>> viz.visualize_dag(pipeline, "pipeline_dag.png")
        >>> viz.visualize_execution(results, "execution_timeline.png")
    """

    def __init__(self, style: str = "default"):
        self.style = style

    def visualize_dag(
        self,
        pipeline: Pipeline,
        output_path: Optional[str] = None,
        show_status: bool = False,
        results: Optional[Dict[str, StepResult]] = None,
    ) -> None:
        """
        Visualize pipeline DAG.

        Args:
            pipeline: Pipeline to visualize
            output_path: Path to save visualization (shows if None)
            show_status: Color nodes by execution status
            results: Step results for status coloring
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(14, 10))

            # Get positions
            pos = nx.spring_layout(pipeline.dag.graph, k=2, iterations=50)

            # Determine node colors
            if show_status and results:
                color_map = {
                    StepStatus.COMPLETED: "lightgreen",
                    StepStatus.FAILED: "lightcoral",
                    StepStatus.RUNNING: "lightblue",
                    StepStatus.PENDING: "lightyellow",
                    StepStatus.SKIPPED: "lightgray",
                    StepStatus.CACHED: "lightcyan",
                }
                node_colors = [
                    color_map.get(
                        results.get(
                            node, StepResult(step_id=node, status=StepStatus.PENDING)
                        ).status,
                        "white",
                    )
                    for node in pipeline.dag.graph.nodes()
                ]
            else:
                node_colors = "lightblue"

            # Draw
            nx.draw_networkx_nodes(
                pipeline.dag.graph,
                pos,
                node_color=node_colors,
                node_size=3000,
                ax=ax,
            )
            nx.draw_networkx_labels(
                pipeline.dag.graph,
                pos,
                font_size=10,
                font_weight="bold",
                ax=ax,
            )
            nx.draw_networkx_edges(
                pipeline.dag.graph,
                pos,
                edge_color="gray",
                arrows=True,
                arrowsize=20,
                ax=ax,
            )

            ax.set_title(f"Pipeline: {pipeline.id}")
            ax.axis("off")

            if show_status and results:
                legend_elements = [
                    mpatches.Patch(facecolor=color, label=status.name)
                    for status, color in color_map.items()
                ]
                ax.legend(handles=legend_elements, loc="upper left")

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"DAG visualization saved to {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib and networkx required for visualization")

    def visualize_execution(
        self,
        results: Dict[str, StepResult],
        output_path: Optional[str] = None,
    ) -> None:
        """
        Visualize execution timeline.

        Args:
            results: Step execution results
            output_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))

            steps = list(results.keys())
            start_times = []
            durations = []

            for step_id in steps:
                result = results[step_id]
                if result.start_time:
                    start_times.append(result.start_time)
                    durations.append(result.execution_time or 0)
                else:
                    start_times.append(datetime.now())
                    durations.append(0)

            # Convert to relative times
            if start_times:
                min_time = min(start_times)
                relative_starts = [(t - min_time).total_seconds() for t in start_times]

                # Create Gantt-like chart
                colors = ["green" if results[s].is_success else "red" for s in steps]
                ax.barh(steps, durations, left=relative_starts, color=colors, alpha=0.7)

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Steps")
            ax.set_title("Pipeline Execution Timeline")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"Execution visualization saved to {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib required for visualization")

    def generate_report(
        self,
        pipeline: Pipeline,
        results: Dict[str, StepResult],
        format: str = "text",
    ) -> str:
        """
        Generate execution report.

        Args:
            pipeline: Executed pipeline
            results: Step results
            format: Report format ('text', 'json', 'markdown')

        Returns:
            Report string
        """
        if format == "json":
            return json.dumps(
                {
                    "pipeline_id": pipeline.id,
                    "steps": {
                        step_id: {
                            "status": result.status.name,
                            "execution_time": result.execution_time,
                            "error": str(result.error) if result.error else None,
                        }
                        for step_id, result in results.items()
                    },
                },
                indent=2,
            )

        elif format == "markdown":
            lines = [
                f"# Pipeline Report: {pipeline.id}",
                "",
                "| Step | Status | Time (s) | Error |",
                "|------|--------|----------|-------|",
            ]
            for step_id, result in results.items():
                error_str = str(result.error)[:50] if result.error else "-"
                lines.append(
                    f"| {step_id} | {result.status.name} | "
                    f"{result.execution_time:.2f} | {error_str} |"
                )
            return "\n".join(lines)

        else:  # text
            lines = [f"Pipeline: {pipeline.id}", "=" * 50]
            for step_id, result in results.items():
                lines.append(f"\nStep: {step_id}")
                lines.append(f"  Status: {result.status.name}")
                lines.append(f"  Time: {result.execution_time:.2f}s")
                if result.error:
                    lines.append(f"  Error: {result.error}")
            return "\n".join(lines)


# =============================================================================
# Section 7: Integration
# =============================================================================


class MLflowIntegration:
    """
    Integration with MLflow for experiment tracking.

    Automatically logs pipeline execution to MLflow.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Name of the experiment

    Example:
        >>> mlflow_int = MLflowIntegration(
        ...     tracking_uri="http://localhost:5000",
        ...     experiment_name="my_experiment"
        ... )
        >>> mlflow_int.log_pipeline(pipeline, results)
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "fishstick_pipelines",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow = None

    def _get_mlflow(self):
        """Lazy import and setup MLflow."""
        if self._mlflow is None:
            try:
                import mlflow

                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow = mlflow
            except ImportError:
                raise ImportError("mlflow is required for MLflowIntegration")
        return self._mlflow

    def log_pipeline(
        self,
        pipeline: Pipeline,
        results: Dict[str, StepResult],
        params: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log pipeline execution to MLflow."""
        mlflow = self._get_mlflow()

        with mlflow.start_run(run_name=pipeline.id):
            # Log parameters
            if params:
                mlflow.log_params(params)

            # Log step metrics
            for step_id, result in results.items():
                if result.execution_time:
                    mlflow.log_metric(f"{step_id}_time", result.execution_time)

                if result.metrics:
                    for metric_name, value in result.metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{step_id}_{metric_name}", value)

            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)

            # Log pipeline structure
            mlflow.log_dict(pipeline.to_dict(), "pipeline.json")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log a model to MLflow."""
        mlflow = self._get_mlflow()

        # Try to infer model flavor
        try:
            import sklearn

            if isinstance(model, sklearn.base.BaseEstimator):
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                )
                return
        except ImportError:
            pass

        # Default to pickle
        mlflow.pyfunc.log_model(
            artifact_path,
            python_model=model,
            registered_model_name=registered_model_name,
        )


class KubeflowIntegration:
    """
    Integration with Kubeflow Pipelines.

    Converts fishstick pipelines to Kubeflow pipeline definitions.

    Args:
        host: Kubeflow Pipelines host

    Example:
        >>> kfp_int = KubeflowIntegration(host="http://localhost:8080")
        >>> kfp_pipeline = kfp_int.convert_pipeline(pipeline)
        >>> kfp_int.run_pipeline(kfp_pipeline, experiment_name="test")
    """

    def __init__(self, host: Optional[str] = None):
        self.host = host
        self._kfp = None

    def _get_kfp(self):
        """Lazy import KFP."""
        if self._kfp is None:
            try:
                import kfp

                self._kfp = kfp
            except ImportError:
                raise ImportError("kfp is required for KubeflowIntegration")
        return self._kfp

    def convert_pipeline(self, pipeline: Pipeline) -> Callable:
        """
        Convert fishstick pipeline to Kubeflow pipeline.

        Returns:
            Kubeflow pipeline function
        """
        kfp = self._get_kfp()

        @kfp.dsl.pipeline(
            name=pipeline.id,
            description=pipeline.description,
        )
        def kfp_pipeline():
            # This is a simplified conversion
            # Full implementation would create proper KFP components
            pass

        return kfp_pipeline

    def run_pipeline(
        self,
        pipeline_func: Callable,
        experiment_name: str,
        run_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Run pipeline on Kubeflow.

        Returns:
            Run ID
        """
        kfp = self._get_kfp()

        client = kfp.Client(host=self.host)

        experiment = client.create_experiment(name=experiment_name)

        run = client.run_pipeline(
            experiment_id=experiment.id,
            job_name=run_name
            or f"{pipeline_func.__name__}_{datetime.now().isoformat()}",
            pipeline_func=pipeline_func,
            arguments=arguments or {},
        )

        return run.id


class AirflowIntegration:
    """
    Integration with Apache Airflow.

    Generates Airflow DAGs from fishstick pipelines.

    Args:
        dag_id: DAG identifier
        schedule_interval: Schedule interval (cron expression or timedelta)

    Example:
        >>> airflow_int = AirflowIntegration(
        ...     dag_id="my_pipeline",
        ...     schedule_interval="@daily"
        ... )
        >>> dag = airflow_int.to_airflow_dag(pipeline)
    """

    def __init__(
        self,
        dag_id: str,
        schedule_interval: Union[str, timedelta] = "@daily",
        default_args: Optional[Dict[str, Any]] = None,
    ):
        self.dag_id = dag_id
        self.schedule_interval = schedule_interval
        self.default_args = default_args or {}

    def to_airflow_dag(self, pipeline: Pipeline) -> Any:
        """
        Convert pipeline to Airflow DAG.

        Returns:
            Airflow DAG object
        """
        try:
            from airflow import DAG
            from airflow.operators.python import PythonOperator
        except ImportError:
            raise ImportError("apache-airflow is required for AirflowIntegration")

        default_args = {
            "owner": "fishstick",
            "depends_on_past": False,
            "email_on_failure": False,
            "email_on_retry": False,
            "retries": 1,
            **self.default_args,
        }

        with DAG(
            dag_id=self.dag_id,
            default_args=default_args,
            description=pipeline.description,
            schedule_interval=self.schedule_interval,
            start_date=datetime.now(),
            catchup=False,
        ) as dag:
            task_map = {}

            # Create tasks
            for step_id in pipeline.get_execution_order():
                step = pipeline.get_step(step_id)

                def make_task(step_func):
                    def task(**context):
                        step_context = StepContext(
                            pipeline_id=pipeline.id,
                            step_id=step_id,
                            inputs=context.get("dag_run", {}).conf or {},
                        )
                        return step_func(step_context)

                    return task

                task = PythonOperator(
                    task_id=step_id,
                    python_callable=make_task(step.func),
                )
                task_map[step_id] = task

            # Set dependencies
            for step_id in pipeline.get_execution_order():
                step = pipeline.get_step(step_id)
                for dep in step.depends_on:
                    if dep in task_map:
                        task_map[dep] >> task_map[step_id]

        return dag

    def generate_dag_file(self, pipeline: Pipeline, output_path: str) -> None:
        """Generate Python DAG file for Airflow."""
        dag_code = f'''"""
DAG generated from fishstick pipeline: {pipeline.id}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {{
    'owner': 'fishstick',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}}

with DAG(
    dag_id='{self.dag_id}',
    default_args=default_args,
    description={repr(pipeline.description)},
    schedule_interval={repr(self.schedule_interval)},
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    # Define tasks here
    pass
'''

        with open(output_path, "w") as f:
            f.write(dag_code)

        logger.info(f"DAG file generated: {output_path}")


# =============================================================================
# Section 8: Utilities
# =============================================================================


class PipelineBuilder:
    """
    Builder pattern for constructing pipelines.

    Provides a fluent interface for building complex pipelines.

    Example:
        >>> builder = PipelineBuilder("training_pipeline")
        >>> pipeline = (builder
        ...     .add_step("load", load_data_func)
        ...     .add_step("preprocess", preprocess_func, depends_on=["load"])
        ...     .add_step("train", train_func, depends_on=["preprocess"])
        ...     .with_description("ML training pipeline")
        ...     .build())
    """

    def __init__(self, pipeline_id: str = "pipeline"):
        self.pipeline_id = pipeline_id
        self.steps: Dict[str, Step] = {}
        self.description: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def add_step(
        self,
        step_id: str,
        func: StepFunction,
        depends_on: Optional[List[str]] = None,
        cache_key: Optional[Callable[[StepContext], str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        fallback: Optional[FallbackStep] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineBuilder:
        """Add a step to the pipeline."""
        step = Step(
            id=step_id,
            func=func,
            depends_on=depends_on,
            cache_key=cache_key,
            retry_policy=retry_policy,
            fallback=fallback,
            metadata=metadata,
        )
        self.steps[step_id] = step
        return self

    def add_conditional_step(
        self,
        step_id: str,
        func: StepFunction,
        condition: Callable[[StepContext], bool],
        depends_on: Optional[List[str]] = None,
        **kwargs,
    ) -> PipelineBuilder:
        """Add a conditional step to the pipeline."""
        step = ConditionalStep(
            id=step_id,
            func=func,
            condition=condition,
            depends_on=depends_on,
            **kwargs,
        )
        self.steps[step_id] = step
        return self

    def add_loop_step(
        self,
        step_id: str,
        func: StepFunction,
        stop_condition: Callable[[StepContext], bool],
        max_iterations: int = 1000,
        depends_on: Optional[List[str]] = None,
        **kwargs,
    ) -> PipelineBuilder:
        """Add a loop step to the pipeline."""
        step = LoopStep(
            id=step_id,
            func=func,
            stop_condition=stop_condition,
            max_iterations=max_iterations,
            depends_on=depends_on,
            **kwargs,
        )
        self.steps[step_id] = step
        return self

    def with_description(self, description: str) -> PipelineBuilder:
        """Set pipeline description."""
        self.description = description
        return self

    def with_metadata(self, **kwargs) -> PipelineBuilder:
        """Add metadata to pipeline."""
        self.metadata.update(kwargs)
        return self

    def build(self) -> Pipeline:
        """Build the pipeline."""
        return Pipeline(
            id=self.pipeline_id,
            steps=self.steps,
            description=self.description,
            metadata=self.metadata,
        )


def run_pipeline(
    pipeline: Pipeline,
    inputs: Optional[Dict[str, Any]] = None,
    executor: Optional[PipelineExecutor] = None,
    cache: Optional[StepCache] = None,
    checkpoint: Optional[Checkpoint] = None,
    monitor: Optional[PipelineMonitor] = None,
    **kwargs,
) -> Dict[str, StepResult]:
    """
    Run a pipeline with the specified configuration.

    This is a convenience function that sets up all components and executes
    the pipeline.

    Args:
        pipeline: Pipeline to execute
        inputs: Initial inputs to the pipeline
        executor: Executor to use (default: SequentialExecutor)
        cache: Step cache for memoization
        checkpoint: Checkpoint manager for persistence
        monitor: Pipeline monitor for tracking
        **kwargs: Additional arguments passed to executor

    Returns:
        Dictionary mapping step IDs to their results

    Example:
        >>> pipeline = PipelineBuilder("train").add_step(...).build()
        >>> results = run_pipeline(
        ...     pipeline,
        ...     inputs={"data_path": "data.csv"},
        ...     executor=ParallelExecutor(max_workers=4),
        ...     cache=StepCache(MemoryCacheBackend()),
        ... )
    """
    # Validate pipeline
    is_valid, errors = validate_pipeline(pipeline)
    if not is_valid:
        raise ValueError(f"Pipeline validation failed: {errors}")

    # Use default executor if not provided
    if executor is None:
        executor = SequentialExecutor(monitor=monitor)

    # Create monitor if not provided
    if monitor is None and executor.monitor is None:
        monitor = PipelineMonitor()
        executor.monitor = monitor

    # Execute pipeline
    try:
        results = executor.execute(pipeline, inputs=inputs, **kwargs)

        # Save checkpoint if provided
        if checkpoint:
            last_step = pipeline.get_execution_order()[-1]
            checkpoint.save(pipeline, results, step_id=last_step)

        return results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


def validate_pipeline(pipeline: Pipeline) -> Tuple[bool, List[str]]:
    """
    Validate a pipeline for errors.

    Args:
        pipeline: Pipeline to validate

    Returns:
        Tuple of (is_valid, list of error messages)

    Example:
        >>> is_valid, errors = validate_pipeline(my_pipeline)
        >>> if not is_valid:
        ...     print("Validation errors:", errors)
    """
    errors = []

    # Check for empty pipeline
    if not pipeline.steps:
        errors.append("Pipeline has no steps")

    # Check for duplicate step IDs
    step_ids = list(pipeline.steps.keys())
    if len(step_ids) != len(set(step_ids)):
        errors.append("Pipeline contains duplicate step IDs")

    # Check for cycles
    if pipeline.dag.has_cycles():
        errors.append("Pipeline contains circular dependencies")

    # Check dependencies exist
    for step_id, step in pipeline.steps.items():
        for dep in step.depends_on:
            if dep not in pipeline.steps:
                errors.append(f"Step '{step_id}' has unknown dependency '{dep}'")

    # Check for orphaned steps (optional warning)
    all_deps = set()
    for step in pipeline.steps.values():
        all_deps.update(step.depends_on)

    roots = set(pipeline.steps.keys()) - all_deps
    if len(roots) == 0 and pipeline.steps:
        errors.append("Pipeline has no root steps (all steps have dependencies)")

    # Validate step functions
    for step_id, step in pipeline.steps.items():
        if not callable(step.func):
            errors.append(f"Step '{step_id}' has non-callable function")

        # Check function signature
        try:
            sig = inspect.signature(step.func)
            params = list(sig.parameters.keys())
            if not params:
                errors.append(
                    f"Step '{step_id}' function must accept at least one parameter"
                )
        except Exception:
            pass

    return len(errors) == 0, errors


# Export all public classes and functions
__all__ = [
    # Pipeline Definition
    "Pipeline",
    "Step",
    "StepContext",
    "StepResult",
    "StepStatus",
    "ConditionalStep",
    "LoopStep",
    "DAG",
    "StepFunction",
    # Execution
    "PipelineExecutor",
    "SequentialExecutor",
    "ParallelExecutor",
    "DistributedExecutor",
    # Scheduling
    "Scheduler",
    "CronScheduler",
    "IntervalScheduler",
    "EventScheduler",
    # Caching
    "StepCache",
    "CacheBackend",
    "MemoryCacheBackend",
    "DiskCacheBackend",
    "memoize",
    "invalidate_cache",
    # Error Handling
    "RetryPolicy",
    "FallbackStep",
    "ErrorHandler",
    "Checkpoint",
    # Monitoring
    "PipelineMonitor",
    "StepMetrics",
    "PipelineLogger",
    "PipelineVisualization",
    # Integration
    "MLflowIntegration",
    "KubeflowIntegration",
    "AirflowIntegration",
    # Utilities
    "PipelineBuilder",
    "run_pipeline",
    "validate_pipeline",
]
