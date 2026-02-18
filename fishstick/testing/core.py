"""
Comprehensive Testing Module for Fishstick

This module provides a complete testing framework including:
- Unit Testing (Model, Layer, Loss, Optimizer)
- Integration Testing (Pipeline, E2E, API, Load)
- Model Testing (Validation, Overfitting, Underfitting, Bias, Robustness)
- Property-Based Testing (Hypothesis, Property, Invariant)
- Snapshot Testing (Snapshot, Model, Output comparison)
- Fuzzing (Input, Model fuzzing)
- Coverage (Code, Model, Branch coverage)
- Utilities (Test suites, runners)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import pickle
import hashlib
import time
import warnings
from collections import defaultdict
import copy
import gc
import tracemalloc
from contextlib import contextmanager
import traceback


# ============================================================================
# Unit Testing
# ============================================================================


class TestStatus(Enum):
    """Status of a test execution."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a test execution."""

    test_name: str
    status: TestStatus
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error_traceback: Optional[str] = None


class UnitTest(ABC):
    """Abstract base class for unit tests."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results: List[TestResult] = []

    @abstractmethod
    def setup(self):
        """Set up test fixtures."""
        pass

    @abstractmethod
    def teardown(self):
        """Tear down test fixtures."""
        pass

    @abstractmethod
    def run(self) -> List[TestResult]:
        """Run all tests and return results."""
        pass

    def assert_true(self, condition: bool, message: str = "") -> bool:
        """Assert condition is True."""
        if not condition:
            raise AssertionError(message or "Assertion failed: expected True")
        return True

    def assert_equal(self, a: Any, b: Any, message: str = "") -> bool:
        """Assert two values are equal."""
        if a != b:
            raise AssertionError(message or f"Assertion failed: {a} != {b}")
        return True

    def assert_almost_equal(
        self,
        a: float,
        b: float,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        message: str = "",
    ) -> bool:
        """Assert two float values are almost equal."""
        if not np.isclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(message or f"Assertion failed: {a} !~= {b}")
        return True

    def assert_tensor_equal(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        message: str = "",
    ) -> bool:
        """Assert two tensors are equal."""
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(message or f"Assertion failed: tensors not equal")
        return True

    def assert_shape(
        self, tensor: torch.Tensor, expected_shape: Tuple[int, ...], message: str = ""
    ) -> bool:
        """Assert tensor has expected shape."""
        if tensor.shape != expected_shape:
            raise AssertionError(
                message or f"Assertion failed: shape {tensor.shape} != {expected_shape}"
            )
        return True

    def assert_no_nan(self, tensor: torch.Tensor, message: str = "") -> bool:
        """Assert tensor has no NaN values."""
        if torch.isnan(tensor).any():
            raise AssertionError(message or "Assertion failed: tensor contains NaN")
        return True

    def assert_no_inf(self, tensor: torch.Tensor, message: str = "") -> bool:
        """Assert tensor has no Inf values."""
        if torch.isinf(tensor).any():
            raise AssertionError(message or "Assertion failed: tensor contains Inf")
        return True


class ModelUnitTest(UnitTest):
    """
    Unit tests for neural network models.

    Tests model initialization, forward pass, backward pass,
    parameter updates, and shape consistency.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Dict[str, Any] = None,
        input_shape: Tuple[int, ...] = (32, 3, 224, 224),
        num_classes: int = 10,
    ):
        super().__init__("ModelUnitTest", f"Unit tests for {model_class.__name__}")
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model: Optional[nn.Module] = None

    def setup(self):
        """Initialize model for testing."""
        self.model = self.model_class(**self.model_kwargs)
        self.model.eval()

    def teardown(self):
        """Clean up model."""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> List[TestResult]:
        """Run all model unit tests."""
        self.results = []
        self.setup()

        tests = [
            ("test_initialization", self.test_initialization),
            ("test_forward_pass", self.test_forward_pass),
            ("test_backward_pass", self.test_backward_pass),
            ("test_parameter_count", self.test_parameter_count),
            ("test_batch_independence", self.test_batch_independence),
            ("test_deterministic_forward", self.test_deterministic_forward),
        ]

        for test_name, test_func in tests:
            start_time = time.time()
            try:
                test_func()
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        duration_ms=duration,
                        message="Test passed",
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                )

        self.teardown()
        return self.results

    def test_initialization(self):
        """Test model initializes correctly."""
        self.assert_true(self.model is not None, "Model should be initialized")

        # Check all parameters are initialized
        for name, param in self.model.named_parameters():
            self.assert_no_nan(param, f"Parameter {name} contains NaN")
            self.assert_no_inf(param, f"Parameter {name} contains Inf")

    def test_forward_pass(self):
        """Test forward pass works correctly."""
        batch_size = self.input_shape[0]
        x = torch.randn(self.input_shape)

        with torch.no_grad():
            output = self.model(x)

        self.assert_true(output is not None, "Output should not be None")
        self.assert_no_nan(output, "Output contains NaN")
        self.assert_no_inf(output, "Output contains Inf")

        # Check output shape
        expected_shape = (batch_size, self.num_classes)
        if output.dim() == 2:
            self.assert_equal(output.shape[0], batch_size, "Output batch size mismatch")

    def test_backward_pass(self):
        """Test backward pass (gradient computation) works."""
        self.model.train()
        x = torch.randn(self.input_shape)
        target = torch.randint(0, self.num_classes, (self.input_shape[0],))

        criterion = nn.CrossEntropyLoss()
        output = self.model(x)
        loss = criterion(output, target)
        loss.backward()

        # Check gradients exist and are not NaN
        has_grad = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                has_grad = True
                self.assert_no_nan(param.grad, f"Gradient for {name} contains NaN")
                self.assert_no_inf(param.grad, f"Gradient for {name} contains Inf")

        self.assert_true(has_grad, "Model should have trainable parameters")

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self.assert_true(total_params > 0, "Model should have parameters")
        self.assert_true(trainable_params > 0, "Model should have trainable parameters")

        return {"total_params": total_params, "trainable_params": trainable_params}

    def test_batch_independence(self):
        """Test that samples in a batch are processed independently."""
        self.model.eval()

        # Single sample
        x_single = torch.randn(1, *self.input_shape[1:])
        with torch.no_grad():
            output_single = self.model(x_single)

        # Batch with same sample repeated
        x_batch = x_single.repeat(3, 1, 1, 1)
        with torch.no_grad():
            output_batch = self.model(x_batch)

        # All outputs should be identical
        for i in range(3):
            self.assert_tensor_equal(
                output_batch[i : i + 1],
                output_single,
                message=f"Sample {i} differs from single forward",
            )

    def test_deterministic_forward(self):
        """Test that forward pass is deterministic in eval mode."""
        self.model.eval()
        x = torch.randn(self.input_shape)

        with torch.no_grad():
            output1 = self.model(x)
            output2 = self.model(x)

        self.assert_tensor_equal(
            output1, output2, message="Forward pass is not deterministic"
        )


class LayerUnitTest(UnitTest):
    """
    Unit tests for individual neural network layers.

    Tests layer initialization, forward pass, shape handling,
    and gradient flow.
    """

    def __init__(
        self,
        layer_class: Type[nn.Module],
        layer_kwargs: Dict[str, Any],
        input_shape: Tuple[int, ...] = (32, 64),
    ):
        super().__init__("LayerUnitTest", f"Unit tests for {layer_class.__name__}")
        self.layer_class = layer_class
        self.layer_kwargs = layer_kwargs
        self.input_shape = input_shape
        self.layer: Optional[nn.Module] = None

    def setup(self):
        """Initialize layer for testing."""
        self.layer = self.layer_class(**self.layer_kwargs)
        self.layer.eval()

    def teardown(self):
        """Clean up layer."""
        self.layer = None
        gc.collect()

    def run(self) -> List[TestResult]:
        """Run all layer unit tests."""
        self.results = []
        self.setup()

        tests = [
            ("test_initialization", self.test_initialization),
            ("test_forward_pass", self.test_forward_pass),
            ("test_backward_pass", self.test_backward_pass),
            ("test_shape_transformation", self.test_shape_transformation),
            ("test_gradient_flow", self.test_gradient_flow),
        ]

        for test_name, test_func in tests:
            start_time = time.time()
            try:
                test_func()
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        duration_ms=duration,
                        message="Test passed",
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                )

        self.teardown()
        return self.results

    def test_initialization(self):
        """Test layer initializes correctly."""
        self.assert_true(self.layer is not None, "Layer should be initialized")

        for name, param in self.layer.named_parameters():
            self.assert_no_nan(param, f"Parameter {name} contains NaN")

    def test_forward_pass(self):
        """Test forward pass works correctly."""
        x = torch.randn(self.input_shape)

        with torch.no_grad():
            output = self.layer(x)

        self.assert_true(output is not None, "Output should not be None")
        self.assert_no_nan(output, "Output contains NaN")

    def test_backward_pass(self):
        """Test backward pass works."""
        self.layer.train()
        x = torch.randn(self.input_shape)

        output = self.layer(x)
        loss = output.sum()
        loss.backward()

        for name, param in self.layer.named_parameters():
            if param.grad is not None:
                self.assert_no_nan(param.grad, f"Gradient contains NaN")

    def test_shape_transformation(self):
        """Test shape transformation is consistent."""
        x = torch.randn(self.input_shape)

        with torch.no_grad():
            output1 = self.layer(x)
            output2 = self.layer(x)

        self.assert_equal(output1.shape, output2.shape, "Output shape not consistent")

    def test_gradient_flow(self):
        """Test gradient flows through layer."""
        self.layer.train()
        x = torch.randn(self.input_shape, requires_grad=True)

        output = self.layer(x)
        loss = output.sum()
        loss.backward()

        self.assert_true(x.grad is not None, "Input gradient should exist")
        self.assert_no_nan(x.grad, "Input gradient contains NaN")


class LossUnitTest(UnitTest):
    """
    Unit tests for loss functions.

    Tests loss computation, gradient computation, and
    edge cases (zero predictions, extreme values).
    """

    def __init__(
        self,
        loss_class: Type[nn.Module],
        loss_kwargs: Dict[str, Any] = None,
        num_classes: int = 10,
        batch_size: int = 32,
    ):
        super().__init__("LossUnitTest", f"Unit tests for {loss_class.__name__}")
        self.loss_class = loss_class
        self.loss_kwargs = loss_kwargs or {}
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.criterion: Optional[nn.Module] = None

    def setup(self):
        """Initialize loss function."""
        self.criterion = self.loss_class(**self.loss_kwargs)

    def teardown(self):
        """Clean up."""
        self.criterion = None

    def run(self) -> List[TestResult]:
        """Run all loss unit tests."""
        self.results = []
        self.setup()

        tests = [
            ("test_loss_computation", self.test_loss_computation),
            ("test_gradient_computation", self.test_gradient_computation),
            ("test_perfect_predictions", self.test_perfect_predictions),
            ("test_zero_loss", self.test_zero_loss),
            ("test_extreme_values", self.test_extreme_values),
        ]

        for test_name, test_func in tests:
            start_time = time.time()
            try:
                test_func()
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        duration_ms=duration,
                        message="Test passed",
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                )

        self.teardown()
        return self.results

    def test_loss_computation(self):
        """Test loss computation produces valid values."""
        predictions = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        loss = self.criterion(predictions, targets)

        self.assert_true(loss.dim() == 0, "Loss should be scalar")
        self.assert_true(loss.item() >= 0, "Loss should be non-negative")
        self.assert_no_nan(loss, "Loss contains NaN")
        self.assert_no_inf(loss, "Loss contains Inf")

    def test_gradient_computation(self):
        """Test loss gradients are computed correctly."""
        predictions = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        loss = self.criterion(predictions, targets)
        loss.backward()

        self.assert_true(predictions.grad is not None, "Gradient should exist")
        self.assert_no_nan(predictions.grad, "Gradient contains NaN")

    def test_perfect_predictions(self):
        """Test loss behavior with perfect predictions."""
        # Create one-hot targets as predictions
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        predictions = torch.zeros(self.batch_size, self.num_classes)
        predictions.scatter_(1, targets.unsqueeze(1), 100.0)  # High confidence

        loss = self.criterion(predictions, targets)

        # Loss should be low (but not necessarily zero for all loss functions)
        self.assert_true(
            loss.item() < 1.0, "Loss should be low for perfect predictions"
        )

    def test_zero_loss(self):
        """Test loss reduction works correctly."""
        # Test with different reduction modes if applicable
        if "reduction" in self.loss_kwargs:
            return

        predictions = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        loss = self.criterion(predictions, targets)
        self.assert_true(
            loss.item() > 0, "Loss should be positive for random predictions"
        )

    def test_extreme_values(self):
        """Test loss handles extreme values."""
        # Very large predictions
        predictions_large = torch.randn(self.batch_size, self.num_classes) * 100
        targets = torch.randint(0, self.num_classes, (self.batch_size,))

        loss = self.criterion(predictions_large, targets)
        self.assert_no_nan(loss, "Loss contains NaN with large values")
        self.assert_no_inf(loss, "Loss contains Inf with large values")


class OptimizerUnitTest(UnitTest):
    """
    Unit tests for optimizers.

    Tests optimizer initialization, parameter updates,
    learning rate scheduling, and state management.
    """

    def __init__(
        self,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        model_class: Type[nn.Module] = None,
        model_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(
            "OptimizerUnitTest", f"Unit tests for {optimizer_class.__name__}"
        )
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.model_class = model_class or nn.Linear
        self.model_kwargs = model_kwargs or {"in_features": 10, "out_features": 5}
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def setup(self):
        """Initialize model and optimizer."""
        self.model = self.model_class(**self.model_kwargs)
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )

    def teardown(self):
        """Clean up."""
        self.model = None
        self.optimizer = None

    def run(self) -> List[TestResult]:
        """Run all optimizer unit tests."""
        self.results = []
        self.setup()

        tests = [
            ("test_initialization", self.test_initialization),
            ("test_parameter_update", self.test_parameter_update),
            ("test_state_dict", self.test_state_dict),
            ("test_zero_grad", self.test_zero_grad),
            ("test_step", self.test_step),
        ]

        for test_name, test_func in tests:
            start_time = time.time()
            try:
                test_func()
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        duration_ms=duration,
                        message="Test passed",
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                )

        self.teardown()
        return self.results

    def test_initialization(self):
        """Test optimizer initializes correctly."""
        self.assert_true(self.optimizer is not None, "Optimizer should be initialized")

        # Check param groups
        self.assert_true(
            len(self.optimizer.param_groups) > 0, "Optimizer should have param groups"
        )

    def test_parameter_update(self):
        """Test optimizer updates parameters."""
        # Get initial parameter values
        initial_params = [p.clone() for p in self.model.parameters()]

        # Forward pass
        x = torch.randn(32, self.model_kwargs.get("in_features", 10))
        output = self.model(x)
        loss = output.sum()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        # Check parameters changed
        for initial, updated in zip(initial_params, self.model.parameters()):
            if updated.grad is not None:
                diff = torch.abs(initial - updated).max().item()
                self.assert_true(diff > 0, "Parameters should be updated")

    def test_state_dict(self):
        """Test optimizer state dict save/load."""
        state_dict = self.optimizer.state_dict()

        self.assert_true(
            "param_groups" in state_dict, "State dict should contain param_groups"
        )

        # Load state dict
        self.optimizer.load_state_dict(state_dict)
        self.assert_true(True, "State dict should load successfully")

    def test_zero_grad(self):
        """Test zero_grad clears gradients."""
        x = torch.randn(32, self.model_kwargs.get("in_features", 10))
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None for p in self.model.parameters())
        self.assert_true(has_grad, "Gradients should exist after backward")

        # Zero gradients
        self.optimizer.zero_grad()

        # Check gradients are zero
        for p in self.model.parameters():
            if p.grad is not None:
                self.assert_equal(
                    p.grad.abs().max().item(), 0.0, "Gradients should be zero"
                )

    def test_step(self):
        """Test optimizer step executes without errors."""
        x = torch.randn(32, self.model_kwargs.get("in_features", 10))
        output = self.model(x)
        loss = output.sum()

        self.optimizer.zero_grad()
        loss.backward()

        try:
            self.optimizer.step()
        except Exception as e:
            raise AssertionError(f"Optimizer step failed: {e}")


def test_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (32, 3, 224, 224),
    num_classes: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quick test function for models.

    Args:
        model: Model to test
        input_shape: Input tensor shape
        num_classes: Number of output classes
        verbose: Whether to print results

    Returns:
        Dictionary with test results
    """
    tester = ModelUnitTest(type(model), {}, input_shape, num_classes)
    tester.model = model
    tester.model.eval()

    results = tester.run()

    summary = {
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.status == TestStatus.PASSED),
        "failed": sum(1 for r in results if r.status == TestStatus.FAILED),
        "duration_ms": sum(r.duration_ms for r in results),
        "results": results,
    }

    if verbose:
        print(f"Model Test Results:")
        print(f"  Total: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Duration: {summary['duration_ms']:.2f}ms")

        for result in results:
            status_icon = "✓" if result.status == TestStatus.PASSED else "✗"
            print(f"  {status_icon} {result.test_name}: {result.message}")

    return summary


# ============================================================================
# Integration Testing
# ============================================================================


class IntegrationTest(ABC):
    """Abstract base class for integration tests."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results: List[TestResult] = []

    @abstractmethod
    def setup(self):
        """Set up test environment."""
        pass

    @abstractmethod
    def teardown(self):
        """Clean up test environment."""
        pass

    @abstractmethod
    def run(self) -> List[TestResult]:
        """Run integration tests."""
        pass


class PipelineIntegrationTest(IntegrationTest):
    """
    Integration tests for ML pipelines.

    Tests data loading, preprocessing, model training,
    evaluation, and inference pipeline end-to-end.
    """

    def __init__(
        self,
        pipeline_components: Dict[str, Callable],
        test_data: Tuple[torch.Tensor, torch.Tensor],
    ):
        super().__init__("PipelineIntegrationTest", "Test complete ML pipeline")
        self.pipeline_components = pipeline_components
        self.test_data = test_data
        self.x_test, self.y_test = test_data

    def setup(self):
        """Set up pipeline components."""
        pass

    def teardown(self):
        """Clean up."""
        pass

    def run(self) -> List[TestResult]:
        """Run pipeline integration tests."""
        self.results = []

        tests = [
            ("test_data_loading", self.test_data_loading),
            ("test_preprocessing", self.test_preprocessing),
            ("test_training_step", self.test_training_step),
            ("test_inference", self.test_inference),
            ("test_pipeline_end_to_end", self.test_pipeline_end_to_end),
        ]

        for test_name, test_func in tests:
            start_time = time.time()
            try:
                test_func()
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        duration_ms=duration,
                        message="Test passed",
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                )

        return self.results

    def test_data_loading(self):
        """Test data loading works correctly."""
        if "data_loader" in self.pipeline_components:
            loader = self.pipeline_components["data_loader"]
            batch = next(iter(loader))
            assert batch is not None, "Data loader should yield batches"

    def test_preprocessing(self):
        """Test preprocessing transforms work."""
        if "preprocess" in self.pipeline_components:
            preprocess = self.pipeline_components["preprocess"]
            x_processed = preprocess(self.x_test)
            assert x_processed is not None, "Preprocessing should return result"
            assert not torch.isnan(x_processed).any(), "Preprocessing produced NaN"

    def test_training_step(self):
        """Test training step executes."""
        if "training_step" in self.pipeline_components:
            step_fn = self.pipeline_components["training_step"]
            result = step_fn(self.x_test, self.y_test)
            assert result is not None, "Training step should return result"

    def test_inference(self):
        """Test inference step executes."""
        if "inference" in self.pipeline_components:
            inference_fn = self.pipeline_components["inference"]
            predictions = inference_fn(self.x_test)
            assert predictions is not None, "Inference should return predictions"
            assert predictions.shape[0] == self.x_test.shape[0], (
                "Predictions should match batch size"
            )

    def test_pipeline_end_to_end(self):
        """Test complete pipeline execution."""
        if all(
            k in self.pipeline_components
            for k in ["data_loader", "preprocess", "training_step", "inference"]
        ):
            # Run complete pipeline
            loader = self.pipeline_components["data_loader"]
            preprocess = self.pipeline_components["preprocess"]
            train_step = self.pipeline_components["training_step"]
            inference = self.pipeline_components["inference"]

            batch = next(iter(loader))
            x, y = batch
            x = preprocess(x)
            loss = train_step(x, y)
            preds = inference(x)

            assert loss is not None and preds is not None, (
                "Pipeline should complete successfully"
            )


class EndToEndTest(IntegrationTest):
    """
    End-to-end tests for complete ML workflows.

    Tests training from scratch, model saving/loading,
    and production inference scenarios.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        num_epochs: int = 2,
    ):
        super().__init__("EndToEndTest", "End-to-end ML workflow test")
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs

    def setup(self):
        """Set up training environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def teardown(self):
        """Clean up."""
        self.model = self.model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> List[TestResult]:
        """Run end-to-end tests."""
        self.results = []
        self.setup()

        tests = [
            ("test_training_convergence", self.test_training_convergence),
            ("test_model_save_load", self.test_model_save_load),
            ("test_inference_latency", self.test_inference_latency),
            ("test_memory_usage", self.test_memory_usage),
        ]

        for test_name, test_func in tests:
            start_time = time.time()
            try:
                test_func()
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        duration_ms=duration,
                        message="Test passed",
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                )

        self.teardown()
        return self.results

    def test_training_convergence(self):
        """Test model trains and loss decreases."""
        initial_loss = None
        final_loss = None

        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(self.train_data):
                if batch_idx >= 5:  # Limit batches for speed
                    break

                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            if epoch == 0:
                initial_loss = avg_loss
            final_loss = avg_loss

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss} -> {final_loss}"
        )

    def test_model_save_load(self):
        """Test model can be saved and loaded."""
        # Save model
        state_dict = self.model.state_dict()

        # Create new model instance
        model_copy = type(self.model)()
        model_copy.load_state_dict(state_dict)

        # Test outputs match
        x = torch.randn(4, 3, 224, 224).to(self.device)

        self.model.eval()
        model_copy = model_copy.to(self.device)
        model_copy.eval()

        with torch.no_grad():
            out1 = self.model(x)
            out2 = model_copy(x)

        assert torch.allclose(out1, out2), "Loaded model should produce same outputs"

    def test_inference_latency(self):
        """Test inference meets latency requirements."""
        self.model.eval()
        x = torch.randn(1, 3, 224, 224).to(self.device)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(x)

        # Measure latency
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = self.model(x)
                times.append(time.time() - start)

        avg_latency = np.mean(times) * 1000  # Convert to ms
        p99_latency = np.percentile(times, 99) * 1000

        assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"

    def test_memory_usage(self):
        """Test memory usage is reasonable."""
        if not torch.cuda.is_available():
            return

        torch.cuda.reset_peak_memory_stats()

        self.model.train()
        for batch_idx, (x, y) in enumerate(self.train_data):
            if batch_idx >= 3:
                break

            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        assert peak_memory < 2048, f"Peak memory usage too high: {peak_memory:.2f}MB"


class APITest(IntegrationTest):
    """
    API integration tests.

    Tests REST API endpoints for model serving,
    including request/response handling, error cases,
    and rate limiting.
    """

    def __init__(self, api_client: Any, endpoints: Dict[str, Dict[str, Any]]):
        super().__init__("APITest", "API endpoint testing")
        self.api_client = api_client
        self.endpoints = endpoints

    def setup(self):
        """Set up API client."""
        pass

    def teardown(self):
        """Clean up."""
        pass

    def run(self) -> List[TestResult]:
        """Run API tests."""
        self.results = []

        for endpoint_name, endpoint_config in self.endpoints.items():
            start_time = time.time()
            try:
                self.test_endpoint(endpoint_name, endpoint_config)
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=f"test_endpoint_{endpoint_name}",
                        status=TestStatus.PASSED,
                        duration_ms=duration,
                        message="Endpoint test passed",
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=f"test_endpoint_{endpoint_name}",
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                )

        return self.results

    def test_endpoint(self, name: str, config: Dict[str, Any]):
        """Test a specific API endpoint."""
        method = config.get("method", "GET")
        url = config.get("url", "")
        data = config.get("data", None)
        expected_status = config.get("expected_status", 200)

        # This is a mock implementation - replace with actual API calls
        # response = self.api_client.request(method, url, json=data)
        # assert response.status_code == expected_status
        pass


def load_test(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    num_requests: int = 1000,
    concurrency: int = 10,
    target_latency_ms: float = 50.0,
) -> Dict[str, Any]:
    """
    Perform load testing on a model.

    Args:
        model: Model to test
        input_shape: Input tensor shape
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
        target_latency_ms: Target latency in milliseconds

    Returns:
        Dictionary with load test results
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    latencies = []
    errors = 0

    with torch.no_grad():
        for i in range(num_requests):
            x = torch.randn(*input_shape).to(device)

            start = time.time()
            try:
                _ = model(x)
                latencies.append((time.time() - start) * 1000)
            except Exception:
                errors += 1

    results = {
        "total_requests": num_requests,
        "successful_requests": len(latencies),
        "failed_requests": errors,
        "avg_latency_ms": np.mean(latencies) if latencies else 0,
        "p50_latency_ms": np.percentile(latencies, 50) if latencies else 0,
        "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
        "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
        "throughput_rps": num_requests / sum(latencies) * 1000 if latencies else 0,
        "meets_target": np.mean(latencies) < target_latency_ms if latencies else False,
    }

    return results


# ============================================================================
# Model Testing
# ============================================================================


class ModelValidationTest:
    """
    Comprehensive model validation tests.

    Validates model architecture, parameters, and behavior
    against expected specifications.
    """

    def __init__(self, model: nn.Module, expected_config: Dict[str, Any]):
        self.model = model
        self.expected_config = expected_config
        self.results: List[TestResult] = []

    def validate(self) -> List[TestResult]:
        """Run all validation tests."""
        tests = [
            ("validate_architecture", self.validate_architecture),
            ("validate_parameters", self.validate_parameters),
            ("validate_input_output", self.validate_input_output),
            ("validate_gradient_flow", self.validate_gradient_flow),
        ]

        for test_name, test_func in tests:
            start_time = time.time()
            try:
                test_func()
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        duration_ms=duration,
                        message="Validation passed",
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.results.append(
                    TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration_ms=duration,
                        message=str(e),
                    )
                )

        return self.results

    def validate_architecture(self):
        """Validate model architecture matches expected."""
        if "num_layers" in self.expected_config:
            actual_layers = sum(1 for _ in self.model.modules())
            assert actual_layers == self.expected_config["num_layers"], (
                f"Layer count mismatch: {actual_layers} vs {self.expected_config['num_layers']}"
            )

        if "expected_modules" in self.expected_config:
            module_names = [name for name, _ in self.model.named_modules()]
            for module in self.expected_config["expected_modules"]:
                assert any(module in name for name in module_names), (
                    f"Expected module {module} not found"
                )

    def validate_parameters(self):
        """Validate model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())

        if "max_params" in self.expected_config:
            assert total_params <= self.expected_config["max_params"], (
                f"Too many parameters: {total_params} > {self.expected_config['max_params']}"
            )

        if "min_params" in self.expected_config:
            assert total_params >= self.expected_config["min_params"], (
                f"Too few parameters: {total_params} < {self.expected_config['min_params']}"
            )

    def validate_input_output(self):
        """Validate model handles expected input/output shapes."""
        input_shape = self.expected_config.get("input_shape", (1, 3, 224, 224))
        output_shape = self.expected_config.get("output_shape")

        x = torch.randn(*input_shape)
        self.model.eval()

        with torch.no_grad():
            output = self.model(x)

        if output_shape:
            assert output.shape == output_shape, (
                f"Output shape mismatch: {output.shape} vs {output_shape}"
            )

    def validate_gradient_flow(self):
        """Validate gradients flow through all layers."""
        self.model.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)

        output = self.model(x)
        loss = output.sum()
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class OverfittingTest:
    """
    Tests for detecting overfitting.

    Checks if model overfits to training data by comparing
    training and validation performance.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        overfit_threshold: float = 0.1,
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.overfit_threshold = overfit_threshold
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def test(self, num_epochs: int = 10) -> Dict[str, Any]:
        """Run overfitting detection test."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for x, y in self.train_data:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += y.size(0)
                train_correct += predicted.eq(y).sum().item()

            # Validate
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x, y in self.val_data:
                    x, y = x.to(device), y.to(device)
                    output = self.model(x)
                    loss = criterion(output, y)

                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += y.size(0)
                    val_correct += predicted.eq(y).sum().item()

            self.history["train_loss"].append(train_loss / len(self.train_data))
            self.history["val_loss"].append(val_loss / len(self.val_data))
            self.history["train_acc"].append(100.0 * train_correct / train_total)
            self.history["val_acc"].append(100.0 * val_correct / val_total)

        # Analyze for overfitting
        train_val_gap = self.history["train_acc"][-1] - self.history["val_acc"][-1]
        is_overfitting = train_val_gap > self.overfit_threshold * 100

        return {
            "is_overfitting": is_overfitting,
            "train_val_gap": train_val_gap,
            "final_train_acc": self.history["train_acc"][-1],
            "final_val_acc": self.history["val_acc"][-1],
            "history": self.history,
        }


class UnderfittingTest:
    """
    Tests for detecting underfitting.

    Checks if model is too simple to capture patterns in data.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        performance_threshold: float = 0.5,
    ):
        self.model = model
        self.train_data = train_data
        self.performance_threshold = performance_threshold

    def test(self, num_epochs: int = 10) -> Dict[str, Any]:
        """Run underfitting detection test."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        accuracies = []

        for epoch in range(num_epochs):
            self.model.train()
            correct = 0
            total = 0

            for x, y in self.train_data:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                _, predicted = output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            acc = correct / total
            accuracies.append(acc)

        # Check for underfitting
        final_acc = accuracies[-1]
        is_underfitting = final_acc < self.performance_threshold

        # Check if accuracy is still improving
        is_improving = accuracies[-1] > accuracies[len(accuracies) // 2]

        return {
            "is_underfitting": is_underfitting,
            "final_accuracy": final_acc,
            "is_improving": is_improving,
            "accuracy_history": accuracies,
        }


class BiasTest:
    """
    Tests for detecting bias in model predictions.

    Checks for performance disparities across different groups.
    """

    def __init__(self, model: nn.Module, sensitive_attributes: List[str]):
        self.model = model
        self.sensitive_attributes = sensitive_attributes

    def test(
        self, data_loader: DataLoader, attribute_values: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Run bias detection tests."""
        self.model.eval()
        device = next(self.model.parameters()).device

        results = {}

        for attr in self.sensitive_attributes:
            group_metrics = {}

            for value in attribute_values.get(attr, []):
                # Filter data for this group
                group_correct = 0
                group_total = 0

                with torch.no_grad():
                    for batch in data_loader:
                        x, y, meta = batch  # Assume metadata includes sensitive attrs

                        mask = meta[attr] == value
                        if mask.any():
                            x_group = x[mask].to(device)
                            y_group = y[mask].to(device)

                            output = self.model(x_group)
                            _, predicted = output.max(1)

                            group_total += y_group.size(0)
                            group_correct += predicted.eq(y_group).sum().item()

                if group_total > 0:
                    group_metrics[value] = {
                        "accuracy": group_correct / group_total,
                        "total": group_total,
                    }

            # Calculate disparity
            if group_metrics:
                accuracies = [m["accuracy"] for m in group_metrics.values()]
                disparity = max(accuracies) - min(accuracies)

                results[attr] = {
                    "group_metrics": group_metrics,
                    "disparity": disparity,
                    "is_biased": disparity > 0.05,  # 5% threshold
                }

        return results


class RobustnessTest:
    """
    Tests for model robustness.

    Checks model performance under various perturbations
    and adversarial conditions.
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.03):
        self.model = model
        self.epsilon = epsilon

    def test(
        self, data_loader: DataLoader, test_types: List[str] = None
    ) -> Dict[str, Any]:
        """Run robustness tests."""
        test_types = test_types or ["noise", "occlusion", "adversarial"]
        self.model.eval()
        device = next(self.model.parameters()).device

        results = {}

        for test_type in test_types:
            if test_type == "noise":
                results["noise"] = self._test_noise_robustness(data_loader, device)
            elif test_type == "occlusion":
                results["occlusion"] = self._test_occlusion_robustness(
                    data_loader, device
                )
            elif test_type == "adversarial":
                results["adversarial"] = self._test_adversarial_robustness(
                    data_loader, device
                )

        return results

    def _test_noise_robustness(
        self, data_loader: DataLoader, device: torch.device
    ) -> Dict:
        """Test robustness to Gaussian noise."""
        clean_correct = 0
        noisy_correct = 0
        total = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)

                # Clean predictions
                output = self.model(x)
                _, predicted = output.max(1)
                clean_correct += predicted.eq(y).sum().item()

                # Noisy predictions
                noise = torch.randn_like(x) * self.epsilon
                output_noisy = self.model(x + noise)
                _, predicted_noisy = output_noisy.max(1)
                noisy_correct += predicted_noisy.eq(y).sum().item()

                total += y.size(0)

        clean_acc = clean_correct / total
        noisy_acc = noisy_correct / total

        return {
            "clean_accuracy": clean_acc,
            "noisy_accuracy": noisy_acc,
            "accuracy_drop": clean_acc - noisy_acc,
            "is_robust": (clean_acc - noisy_acc) < 0.1,
        }

    def _test_occlusion_robustness(
        self, data_loader: DataLoader, device: torch.device
    ) -> Dict:
        """Test robustness to occlusion."""
        clean_correct = 0
        occluded_correct = 0
        total = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)

                # Clean predictions
                output = self.model(x)
                _, predicted = output.max(1)
                clean_correct += predicted.eq(y).sum().item()

                # Occluded predictions (zero out center patch)
                x_occluded = x.clone()
                h, w = x.shape[2], x.shape[3]
                x_occluded[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0

                output_occluded = self.model(x_occluded)
                _, predicted_occluded = output_occluded.max(1)
                occluded_correct += predicted_occluded.eq(y).sum().item()

                total += y.size(0)

        clean_acc = clean_correct / total
        occluded_acc = occluded_correct / total

        return {
            "clean_accuracy": clean_acc,
            "occluded_accuracy": occluded_acc,
            "accuracy_drop": clean_acc - occluded_acc,
            "is_robust": (clean_acc - occluded_acc) < 0.15,
        }

    def _test_adversarial_robustness(
        self, data_loader: DataLoader, device: torch.device
    ) -> Dict:
        """Test robustness to FGSM adversarial attacks."""
        clean_correct = 0
        adversarial_correct = 0
        total = 0

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            x.requires_grad = True

            # Clean predictions
            output = self.model(x)
            _, predicted = output.max(1)
            clean_correct += predicted.eq(y).sum().item()

            # Generate adversarial examples (FGSM)
            loss = nn.CrossEntropyLoss()(output, y)
            self.model.zero_grad()
            loss.backward()

            x_adv = x + self.epsilon * x.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)

            # Adversarial predictions
            with torch.no_grad():
                output_adv = self.model(x_adv)
                _, predicted_adv = output_adv.max(1)
                adversarial_correct += predicted_adv.eq(y).sum().item()

            total += y.size(0)

            if total >= 1000:  # Limit samples
                break

        clean_acc = clean_correct / total
        adv_acc = adversarial_correct / total

        return {
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "accuracy_drop": clean_acc - adv_acc,
            "is_robust": (clean_acc - adv_acc) < 0.2,
        }


# ============================================================================
# Property-Based Testing
# ============================================================================


class HypothesisTest:
    """
    Property-based testing using hypothesis generation.

    Generates random inputs to test model invariants.
    """

    def __init__(self, model: nn.Module, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples
        self.properties: List[Tuple[str, Callable]] = []

    def add_property(
        self, name: str, property_fn: Callable[[torch.Tensor, torch.Tensor], bool]
    ):
        """Add a property to test."""
        self.properties.append((name, property_fn))

    def test(self, input_generator: Callable[[], torch.Tensor]) -> List[TestResult]:
        """Run hypothesis tests."""
        results = []
        self.model.eval()

        for prop_name, prop_fn in self.properties:
            violations = 0

            for i in range(self.num_samples):
                x = input_generator()

                with torch.no_grad():
                    output = self.model(x)

                if not prop_fn(x, output):
                    violations += 1

            status = TestStatus.PASSED if violations == 0 else TestStatus.FAILED
            results.append(
                TestResult(
                    test_name=f"hypothesis_{prop_name}",
                    status=status,
                    duration_ms=0,
                    message=f"Violations: {violations}/{self.num_samples}",
                )
            )

        return results


class PropertyTest:
    """
    Property-based testing with custom generators.

    Allows defining custom input generators and properties.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.generators: Dict[str, Callable[[], torch.Tensor]] = {}
        self.properties: Dict[str, List[Tuple[str, Callable]]] = {}

    def add_generator(self, name: str, generator: Callable[[], torch.Tensor]):
        """Add an input generator."""
        self.generators[name] = generator

    def add_property(
        self, name: str, property_fn: Callable, generator_name: str = "default"
    ):
        """Add a property for a specific generator."""
        if generator_name not in self.properties:
            self.properties[generator_name] = []
        self.properties[generator_name].append((name, property_fn))

    def test(self, num_samples: int = 100) -> Dict[str, List[TestResult]]:
        """Run all property tests."""
        results = {}
        self.model.eval()

        for gen_name, generator in self.generators.items():
            gen_results = []

            for prop_name, prop_fn in self.properties.get(gen_name, []):
                violations = 0

                for _ in range(num_samples):
                    x = generator()

                    with torch.no_grad():
                        output = self.model(x)

                    if not prop_fn(x, output):
                        violations += 1

                status = TestStatus.PASSED if violations == 0 else TestStatus.FAILED
                gen_results.append(
                    TestResult(
                        test_name=f"{gen_name}_{prop_name}",
                        status=status,
                        duration_ms=0,
                        message=f"Violations: {violations}/{num_samples}",
                    )
                )

            results[gen_name] = gen_results

        return results


def invariant_check(
    model: nn.Module,
    data: torch.Tensor,
    invariants: List[Callable[[torch.Tensor, torch.Tensor], bool]],
) -> Dict[str, Any]:
    """
    Check model invariants on given data.

    Args:
        model: Model to test
        data: Input data
        invariants: List of invariant functions that return True/False

    Returns:
        Dictionary with invariant check results
    """
    model.eval()

    with torch.no_grad():
        output = model(data)

    results = {}
    for i, invariant in enumerate(invariants):
        try:
            passed = invariant(data, output)
            results[f"invariant_{i}"] = {"passed": passed, "error": None}
        except Exception as e:
            results[f"invariant_{i}"] = {"passed": False, "error": str(e)}

    results["all_passed"] = all(
        r["passed"] for r in results.values() if isinstance(r, dict)
    )

    return results


# ============================================================================
# Snapshot Testing
# ============================================================================


class SnapshotTest:
    """
    Snapshot testing for model outputs.

    Captures and compares model outputs against stored snapshots.
    """

    def __init__(self, snapshot_dir: str = "./snapshots"):
        self.snapshot_dir = snapshot_dir
        self.snapshots: Dict[str, Any] = {}

    def capture(self, name: str, data: Any):
        """Capture a snapshot of data."""
        self.snapshots[name] = self._serialize(data)

    def save(self, filename: str):
        """Save snapshots to file."""
        import os

        os.makedirs(self.snapshot_dir, exist_ok=True)
        filepath = os.path.join(self.snapshot_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(self.snapshots, f)

    def load(self, filename: str):
        """Load snapshots from file."""
        import os

        filepath = os.path.join(self.snapshot_dir, filename)

        with open(filepath, "rb") as f:
            self.snapshots = pickle.load(f)

    def compare(self, name: str, data: Any, tolerance: float = 1e-5) -> bool:
        """Compare data against stored snapshot."""
        if name not in self.snapshots:
            return False

        serialized = self._serialize(data)
        expected = self.snapshots[name]

        return self._compare_values(expected, serialized, tolerance)

    def _serialize(self, data: Any) -> Any:
        """Serialize data for storage."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (int, float, str, bool)):
            return data
        elif isinstance(data, (list, tuple)):
            return [self._serialize(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._serialize(v) for k, v in data.items()}
        else:
            return str(data)

    def _compare_values(self, expected: Any, actual: Any, tolerance: float) -> bool:
        """Compare two values with tolerance."""
        if type(expected) != type(actual):
            return False

        if isinstance(expected, np.ndarray):
            return np.allclose(expected, actual, rtol=tolerance, atol=tolerance)
        elif isinstance(expected, (int, float)):
            return abs(expected - actual) < tolerance
        elif isinstance(expected, (list, tuple)):
            if len(expected) != len(actual):
                return False
            return all(
                self._compare_values(e, a, tolerance) for e, a in zip(expected, actual)
            )
        elif isinstance(expected, dict):
            if set(expected.keys()) != set(actual.keys()):
                return False
            return all(
                self._compare_values(expected[k], actual[k], tolerance)
                for k in expected.keys()
            )
        else:
            return expected == actual


class ModelSnapshot:
    """
    Snapshot testing specifically for model parameters.

    Captures model weights and architecture for regression testing.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.state_snapshot = None

    def capture(self):
        """Capture current model state."""
        self.state_snapshot = {
            "state_dict": copy.deepcopy(self.model.state_dict()),
            "architecture": str(self.model),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }

    def save(self, filepath: str):
        """Save snapshot to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.state_snapshot, f)

    def load(self, filepath: str):
        """Load snapshot from file."""
        with open(filepath, "rb") as f:
            self.state_snapshot = pickle.load(f)

    def compare(self, tolerance: float = 1e-6) -> Dict[str, Any]:
        """Compare current model against snapshot."""
        if self.state_snapshot is None:
            raise ValueError("No snapshot loaded")

        results = {
            "architecture_match": str(self.model)
            == self.state_snapshot["architecture"],
            "parameter_count_match": sum(p.numel() for p in self.model.parameters())
            == self.state_snapshot["num_parameters"],
            "state_dict_match": True,
            "differences": [],
        }

        current_state = self.model.state_dict()
        expected_state = self.state_snapshot["state_dict"]

        for key in expected_state.keys():
            if key not in current_state:
                results["state_dict_match"] = False
                results["differences"].append(f"Missing key: {key}")
                continue

            if not torch.allclose(
                current_state[key], expected_state[key], rtol=tolerance, atol=tolerance
            ):
                results["state_dict_match"] = False
                diff = (current_state[key] - expected_state[key]).abs().max().item()
                results["differences"].append(f"Mismatch in {key}: max diff {diff}")

        return results


class OutputSnapshot:
    """
    Snapshot testing for model outputs.

    Captures and compares model predictions on fixed inputs.
    """

    def __init__(self, model: nn.Module, input_data: torch.Tensor):
        self.model = model
        self.input_data = input_data
        self.output_snapshot = None

    def capture(self):
        """Capture model output on input data."""
        self.model.eval()

        with torch.no_grad():
            output = self.model(self.input_data)

        self.output_snapshot = {
            "input": self.input_data.detach().cpu().numpy(),
            "output": output.detach().cpu().numpy(),
            "input_hash": hashlib.md5(
                self.input_data.cpu().numpy().tobytes()
            ).hexdigest(),
        }

    def save(self, filepath: str):
        """Save snapshot to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.output_snapshot, f)

    def load(self, filepath: str):
        """Load snapshot from file."""
        with open(filepath, "rb") as f:
            self.output_snapshot = pickle.load(f)

    def compare(self, tolerance: float = 1e-5) -> Dict[str, Any]:
        """Compare current output against snapshot."""
        if self.output_snapshot is None:
            raise ValueError("No snapshot loaded")

        # Verify input is same
        current_hash = hashlib.md5(self.input_data.cpu().numpy().tobytes()).hexdigest()

        if current_hash != self.output_snapshot["input_hash"]:
            return {
                "input_match": False,
                "output_match": False,
                "message": "Input data has changed",
            }

        # Compare outputs
        self.model.eval()
        with torch.no_grad():
            current_output = self.model(self.input_data)

        expected_output = torch.from_numpy(self.output_snapshot["output"])

        match = torch.allclose(
            current_output.cpu(), expected_output, rtol=tolerance, atol=tolerance
        )

        diff = (
            (current_output.cpu() - expected_output).abs().max().item()
            if not match
            else 0
        )

        return {
            "input_match": True,
            "output_match": match,
            "max_difference": diff,
            "message": "Outputs match" if match else f"Max difference: {diff}",
        }


def compare_snapshot(
    current: Any, expected: Any, tolerance: float = 1e-5
) -> Dict[str, Any]:
    """
    Compare current value against expected snapshot value.

    Args:
        current: Current value
        expected: Expected snapshot value
        tolerance: Comparison tolerance

    Returns:
        Dictionary with comparison results
    """

    def compare_recursive(curr, exp, tol):
        if isinstance(exp, np.ndarray):
            return np.allclose(curr, exp, rtol=tol, atol=tol)
        elif isinstance(exp, torch.Tensor):
            return torch.allclose(curr, exp, rtol=tol, atol=tol)
        elif isinstance(exp, (int, float)):
            return abs(curr - exp) < tol
        elif isinstance(exp, (list, tuple)):
            return len(curr) == len(exp) and all(
                compare_recursive(c, e, tol) for c, e in zip(curr, exp)
            )
        elif isinstance(exp, dict):
            return curr.keys() == exp.keys() and all(
                compare_recursive(curr[k], exp[k], tol) for k in exp.keys()
            )
        else:
            return curr == exp

    try:
        match = compare_recursive(current, expected, tolerance)
        return {"match": match, "error": None}
    except Exception as e:
        return {"match": False, "error": str(e)}


# ============================================================================
# Fuzzing
# ============================================================================


class FuzzTester:
    """
    Base class for fuzz testing.

    Generates random inputs to find edge cases and crashes.
    """

    def __init__(self, model: nn.Module, num_iterations: int = 1000):
        self.model = model
        self.num_iterations = num_iterations
        self.crashes: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        """Run fuzz testing."""
        self.model.eval()

        for i in range(self.num_iterations):
            try:
                input_data = self.generate_input()
                result = self.test_input(input_data)

                if self.is_anomalous(result):
                    self.anomalies.append(
                        {"iteration": i, "input": input_data, "result": result}
                    )
            except Exception as e:
                self.crashes.append(
                    {
                        "iteration": i,
                        "input": input_data if "input_data" in locals() else None,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )

        return {
            "iterations": self.num_iterations,
            "crashes": len(self.crashes),
            "anomalies": len(self.anomalies),
            "crash_rate": len(self.crashes) / self.num_iterations,
            "anomaly_rate": len(self.anomalies) / self.num_iterations,
            "crash_details": self.crashes[:10],  # First 10 crashes
            "anomaly_details": self.anomalies[:10],  # First 10 anomalies
        }

    @abstractmethod
    def generate_input(self) -> Any:
        """Generate random input for fuzzing."""
        pass

    @abstractmethod
    def test_input(self, input_data: Any) -> Any:
        """Test model with input."""
        pass

    @abstractmethod
    def is_anomalous(self, result: Any) -> bool:
        """Check if result is anomalous."""
        pass


class InputFuzzer(FuzzTester):
    """
    Fuzz testing for model inputs.

    Generates various types of random inputs including:
    - Random noise
    - Extreme values
    - Invalid shapes
    - NaN/Inf values
    """

    def __init__(
        self, model: nn.Module, input_shape: Tuple[int, ...], num_iterations: int = 1000
    ):
        super().__init__(model, num_iterations)
        self.input_shape = input_shape
        self.tested_inputs: List[torch.Tensor] = []

    def generate_input(self) -> torch.Tensor:
        """Generate random input tensor."""
        import random

        strategy = random.choice(
            ["normal", "uniform", "extreme", "sparse", "edge_values", "invalid_shape"]
        )

        if strategy == "normal":
            return torch.randn(*self.input_shape)
        elif strategy == "uniform":
            return torch.rand(*self.input_shape)
        elif strategy == "extreme":
            return torch.randn(*self.input_shape) * 1000
        elif strategy == "sparse":
            x = torch.zeros(*self.input_shape)
            mask = torch.rand(*self.input_shape) < 0.1
            x[mask] = torch.randn(mask.sum())
            return x
        elif strategy == "edge_values":
            x = torch.randn(*self.input_shape)
            if random.random() < 0.5:
                x[0, 0] = float("nan")
            if random.random() < 0.5:
                x[0, 1] = float("inf")
            return x
        elif strategy == "invalid_shape":
            # Randomly modify shape
            shape = list(self.input_shape)
            idx = random.randint(0, len(shape) - 1)
            shape[idx] = random.randint(1, shape[idx] * 2)
            return torch.randn(*shape)

    def test_input(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Test model with input."""
        device = next(self.model.parameters()).device
        x = input_data.to(device)

        with torch.no_grad():
            output = self.model(x)

        return {
            "output_shape": output.shape,
            "has_nan": torch.isnan(output).any().item(),
            "has_inf": torch.isinf(output).any().item(),
            "output_range": (output.min().item(), output.max().item()),
        }

    def is_anomalous(self, result: Dict[str, Any]) -> bool:
        """Check for anomalous outputs."""
        return (
            result["has_nan"]
            or result["has_inf"]
            or abs(result["output_range"][0]) > 10000
            or abs(result["output_range"][1]) > 10000
        )


class ModelFuzzer(FuzzTester):
    """
    Fuzz testing for model parameters.

    Randomly modifies model weights to test robustness.
    """

    def __init__(
        self, model: nn.Module, input_shape: Tuple[int, ...], num_iterations: int = 100
    ):
        super().__init__(model, num_iterations)
        self.input_shape = input_shape
        self.original_state = copy.deepcopy(model.state_dict())
        self.fixed_input = torch.randn(*input_shape)

    def generate_input(self) -> Dict[str, Any]:
        """Generate parameter perturbations."""
        import random

        perturbations = {}
        for name, param in self.model.named_parameters():
            if random.random() < 0.1:  # 10% chance to perturb each parameter
                strategy = random.choice(["noise", "scale", "zero", "extreme"])

                if strategy == "noise":
                    perturbations[name] = param + torch.randn_like(param) * 0.1
                elif strategy == "scale":
                    perturbations[name] = param * random.uniform(0.5, 2.0)
                elif strategy == "zero":
                    perturbations[name] = torch.zeros_like(param)
                elif strategy == "extreme":
                    perturbations[name] = torch.randn_like(param) * 100

        return perturbations

    def test_input(self, perturbations: Dict[str, Any]) -> Dict[str, Any]:
        """Test model with perturbed parameters."""
        # Apply perturbations
        state_dict = self.model.state_dict()
        for name, value in perturbations.items():
            state_dict[name] = value
        self.model.load_state_dict(state_dict)

        device = next(self.model.parameters()).device
        x = self.fixed_input.to(device)

        try:
            with torch.no_grad():
                output = self.model(x)

            result = {
                "success": True,
                "has_nan": torch.isnan(output).any().item(),
                "has_inf": torch.isinf(output).any().item(),
            }
        except Exception as e:
            result = {"success": False, "error": str(e)}

        # Restore original state
        self.model.load_state_dict(self.original_state)

        return result

    def is_anomalous(self, result: Dict[str, Any]) -> bool:
        """Check for anomalous behavior."""
        return (
            not result.get("success", True)
            or result.get("has_nan", False)
            or result.get("has_inf", False)
        )


def fuzz_test(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    num_iterations: int = 100,
    test_type: str = "input",
) -> Dict[str, Any]:
    """
    Convenience function for fuzz testing.

    Args:
        model: Model to test
        input_shape: Input tensor shape
        num_iterations: Number of fuzz iterations
        test_type: 'input' or 'model'

    Returns:
        Fuzz test results
    """
    if test_type == "input":
        fuzzer = InputFuzzer(model, input_shape, num_iterations)
    elif test_type == "model":
        fuzzer = ModelFuzzer(model, input_shape, num_iterations)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return fuzzer.run()


# ============================================================================
# Coverage
# ============================================================================


class CodeCoverage:
    """
    Code coverage measurement.

    Tracks which lines of code are executed during testing.
    """

    def __init__(self):
        self.coverage_data: Dict[str, set] = {}
        self.tracer = None

    def start(self):
        """Start coverage tracking."""
        import sys

        self.tracer = sys.settrace(self._trace_calls)

    def stop(self):
        """Stop coverage tracking."""
        import sys

        sys.settrace(None)

    def _trace_calls(self, frame, event, arg):
        """Trace function calls."""
        if event == "line":
            filename = frame.f_code.co_filename
            line_no = frame.f_lineno

            if filename not in self.coverage_data:
                self.coverage_data[filename] = set()

            self.coverage_data[filename].add(line_no)

        return self._trace_calls

    def report(self) -> Dict[str, Any]:
        """Generate coverage report."""
        return {
            "files": len(self.coverage_data),
            "lines_executed": sum(len(lines) for lines in self.coverage_data.values()),
            "coverage_by_file": {
                filename: len(lines) for filename, lines in self.coverage_data.items()
            },
        }


class ModelCoverage:
    """
    Model coverage measurement.

    Tracks which parts of a model are activated during testing.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_counts: Dict[str, int] = {}
        self.layer_outputs: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List = []

    def register_hooks(self):
        """Register forward hooks on all layers."""

        def hook_fn(name):
            def hook(module, input, output):
                self.activation_counts[name] = self.activation_counts.get(name, 0) + 1

                if name not in self.layer_outputs:
                    self.layer_outputs[name] = []

                # Store output statistics
                if isinstance(output, torch.Tensor):
                    self.layer_outputs[name].append(
                        {
                            "mean": output.mean().item(),
                            "std": output.std().item(),
                            "shape": tuple(output.shape),
                        }
                    )

            return hook

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def report(self) -> Dict[str, Any]:
        """Generate coverage report."""
        total_layers = len(
            [m for m in self.model.modules() if len(list(m.children())) == 0]
        )
        activated_layers = len(self.activation_counts)

        return {
            "total_layers": total_layers,
            "activated_layers": activated_layers,
            "coverage_percentage": (activated_layers / total_layers * 100)
            if total_layers > 0
            else 0,
            "activation_counts": self.activation_counts,
            "layer_statistics": self.layer_outputs,
            "unused_layers": [
                name
                for name in dict(self.model.named_modules()).keys()
                if name not in self.activation_counts and len(name) > 0
            ],
        }


class BranchCoverage:
    """
    Branch coverage measurement for conditional logic.

    Tracks which branches of if/else statements are executed.
    """

    def __init__(self):
        self.branches: Dict[str, Dict[str, int]] = {}

    def record_branch(self, branch_id: str, taken: str):
        """Record that a branch was taken."""
        if branch_id not in self.branches:
            self.branches[branch_id] = {"true": 0, "false": 0}

        self.branches[branch_id][taken] += 1

    def report(self) -> Dict[str, Any]:
        """Generate branch coverage report."""
        total_branches = len(self.branches) * 2
        covered_branches = sum(
            1 for b in self.branches.values() for count in b.values() if count > 0
        )

        return {
            "total_branches": total_branches,
            "covered_branches": covered_branches,
            "coverage_percentage": (covered_branches / total_branches * 100)
            if total_branches > 0
            else 0,
            "branch_details": self.branches,
        }


def measure_coverage(
    model: nn.Module, test_data: DataLoader, coverage_types: List[str] = None
) -> Dict[str, Any]:
    """
    Measure different types of coverage.

    Args:
        model: Model to test
        test_data: Test data loader
        coverage_types: Types of coverage to measure ('model', 'activation')

    Returns:
        Dictionary with coverage results
    """
    coverage_types = coverage_types or ["model"]
    results = {}

    if "model" in coverage_types:
        model_coverage = ModelCoverage(model)
        model_coverage.register_hooks()

        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for x, _ in test_data:
                x = x.to(device)
                _ = model(x)

        results["model_coverage"] = model_coverage.report()
        model_coverage.remove_hooks()

    return results


# ============================================================================
# Utilities
# ============================================================================


class TestSuite:
    """
    Collection of tests to run together.
    """

    def __init__(self, name: str):
        self.name = name
        self.tests: List[Union[UnitTest, IntegrationTest]] = []

    def add_test(self, test: Union[UnitTest, IntegrationTest]):
        """Add a test to the suite."""
        self.tests.append(test)

    def run(self, stop_on_failure: bool = False) -> Dict[str, List[TestResult]]:
        """Run all tests in the suite."""
        results = {}

        for test in self.tests:
            test_results = test.run()
            results[test.name] = test_results

            if stop_on_failure:
                failed = sum(1 for r in test_results if r.status == TestStatus.FAILED)
                if failed > 0:
                    break

        return results


class TestRunner:
    """
    Test runner with reporting capabilities.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.suites: List[TestSuite] = []

    def add_suite(self, suite: TestSuite):
        """Add a test suite."""
        self.suites.append(suite)

    def run(self, stop_on_failure: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        all_results = {}
        start_time = time.time()

        for suite in self.suites:
            if self.verbose:
                print(f"\nRunning suite: {suite.name}")
                print("=" * 60)

            suite_results = suite.run(stop_on_failure)
            all_results[suite.name] = suite_results

            if self.verbose:
                for test_name, results in suite_results.items():
                    print(f"\n  {test_name}:")
                    for result in results:
                        icon = "✓" if result.status == TestStatus.PASSED else "✗"
                        print(
                            f"    {icon} {result.test_name} ({result.duration_ms:.1f}ms)"
                        )
                        if result.status == TestStatus.FAILED:
                            print(f"      Error: {result.message}")

        total_time = time.time() - start_time

        # Calculate summary
        total_tests = sum(
            len(results)
            for suite_results in all_results.values()
            for results in suite_results.values()
        )
        passed_tests = sum(
            sum(1 for r in results if r.status == TestStatus.PASSED)
            for suite_results in all_results.values()
            for results in suite_results.values()
        )
        failed_tests = sum(
            sum(1 for r in results if r.status == TestStatus.FAILED)
            for suite_results in all_results.values()
            for results in suite_results.values()
        )

        summary = {
            "total_suites": len(self.suites),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "duration_seconds": total_time,
            "results": all_results,
        }

        if self.verbose:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Total Suites: {summary['total_suites']}")
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Passed: {summary['passed']}")
            print(f"Failed: {summary['failed']}")
            print(f"Success Rate: {summary['success_rate'] * 100:.1f}%")
            print(f"Duration: {summary['duration_seconds']:.2f}s")

        return summary


def test_suite(name: str = "Default Test Suite") -> TestSuite:
    """
    Create a new test suite.

    Args:
        name: Name of the test suite

    Returns:
        New TestSuite instance
    """
    return TestSuite(name)


def run_tests(
    suite: TestSuite, verbose: bool = True, stop_on_failure: bool = False
) -> Dict[str, Any]:
    """
    Run a test suite and return results.

    Args:
        suite: Test suite to run
        verbose: Whether to print progress
        stop_on_failure: Whether to stop on first failure

    Returns:
        Test results dictionary
    """
    runner = TestRunner(verbose=verbose)
    runner.add_suite(suite)
    return runner.run(stop_on_failure)


# ============================================================================
# Convenience Functions
# ============================================================================


def quick_test(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    verbose: bool = True,
) -> bool:
    """
    Quick sanity check for a model.

    Args:
        model: Model to test
        input_shape: Input shape
        verbose: Print results

    Returns:
        True if all tests pass
    """
    results = test_model(model, input_shape, verbose=verbose)
    return results["failed"] == 0


def validate_model(model: nn.Module, validation_config: Dict[str, Any]) -> bool:
    """
    Validate model against configuration.

    Args:
        model: Model to validate
        validation_config: Validation configuration

    Returns:
        True if validation passes
    """
    validator = ModelValidationTest(model, validation_config)
    results = validator.validate()
    return all(r.status == TestStatus.PASSED for r in results)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Unit Testing
    "UnitTest",
    "ModelUnitTest",
    "LayerUnitTest",
    "LossUnitTest",
    "OptimizerUnitTest",
    "test_model",
    # Integration Testing
    "IntegrationTest",
    "PipelineIntegrationTest",
    "EndToEndTest",
    "APITest",
    "load_test",
    # Model Testing
    "ModelValidationTest",
    "OverfittingTest",
    "UnderfittingTest",
    "BiasTest",
    "RobustnessTest",
    # Property-Based Testing
    "HypothesisTest",
    "PropertyTest",
    "invariant_check",
    # Snapshot Testing
    "SnapshotTest",
    "ModelSnapshot",
    "OutputSnapshot",
    "compare_snapshot",
    # Fuzzing
    "FuzzTester",
    "InputFuzzer",
    "ModelFuzzer",
    "fuzz_test",
    # Coverage
    "CodeCoverage",
    "ModelCoverage",
    "BranchCoverage",
    "measure_coverage",
    # Utilities
    "TestSuite",
    "TestRunner",
    "test_suite",
    "run_tests",
    "TestResult",
    "TestStatus",
    # Convenience
    "quick_test",
    "validate_model",
]
