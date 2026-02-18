"""
Base Classes for Neural Operators.

Abstract base classes and interfaces for neural operator implementations.
Provides common APIs and utilities for all operator types.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any, Dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


@dataclass
class OperatorConfig:
    """Configuration for neural operators."""

    in_channels: int
    out_channels: int
    hidden_dim: int = 64
    num_layers: int = 4
    activation: str = "gelu"
    dropout: float = 0.0
    use_norm: bool = True
    bias: bool = True


class BaseNeuralOperator(nn.Module, ABC):
    """Abstract base class for all neural operators.

    All neural operator implementations should inherit from this class
    and implement the forward method.
    """

    def __init__(self, config: OperatorConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the operator.

        Args:
            x: Input tensor

        Returns:
            Output tensor after applying the operator
        """
        pass

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


class FunctionToFunctionOperator(BaseNeuralOperator, ABC):
    """Base class for operators that map functions to functions.

    These operators learn mappings between infinite-dimensional
    function spaces, such as operators for solving PDEs.
    """

    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.spatial_dim: Optional[int] = None

    @abstractmethod
    def evaluate_at_points(
        self,
        function_values: Tensor,
        query_points: Tensor,
    ) -> Tensor:
        """Evaluate the operator at specific query points.

        Args:
            function_values: Values of input function at sensor locations
            query_points: Locations where to evaluate the output

        Returns:
            Operator output at query points
        """
        pass


class OperatorOutput:
    """Container for operator output with additional metadata."""

    def __init__(
        self,
        values: Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.values = values
        self.metadata = metadata or {}

    def __getitem__(self, key: str) -> Any:
        return self.metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.metadata[key] = value


class TimeSeriesOperator(BaseNeuralOperator, ABC):
    """Base class for operators on time series / sequential data."""

    def __init__(
        self,
        config: OperatorConfig,
        sequence_length: Optional[int] = None,
    ):
        super().__init__(config)
        self.sequence_length = sequence_length

    @abstractmethod
    def forward_sequence(
        self,
        x: Tensor,
        times: Optional[Tensor] = None,
    ) -> Tensor:
        """Process entire sequence.

        Args:
            x: Input sequence [batch, seq_len, features]
            times: Time stamps [batch, seq_len]

        Returns:
            Output sequence
        """
        pass


class OperatorLoss(nn.Module):
    """Base class for operator losses."""

    def __init__(self):
        super().__init__()

    def forward(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute loss between predictions and targets.

        Args:
            predictions: Predicted values
            targets: Ground truth values
            **kwargs: Additional loss-specific arguments

        Returns:
            Loss value
        """
        raise NotImplementedError


class OperatorL2Loss(OperatorLoss):
    """L2 loss for operator learning."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.mse_loss(predictions, targets, reduction=self.reduction)


class OperatorRelativeLoss(OperatorLoss):
    """Relative L2 loss for operator learning."""

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        diff = predictions - targets
        relative = (diff**2).sum(dim=-1) / ((targets**2).sum(dim=-1) + self.epsilon)
        return relative.mean()


class OperatorValidationMonitor:
    """Monitor for tracking operator training progress."""

    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss: float = float("inf")
        self.best_model_state: Optional[Dict[str, Tensor]] = None

    def update(
        self,
        train_loss: float,
        val_loss: float,
        model: nn.Module,
    ) -> None:
        """Update metrics with new loss values."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }

    def get_best_model(self) -> Optional[Dict[str, Tensor]]:
        """Get the best model state based on validation loss."""
        return self.best_model_state


class OperatorDataset(torch.utils.data.Dataset):
    """Base dataset class for operator learning."""

    def __init__(
        self,
        input_functions: List[Tensor],
        output_functions: List[Tensor],
    ):
        self.input_functions = input_functions
        self.output_functions = output_functions

        assert len(input_functions) == len(output_functions), (
            "Input and output functions must have same length"
        )

    def __len__(self) -> int:
        return len(self.input_functions)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.input_functions[idx], self.output_functions[idx]


class Collator:
    """Custom collator for batching operator data."""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        inputs, targets = zip(*batch)

        max_len = max(inp.size(-1) for inp in inputs)

        padded_inputs = []
        for inp in inputs:
            if inp.size(-1) < max_len:
                padded = torch.full(
                    (*inp.shape[:-1], max_len),
                    self.pad_value,
                    dtype=inp.dtype,
                    device=inp.device,
                )
                padded[..., : inp.size(-1)] = inp
                padded_inputs.append(padded)
            else:
                padded_inputs.append(inp)

        inputs_tensor = torch.stack(padded_inputs)
        targets_tensor = torch.stack(targets)

        return inputs_tensor, targets_tensor


class OperatorTrainer:
    """Generic trainer for neural operators."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: OperatorLoss,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or torch.device("cpu")

        self.model.to(self.device)
        self.monitor = OperatorValidationMonitor()

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.monitor.update(train_loss, val_loss, self.model)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
                )

        return history


class FourierFeatures(nn.Module):
    """Random Fourier features for kernel approximation."""

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.scale = scale

        self.register_buffer(
            "weights",
            torch.randn(num_features // 2, input_dim) * scale * 2 * np.pi,
        )
        self.register_buffer(
            "bias",
            torch.rand(num_features // 2) * 2 * np.pi,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_proj = torch.matmul(x, self.weights.T) + self.bias
        features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return features


import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for spatial/temporal coordinates."""

    def __init__(
        self,
        num_frequencies: int = 16,
        include_original: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_original = include_original

        freq_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: Tensor) -> Tensor:
        encoded = []

        if self.include_original:
            encoded.append(x)

        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))

        return torch.cat(encoded, dim=-1)


class DomainTransformer:
    """Transform between physical and spectral domains."""

    @staticmethod
    def to_fourier(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor]:
        """Transform to Fourier domain.

        Returns:
            Real and imaginary parts
        """
        x_ft = torch.fft.rfft(x, dim=dim)
        return x_ft.real, x_ft.imag

    @staticmethod
    def from_fourier(
        real: Tensor,
        imag: Tensor,
        dim: int = -1,
        n: Optional[int] = None,
    ) -> Tensor:
        """Transform from Fourier domain."""
        x_ft = torch.complex(real, imag)
        return torch.fft.irfft(x_ft, dim=dim, n=n)

    @staticmethod
    def get_frequencies(
        num_points: int,
        domain_length: float = 2 * np.pi,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Get frequency values for FFT."""
        freqs = torch.fft.rfftfreq(
            num_points, d=domain_length / num_points, device=device
        )
        return freqs


class IntegralTransform(nn.Module):
    """Learnable integral transform layer."""

    def __init__(
        self,
        input_dim: int,
        num_basis: int = 32,
        kernel_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_basis = num_basis
        self.kernel_scale = kernel_scale

        self.kernel = nn.Parameter(torch.randn(input_dim, num_basis) * kernel_scale)

    def forward(
        self,
        x: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        if weights is None:
            weights = torch.softmax(self.kernel, dim=-1)

        return torch.matmul(x, weights)


class KernelIntegration(nn.Module):
    """Kernel integration for operator approximation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_centers: int = 64,
        kernel_type: str = "gaussian",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.kernel_type = kernel_type

        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.weights = nn.Parameter(torch.randn(num_centers, output_dim))

    def _gaussian_kernel(self, x: Tensor) -> Tensor:
        x_exp = x.unsqueeze(1)
        c_exp = self.centers.unsqueeze(0)
        dist = ((x_exp - c_exp) ** 2).sum(dim=-1)
        return torch.exp(-dist / 2)

    def forward(self, x: Tensor) -> Tensor:
        if self.kernel_type == "gaussian":
            kernel = self._gaussian_kernel(x)
        else:
            kernel = torch.ones(x.size(0), self.num_centers, device=x.device)

        return torch.matmul(kernel, self.weights)


__all__ = [
    "OperatorConfig",
    "BaseNeuralOperator",
    "FunctionToFunctionOperator",
    "TimeSeriesOperator",
    "OperatorOutput",
    "OperatorLoss",
    "OperatorL2Loss",
    "OperatorRelativeLoss",
    "OperatorValidationMonitor",
    "OperatorDataset",
    "Collator",
    "OperatorTrainer",
    "FourierFeatures",
    "PositionalEncoding",
    "DomainTransformer",
    "IntegralTransform",
    "KernelIntegration",
]
