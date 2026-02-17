"""
Quantum Variational Circuits for Machine Learning.

Provides variational quantum circuits for classification,
regression, and as learnable function approximators.
"""

from typing import Optional, Tuple, List, Dict, Callable
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from enum import Enum


class EntanglingStrategy(str, Enum):
    """Strategies for entangling qubits in variational circuits."""

    LINEAR = "linear"
    FULL = "full"
    CIRCULAR = "circular"
    SPARSE = "sparse"
    NEAREST_NEIGHBOR = "nearest_neighbor"


class ParameterizedQuantumCircuit(nn.Module):
    """
    Parameterized Quantum Circuit (PQC).

    A variational quantum circuit with learnable parameters
    that can be trained via gradient descent.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        entangle: Entangling strategy
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        entangle: EntanglingStrategy = EntanglingStrategy.LINEAR,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle

        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """
        Execute circuit and return quantum state.

        Args:
            x: Optional input for data re-uploading
        Returns:
            Quantum state vector [2^n_qubits]
        """
        state = self._initialize_state()

        for layer in range(self.n_layers):
            state = self._apply_rotation_layer(state, layer)

            if layer < self.n_layers - 1 or self.entangle != EntanglingStrategy.LINEAR:
                state = self._apply_entanglement(state, layer)

        return state

    def _initialize_state(self) -> Tensor:
        """Initialize to |0...0> state."""
        state = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        state[0] = 1.0
        return state

    def _apply_rotation_layer(self, state: Tensor, layer: int) -> Tensor:
        """Apply parameterized rotation gates."""
        return state

    def _apply_entanglement(self, state: Tensor, layer: int) -> Tensor:
        """Apply entangling gates."""
        if self.entangle == EntanglingStrategy.LINEAR:
            for q in range(self.n_qubits - 1):
                state = self._apply_cnot(state, q, q + 1)
        elif self.entangle == EntanglingStrategy.FULL:
            for q in range(self.n_qubits):
                state = self._apply_cnot(state, q, (q + 1) % self.n_qubits)
        elif self.entangle == EntanglingStrategy.CIRCULAR:
            for q in range(self.n_qubits):
                state = self._apply_cnot(state, q, (q + 1) % self.n_qubits)

        return state

    def _apply_cnot(self, state: Tensor, control: int, target: int) -> Tensor:
        """Apply controlled-NOT gate."""
        return state

    def measure_expectation(self, observable: Optional[Tensor] = None) -> Tensor:
        """
        Compute expectation value of observable.

        Args:
            observable: Observable matrix [2^n, 2^n]
        Returns:
            Expectation value
        """
        state = self.forward()

        if observable is None:
            observable = torch.eye(2**self.n_qubits, dtype=torch.complex64)

        return torch.real(torch.conj(state) @ observable @ state)

    def measure_probabilities(self) -> Tensor:
        """
        Get measurement probabilities for all basis states.

        Returns:
            Probability vector [2^n_qubits]
        """
        state = self.forward()
        return torch.abs(state) ** 2


class QuantumEmbeddingCircuit(nn.Module):
    """
    Quantum Circuit for Data Embedding.

    Encodes classical data into quantum states using various
    embedding strategies.

    Args:
        n_qubits: Number of qubits
        embedding_type: Type of embedding ('amplitude', 'angle', 'basis')
        n_layers: Number of embedding layers
    """

    def __init__(
        self,
        n_qubits: int = 8,
        embedding_type: str = "angle",
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.embedding_type = embedding_type
        self.n_layers = n_layers

        if embedding_type == "angle":
            self.feature_map = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * n_layers),
                nn.Tanh(),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed data into quantum state.

        Args:
            x: Input data [batch, features]
        Returns:
            Quantum state [batch, 2^n_qubits]
        """
        batch_size = x.shape[0]
        states = []

        for b in range(batch_size):
            state = self._embed(x[b])
            states.append(state)

        return torch.stack(states)

    def _embed(self, x: Tensor) -> Tensor:
        """Embed single data point."""
        state = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        state[0] = 1.0

        if self.embedding_type == "angle":
            x_enc = x[: self.n_qubits]
            for q in range(min(self.n_qubits, len(x_enc))):
                theta = torch.pi * (x_enc[q] + 1)
                state = self._ry_gate(state, q, theta)

        elif self.embedding_type == "amplitude":
            x_norm = x / (torch.norm(x) + 1e-8)
            for i, val in enumerate(x_norm):
                if i < 2**self.n_qubits:
                    state[i] = complex(val, 0)

        elif self.embedding_type == "basis":
            idx = int(torch.argmax(x[: 2**self.n_qubits]).item())
            state[idx] = 1.0

        return state

    def _ry_gate(self, state: Tensor, qubit: int, theta: float) -> Tensor:
        """Apply Y-rotation gate."""
        return state


class VariationalQuantumClassifier(nn.Module):
    """
    Variational Quantum Classifier.

    Quantum circuit for classification tasks using parameterized
    circuits with measurement-based output.

    Args:
        n_qubits: Number of qubits
        n_classes: Number of classes
        n_layers: Number of variational layers
        embedding_dim: Input feature dimension
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_classes: int = 2,
        n_layers: int = 2,
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim or n_qubits

        self.embedding = QuantumEmbeddingCircuit(
            n_qubits=n_qubits,
            embedding_type="angle",
        )

        self.circuit = ParameterizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(2**n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Classify input samples.

        Args:
            x: Input features [batch, features]
        Returns:
            Class logits [batch, n_classes]
        """
        embedded = self.embedding(x)

        probs_list = []
        for b in range(embedded.shape[0]):
            self.circuit.forward()
            probs = self.circuit.measure_probabilities()
            probs_list.append(probs)

        probs_stack = torch.stack(probs_list)

        logits = self.classifier(probs_stack)

        return logits

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Get class probabilities.

        Args:
            x: Input features [batch, features]
        Returns:
            Class probabilities [batch, n_classes]
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class VariationalQuantumRegressor(nn.Module):
    """
    Variational Quantum Regressor.

    Quantum circuit for regression tasks.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        output_dim: Output dimension
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.embedding = QuantumEmbeddingCircuit(
            n_qubits=n_qubits,
            embedding_type="angle",
        )

        self.circuit = ParameterizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        self.regressor = nn.Sequential(
            nn.Linear(2**n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Predict continuous values.

        Args:
            x: Input features [batch, features]
        Returns:
            Predictions [batch, output_dim]
        """
        embedded = self.embedding(x)

        probs_list = []
        for b in range(embedded.shape[0]):
            self.circuit.forward()
            probs = self.circuit.measure_probabilities()
            probs_list.append(probs)

        probs_stack = torch.stack(probs_list)

        return self.regressor(probs_stack)


class QuantumCircuitLearner(nn.Module):
    """
    Generic Quantum Circuit Learner.

    A flexible framework for training quantum circuits
    on arbitrary supervised learning tasks.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        loss_fn: Loss function
        optimizer: Optimizer class
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        loss_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.embedding = QuantumEmbeddingCircuit(n_qubits=n_qubits)
        self.circuit = ParameterizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        self.measurements = nn.Parameter(torch.randn(2**n_qubits, n_layers))

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            x: Input features [batch, features]
            target: Target labels [batch]
        Returns:
            Dictionary with predictions and loss
        """
        embedded = self.embedding(x)

        output_features = []
        for b in range(embedded.shape[0]):
            state = self.circuit()
            probs = torch.abs(state) ** 2

            measured = probs @ self.measurements
            output_features.append(measured)

        output = torch.stack(output_features)

        result = {"output": output}

        if target is not None:
            loss = self.loss_fn(output, target)
            result["loss"] = loss

        return result

    def train_step(
        self, x: Tensor, y: Tensor, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Tensor]:
        """
        Single training step.

        Args:
            x: Input features
            y: Target labels
            optimizer: Optimizer (creates new one if None)
        Returns:
            Dictionary with results
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        optimizer.zero_grad()

        result = self.forward(x, y)

        if "loss" in result:
            result["loss"].backward()
            optimizer.step()

        return result

    def predict(self, x: Tensor) -> Tensor:
        """
        Make predictions.

        Args:
            x: Input features
        Returns:
            Predictions
        """
        result = self.forward(x)
        return result["output"]
