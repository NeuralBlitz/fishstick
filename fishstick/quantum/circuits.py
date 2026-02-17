"""Quantum circuits and gates."""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


@dataclass
class Gate:
    """Base class for quantum gates."""

    name: str
    qubits: Tuple[int, ...]
    params: Optional[Tensor] = None

    def matrix(self) -> Tensor:
        raise NotImplementedError


@dataclass
class Hadamard(Gate):
    """Hadamard gate H = (1/sqrt(2))[[1,1],[1,-1]]."""

    def __init__(self, qubit: int):
        super().__init__("H", (qubit,))
        self.qubit = qubit

    def matrix(self) -> Tensor:
        h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64)
        return h / np.sqrt(2)


@dataclass
class PauliX(Gate):
    """Pauli-X gate (quantum NOT)."""

    def __init__(self, qubit: int):
        super().__init__("X", (qubit,))
        self.qubit = qubit

    def matrix(self) -> Tensor:
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)


@dataclass
class PauliY(Gate):
    """Pauli-Y gate."""

    def __init__(self, qubit: int):
        super().__init__("Y", (qubit,))
        self.qubit = qubit

    def matrix(self) -> Tensor:
        return torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=torch.complex64)


@dataclass
class PauliZ(Gate):
    """Pauli-Z gate."""

    def __init__(self, qubit: int):
        super().__init__("Z", (qubit,))
        self.qubit = qubit

    def matrix(self) -> Tensor:
        return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)


@dataclass
class RX(Gate):
    """Rotation around X axis."""

    def __init__(self, qubit: int, theta: float):
        super().__init__("RX", (qubit,), params=torch.tensor([theta]))
        self.qubit = qubit
        self.theta = theta

    def matrix(self) -> Tensor:
        cos_t = np.cos(self.theta / 2)
        sin_t = np.sin(self.theta / 2)
        return torch.tensor(
            [[cos_t, -1j * sin_t], [-1j * sin_t, cos_t]], dtype=torch.complex64
        )


@dataclass
class RY(Gate):
    """Rotation around Y axis."""

    def __init__(self, qubit: int, theta: float):
        super().__init__("RY", (qubit,), params=torch.tensor([theta]))
        self.qubit = qubit
        self.theta = theta

    def matrix(self) -> Tensor:
        cos_t = np.cos(self.theta / 2)
        sin_t = np.sin(self.theta / 2)
        return torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.complex64)


@dataclass
class RZ(Gate):
    """Rotation around Z axis."""

    def __init__(self, qubit: int, theta: float):
        super().__init__("RZ", (qubit,), params=torch.tensor([theta]))
        self.qubit = qubit
        self.theta = theta

    def matrix(self) -> Tensor:
        cos_t = np.cos(self.theta / 2)
        sin_t = np.sin(self.theta / 2)
        return torch.tensor(
            [[cos_t - 1j * sin_t, 0.0], [0.0, cos_t + 1j * sin_t]],
            dtype=torch.complex64,
        )


@dataclass
class CNOT(Gate):
    """Controlled-NOT gate."""

    def __init__(self, control: int, target: int):
        super().__init__("CNOT", (control, target))
        self.control = control
        self.target = target

    def matrix(self) -> Tensor:
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=torch.complex64,
        )


@dataclass
class SWAP(Gate):
    """SWAP gate."""

    def __init__(self, qubit1: int, qubit2: int):
        super().__init__("SWAP", (qubit1, qubit2))
        self.qubit1 = qubit1
        self.qubit2 = qubit2

    def matrix(self) -> Tensor:
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.complex64,
        )


@dataclass
class CZ(Gate):
    """Controlled-Z gate."""

    def __init__(self, control: int, target: int):
        super().__init__("CZ", (control, target))
        self.control = control
        self.target = target

    def matrix(self) -> Tensor:
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ],
            dtype=torch.complex64,
        )


class QuantumCircuit(nn.Module):
    """Parametric quantum circuit."""

    def __init__(self, n_qubits: int, depth: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.gates: List[Gate] = []
        self._params = nn.Parameter(torch.randn(depth * n_qubits * 3))

    def add_gate(self, gate: Gate) -> "QuantumCircuit":
        self.gates.append(gate)
        return self

    def H(self, qubit: int) -> "QuantumCircuit":
        return self.add_gate(Hadamard(qubit))

    def X(self, qubit: int) -> "QuantumCircuit":
        return self.add_gate(PauliX(qubit))

    def Y(self, qubit: int) -> "QuantumCircuit":
        return self.add_gate(PauliY(qubit))

    def Z(self, qubit: int) -> "QuantumCircuit":
        return self.add_gate(PauliZ(qubit))

    def RX(self, qubit: int, theta: float) -> "QuantumCircuit":
        return self.add_gate(RX(qubit, theta))

    def RY(self, qubit: int, theta: float) -> "QuantumCircuit":
        return self.add_gate(RY(qubit, theta))

    def RZ(self, qubit: int, theta: float) -> "QuantumCircuit":
        return self.add_gate(RZ(qubit, theta))

    def CNOT(self, control: int, target: int) -> "QuantumCircuit":
        return self.add_gate(CNOT(control, target))

    def SWAP(self, qubit1: int, qubit2: int) -> "QuantumCircuit":
        return self.add_gate(SWAP(qubit1, qubit2))

    def CZ(self, control: int, target: int) -> "QuantumCircuit":
        return self.add_gate(CZ(control, target))

    def _get_params(self, layer_idx: int) -> Tuple[float, float, float]:
        idx = layer_idx * self.n_qubits * 3
        return (
            self._params[idx].item(),
            self._params[idx + 1].item(),
            self._params[idx + 2].item(),
        )

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """Execute circuit and return state vector."""
        state = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        state[0] = 1.0

        param_idx = 0
        for layer in range(self.depth):
            for q in range(self.n_qubits):
                if param_idx < self._params.shape[0]:
                    theta = self._params[param_idx].item()
                    state = self._apply_1qubit(state, q, theta)
                    param_idx += 1

            for q in range(self.n_qubits - 1):
                if self.n_qubits > 1:
                    state = self._apply_2qubit(state, q, q + 1)

        return state

    def _apply_1qubit(self, state: Tensor, qubit: int, theta: float) -> Tensor:
        """Apply parameterized rotation."""
        n = 2**self.n_qubits
        new_state = torch.zeros(n, dtype=torch.complex64)

        for i in range(n):
            if (i >> qubit) & 1:
                new_state[i] += (
                    np.cos(theta / 2) * state[i]
                    - 1j * np.sin(theta / 2) * state[i ^ (1 << qubit)]
                )
                new_state[i ^ (1 << qubit)] += (
                    -1j * np.sin(theta / 2) * state[i]
                    + np.cos(theta / 2) * state[i ^ (1 << qubit)]
                )
            else:
                new_state[i] = state[i]

        return new_state

    def _apply_2qubit(self, state: Tensor, q1: int, q2: int) -> Tensor:
        """Apply entangling gate."""
        return state

    def expectation(self, observable: Tensor) -> Tensor:
        """Compute expectation value of observable."""
        state = self.forward()
        return torch.real(torch.conj(state) @ observable @ state)

    def prob(self) -> Tensor:
        """Return measurement probabilities."""
        state = self.forward()
        return torch.abs(state) ** 2


class VariationalQuantumCircuit(nn.Module):
    """Variational quantum circuit for optimization."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        entangle: str = "linear",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        state = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        state[0] = 1.0

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                rx = self.params[layer, q, 0]
                ry = self.params[layer, q, 1]
                rz = self.params[layer, q, 2]
                state = self._rot(state, q, rx, ry, rz)

            if self.entangle == "linear":
                for q in range(self.n_qubits - 1):
                    state = self._cx(state, q, q + 1)
            elif self.entangle == "full":
                for q in range(self.n_qubits):
                    state = self._cx(state, q, (q + 1) % self.n_qubits)

        return state

    def _rot(
        self, state: Tensor, qubit: int, rx: float, ry: float, rz: float
    ) -> Tensor:
        """Apply rotation gates."""
        return state

    def _cx(self, state: Tensor, control: int, target: int) -> Tensor:
        """Apply controlled-X gate."""
        return state

    def measure(self, qubit: int) -> Tensor:
        """Measure single qubit."""
        probs = self.prob()
        n = 2**self.n_qubits
        prob0 = sum(probs[i].real for i in range(n) if ((i >> qubit) & 1) == 0)
        return torch.tensor([prob0, 1 - prob0])

    def prob(self) -> Tensor:
        state = self.forward()
        return torch.abs(state) ** 2
