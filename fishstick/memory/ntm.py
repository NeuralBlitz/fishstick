"""
Neural Turing Machine Implementation.

Implements a differentiable external memory system with:
- Memory matrix for storing information
- Read heads with content-based addressing
- Write heads with content-based and location-based addressing
- Sharp, soft, and lookup-free attention mechanisms

Based on: Neural Turing Machines (Graves et al., 2014)
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ContentAddressing(nn.Module):
    """
    Content-based memory addressing.

    Reads from memory locations based on similarity with key vector.
    """

    def __init__(self, memory_size: int, key_size: int, temperature: float = 1.0):
        super().__init__()
        self.memory_size = memory_size
        self.key_size = key_size
        self.temperature = temperature

        self.key_projection = nn.Linear(key_size, key_size)
        nn.init.xavier_uniform_(self.key_projection.weight)

    def forward(
        self,
        memory: Tensor,
        key: Tensor,
        beta: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Content-based addressing.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            key: Key vector [batch, key_size] or [batch, item_size]
            beta: Sharpness parameter [batch] (optional)

        Returns:
            attention_weights: Weights over memory locations [batch, memory_size]
        """
        if key.size(-1) != memory.size(-1):
            key = self.key_projection(key)

        key = key.unsqueeze(1)

        similarity = torch.sum(memory * key, dim=-1)

        if beta is None:
            beta = torch.ones(memory.size(0), device=memory.device)
        beta = beta.unsqueeze(-1)

        weights = F.softmax(beta * similarity / self.temperature, dim=-1)

        return weights


class LocationAddressing(nn.Module):
    """
    Location-based memory addressing.

    Enables iterative navigation through memory locations.
    """

    def __init__(
        self,
        memory_size: int,
        interpolation_strength: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.interpolation_strength = interpolation_strength

    def forward(
        self,
        prev_weights: Tensor,
        shift: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        """
        Location-based addressing with shift and sharpening.

        Args:
            prev_weights: Previous attention weights [batch, memory_size]
            shift: Shift vector [batch, shift_range] (e.g., [-1, 0, 1])
            gamma: Sharpening factor [batch]

        Returns:
            new_weights: Updated weights [batch, memory_size]
        """
        batch_size = prev_weights.size(0)
        shift_range = shift.size(-1)

        shifted_weights = self._circular_convolution(prev_weights, shift)

        interpolation = torch.sigmoid(self.interpolation_strength)
        weights = interpolation * shifted_weights + (1 - interpolation) * prev_weights

        weights = weights ** gamma.unsqueeze(-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights

    def _circular_convolution(self, weights: Tensor, shift: Tensor) -> Tensor:
        """Circular convolution for shift operation."""
        batch_size, memory_size = weights.shape
        shift_range = shift.size(-1)

        shift_center = shift_range // 2

        result = torch.zeros_like(weights)

        for i in range(shift_range):
            shift_amount = i - shift_center
            if shift_amount >= 0:
                result += shift[:, i : i + 1] * torch.roll(
                    weights, shifts=shift_amount, dims=-1
                )
            else:
                result += shift[:, i : i + 1] * torch.roll(
                    weights, shifts=shift_amount, dims=-1
                )

        return result


class ReadHead(nn.Module):
    """
    Read head for Neural Turing Machine.

    Reads content from memory based on attention weights.
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        controller_size: int,
        addressing_type: str = "content",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.controller_size = controller_size
        self.addressing_type = addressing_type
        self.temperature = temperature

        self.content_addressing = ContentAddressing(memory_size, item_size, temperature)

        self.key_net = nn.Linear(controller_size, item_size)
        self.beta_net = nn.Linear(controller_size, 1)

        if addressing_type == "location":
            self.location_addressing = LocationAddressing(memory_size)
            self.shift_net = nn.Linear(controller_size, 3)
            self.gamma_net = nn.Linear(controller_size, 1)

        self.read_vector_size = item_size

    def forward(
        self,
        memory: Tensor,
        prev_weights: Optional[Tensor],
        controller_output: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Read from memory.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            prev_weights: Previous attention weights [batch, memory_size]
            controller_output: Controller output [batch, controller_size]

        Returns:
            read_vector: Read content [batch, item_size]
            weights: Attention weights [batch, memory_size]
            info: Debug information
        """
        key = torch.tanh(self.key_net(controller_output))
        beta = F.softplus(self.beta_net(controller_output)).squeeze(-1) + 1e-3

        weights = self.content_addressing(memory, key, beta)

        if self.addressing_type == "location" and prev_weights is not None:
            shift = F.softmax(self.shift_net(controller_output), dim=-1)
            gamma = F.softplus(self.gamma_net(controller_output)).squeeze(-1) + 1e-3

            weights = self.location_addressing(prev_weights, shift, gamma)

        read_vector = torch.einsum("bm,bmi->bi", weights, memory)

        info = {
            "key": key,
            "beta": beta,
            "weights": weights,
        }

        if self.addressing_type == "location":
            info["shift"] = shift
            info["gamma"] = gamma

        return read_vector, weights, info


class WriteHead(nn.Module):
    """
    Write head for Neural Turing Machine.

    Writes content to memory with erase and add operations.
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        controller_size: int,
        addressing_type: str = "content",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.controller_size = controller_size
        self.addressing_type = addressing_type
        self.temperature = temperature

        self.content_addressing = ContentAddressing(memory_size, item_size, temperature)

        self.key_net = nn.Linear(controller_size, item_size)
        self.beta_net = nn.Linear(controller_size, 1)
        self.erase_net = nn.Linear(controller_size, item_size)
        self.add_net = nn.Linear(controller_size, item_size)

        if addressing_type == "location":
            self.location_addressing = LocationAddressing(memory_size)
            self.shift_net = nn.Linear(controller_size, 3)
            self.gamma_net = nn.Linear(controller_size, 1)

    def forward(
        self,
        memory: Tensor,
        prev_weights: Optional[Tensor],
        controller_output: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Write to memory.

        Args:
            memory: Current memory [batch, memory_size, item_size]
            prev_weights: Previous attention weights [batch, memory_size]
            controller_output: Controller output [batch, controller_size]

        Returns:
            memory: Updated memory [batch, memory_size, item_size]
            weights: Attention weights [batch, memory_size]
            info: Debug information
        """
        key = torch.tanh(self.key_net(controller_output))
        beta = F.softplus(self.beta_net(controller_output)).squeeze(-1) + 1e-3

        weights = self.content_addressing(memory, key, beta)

        if self.addressing_type == "location" and prev_weights is not None:
            shift = F.softmax(self.shift_net(controller_output), dim=-1)
            gamma = F.softplus(self.gamma_net(controller_output)).squeeze(-1) + 1e-3

            weights = self.location_addressing(prev_weights, shift, gamma)

        erase_vector = torch.sigmoid(self.erase_net(controller_output))
        add_vector = torch.tanh(self.add_net(controller_output))

        erase = weights.unsqueeze(-1) * erase_vector.unsqueeze(1)
        add = weights.unsqueeze(-1) * add_vector.unsqueeze(1)

        new_memory = memory * (1 - erase) + add

        info = {
            "key": key,
            "beta": beta,
            "weights": weights,
            "erase": erase_vector,
            "add": add_vector,
        }

        if self.addressing_type == "location":
            info["shift"] = shift
            info["gamma"] = gamma

        return new_memory, weights, info


class NTMController(nn.Module):
    """
    Controller for Neural Turing Machine.

    Processes inputs and generates outputs using both external memory
    and internal state.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        controller_size: int,
        num_read_heads: int,
        memory_read_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size
        self.num_read_heads = num_read_heads
        self.memory_read_size = memory_read_size

        self.input_net = nn.Linear(input_size, controller_size)
        self.prev_output_net = nn.Linear(controller_size, controller_size)
        self.read_net = nn.Linear(memory_read_size * num_read_heads, controller_size)

        self.output_net = nn.Linear(controller_size, output_size)

        self.init_state = nn.Parameter(torch.zeros(controller_size))

    def forward(
        self,
        x: Tensor,
        prev_output: Optional[Tensor],
        read_vectors: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Process input with memory context.

        Args:
            x: Input [batch, input_size]
            prev_output: Previous output [batch, controller_size]
            read_vectors: List of read vectors from each head

        Returns:
            output: Controller output [batch, output_size]
            state: New internal state [batch, controller_size]
        """
        input_feat = self.input_net(x)

        if prev_output is None:
            prev_output = self.init_state.expand(x.size(0), -1)
        prev_feat = self.prev_output_net(prev_output)

        if read_vectors:
            read_feat = self.read_net(torch.cat(read_vectors, dim=-1))
        else:
            read_feat = torch.zeros_like(input_feat)

        state = input_feat + prev_feat + read_feat
        state = torch.tanh(state)

        output = self.output_net(state)

        return output, state


class NeuralTuringMachine(nn.Module):
    """
    Complete Neural Turing Machine.

    Combines controller, memory matrix, and read/write heads.
    Supports multiple read/write heads and various addressing schemes.

    Attributes:
        memory_size: Number of memory locations
        item_size: Size of each memory location
        num_read_heads: Number of read heads
        num_write_heads: Number of write heads
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        memory_size: int = 128,
        item_size: int = 64,
        controller_size: int = 256,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
        addressing_type: str = "content",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.item_size = item_size
        self.controller_size = controller_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.addressing_type = addressing_type
        self.temperature = temperature

        self.memory = nn.Parameter(torch.randn(memory_size, item_size) * 0.01)

        self.read_heads = nn.ModuleList(
            [
                ReadHead(
                    memory_size,
                    item_size,
                    controller_size,
                    addressing_type,
                    temperature,
                )
                for _ in range(num_read_heads)
            ]
        )

        self.write_heads = nn.ModuleList(
            [
                WriteHead(
                    memory_size,
                    item_size,
                    controller_size,
                    addressing_type,
                    temperature,
                )
                for _ in range(num_write_heads)
            ]
        )

        self.controller = NTMController(
            input_size, output_size, controller_size, num_read_heads, item_size
        )

        self.prev_weights = None
        self.prev_output = None

        self._initialize_memory()

    def _initialize_memory(self) -> None:
        """Initialize memory to small random values."""
        nn.init.xavier_uniform_(self.memory)

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset NTM state for new sequence."""
        self.memory.data = torch.randn_like(self.memory) * 0.01

        self.prev_weights = None
        self.prev_output = None

    def forward(
        self,
        x: Tensor,
        reset: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Process input through NTM.

        Args:
            x: Input [batch, input_size]
            reset: Whether to reset memory state

        Returns:
            output: Model output [batch, output_size]
            info: Dictionary with memory state and debug info
        """
        if reset:
            self.reset(x.size(0), x.device)

        batch_size = x.size(0)

        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        read_vectors = []
        weights_list = []

        controller_output, controller_state = self.controller(x, self.prev_output, [])

        for read_head in self.read_heads:
            read_vec, weights, info = read_head(
                memory, self.prev_weights, controller_state
            )
            read_vectors.append(read_vec)
            weights_list.append(weights)

        controller_output, controller_state = self.controller(
            x, self.prev_output, read_vectors
        )

        for write_head in self.write_heads:
            memory, weights, info = write_head(
                memory, self.prev_weights, controller_state
            )
            weights_list.append(weights)

        self.memory.data = memory[0]
        self.prev_weights = weights_list[-1]
        self.prev_output = controller_state

        info = {
            "read_vectors": read_vectors,
            "weights": weights_list,
            "memory": memory,
            "state": controller_state,
        }

        return controller_output, info


class LookupFreeNTM(nn.Module):
    """
    Lookup-free Neural Turing Machine variant.

    Uses continuous attention without discrete lookup operations.
    More memory efficient for large memory sizes.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        memory_size: int = 64,
        item_size: int = 64,
        controller_size: int = 128,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.item_size = item_size

        self.memory = nn.Parameter(torch.randn(memory_size, item_size) * 0.01)

        self.key_net = nn.Linear(controller_size, item_size)
        self.strength_net = nn.Linear(controller_size, 1)

        self.erase_net = nn.Linear(controller_size, item_size)
        self.add_net = nn.Linear(controller_size, item_size)

        self.controller = nn.Sequential(
            nn.Linear(input_size + item_size, controller_size),
            nn.Tanh(),
            nn.Linear(controller_size, controller_size),
            nn.Tanh(),
        )

        self.output_net = nn.Linear(controller_size + item_size, output_size)

        self.controller_state = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with lookup-free memory access."""
        batch_size = x.size(0)

        if self.controller_state is None:
            self.controller_state = torch.zeros(
                batch_size, self.item_size, device=x.device
            )

        x_with_read = torch.cat([x, self.controller_state], dim=-1)

        controller_h = self.controller(x_with_read)

        key = torch.tanh(self.key_net(controller_h))
        strength = F.softplus(self.strength_net(controller_h)).squeeze(-1) + 1e-3

        key_expanded = key.unsqueeze(1)
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        similarity = torch.sum(memory_expanded * key_expanded, dim=-1)
        attention = F.softmax(strength * similarity, dim=-1)

        read_vector = torch.einsum("bm,bmi->bi", attention, memory_expanded)

        erase = torch.sigmoid(self.erase_net(controller_h))
        add = torch.tanh(self.add_net(controller_h))

        erase_matrix = attention.unsqueeze(-1) * erase.unsqueeze(1)
        add_matrix = attention.unsqueeze(-1) * add.unsqueeze(1)

        memory_expanded = memory_expanded * (1 - erase_matrix) + add_matrix

        self.memory.data = memory_expanded[0].detach()

        output_input = torch.cat([controller_h, read_vector], dim=-1)
        output = self.output_net(output_input)

        self.controller_state = read_vector.detach()

        info = {
            "key": key,
            "strength": strength,
            "attention": attention,
            "read_vector": read_vector,
            "memory": memory_expanded,
        }

        return output, info


def create_ntm(
    input_size: int,
    output_size: int,
    memory_size: int = 128,
    item_size: int = 64,
    num_read_heads: int = 1,
    num_write_heads: int = 1,
    **kwargs,
) -> NeuralTuringMachine:
    """Factory function to create NTM model."""
    return NeuralTuringMachine(
        input_size=input_size,
        output_size=output_size,
        memory_size=memory_size,
        item_size=item_size,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads,
        **kwargs,
    )
