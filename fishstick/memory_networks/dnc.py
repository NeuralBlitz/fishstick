"""
Differentiable Neural Computer (DNC) Implementation.

Based on: Hybrid Computing using a Neural Network (Graves et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class UsageGate(nn.Module):
    def __init__(self, memory_size: int, controller_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.gate_net = nn.Linear(controller_size, 1)

    def forward(self, prev_usage: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return prev_usage + (1 - prev_usage) * weights


class LinkMatrix(nn.Module):
    def __init__(self, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.link = None

    def forward(
        self,
        prev_link: Optional[torch.Tensor],
        prev_weights: torch.Tensor,
        write_weights: torch.Tensor,
        precedence_weights: torch.Tensor,
    ) -> torch.Tensor:
        if prev_link is None:
            batch_size = write_weights.size(0)
            prev_link = torch.zeros(
                batch_size,
                self.memory_size,
                self.memory_size,
                device=write_weights.device,
            )

        link = (1 - write_weights.unsqueeze(2)) * prev_link
        link = link + torch.matmul(
            write_weights.unsqueeze(2), precedence_weights.unsqueeze(1)
        )

        diag = torch.eye(self.memory_size, device=link.device).unsqueeze(0)
        link = link * (1 - diag)

        return link


class TemporalLinkage(nn.Module):
    def __init__(self, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.link_matrix = LinkMatrix(memory_size)
        self.precedence = None

    def forward(
        self,
        prev_link: Optional[torch.Tensor],
        prev_weights: torch.Tensor,
        write_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = write_weights.size(0)

        if self.precedence is None:
            self.precedence = torch.zeros(
                batch_size, self.memory_size, device=write_weights.device
            )

        precedence = (
            1 - torch.sum(write_weights, dim=1, keepdim=True)
        ) * self.precedence + write_weights

        link = self.link_matrix(prev_link, prev_weights, write_weights, precedence)

        self.precedence = precedence
        return link, precedence


class FreeGate(nn.Module):
    def __init__(self, memory_size: int, controller_size: int, num_reads: int):
        super().__init__()
        self.memory_size = memory_size
        self.num_reads = num_reads
        self.free_gate_net = nn.Linear(controller_size, num_reads)

    def forward(
        self,
        controller_output: torch.Tensor,
        read_weights: torch.Tensor,
        usage: torch.Tensor,
    ) -> torch.Tensor:
        free_gate = F.sigmoid(self.free_gate_net(controller_output))

        free_weights = free_gate.unsqueeze(2) * read_weights
        free_mem = (1 - usage) * torch.prod(1 - free_weights, dim=1)

        return free_mem


class DNCMemory(nn.Module):
    def __init__(self, memory_size: int, word_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size
        self.memory = nn.Parameter(torch.zeros(memory_size, word_size))
        nn.init.normal_(self.memory, mean=0.0, std=0.5)

    def read(self, weights: torch.Tensor) -> torch.Tensor:
        return torch.matmul(weights.unsqueeze(1), self.memory).squeeze(1)

    def write(self, weights: torch.Tensor, erase: torch.Tensor, add: torch.Tensor):
        erase_matrix = torch.ger(weights.squeeze(), erase.squeeze())
        add_matrix = torch.ger(weights.squeeze(), add.squeeze())
        self.memory.data = self.memory.data * (1 - erase_matrix) + add_matrix


class DNCAddress(nn.Module):
    def __init__(
        self, memory_size: int, word_size: int, controller_size: int, num_reads: int
    ):
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads

        self.key_net = nn.Linear(controller_size, word_size)
        self.beta_net = nn.Linear(controller_size, 1)
        self.shift_net = nn.Linear(controller_size, 3)
        self.gate_net = nn.Linear(controller_size, 1)

        self.usage_gate = UsageGate(memory_size, controller_size)
        self.temporal_linkage = TemporalLinkage(memory_size)
        self.free_gate = FreeGate(memory_size, controller_size, num_reads)

    def content_addressing(
        self, memory: torch.Tensor, key: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        similarity = torch.matmul(memory, key.unsqueeze(2)).squeeze(2)
        return F.softmax(similarity * beta, dim=-1)

    def address(
        self,
        controller_output: torch.Tensor,
        prev_read_weights: torch.Tensor,
        prev_write_weight: torch.Tensor,
        memory: torch.Tensor,
        usage: torch.Tensor,
        link: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = F.tanh(self.key_net(controller_output))
        beta = F.softplus(self.beta_net(controller_output)).squeeze(-1)

        content_weights = self.content_addressing(memory, key, beta)

        shift = F.softmax(self.shift_net(controller_output), dim=-1)
        shift_matrix = self._create_shift_matrix(shift, self.memory_size)

        directional_weights = torch.matmul(
            prev_read_weights.unsqueeze(1), shift_matrix
        ).squeeze(1)

        allocation_weights = self._allocation(usage)

        write_gate = F.sigmoid(self.gate_net(controller_output)).squeeze(-1)

        write_weights = write_gate * (
            0.5 * content_weights + 0.5 * directional_weights * allocation_weights
        )

        return write_weights, content_weights, directional_weights, allocation_weights

    def _create_shift_matrix(
        self, shift: torch.Tensor, memory_size: int
    ) -> torch.Tensor:
        batch_size = shift.size(0)
        device = shift.device

        indices = torch.arange(memory_size, device=device).float()

        shift_centered = shift[:, 0] * -1 + shift[:, 1] * 0 + shift[:, 2] * 1
        shift_centered = shift_centered.unsqueeze(1)

        distance = indices.unsqueeze(0).unsqueeze(2) - indices.unsqueeze(0).unsqueeze(1)
        distance = distance + memory_size / 2
        distance = distance % memory_size

        kernel = torch.exp(-distance.pow(2) / (2 * shift_centered.pow(2) + 1e-8))
        kernel = kernel / (kernel.sum(dim=2, keepdim=True) + 1e-8)

        return kernel

    def _allocation(self, usage: torch.Tensor) -> torch.Tensor:
        sorted_usage, indices = torch.sort(usage, dim=-1)
        allocation_weights = torch.zeros_like(usage)

        cumprod = torch.cumprod(1 - sorted_usage, dim=-1)
        allocation_weights.scatter_(-1, indices, cumprod)

        return allocation_weights


class AccessModule(nn.Module):
    def __init__(
        self, memory_size: int, word_size: int, controller_size: int, num_reads: int
    ):
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads

        self.memory = DNCMemory(memory_size, word_size)

        self.read_heads = nn.ModuleList(
            [
                DNCAddress(memory_size, word_size, controller_size, num_reads)
                for _ in range(num_reads)
            ]
        )

        self.write_head = DNCAddress(memory_size, word_size, controller_size, num_reads)

        self.erase_net = nn.Linear(controller_size, word_size)
        self.add_net = nn.Linear(controller_size, word_size)

        self.usage = None
        self.link = None

    def init_state(self, batch_size: int, device: torch.device):
        self.usage = torch.zeros(batch_size, self.memory_size, device=device)
        self.link = None

    def forward(
        self, controller_output: torch.Tensor, prev_read_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        if self.usage is None:
            self.init_state(controller_output.size(0), controller_output.device)

        write_weights, _, _, _ = self.write_head.address(
            controller_output,
            prev_read_weights,
            torch.zeros_like(prev_read_weights[:, 0]),
            self.memory.memory,
            self.usage,
            self.link,
        )

        erase = F.sigmoid(self.erase_net(controller_output))
        add = F.tanh(self.add_net(controller_output))

        self.memory.write(write_weights, erase, add)

        self.link, _ = self.temporal_linkage(
            self.link, prev_read_weights, write_weights
        )

        self.usage = self.usage_gate(self.usage, write_weights)

        read_weights_list = []
        read_vectors_list = []

        for i, head in enumerate(self.read_heads):
            read_weights, _, _, _ = head.address(
                controller_output,
                prev_read_weights,
                write_weights,
                self.memory.memory,
                self.usage,
                self.link,
            )
            read_weights_list.append(read_weights)
            read_vectors_list.append(self.memory.read(read_weights))

        read_weights = torch.stack(read_weights_list, dim=1)
        read_vectors = torch.stack(read_vectors_list, dim=1)

        state = {
            "usage": self.usage,
            "link": self.link,
            "write_weights": write_weights,
            "read_weights": read_weights,
        }

        return read_vectors, state

    def usage_gate(
        self, prev_usage: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return prev_usage + (1 - prev_usage) * weights


class DNCCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        memory_size: int = 256,
        word_size: int = 64,
        num_reads: int = 4,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads

        self.controller = nn.LSTMCell(input_size + word_size * num_reads, hidden_size)

        self.access = AccessModule(memory_size, word_size, hidden_size, num_reads)

        self.output_net = nn.Linear(hidden_size + word_size * num_reads, output_size)

        self.h_state = None
        self.c_state = None
        self.prev_read_weights = None

    def init_state(self, batch_size: int, device: torch.device):
        self.h_state = torch.zeros(batch_size, self.hidden_size, device=device)
        self.c_state = torch.zeros(batch_size, self.hidden_size, device=device)
        self.prev_read_weights = torch.zeros(
            batch_size, self.num_reads, self.memory_size, device=device
        )
        self.access.init_state(batch_size, device)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.h_state is None:
            self.init_state(x.size(0), x.device)

        prev_read_vectors = self.access.memory.read(self.prev_read_weights)
        prev_read_flat = prev_read_vectors.view(x.size(0), -1)

        controller_input = torch.cat([x, prev_read_flat], dim=-1)

        self.h_state, self.c_state = self.controller(
            controller_input, (self.h_state, self.c_state)
        )

        read_vectors, access_state = self.access(self.h_state, self.prev_read_weights)

        self.prev_read_weights = access_state["read_weights"]

        output_input = torch.cat(
            [self.h_state, read_vectors.view(x.size(0), -1)], dim=-1
        )
        output = self.output_net(output_input)

        return output, (self.h_state, self.c_state)


class DifferentiableNeuralComputer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        memory_size: int = 256,
        word_size: int = 64,
        num_reads: int = 4,
    ):
        super().__init__()
        self.dnc_cell = DNCCell(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
            word_size=word_size,
            num_reads=num_reads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.dnc_cell(x)
        return output

    def reset(self):
        self.dnc_cell.h_state = None
        self.dnc_cell.c_state = None
        self.dnc_cell.prev_read_weights = None
