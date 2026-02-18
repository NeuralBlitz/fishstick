"""
Neural Turing Machine (NTM) Implementation.

Based on: Neural Turing Machines (Graves et al., 2014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NTMMemory(nn.Module):
    def __init__(self, memory_size: int, word_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size
        self.memory = nn.Parameter(torch.zeros(memory_size, word_size))
        nn.init.normal_(self.memory, mean=0.0, std=0.5)

    def read(self, weights: torch.Tensor) -> torch.Tensor:
        return torch.matmul(weights.unsqueeze(1), self.memory).squeeze(1)

    def write(
        self,
        weights: torch.Tensor,
        erase_vector: torch.Tensor,
        add_vector: torch.Tensor,
    ):
        erase_matrix = torch.ger(weights.squeeze(), erase_vector.squeeze())
        add_matrix = torch.ger(weights.squeeze(), add_vector.squeeze())
        self.memory.data = self.memory.data * (1 - erase_matrix) + add_matrix


class ContentAddressing(nn.Module):
    def __init__(self, word_size: int):
        super().__init__()
        self.word_size = word_size

    def forward(
        self, memory: torch.Tensor, key: torch.Tensor, beta: float = 1.0
    ) -> torch.Tensor:
        key = key.unsqueeze(1)
        similarity = torch.matmul(memory, key.t()) / self.word_size
        weights = F.softmax(beta * similarity, dim=-1)
        return weights


class LocationAddressing(nn.Module):
    def __init__(self, memory_size: int):
        super().__init__()
        self.memory_size = memory_size

    def forward(self, prev_weights: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        conv_weights = F.conv1d(
            prev_weights.unsqueeze(0).unsqueeze(0),
            shift.unsqueeze(0).unsqueeze(0),
            padding=1,
        )
        return F.softmax(conv_weights.squeeze(), dim=-1)


class ReadHead(nn.Module):
    def __init__(self, memory_size: int, word_size: int, controller_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size

        self.key_net = nn.Linear(controller_size, word_size)
        self.beta_net = nn.Linear(controller_size, 1)

        self.content_addressing = ContentAddressing(word_size)
        self.location_addressing = LocationAddressing(memory_size)

    def forward(
        self,
        controller_output: torch.Tensor,
        prev_weights: torch.Tensor,
        memory: NTMMemory,
        gamma: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = F.tanh(self.key_net(controller_output))
        beta = F.softplus(self.beta_net(controller_output)).squeeze(-1)

        content_weights = self.content_addressing(memory.memory, key, beta)
        shift = F.softmax(controller_output[:, :3], dim=-1)

        location_weights = self.location_addressing(prev_weights, shift)
        weights = (content_weights**gamma) * (location_weights ** (1 - gamma))
        weights = F.normalize(weights, p=1, dim=-1)

        read_vector = memory.read(weights)
        return read_vector, weights


class WriteHead(nn.Module):
    def __init__(self, memory_size: int, word_size: int, controller_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size

        self.key_net = nn.Linear(controller_size, word_size)
        self.beta_net = nn.Linear(controller_size, 1)
        self.erase_net = nn.Linear(controller_size, word_size)
        self.add_net = nn.Linear(controller_size, word_size)

        self.content_addressing = ContentAddressing(word_size)
        self.location_addressing = LocationAddressing(memory_size)

    def forward(
        self,
        controller_output: torch.Tensor,
        prev_weights: torch.Tensor,
        memory: NTMMemory,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        key = F.tanh(self.key_net(controller_output))
        beta = F.softplus(self.beta_net(controller_output)).squeeze(-1)

        content_weights = self.content_addressing(memory.memory, key, beta)
        shift = F.softmax(controller_output[:, :3], dim=-1)

        location_weights = self.location_addressing(prev_weights, shift)
        weights = (content_weights**gamma) * (location_weights ** (1 - gamma))
        weights = F.normalize(weights, p=1, dim=-1)

        erase = F.sigmoid(self.erase_net(controller_output))
        add = F.tanh(self.add_net(controller_output))

        memory.write(weights, erase, add)
        return weights


class NTMController(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, num_heads: int = 1
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.x_to_h = nn.Linear(input_size, hidden_size)
        self.r_to_h = nn.Linear(word_size * num_heads, hidden_size)
        self.h_to_o = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, read_vectors: torch.Tensor) -> torch.Tensor:
        x_hidden = self.x_to_h(x)
        if read_vectors.numel() > 0:
            r_hidden = self.r_to_h(read_vectors)
            hidden = F.relu(x_hidden + r_hidden)
        else:
            hidden = F.relu(x_hidden)
        return self.h_to_o(hidden)


class NTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        memory_size: int = 128,
        word_size: int = 20,
        controller_size: int = 100,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.word_size = word_size
        self.controller_size = controller_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads

        self.memory = NTMMemory(memory_size, word_size)

        self.controller = nn.LSTM(
            input_size + word_size * num_read_heads, controller_size, batch_first=True
        )

        self.read_heads = nn.ModuleList(
            [
                ReadHead(memory_size, word_size, controller_size)
                for _ in range(num_read_heads)
            ]
        )

        self.write_heads = nn.ModuleList(
            [
                WriteHead(memory_size, word_size, controller_size)
                for _ in range(num_write_heads)
            ]
        )

        self.read_to_output = nn.Linear(word_size * num_read_heads, output_size)

        self.prev_weights = None

    def init_weights(self, batch_size: int, device: torch.device):
        self.prev_weights = torch.zeros(batch_size, self.memory_size, device=device)
        self.controller.flatten_parameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prev_weights is None:
            self.init_weights(x.size(0), x.device)

        read_vectors_list = []
        for head in self.read_heads:
            read_vec, self.prev_weights = head(
                torch.zeros_like(x[:, : self.controller_size]),
                self.prev_weights,
                self.memory,
            )
            read_vectors_list.append(read_vec)

        read_vectors = torch.cat(read_vectors_list, dim=-1)

        controller_input = torch.cat([x, read_vectors], dim=-1)
        controller_output, _ = self.controller(controller_input.unsqueeze(1))
        controller_output = controller_output.squeeze(1)

        for head in self.write_heads:
            self.prev_weights = head(controller_output, self.prev_weights, self.memory)

        read_vectors_list = []
        for head in self.read_heads:
            read_vec, _ = head(controller_output, self.prev_weights, self.memory)
            read_vectors_list.append(read_vec)

        read_vectors = torch.cat(read_vectors_list, dim=-1)
        output = self.read_to_output(read_vectors)

        return output, controller_output


class NeuralTuringMachine(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        memory_size: int = 128,
        word_size: int = 20,
        controller_size: int = 100,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
    ):
        super().__init__()
        self.ntm_cell = NTMCell(
            input_size=input_size,
            output_size=output_size,
            memory_size=memory_size,
            word_size=word_size,
            controller_size=controller_size,
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.ntm_cell(x)
        return output

    def reset(self):
        self.ntm_cell.prev_weights = None
