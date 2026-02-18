import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional, Dict, List, Tuple, Any
import random
import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_samples = 0

    def add(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        for inputs, targets in zip(*batch):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((inputs.to(self.device), targets.to(self.device)))
            else:
                idx = random.randint(0, self.seen_samples)
                if idx < self.buffer_size:
                    self.buffer[idx] = (inputs.to(self.device), targets.to(self.device))
            self.seen_samples += 1

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.buffer) == 0:
            raise RuntimeError("Buffer is empty")

        indices = random.sample(
            range(len(self.buffer)), min(batch_size, len(self.buffer))
        )
        inputs = torch.stack([self.buffer[i][0] for i in indices])
        targets = torch.stack([self.buffer[i][1] for i in indices])

        return inputs, targets

    def __len__(self):
        return len(self.buffer)


class ReservoirReplay:
    def __init__(self, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_samples = 0

    def add_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        batch_size = inputs.size(0)

        for i in range(batch_size):
            self._add_single(inputs[i], targets[i])

    def _add_single(self, input: torch.Tensor, target: torch.Tensor):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((input.to(self.device), target.to(self.device)))
        else:
            idx = random.randint(0, self.seen_samples)
            if idx < self.buffer_size:
                self.buffer[idx] = (input.to(self.device), target.to(self.device))

        self.seen_samples += 1

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.buffer) == 0:
            raise RuntimeError("Buffer is empty")

        indices = random.sample(
            range(len(self.buffer)), min(batch_size, len(self.buffer))
        )
        inputs = torch.stack([self.buffer[i][0] for i in indices])
        targets = torch.stack([self.buffer[i][1] for i in indices])

        return inputs, targets

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.buffer) == 0:
            raise RuntimeError("Buffer is empty")

        inputs = torch.stack([self.buffer[i][0] for i in range(len(self.buffer))])
        targets = torch.stack([self.buffer[i][1] for i in range(len(self.buffer))])

        return inputs, targets

    def __len__(self):
        return len(self.buffer)


class MemoryAwareSynapses:
    def __init__(
        self,
        model: nn.Module,
        buffer_size: int,
        device: str = "cpu",
        alpha: float = 0.5,
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.device = device
        self.alpha = alpha

        self.replay = ReservoirReplay(buffer_size, device)
        self.importance_matrix: Dict[str, torch.Tensor] = {}
        self._init_importance_matrix()

    def _init_importance_matrix(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importance_matrix[name] = torch.ones_like(param.data)

    def compute_parameter_importance(self, dataloader: DataLoader):
        self.model.eval()
        param_grads: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_grads[name] = torch.zeros_like(param.data)

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)

            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_grads[name] += param.grad.data**2

        for name in param_grads:
            param_grads[name] = param_grads[name] / len(dataloader)

        for name in self.importance_matrix:
            if name in param_grads:
                self.importance_matrix[name] = (
                    self.alpha * self.importance_matrix[name]
                    + (1 - self.alpha) * param_grads[name]
                )

    def mas_penalty(self) -> torch.Tensor:
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance_matrix:
                importance = self.importance_matrix[name]
                penalty += torch.sum(importance * (param**2))
        return penalty

    def update_buffer(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.replay.add_batch(inputs, targets)

    def sample_from_buffer(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.replay.sample(batch_size)


class GradientEpisodicMemory:
    def __init__(
        self,
        buffer_size: int,
        n_tasks: int,
        device: str = "cpu",
        gamma: float = 0.5,
    ):
        self.buffer_size = buffer_size
        self.n_tasks = n_tasks
        self.device = device
        self.gamma = gamma

        self.memory_per_task = buffer_size // n_tasks

        self.task_buffers: List[ReservoirReplay] = [
            ReservoirReplay(self.memory_per_task, device) for _ in range(n_tasks)
        ]

        self.saved_gradients: Dict[str, torch.Tensor] = {}
        self._init_saved_gradients()

    def _init_saved_gradients(self):
        self.saved_gradients = {}

    def store_gradients(self, model: nn.Module, task_id: int):
        self.saved_gradients[task_id] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.saved_gradients[task_id][name] = param.data.clone()

    def add_task_data(
        self,
        task_id: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        if task_id >= self.n_tasks:
            raise ValueError(f"task_id {task_id} exceeds n_tasks {self.n_tasks}")

        self.task_buffers[task_id].add_batch(inputs, targets)

    def sample_task_batch(
        self,
        task_id: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if task_id >= len(self.task_buffers):
            raise ValueError(f"task_id {task_id} is out of range")

        return self.task_buffers[task_id].sample(batch_size)

    def compute_ort_penalty(self, model: nn.Module) -> torch.Tensor:
        if not self.saved_gradients:
            return torch.tensor(0.0, device=self.device)

        current_params = {name: param.data for name, param in model.named_parameters()}

        penalty = torch.tensor(0.0, device=self.device)

        for task_id, saved_params in self.saved_gradients.items():
            if task_id >= len(self.task_buffers):
                continue

            buffer = self.task_buffers[task_id]
            if len(buffer) == 0:
                continue

            inputs, targets = buffer.sample(min(32, len(buffer)))

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            model.zero_grad()
            loss.backward()

            task_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    task_grad_norm += torch.sum(param.grad**2)

            saved_params_tensor = torch.cat(
                [p.flatten() for p in saved_params.values()]
            )
            current_params_tensor = torch.cat(
                [current_params[name].flatten() for name in saved_params.keys()]
            )

            diff = torch.norm(current_params_tensor - saved_params_tensor)
            penalty += self.gamma * diff * torch.sqrt(task_grad_norm + 1e-8)

        return penalty

    def get_all_task_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        all_data = []
        for buffer in self.task_buffers:
            if len(buffer) > 0:
                inputs, targets = buffer.get_all()
                all_data.append((inputs, targets))
        return all_data


class GEM:
    def __init__(
        self,
        model: nn.Module,
        buffer_size: int,
        n_tasks: int,
        device: str = "cpu",
        gamma: float = 0.5,
    ):
        self.model = model
        self.gem = GradientEpisodicMemory(buffer_size, n_tasks, device, gamma)

    def store_task_data(
        self,
        task_id: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        self.gem.add_task_data(task_id, inputs, targets)
        self.gem.store_gradients(self.model, task_id)

    def compute_loss(
        self,
        primary_loss: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        ort_penalty = self.gem.compute_ort_penalty(self.model)

        if ort_penalty > 0:
            return primary_loss + ort_penalty
        return primary_loss

    def sample_from_buffer(
        self,
        task_id: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gem.sample_task_batch(task_id, batch_size)


class ClassBalancedReplay:
    def __init__(self, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.class_buffers: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.class_counts: Dict[int, int] = {}

    def add_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        unique_classes = torch.unique(targets)

        for cls in unique_classes:
            cls = cls.item()
            mask = targets == cls
            cls_inputs = inputs[mask]
            cls_targets = targets[mask]

            if cls not in self.class_buffers:
                self.class_buffers[cls] = []
                self.class_counts[cls] = 0

            per_class_buffer = self.buffer_size // len(unique_classes)

            for inp, tgt in zip(cls_inputs, cls_targets):
                if len(self.class_buffers[cls]) < per_class_buffer:
                    self.class_buffers[cls].append((inp, tgt))
                else:
                    idx = random.randint(0, self.class_counts[cls])
                    if idx < per_class_buffer:
                        self.class_buffers[cls][idx] = (inp, tgt)
                self.class_counts[cls] += 1

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        all_samples = []
        for cls, buffer in self.class_buffers.items():
            all_samples.extend(buffer)

        if len(all_samples) == 0:
            raise RuntimeError("Buffer is empty")

        indices = random.sample(
            range(len(all_samples)), min(batch_size, len(all_samples))
        )

        inputs = torch.stack([all_samples[i][0] for i in indices])
        targets = torch.stack([all_samples[i][1] for i in indices])

        return inputs, targets
