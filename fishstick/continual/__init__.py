"""
Continual Learning Module

Methods for learning incrementally without forgetting.
"""

from typing import Optional, List, Dict, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy


class EWC:
    """Elastic Weight Consolidation (EWC) for continual learning.

    Args:
        model: Model to protect
        importance: Importance of old tasks
    """

    def __init__(self, model: nn.Module, importance: float = 1000):
        self.model = model
        self.importance = importance
        self.params = {
            n: p.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def compute_fisher(self, data_loader):
        """Compute Fisher Information Matrix."""
        self.model.eval()
        for x, y in data_loader:
            self.model.zero_grad()
            out = self.model(x)
            out[y].backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.data**2

    def penalty(self) -> Tensor:
        """Compute EWC penalty."""
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.importance * loss


class PackNet(nn.Module):
    """PackNet: Prune and retain for continual learning.

    Args:
        model: Base model
        num_tasks: Number of tasks to learn
    """

    def __init__(self, model: nn.Module, num_tasks: int = 10):
        super().__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.task_masks = {}

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        out = self.model(x)
        if task_id in self.task_masks:
            for name, mask in self.task_masks[task_id].items():
                if name in out:
                    out = out * mask
        return out

    def prune_and_freeze(self, task_id: int, sparsity: float = 0.5):
        """Prune and freeze weights for current task."""
        masks = {}
        for name, param in self.model.named_parameters():
            if "weight" in name:
                importance = param.data.abs()
                threshold = torch.quantile(importance.flatten(), sparsity)
                mask = importance > threshold
                masks[name] = mask.float()
                param.data *= mask.float()
                param.requires_grad = False
        self.task_masks[task_id] = masks


class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Networks for continual learning.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_tasks: Number of tasks
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_tasks: int,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.columns = nn.ModuleList()

        for task in range(num_tasks):
            if task == 0:
                column = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            else:
                column = nn.ModuleDict(
                    {
                        "adapter": nn.Linear(input_dim, hidden_dim),
                        "fc1": nn.Linear(hidden_dim * task, hidden_dim),
                        "fc2": nn.Linear(hidden_dim, hidden_dim),
                    }
                )
            self.columns.append(column)

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        if task_id == 0:
            return self.columns[task_id](x)
        else:
            features = [self.columns[i](x) for i in range(task_id + 1)]
            combined = torch.cat(features, dim=-1)
            return self.columns[task_id]["fc2"](
                F.relu(self.columns[task_id]["fc1"](combined))
            )


class MemoryReplay(nn.Module):
    """Memory-based replay for continual learning.

    Args:
        model: Base model
        memory_size: Size of replay memory
    """

    def __init__(self, model: nn.Module, memory_size: int = 1000):
        super().__init__()
        self.model = model
        self.memory_size = memory_size
        self.memory_x = []
        self.memory_y = []

    def add_to_memory(self, x: Tensor, y: Tensor):
        """Add samples to replay memory."""
        self.memory_x.append(x.detach().cpu())
        self.memory_y.append(y.detach().cpu())
        if len(self.memory_x) > self.memory_size:
            self.memory_x = self.memory_x[-self.memory_size :]
            self.memory_y = self.memory_y[-self.memory_size :]

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from memory."""
        idx = torch.randint(0, len(self.memory_x), (batch_size,))
        x = torch.stack([self.memory_x[i] for i in idx])
        y = torch.stack([self.memory_y[i] for i in idx])
        return x, y


class GEM:
    """Gradient Episodic Memory (GEM) for continual learning.

    Args:
        model: Base model
        memory_size: Size of episodic memory
    """

    def __init__(self, model: nn.Module, memory_per_task: int = 200):
        self.model = model
        self.memory_per_task = memory_per_task
        self.memory = {}

    def store(self, task_id: int, x: Tensor, y: Tensor):
        """Store samples for a task."""
        indices = torch.randperm(len(x))[: self.memory_per_task]
        self.memory[task_id] = (x[indices], y[indices])

    def compute_gradient_penalty(self, grad_old: Tensor, grad_new: Tensor) -> Tensor:
        """Compute gradient projection to avoid forgetting."""
        dot = (grad_old * grad_new).sum()
        if dot < 0:
            return grad_new - (dot / (grad_old**2).sum()) * grad_old
        return grad_new


class LwF:
    """Learning without Forgetting (LwF) for continual learning.

    Args:
        model: Base model
        temperature: Knowledge distillation temperature
        alpha: Balancing factor
    """

    def __init__(self, model: nn.Module, temperature: float = 2.0, alpha: float = 0.5):
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.old_model = None

    def preserve_knowledge(self, x: Tensor) -> Tensor:
        """Preserve old knowledge using knowledge distillation."""
        if self.old_model is None:
            return 0

        with torch.no_grad():
            old_logits = self.old_model(x)
            new_logits = self.model(x)

            T = self.temperature
            soft_old = F.softmax(old_logits / T, dim=-1)
            soft_new = F.log_softmax(new_logits / T, dim=-1)

            kd_loss = F.kl_div(soft_new, soft_old, reduction="batchmean") * (T**2)
            return kd_loss

    def save_old_model(self):
        """Save a copy of model for knowledge distillation."""
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()


from typing import Tuple


class AGEM:
    """Averaged GEM for efficient continual learning."""

    def __init__(self, model: nn.Module, memory_size: int = 1000):
        self.model = model
        self.memory_size = memory_size
        self.memory_x = []
        self.memory_y = []

    def store(self, x: Tensor, y: Tensor):
        self.memory_x.append(x)
        self.memory_y.append(y)

    def project_gradient(self, grad: Tensor) -> Tensor:
        if len(self.memory_x) == 0:
            return grad
        return grad


class PackMemory:
    """PackNet-style memory for weight preservation."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.masks = {}
        self.active_params = {}

    def save_active(self, task_id: int):
        """Save which params are active for task."""
        active = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                active[n] = p.clone()
        self.active_params[task_id] = active
