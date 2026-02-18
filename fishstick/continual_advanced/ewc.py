import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import copy


class FisherInformation:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.param_names: List[str] = []

    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: int = 200,
        ema_decay: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        self.fisher_dict = {}
        param_names = []
        ema_fisher: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param.data)
                param_names.append(name)
                ema_fisher[name] = torch.zeros_like(param.data)

        self.model.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)

            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad.data.clone()
                    ema_fisher[name] = ema_decay * ema_fisher[name] + (
                        1 - ema_decay
                    ) * (grad**2)

        for name in param_names:
            self.fisher_dict[name] = ema_fisher[name]

        self.param_names = param_names
        return self.fisher_dict

    def get_fisher_diagonal(self) -> Dict[str, torch.Tensor]:
        return self.fisher_dict

    def get_fisher_norm(self) -> torch.Tensor:
        total_fisher = 0.0
        for fisher in self.fisher_dict.values():
            total_fisher += torch.sum(fisher)
        return total_fisher


def compute_fisher_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    num_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    fisher_computer = FisherInformation(model, device)
    return fisher_computer.compute_fisher(dataloader, num_samples)


class EWC:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        ewc_lambda: float = 1000,
        fisher_decay: float = 0.9,
    ):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.fisher_decay = fisher_decay

        self.params: Dict[str, torch.Tensor] = {}
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.task_counts: Dict[str, int] = defaultdict(int)

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()

    def register_task(self, dataloader: DataLoader, num_samples: int = 200):
        fisher_computer = FisherInformation(self.model, self.device)
        fisher = fisher_computer.compute_fisher(
            dataloader, num_samples, self.fisher_decay
        )

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.fisher_dict:
                    self.fisher_dict[name] = torch.zeros_like(param.data)
                    self.params[name] = param.data.clone()

                self.fisher_dict[name] = (
                    self.fisher_decay * self.fisher_dict[name]
                    + (1 - self.fisher_decay) * fisher[name]
                )
                self.task_counts[name] += 1

    def penalty(self) -> torch.Tensor:
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                old_param = self.params[name]
                penalty += torch.sum(fisher * (param - old_param) ** 2)
        return penalty

    def ewc_loss(self, primary_loss: torch.Tensor) -> torch.Tensor:
        return primary_loss + self.ewc_lambda * self.penalty()

    def compute_importance_weights(self) -> Dict[str, torch.Tensor]:
        importance_weights = {}
        for name, fisher in self.fisher_dict.items():
            importance_weights[name] = fisher / (fisher.max() + 1e-8)
        return importance_weights

    def consolidate(self, dataloader: DataLoader, num_samples: int = 200):
        self.register_task(dataloader, num_samples)

    def get_consolidated_params(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }


class OnlineEWC:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        ewc_lambda: float = 1000,
        gamma: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma

        self.params: Dict[str, torch.Tensor] = {}
        self.fisher_dict: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
                self.fisher_dict[name] = torch.zeros_like(param.data)

    def register_task(self, dataloader: DataLoader, num_samples: int = 200):
        fisher_computer = FisherInformation(self.model, self.device)
        fisher = fisher_computer.compute_fisher(dataloader, num_samples)

        for name in self.fisher_dict.keys():
            if name in fisher:
                self.fisher_dict[name] = (
                    self.gamma * self.fisher_dict[name] + fisher[name]
                )
                self.params[name] = self.model.state_dict()[name].clone()

    def penalty(self) -> torch.Tensor:
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                old_param = self.params[name]
                penalty += torch.sum(fisher * (param - old_param) ** 2)
        return penalty


class EWCWithElasticity:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        ewc_lambda: float = 1000,
        elasticity: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.elasticity = elasticity

        self.params: Dict[str, torch.Tensor] = {}
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.stiffness: Dict[str, float] = {}

    def compute_adaptive_lambda(self, param_name: str) -> float:
        if param_name not in self.fisher_dict:
            return self.ewc_lambda

        fisher = self.fisher_dict[param_name]
        mean_fisher = torch.mean(fisher).item()

        if param_name in self.stiffness:
            self.stiffness[param_name] = (
                self.elasticity * self.stiffness[param_name]
                + (1 - self.elasticity) * mean_fisher
            )
        else:
            self.stiffness[param_name] = mean_fisher

        return self.ewc_lambda * (1.0 + self.stiffness[param_name])

    def penalty(self) -> torch.Tensor:
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                old_param = self.params[name]
                adaptive_lambda = self.compute_adaptive_lambda(name)
                penalty += (
                    torch.sum(fisher * (param - old_param) ** 2) * adaptive_lambda
                )
        return penalty
