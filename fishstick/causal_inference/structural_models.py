"""
Advanced Structural Causal Model Implementations.

Provides various SCM architectures:
- Linear SCM
- Additive SCM
- Nonlinear SCM with neural networks
- Gaussian Process SCM
- SCM ensembles
"""

from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, Laplace, Uniform
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class NoiseDistribution(Enum):
    """Type of noise distribution for SCM."""

    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    MIXTURE = "mixture"


class BaseSCM(nn.Module, ABC):
    """Abstract base class for structural causal models."""

    def __init__(self, n_variables: int, variable_names: Optional[List[str]] = None):
        super().__init__()
        self.n_variables = n_variables
        if variable_names is None:
            self.variable_names = [f"X{i}" for i in range(n_variables)]
        else:
            self.variable_names = variable_names

    @abstractmethod
    def forward(
        self,
        n_samples: int,
        interventions: Optional[Dict[int, Tensor]] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Sample from the SCM."""
        pass

    @abstractmethod
    def causal_effect(
        self,
        treatment: int,
        outcome: int,
        confounders: Optional[List[int]] = None,
    ) -> Tensor:
        """Compute causal effect from treatment to outcome."""
        pass

    def do_intervention(
        self,
        intervention: Dict[int, Tensor],
        n_samples: int,
    ) -> Dict[str, Tensor]:
        """Perform do-intervention."""
        return self.forward(n_samples, interventions=intervention)


class LinearSCM(BaseSCM):
    """
    Linear structural causal model: X_i = sum_j w_ij * X_j + epsilon_i.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        noise_std: Union[float, List[float]] = 1.0,
        variable_names: Optional[List[str]] = None,
    ):
        n_variables = adjacency_matrix.shape[0]
        super().__init__(n_variables, variable_names)

        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

        if isinstance(noise_std, float):
            self.noise_std = torch.tensor([noise_std] * n_variables)
        else:
            self.noise_std = torch.tensor(noise_std)

        self._validate_dag()

    def _validate_dag(self) -> None:
        """Verify the graph is acyclic."""
        W = self.adjacency_matrix.numpy()
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d)
        h = np.trace(E) - d
        if h > 1e-6:
            raise ValueError("Adjacency matrix must represent a DAG")

    def forward(
        self,
        n_samples: int,
        interventions: Optional[Dict[int, Tensor]] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if interventions is None:
            interventions = {}

        device = self.adjacency_matrix.device
        if noise is None:
            noise = torch.randn(n_samples, self.n_variables, device=device)
            noise = noise * self.noise_std.unsqueeze(0)

        samples = {}
        order = self._topological_order()

        for var_idx in order:
            if var_idx in interventions:
                samples[var_idx] = interventions[var_idx]
            else:
                parents = torch.where(self.adjacency_matrix[:, var_idx] > 0)[0]
                if len(parents) > 0:
                    parent_values = torch.stack(
                        [samples[int(p)] for p in parents], dim=-1
                    )
                    parent_weights = self.adjacency_matrix[parents, var_idx]
                    linear_term = parent_values @ parent_weights
                else:
                    linear_term = torch.zeros(n_samples, device=device)

                samples[var_idx] = linear_term + noise[:, var_idx : var_idx + 1]

        return {self.variable_names[i]: samples[i] for i in range(self.n_variables)}

    def causal_effect(
        self,
        treatment: int,
        outcome: int,
        confounders: Optional[List[int]] = None,
    ) -> Tensor:
        W = self.adjacency_matrix.numpy()
        effect = float(W[treatment, outcome])

        for k in range(self.n_variables):
            if k != treatment and k != outcome:
                if confounders is None or k in confounders:
                    effect += W[treatment, k] * W[k, outcome]

        return torch.tensor(effect)

    def _topological_order(self) -> List[int]:
        n = self.n_variables
        in_degree = torch.sum(self.adjacency_matrix > 0, dim=0).tolist()
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in torch.where(self.adjacency_matrix[node] > 0)[0]:
                child_int = int(child)
                in_degree[child_int] -= 1
                if in_degree[child_int] == 0:
                    queue.append(child_int)

        return order


class AdditiveSCM(BaseSCM):
    """
    Additive nonlinear SCM: X_i = f_i(PA_i) + epsilon_i.
    Uses neural networks for nonlinear functions.
    """

    def __init__(
        self,
        n_variables: int,
        adjacency_matrix: np.ndarray,
        hidden_dim: int = 64,
        noise_type: NoiseDistribution = NoiseDistribution.GAUSSIAN,
        noise_std: float = 1.0,
        variable_names: Optional[List[str]] = None,
    ):
        super().__init__(n_variables, variable_names)

        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim

        self._build_structural_equations(hidden_dim)
        self._validate_dag()

    def _build_structural_equations(self, hidden_dim: int) -> None:
        """Build neural network for each structural equation."""
        self.equations = nn.ModuleList()

        for i in range(self.n_variables):
            n_parents = torch.sum(self.adjacency_matrix[:, i] > 0).item()

            if n_parents == 0:
                self.equations.append(nn.Identity())
            else:
                eq = nn.Sequential(
                    nn.Linear(n_parents, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, 1),
                )
                self.equations.append(eq)

    def _validate_dag(self) -> None:
        W = self.adjacency_matrix.numpy()
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d)
        h = np.trace(E) - d
        if h > 1e-6:
            raise ValueError("Adjacency matrix must represent a DAG")

    def forward(
        self,
        n_samples: int,
        interventions: Optional[Dict[int, Tensor]] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if interventions is None:
            interventions = {}

        device = next(self.parameters()).device

        if noise is None:
            noise = self._sample_noise(n_samples, device)

        samples = {}
        order = self._topological_order()

        for var_idx in order:
            if var_idx in interventions:
                samples[var_idx] = interventions[var_idx]
            else:
                parent_indices = torch.where(self.adjacency_matrix[:, var_idx] > 0)[0]

                if len(parent_indices) > 0:
                    parent_values = torch.stack(
                        [samples[p] for p in parent_indices], dim=-1
                    )
                    structural_value = self.equations[var_idx](parent_values)
                else:
                    structural_value = torch.zeros(n_samples, 1, device=device)

                samples[var_idx] = structural_value + noise[:, var_idx : var_idx + 1]

        return {self.variable_names[i]: samples[i] for i in range(self.n_variables)}

    def _sample_noise(self, n_samples: int, device: torch.device) -> Tensor:
        if self.noise_type == NoiseDistribution.GAUSSIAN:
            noise = (
                torch.randn(n_samples, self.n_variables, device=device) * self.noise_std
            )
        elif self.noise_type == NoiseDistribution.LAPLACIAN:
            noise = torch.randn(n_samples, self.n_variables, device=device)
            noise = torch.distributions.Laplace(0, self.noise_std).sample(
                (n_samples, self.n_variables)
            )
        elif self.noise_type == NoiseDistribution.UNIFORM:
            noise = torch.rand(n_samples, self.n_variables, device=device) - 0.5
            noise = noise * self.noise_std * 2
        else:
            noise = (
                torch.randn(n_samples, self.n_variables, device=device) * self.noise_std
            )
        return noise

    def causal_effect(
        self,
        treatment: int,
        outcome: int,
        confounders: Optional[List[int]] = None,
    ) -> Tensor:
        with torch.no_grad():
            test_points = torch.zeros(1, self.n_variables)
            test_points[0, treatment] = 1.0

            order = self._topological_order()
            samples = {}

            for var_idx in order:
                parent_indices = torch.where(self.adjacency_matrix[:, var_idx] > 0)[0]
                if len(parent_indices) > 0:
                    parent_values = test_points[:, parent_indices]
                    structural_value = self.equations[var_idx](parent_values)
                else:
                    structural_value = torch.zeros(1, 1)
                samples[var_idx] = structural_value

            effect = samples[outcome][0, 0].item()
            return torch.tensor(effect)

    def _topological_order(self) -> List[int]:
        n = self.n_variables
        in_degree = torch.sum(self.adjacency_matrix > 0, dim=0).tolist()
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in torch.where(self.adjacency_matrix[node] > 0)[0]:
                child_int = int(child)
                in_degree[child_int] -= 1
                if in_degree[child_int] == 0:
                    queue.append(child_int)

        return order


class NonlinearSCM(BaseSCM):
    """
    General nonlinear SCM with flexible neural network architectures.
    """

    def __init__(
        self,
        n_variables: int,
        adjacency_matrix: np.ndarray,
        hidden_dims: List[int] = [64, 64],
        activation: str = "leaky_relu",
        noise_std: float = 0.1,
        variable_names: Optional[List[str]] = None,
    ):
        super().__init__(n_variables, variable_names)

        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.noise_std = noise_std

        activation_fn = nn.LeakyReLU(0.2) if activation == "leaky_relu" else nn.ReLU()

        self.equations = nn.ModuleList()
        for i in range(n_variables):
            n_parents = int(torch.sum(self.adjacency_matrix[:, i] > 0).item())

            if n_parents == 0:
                self.equations.append(
                    nn.Sequential(
                        nn.Linear(1, hidden_dims[0]),
                        activation_fn,
                        nn.Linear(hidden_dims[-1], 1),
                    )
                )
            else:
                layers = []
                prev_dim = n_parents
                for h_dim in hidden_dims:
                    layers.extend(
                        [
                            nn.Linear(prev_dim, h_dim),
                            activation_fn,
                        ]
                    )
                    prev_dim = h_dim
                layers.append(nn.Linear(prev_dim, 1))
                self.equations.append(nn.Sequential(*layers))

        self._validate_dag()

    def _validate_dag(self) -> None:
        W = self.adjacency_matrix.numpy()
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d)
        h = np.trace(E) - d
        if h > 1e-6:
            raise ValueError("Adjacency matrix must represent a DAG")

    def forward(
        self,
        n_samples: int,
        interventions: Optional[Dict[int, Tensor]] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if interventions is None:
            interventions = {}

        device = next(self.parameters()).device

        if noise is None:
            noise = (
                torch.randn(n_samples, self.n_variables, device=device) * self.noise_std
            )

        samples = {}
        order = self._topological_order()

        for var_idx in order:
            if var_idx in interventions:
                samples[var_idx] = interventions[var_idx]
            else:
                parent_indices = torch.where(self.adjacency_matrix[:, var_idx] > 0)[0]

                if len(parent_indices) > 0:
                    parent_values = torch.cat(
                        [samples[p] for p in parent_indices], dim=-1
                    )
                else:
                    parent_values = torch.zeros(n_samples, 1, device=device)

                samples[var_idx] = (
                    self.equations[var_idx](parent_values)
                    + noise[:, var_idx : var_idx + 1]
                )

        return {self.variable_names[i]: samples[i] for i in range(self.n_variables)}

    def causal_effect(
        self,
        treatment: int,
        outcome: int,
        confounders: Optional[List[int]] = None,
    ) -> Tensor:
        device = next(self.parameters()).device
        x0 = torch.zeros(1, self.n_variables, device=device)
        x1 = torch.zeros(1, self.n_variables, device=device)
        x1[0, treatment] = 1.0

        y0 = self._compute_output(outcome, x0, device)
        y1 = self._compute_output(outcome, x1, device)

        return y1 - y0

    def _compute_output(
        self, var_idx: int, inputs: Tensor, device: torch.device
    ) -> Tensor:
        samples = {}
        order = self._topological_order()

        for v_idx in order:
            parent_indices = torch.where(self.adjacency_matrix[:, v_idx] > 0)[0]

            if len(parent_indices) > 0:
                parent_values = inputs[:, parent_indices]
            else:
                parent_values = torch.zeros(inputs.size(0), 1, device=device)

            samples[v_idx] = self.equations[v_idx](parent_values)

        return samples[var_idx]

    def _topological_order(self) -> List[int]:
        n = self.n_variables
        in_degree = torch.sum(self.adjacency_matrix > 0, dim=0).tolist()
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in torch.where(self.adjacency_matrix[node] > 0)[0]:
                child_int = int(child)
                in_degree[child_int] -= 1
                if in_degree[child_int] == 0:
                    queue.append(child_int)

        return order


class GaussianProcessSCM(BaseSCM):
    """
    SCM with Gaussian Process structural equations.
    """

    def __init__(
        self,
        n_variables: int,
        adjacency_matrix: np.ndarray,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise_std: float = 0.1,
        variable_names: Optional[List[str]] = None,
    ):
        super().__init__(n_variables, variable_names)

        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_std = noise_std

        self._build_gp_equations()
        self._validate_dag()

    def _build_gp_equations(self) -> None:
        """Initialize GP hyperparameters."""
        self.gp_params = nn.ParameterDict()

        for i in range(self.n_variables):
            n_parents = int(torch.sum(self.adjacency_matrix[:, i] > 0).item())
            if n_parents > 0:
                self.gp_params[f"lengthscale_{i}"] = nn.Parameter(
                    torch.tensor(self.lengthscale)
                )
                self.gp_params[f"variance_{i}"] = nn.Parameter(
                    torch.tensor(self.variance)
                )

    def _validate_dag(self) -> None:
        W = self.adjacency_matrix.numpy()
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d)
        h = np.trace(E) - d
        if h > 1e-6:
            raise ValueError("Adjacency matrix must represent a DAG")

    def _rbf_kernel(
        self, x1: Tensor, x2: Tensor, lengthscale: float, variance: float
    ) -> Tensor:
        dist = torch.cdist(x1, x2) / lengthscale
        return variance * torch.exp(-0.5 * dist**2)

    def forward(
        self,
        n_samples: int,
        interventions: Optional[Dict[int, Tensor]] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if interventions is None:
            interventions = {}

        device = next(self.parameters()).device

        if noise is None:
            noise = (
                torch.randn(n_samples, self.n_variables, device=device) * self.noise_std
            )

        samples = {}
        order = self._topological_order()

        for var_idx in order:
            if var_idx in interventions:
                samples[var_idx] = interventions[var_idx]
            else:
                parent_indices = torch.where(self.adjacency_matrix[:, var_idx] > 0)[0]

                if len(parent_indices) > 0:
                    parent_values = torch.cat(
                        [samples[p] for p in parent_indices], dim=-1
                    )

                    lengthscale = F.softplus(self.gp_params[f"lengthscale_{var_idx}"])
                    variance = F.softplus(self.gp_params[f"variance_{var_idx}"])

                    gp_mean = torch.zeros(parent_values.size(0), device=device)
                    parent_std = parent_values.std(dim=0, keepdim=True) + 1e-6
                    normalized_parents = parent_values / parent_std

                    structural_value = gp_mean.unsqueeze(-1) + variance * torch.tanh(
                        normalized_parents
                    )
                else:
                    structural_value = torch.zeros(n_samples, 1, device=device)

                samples[var_idx] = structural_value + noise[:, var_idx : var_idx + 1]

        return {self.variable_names[i]: samples[i] for i in range(self.n_variables)}

    def causal_effect(
        self,
        treatment: int,
        outcome: int,
        confounders: Optional[List[int]] = None,
    ) -> Tensor:
        device = next(self.parameters()).device

        x0 = torch.zeros(2, self.n_variables, device=device)
        x1 = x0.clone()
        x1[1, treatment] = 1.0

        y0 = self._compute_output_for_effect(x0, device)
        y1 = self._compute_output_for_effect(x1, device)

        return y1 - y0

    def _compute_output_for_effect(
        self, inputs: Tensor, device: torch.device
    ) -> Tensor:
        samples = {}
        order = self._topological_order()

        for v_idx in order:
            parent_indices = torch.where(self.adjacency_matrix[:, v_idx] > 0)[0]

            if len(parent_indices) > 0:
                parent_values = inputs[:, parent_indices]
                lengthscale = F.softplus(self.gp_params[f"lengthscale_{v_idx}"])
                variance = F.softplus(self.gp_params[f"variance_{v_idx}"])

                gp_mean = torch.zeros(inputs.size(0), device=device)
                parent_std = parent_values.std(dim=0, keepdim=True) + 1e-6
                normalized_parents = parent_values / parent_std

                samples[v_idx] = gp_mean.unsqueeze(-1) + variance * torch.tanh(
                    normalized_parents
                )
            else:
                samples[v_idx] = torch.zeros(inputs.size(0), 1, device=device)

        return samples[outcome]

    def _topological_order(self) -> List[int]:
        n = self.n_variables
        in_degree = torch.sum(self.adjacency_matrix > 0, dim=0).tolist()
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in torch.where(self.adjacency_matrix[node] > 0)[0]:
                child_int = int(child)
                in_degree[child_int] -= 1
                if in_degree[child_int] == 0:
                    queue.append(child_int)

        return order


class SCMEnsemble(nn.Module):
    """
    Ensemble of SCMs for robust causal inference.
    """

    def __init__(
        self,
        scms: List[BaseSCM],
        weights: Optional[Tensor] = None,
    ):
        super().__init__()
        self.scms = nn.ModuleList(scms)

        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(scms)) / len(scms))
        else:
            self.weights = nn.Parameter(weights)

    def forward(
        self,
        n_samples: int,
        interventions: Optional[Dict[int, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        normalized_weights = F.softmax(self.weights, dim=0)

        all_samples = []
        for scm in self.scms:
            samples = scm.forward(n_samples, interventions)
            all_samples.append(samples)

        aggregated = {}
        var_names = all_samples[0].keys()

        for var_name in var_names:
            var_tensors = [s[var_name] for s in all_samples]
            stacked = torch.stack(var_tensors)
            weighted = (stacked * normalized_weights.view(-1, 1, 1)).sum(dim=0)
            aggregated[var_name] = weighted

        return aggregated

    def causal_effect_distribution(
        self,
        treatment: int,
        outcome: int,
        n_samples: int = 100,
    ) -> Tensor:
        effects = []
        for scm in self.scms:
            effect = scm.causal_effect(treatment, outcome)
            effects.append(effect)

        return torch.stack(effects)

    def add_scms(self, new_scms: List[BaseSCM]) -> None:
        """Add new SCMs to the ensemble."""
        for scm in new_scms:
            self.scms.append(scm)

        new_weights = torch.ones(len(self.scms)) / len(self.scms)
        self.weights = nn.Parameter(new_weights.to(self.weights.device))


class SCMTrainer:
    """
    Trainer for fitting SCMs to observational data.
    """

    def __init__(
        self,
        scm: BaseSCM,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
    ):
        self.scm = scm
        self.optimizer = optimizer if optimizer else torch.optim.Adam(scm.parameters())

        if loss_fn is None:
            self.loss_fn = F.mse_loss
        else:
            self.loss_fn = loss_fn

    def fit(
        self,
        data: Dict[str, Tensor],
        n_epochs: int = 1000,
        batch_size: int = 256,
        intervention_targets: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        device = next(self.scm.parameters()).device

        data_tensors = {k: v.to(device) for k, v in data.items()}
        variable_names = list(data_tensors.keys())

        losses = []

        for epoch in range(n_epochs):
            self.scm.train()
            epoch_losses = []

            n_samples = next(iter(data_tensors.values())).size(0)
            indices = torch.randperm(n_samples)[:batch_size]

            batch_data = {k: v[indices] for k, v in data_tensors.items()}

            self.optimizer.zero_grad()

            sampled = self.scm.forward(batch_size)

            loss = 0.0
            for var_name in variable_names:
                if var_name in batch_data:
                    pred = sampled[var_name]
                    target = batch_data[var_name]
                    loss += self.loss_fn(pred.squeeze(), target.squeeze())

            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())
            losses.append(np.mean(epoch_losses))

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Loss: {losses[-1]:.4f}")

        return {"final_loss": losses[-1], "loss_history": losses}

    def evaluate(
        self,
        data: Dict[str, Tensor],
        intervention: Optional[Dict[int, Tensor]] = None,
    ) -> Dict[str, float]:
        self.scm.eval()
        device = next(self.scm.parameters()).device

        data_tensors = {k: v.to(device) for k, v in data.items()}

        with torch.no_grad():
            if intervention:
                predicted = self.scm.do_intervention(
                    intervention, data_tensors[next(iter(data_tensors))].size(0)
                )
            else:
                predicted = self.scm.forward(
                    data_tensors[next(iter(data_tensors))].size(0)
                )

        mse = 0.0
        for var_name in data_tensors:
            if var_name in predicted:
                mse += F.mse_loss(
                    predicted[var_name].squeeze(), data_tensors[var_name].squeeze()
                ).item()

        return {"mse": mse / len(data_tensors)}
