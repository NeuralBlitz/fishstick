import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class PdeNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: Optional[int] = None,
        activation: str = "tanh",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim or input_dim

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = torch.tanh

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(self.activation(x))

    def compute_derivative(
        self,
        x: torch.Tensor,
        order: int = 1,
    ) -> torch.Tensor:
        x.requires_grad_(True)
        output = self.forward(x)

        for _ in range(order):
            grad = torch.autograd.grad(
                output.sum(),
                x,
                create_graph=True,
                retain_graph=True,
            )[0]
            output = grad

        return output


class PdeNetResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim or input_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.tanh(x)

            if i > 0 and i % 2 == 0:
                x = x + identity
                identity = x

        return self.output_layer(x)


class PodNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pod_modes: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.pod_modes = pod_modes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)],
        )

        self.pod_basis = nn.Parameter(
            torch.randn(pod_modes, hidden_dim),
            requires_grad=True,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)

        coeffs = torch.matmul(encoded, self.pod_basis.T)

        reconstructed = torch.matmul(coeffs, self.pod_basis)

        return self.decoder(reconstructed)

    def compute_pod(self, snapshots: torch.Tensor, num_modes: int = None):
        num_modes = num_modes or self.pod_modes

        centered = snapshots - snapshots.mean(dim=0, keepdim=True)

        cov = torch.matmul(centered.T, centered) / snapshots.size(0)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        eigenvectors = eigenvectors[:, -num_modes:]

        with torch.no_grad():
            self.pod_basis.data = eigenvectors.T[: self.pod_modes].clone()

        return eigenvectors, eigenvalues[-num_modes:]


class NeuralGalerkin(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_modes: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.modes = nn.Parameter(
            torch.randn(num_modes, hidden_dim),
            requires_grad=True,
        )

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.feature_extractor = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        features = self.feature_extractor(x)

        mode_activations = torch.matmul(features, self.modes.T)
        mode_activations = torch.tanh(mode_activations)

        output = self.output_layer(features)

        if return_features:
            return output, features, mode_activations

        return output

    def compute_galerkin_projection(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        features = self.feature_extractor(x)

        mode_features = torch.matmul(features, self.modes.T)

        projection = torch.matmul(mode_features.T, mode_features)
        projection = projection / x.size(0)

        residual = target - self.forward(x)
        galerkin_term = torch.matmul(mode_features.T, residual)

        return galerkin_term / x.size(0)


class NeuralOperatorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class AdaptiveRefinementNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_refinements: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_refinements = num_refinements

        self.base_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.refinement_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + 1, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_refinements)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_network(x[:, :-1])
        current_output = base_output

        for refinement in self.refinement_layers:
            residual = current_output
            combined = torch.cat([x[:, :-1], current_output], dim=-1)
            refined_output = refinement(combined)
            current_output = residual + refined_output

        return current_output
