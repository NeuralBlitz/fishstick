import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torchdiffeq import odeint


class CouplingLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.register_buffer("mask", mask)

        self.net = nn.Sequential(
            nn.Linear(in_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        log_s = self.net(x_masked)
        log_s = torch.tanh(log_s)
        s = torch.exp(log_s)
        t = self.net(x_masked)
        t = t * (1 - self.mask)

        if not reverse:
            y = x * self.mask + (x * s + t) * (1 - self.mask)
            log_det = ((1 - self.mask) * log_s).sum(dim=-1)
            return y, log_det
        else:
            y = x * self.mask + (x - t) / s * (1 - self.mask)
            log_det = -((1 - self.mask) * log_s).sum(dim=-1)
            return y, log_det


class RealNVP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 6,
    ):
        super().__init__()
        self.dim = dim

        masks = []
        for i in range(n_blocks):
            mask = torch.zeros(dim)
            mask[dim // 2 :] = 1
            if i % 2 == 1:
                mask = 1 - mask
            masks.append(mask)

        self.coupling_layers = nn.ModuleList(
            [CouplingLayer(dim, hidden_dim, masks[i]) for i in range(n_blocks)]
        )

        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), device=x.device)
        for layer in self.coupling_layers:
            x, ld = layer(x, reverse=False)
            log_det += ld
        x = x * self.scale
        return x, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z / self.scale
        log_det = -torch.log(torch.abs(self.scale)).sum() * torch.ones(
            z.size(0), device=z.device
        )

        for layer in reversed(self.coupling_layers):
            z, ld = layer(z, reverse=True)
            log_det += ld

        return z, log_det

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_prob = -0.5 * (z**2).sum(dim=-1) + log_det
        return log_prob


class GlowBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim // 2)
        self.conv1 = nn.Conv1d(dim // 2, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.conv3 = nn.Conv1d(hidden_dim, dim, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(dim)

        self.act_norm = nn.Parameter(torch.ones(dim // 2))

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=1)

        if not reverse:
            h = self.norm1(x1)
            h = self.conv1(h)
            h = F.relu(h)
            h = self.conv2(h)
            h = F.relu(h)
            h = self.conv3(h)

            log_s = torch.tanh(h[:, : x2.size(1), :])
            t = h[:, x2.size(1) :, :]

            x2 = x2 * torch.exp(log_s) + t
            x2 = x2 * self.act_norm

            log_det = log_s.sum(dim=(1, 2))
            return torch.cat([x1, x2], dim=1), log_det
        else:
            x2 = x2 / self.act_norm

            h = self.norm1(x1)
            h = self.conv1(h)
            h = F.relu(h)
            h = self.conv2(h)
            h = F.relu(h)
            h = self.conv3(h)

            log_s = torch.tanh(h[:, : x2.size(1), :])
            t = h[:, x2.size(1) :, :]

            x2 = (x2 - t) * torch.exp(-log_s)

            log_det = -log_s.sum(dim=(1, 2))
            return torch.cat([x1, x2], dim=1), log_det


class Glow(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks

        self.blocks = nn.ModuleList(
            [GlowBlock(dim, hidden_dim) for _ in range(n_blocks)]
        )

        self.prior = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim * 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = x.shape
        x = x.unsqueeze(-1)

        log_det = torch.zeros(B, device=x.device)
        for block in self.blocks:
            x, ld = block(x, reverse=False)
            log_det += ld

        mean, log_scale = self.prior(x.squeeze(-1)).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)
        log_det -= 0.5 * (
            log_scale + 0.5 * (x.squeeze(-1) - mean) ** 2 * torch.exp(-2 * log_scale)
        ).sum(dim=-1)

        return x.squeeze(-1), log_det

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.dim, device=device)
        x = z

        for block in reversed(self.blocks):
            x, _ = block(x, reverse=True)

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_prob = -0.5 * (z**2).sum(dim=-1) + log_det
        return log_prob


class MAF(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 5,
    ):
        super().__init__()
        self.dim = dim

        self.layers = nn.ModuleList()
        for i in range(n_blocks):
            mask = torch.zeros(dim)
            mask[: dim // 2] = 1 if i % 2 == 0 else 0
            if i % 2 == 1:
                mask = 1 - mask
            self.register_buffer(f"mask_{i}", mask)

            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim * 2),
                )
            )

        self.base_dist_mean = nn.Parameter(torch.zeros(dim))
        self.base_dist_logstd = nn.Parameter(torch.ones(dim))

    def get_mask(self, idx: int) -> torch.Tensor:
        return getattr(self, f"mask_{idx}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), device=x.device)
        for i, layer in enumerate(self.layers):
            mask = self.get_mask(i)
            x_masked = x * mask

            params = layer(x_masked)
            mu = params[:, : self.dim]
            log_sigma = torch.tanh(params[:, self.dim :])

            x = x * mask + (x * torch.exp(log_sigma) + mu) * (1 - mask)
            log_det += ((1 - mask) * log_sigma).sum(dim=-1)

        return x, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.size(0), device=z.device)
        x = z

        for i, layer in enumerate(reversed(self.layers)):
            idx = len(self.layers) - 1 - i
            mask = self.get_mask(idx)
            x_masked = x * mask

            params = layer(x_masked)
            mu = params[:, : self.dim]
            log_sigma = torch.tanh(params[:, self.dim :])

            x = x * mask + (x - mu) * torch.exp(-log_sigma) * (1 - mask)
            log_det -= ((1 - mask) * log_sigma).sum(dim=-1)

        return x, log_det

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = self.base_dist_mean + torch.exp(self.base_dist_logstd) * torch.randn(
            num_samples, self.dim, device=device
        )
        x, _ = self.inverse(z)
        return x


class NeuralODEFunc(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralODEFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        time_steps: int = 10,
        solver: str = "euler",
    ):
        super().__init__()
        self.dim = dim
        self.time_steps = time_steps
        self.solver = solver

        self.func = NeuralODEFunc(dim, hidden_dim)

        self.register_buffer("t", torch.linspace(0, 1, time_steps))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), device=x.device)

        z = x
        for i in range(len(self.t) - 1):
            t0 = self.t[i]
            t1 = self.t[i + 1]

            if self.solver == "euler":
                dt = t1 - t0
                dz = self.func(t0, z) * dt
                z = z + dz

                trace = torch.zeros(x.size(0), device=x.device)
                for j in range(self.dim):
                    h = 1e-5
                    z_plus = z.clone()
                    z_plus[:, j] += h
                    dz_plus = self.func(t0, z_plus) * dt
                    trace_j = (dz_plus[:, j] - dz[:, j]) / h
                    trace += trace_j

                log_det -= trace

        return z, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.size(0), device=z.device)

        x = z
        for i in range(len(self.t) - 2, -1, -1):
            t0 = self.t[i + 1]
            t1 = self.t[i]

            dt = t1 - t0
            dx = -self.func(t0, x) * dt
            x = x + dx

            trace = torch.zeros(z.size(0), device=z.device)
            for j in range(self.dim):
                h = 1e-5
                x_plus = x.clone()
                x_plus[:, j] += h
                dx_plus = -self.func(t0, x_plus) * dt
                trace_j = (dx_plus[:, j] - dx[:, j]) / h
                trace += trace_j

            log_det -= trace

        return x, log_det

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_prob = -0.5 * (z**2).sum(dim=-1) + log_det
        return log_prob
