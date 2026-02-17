"""
Neural Ordinary Differential Equations (Neural ODEs).

Continuous-depth models that use ODE solvers for forward propagation.
Combines naturally with the Hamiltonian framework.

Based on: Neural Ordinary Differential Equations (Chen et al., NeurIPS 2018)
"""

from typing import Optional, Tuple, Callable, Dict, Any, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
import numpy as np

from ..core.types import PhaseSpaceState


class ODEFunction(nn.Module):
    """
    Learnable ODE dynamics function: dz/dt = f(z, t, θ).

    This is the neural network that defines the vector field.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 200,
        n_layers: int = 3,
        time_invariant: bool = True,
        activation: str = "tanh",
    ):
        super().__init__()
        self.dim = dim
        self.time_invariant = time_invariant

        layers = []
        input_dim = dim if time_invariant else dim + 1

        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == n_layers - 1:
                layers.append(nn.Linear(hidden_dim, dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < n_layers - 1:
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "softplus":
                    layers.append(nn.Softplus())

        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """
        Compute dz/dt = f(z, t, θ).

        Args:
            t: Time (scalar)
            z: State [batch, dim]

        Returns:
            dz/dt [batch, dim]
        """
        if not self.time_invariant:
            t_expanded = t.expand(z.size(0), 1)
            z = torch.cat([z, t_expanded], dim=-1)

        return self.net(z)


class NeuralODE(nn.Module):
    """
    Neural ODE layer with various integration methods.

    Forward pass: z(t1) = z(t0) + ∫_{t0}^{t1} f(z(t), t, θ) dt

    Backward pass uses adjoint method for memory efficiency.
    """

    def __init__(
        self,
        odefunc: ODEFunction,
        t_span: Tuple[float, float] = (0.0, 1.0),
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-6,
        adjoint: bool = True,
        return_trajectory: bool = False,
    ):
        super().__init__()
        self.odefunc = odefunc
        self.t_span = t_span
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.return_trajectory = return_trajectory

    def forward(self, z0: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Integrate from t0 to t1.

        Args:
            z0: Initial state [batch, dim]

        Returns:
            z1: Final state [batch, dim]
            trajectory: Full trajectory [n_points, batch, dim] if return_trajectory=True
        """
        t = torch.linspace(
            self.t_span[0],
            self.t_span[1],
            2 if not self.return_trajectory else 20,
            device=z0.device,
        )

        integration_fn = odeint_adjoint if self.adjoint else odeint

        trajectory = integration_fn(
            self.odefunc,
            z0,
            t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        # trajectory shape: [n_points, batch, dim]
        if self.return_trajectory:
            return trajectory[-1], trajectory
        return trajectory[-1]


class AugmentedNeuralODE(nn.Module):
    """
    Augmented Neural ODE with higher-dimensional space.

    Addresses limitations of Neural ODEs by augmenting state space,
    enabling more expressive transformations.
    """

    def __init__(
        self,
        dim: int,
        augment_dim: int = 10,
        hidden_dim: int = 200,
        **ode_kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.augment_dim = augment_dim

        # ODE operates in augmented space
        self.odefunc = ODEFunction(
            dim=dim + augment_dim,
            hidden_dim=hidden_dim,
            time_invariant=True,
        )
        self.node = NeuralODE(self.odefunc, **ode_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with augmentation.

        Args:
            x: Input [batch, dim]

        Returns:
            Output [batch, dim]
        """
        # Augment with zeros
        aug = torch.zeros(x.size(0), self.augment_dim, device=x.device)
        z0 = torch.cat([x, aug], dim=-1)

        # Integrate
        z1 = self.node(z0)

        # Return original dimensions
        return z1[:, : self.dim]


class LatentODE(nn.Module):
    """
    Latent ODE model for time series.

    Encoder → ODE in latent space → Decoder
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 20,
        encoder_hidden: int = 200,
        decoder_hidden: int = 200,
        ode_hidden: int = 200,
        obsrv_std: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.obsrv_std = obsrv_std

        # Encoder: x → (μ, logσ)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, latent_dim * 2),
        )

        # ODE in latent space
        self.odefunc = ODEFunction(
            dim=latent_dim,
            hidden_dim=ode_hidden,
            time_invariant=True,
        )
        self.node = NeuralODE(self.odefunc, method="dopri5")

        # Decoder: z → x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, input_dim),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode observation to latent distribution."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: Tensor, t_future: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Observations [batch, input_dim]
            t_future: Future times to predict [n_future]

        Returns:
            Dict with reconstruction, latent, etc.
        """
        # Encode
        mu, logvar = self.encode(x)
        z0 = self.reparameterize(mu, logvar)

        # Integrate
        z1 = self.node(z0)

        # Decode
        x_recon = self.decoder(z1)

        return {
            "x_recon": x_recon,
            "z0": z0,
            "z1": z1,
            "mu": mu,
            "logvar": logvar,
        }

    def sample_trajectory(
        self,
        x0: Tensor,
        t_span: Tuple[float, float],
        n_points: int = 100,
    ) -> Tensor:
        """Sample full trajectory from initial observation."""
        mu, logvar = self.encode(x0)
        z0 = self.reparameterize(mu, logvar)

        # Integrate with trajectory
        self.node.return_trajectory = True
        z1, trajectory = self.node(z0)
        self.node.return_trajectory = False

        # Decode trajectory
        decoded = self.decoder(trajectory)

        return decoded


class SecondOrderNeuralODE(nn.Module):
    """
    Second-order Neural ODE for learning second-order dynamics.

    Models: d²z/dt² = f(z, dz/dt, t)
    which is equivalent to Hamiltonian dynamics with learnable H.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 200,
        method: str = "dopri5",
    ):
        super().__init__()
        self.dim = dim

        # State is [position, velocity]
        self.odefunc = ODEFunction(
            dim=dim * 2,
            hidden_dim=hidden_dim,
            time_invariant=True,
        )
        self.node = NeuralODE(self.odefunc, method=method)

    def forward(self, q0: Tensor, p0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Integrate second-order dynamics.

        Args:
            q0: Initial position [batch, dim]
            p0: Initial velocity [batch, dim]

        Returns:
            q1, p1: Final position and velocity
        """
        z0 = torch.cat([q0, p0], dim=-1)
        z1 = self.node(z0)

        d = self.dim
        q1, p1 = z1[:, :d], z1[:, d:]
        return q1, p1


class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous Normalizing Flow using Neural ODEs.

    Based on FFJORD (Free-form Jacobian of Reversible Dynamics).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 200,
        n_layers: int = 3,
        trace_estimator: str = "hutchinson",
    ):
        super().__init__()
        self.dim = dim
        self.trace_estimator = trace_estimator

        # ODE function for the flow
        self.odefunc = ODEFunction(
            dim=dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_invariant=False,
        )
        self.node = NeuralODE(
            self.odefunc,
            t_span=(0.0, 1.0),
            method="dopri5",
            adjoint=True,
        )

    def forward(self, z0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Transform z0 through the flow.

        Returns:
            z1: Transformed variable
            logdet: Log determinant of Jacobian
        """
        # Need to compute trace of Jacobian
        if self.trace_estimator == "exact":
            logdet = self._compute_logdet_exact(z0)
        else:
            logdet = self._compute_logdet_hutchinson(z0)

        z1 = self.node(z0)
        return z1, logdet

    def _compute_logdet_exact(self, z: Tensor) -> Tensor:
        """Compute exact log determinant (expensive for high dim)."""
        z = z.requires_grad_(True)
        dzdt = self.odefunc(torch.tensor(0.0), z)

        # Compute Jacobian
        jacobian = []
        for i in range(z.size(-1)):
            grad = torch.autograd.grad(
                dzdt[:, i].sum(), z, create_graph=True, retain_graph=True
            )[0]
            jacobian.append(grad)

        jacobian = torch.stack(jacobian, dim=-1)
        trace = torch.diagonal(jacobian, dim1=-2, dim2=-1).sum(-1)

        return trace

    def _compute_logdet_hutchinson(self, z: Tensor) -> Tensor:
        """Estimate trace using Hutchinson's trace estimator."""
        # Sample random vector
        eps = torch.randn_like(z)

        z = z.requires_grad_(True)
        dzdt = self.odefunc(torch.tensor(0.0), z)

        # Compute Jacobian-vector product
        grad = torch.autograd.grad(dzdt, z, grad_outputs=eps, create_graph=True)[0]

        # Trace estimate: E[ε^T J ε]
        trace = (grad * eps).sum(-1)
        return trace

    def inverse(self, z1: Tensor) -> Tensor:
        """Inverse transformation (integrate backwards)."""
        # Temporarily reverse time span
        original_span = self.node.t_span
        self.node.t_span = (original_span[1], original_span[0])

        z0 = self.node(z1)

        # Restore
        self.node.t_span = original_span
        return z0


def create_neural_ode_model(
    dim: int,
    hidden_dim: int = 200,
    augment: bool = False,
    augment_dim: int = 10,
    **ode_kwargs,
) -> nn.Module:
    """Factory function to create Neural ODE model."""
    if augment:
        return AugmentedNeuralODE(
            dim=dim,
            augment_dim=augment_dim,
            hidden_dim=hidden_dim,
            **ode_kwargs,
        )
    else:
        odefunc = ODEFunction(dim, hidden_dim)
        return NeuralODE(odefunc, **ode_kwargs)
