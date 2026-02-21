# Physics-Informed Neural Networks (PINN) Module

Implementation of physics-informed machine learning for solving PDEs and inverse problems with neural networks.

## Overview

This module provides comprehensive tools for:

- **Physics-Informed Neural Networks (PINNs)**: Neural networks trained to satisfy physical laws
- **PDE Specification**: Define arbitrary PDEs with automatic differentiation
- **Boundary Conditions**: Dirichlet, Neumann, Robin, Periodic
- **Inverse Problems**: Learn parameters from observed data
- **Conservation Laws**: Enforce physical conservation principles
- **Pre-built Solvers**: Common PDE equations

## Installation

```bash
# Core dependencies (included in fishstick)
pip install torch numpy scipy
```

## Quick Start

### Basic PINN for PDE Solving

```python
import torch
from fishstick.physics_informed import (
    PINN,
    PINNLoss,
    CollocationPoints,
    DirichletBC,
    grad
)

# Define PDE: u_t + u*u_x - 0.01*u_xx = 0 (Burgers equation)
class BurgersPDE:
    def residual(self, x, u):
        u_t = grad(u, x, create_graph=True)
        u_x = grad(u, x, create_graph=True)
        u_xx = grad(u_x, x, create_graph=True)
        return u_t + u * u_x - 0.01 * u_xx

# Create PINN
pinn = PINN(
    input_dim=2,  # (x, t)
    output_dim=1, # u(x,t)
    hidden_dims=[64, 64, 64, 64]
)

# Generate collocation points
domain = CollocationPoints(
    x_bounds=(-1.0, 1.0),
    t_bounds=(0.0, 1.0),
    n_points=10000
)
domain.sample()

# Define boundary conditions
bc = DirichletBC(boundary_fn=lambda x: torch.sin(torch.pi * x))

# Training
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

for epoch in range(10000):
    optimizer.zero_grad()
    
    # Physics loss
    x_coll = domain.x
    u_pred = pinn(x_coll)
    residual = burgers_pde.residual(x_coll, u_pred)
    physics_loss = torch.mean(residual ** 2)
    
    # Boundary loss
    x_bc = bc.sample_points(100)
    u_bc_pred = pinn(x_bc)
    bc_loss = torch.mean((u_bc_pred - bc.target) ** 2)
    
    # Total loss
    loss = physics_loss + bc_loss
    loss.backward()
    optimizer.step()
```

### Solving the Heat Equation

```python
from fishstick.physics_informed import (
    HeatEquationSolver,
    NeumannBC
)

# Solve heat equation: u_t = alpha * u_xx
solver = HeatEquationSolver(
    alpha=0.01,  # Thermal diffusivity
    domain=[0, 1],
    nx=100
)

# Initial condition: u(x,0) = sin(pi*x)
ic = lambda x: torch.sin(torch.pi * x)

# Neumann BC: u_x(0,t) = u_x(1,t) = 0
bc_left = NeumannBC(lambda x: torch.zeros_like(x))
bc_right = NeumannBC(lambda x: torch.zeros_like(x))

# Solve
solution = solver.solve(
    initial_condition=ic,
    boundary_conditions=[bc_left, bc_right],
    t_final=1.0,
    n_steps=100
)
```

### Inverse Problem: Learn PDE Parameters

```python
from fishstick.physics_informed import InversePINN

# Given sparse observations, learn the PDE coefficient
pinn = InversePINN(
    input_dim=2,
    output_dim=1,
    learnable_params=['nu'],  # Learn viscosity
    hidden_dims=[64, 64, 64]
)

# Observed data
x_data = torch.tensor([[0.1], [0.5], [0.9]])
t_data = torch.tensor([[0.0], [0.0], [0.0]])
u_data = torch.tensor([[0.1], [0.5], [0.1]])

# Training with inverse loss
loss_fn = PINNLoss(pinn)

for epoch in range(10000):
    # Data loss
    u_pred = pinn(torch.cat([x_data, t_data], dim=1))
    data_loss = torch.mean((u_pred - u_data) ** 2)
    
    # Physics loss
    physics_loss = loss_fn.compute_physics_loss(pinn)
    
    loss = data_loss + physics_loss
    loss.backward()

print(f"Learned nu: {pinn.get_param('nu')}")
```

### Conservation Law Enforcement

```python
from fishstick.physics_informed import (
    ConservationLaw,
    MomentumConservation,
    EnergyConservation,
    ConservationPenalty
)

# Define conservation law
conservation = ConservationLaw(
    name="energy",
    quantity_fn=lambda u: 0.5 * u ** 2,  # Energy
    flux_fn=lambda u: 0.5 * u ** 3         # Energy flux
)

# Add to loss
conservation_loss = ConservationPenalty(
    conservation_laws=[conservation],
    penalty_weight=1.0
)
```

### Domain-Adaptive PINN

```python
from fishstick.physics_informed import (
    FourierFeatures,
    DomainAdaptiveLayer,
    AdaptiveSampler
)

# Use Fourier features for periodic solutions
fourier = FourierFeatures(
    n_features=64,
    scale=1.0
)

# Adaptive sampling for better resolution
sampler = AdaptiveSampler(
    n_initial=1000,
    n_adaptive=100,
    residual_threshold=0.1
)

# Update collocation points during training
for epoch in range(10000):
    # Compute residuals
    residuals = compute_residuals(pinn, collocation_points)
    
    # Add high-residual points
    sampler.adapt(residuals)
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `PINN` | Main PINN architecture |
| `PhysicsInformedNeuralNetwork` | Full-featured PINN |
| `InversePINN` | For inverse problems |

### PDE Definition

| Class | Description |
|-------|-------------|
| `PDEDescriptor` | Define custom PDEs |
| `TimeDependentPDE` | Time-dependent PDEs |
| `InversePDE` | Inverse problem setup |

### Autodiff

| Function | Description |
|----------|-------------|
| `grad` | First derivative |
| `jacobian` | Jacobian matrix |
| `hessian` | Hessian matrix |
| `laplacian` | Laplacian operator |
| `divergence` | Divergence |

### Loss Functions

| Class | Description |
|-------|-------------|
| `PhysicsLoss` | PDE residual loss |
| `DataLoss` | Data fitting loss |
| `BoundaryLoss` | Boundary condition loss |
| `InitialLoss` | Initial condition loss |
| `CombinedLoss` | Weighted combination |

### Boundary Conditions

| Class | Description |
|-------|-------------|
| `DirichletBC` | u = specified value |
| `NeumannBC` | du/dn = specified |
| `RobinBC` | du/dn + alpha*u = beta |
| `PeriodicBC` | Periodic boundary |

### Pre-built Solvers

| Class | Description |
|-------|-------------|
| `HeatEquationSolver` | Heat equation |
| `WaveEquationSolver` | Wave equation |
| `BurgersEquationSolver` | Burgers equation |
| `NavierStokesSolver` | Navier-Stokes |
| `SchrodingerSolver` | Schrödinger equation |

## Examples

```python
# Solve Poisson equation
from fishstick.physics_informed import PoissonSolver

solver = PoissonSolver(domain=[0, 1])
solution = solver.solve(rhs=lambda x: -torch.pi**2 * torch.sin(torch.pi * x))

# Solve inverse Navier-Stokes
from fishstick.physics_informed import NavierStokesSolver
```

## References

- Raissi et al., "Physics-Informed Neural Networks" (JCP 2019)
- Raissi et al., "Physics-Informed Deep Learning" (Nature 2021)
- Karniadakis et al., "Physics-informed machine learning" (Nature 2021)

## License

MIT License - see project root for details.
