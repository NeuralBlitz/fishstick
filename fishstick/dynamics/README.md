# dynamics - Hamiltonian and Thermodynamic Dynamics Module

## Overview

The `dynamics` module implements Hamiltonian neural networks, symplectic integration, and thermodynamic gradient flow for physics-informed learning with conservation guarantees.

## Purpose and Scope

This module enables:
- Hamiltonian neural networks with energy conservation
- Symplectic integrators (leapfrog, Yoshida)
- Thermodynamic gradient flow optimization
- Noether conservation law enforcement
- Free energy minimization

## Key Classes and Functions

### Hamiltonian Dynamics (`hamiltonian.py`)

#### `HamiltonianNeuralNetwork`
Neural network that learns Hamiltonian dynamics.

```python
from fishstick.dynamics import HamiltonianNeuralNetwork

hnn = HamiltonianNeuralNetwork(
    input_dim=2,      # Phase space dimension
    hidden_dim=200,
    n_hidden=2
)

# Forward pass: computes dz/dt from Hamilton's equations
z = torch.randn(32, 4)  # [batch, 2*dim] (q, p concatenated)
dzdt = hnn(z)

# Compute Hamiltonian
H = hnn.hamiltonian(z)

# Integrate trajectory
z0 = torch.randn(1, 4)
trajectory = hnn.integrate(z0, n_steps=100, dt=0.1, method="leapfrog")

# Energy conservation loss
loss = hnn.energy_conservation_loss(z_before, z_after)
```

#### `SymplecticIntegrator`
Standalone symplectic integrator.

```python
from fishstick.dynamics import SymplecticIntegrator

def hamiltonian(z):
    q, p = z[..., :2], z[..., 2:]
    return 0.5 * (p**2).sum() + potential(q)

integrator = SymplecticIntegrator(
    hamiltonian=hamiltonian,
    dim=2,
    method="leapfrog"  # or "yoshida4", "symplectic_euler"
)

z1 = integrator.step(z0, dt=0.01)
```

#### `HamiltonianLayer`
Neural network layer with Hamiltonian structure.

```python
from fishstick.dynamics import HamiltonianLayer

layer = HamiltonianLayer(dim=10, dt=0.1)
z_next = layer(z)  # Symplectic update
```

#### `NoetherConservation`
Enforce Noether's theorem.

```python
from fishstick.dynamics import NoetherConservation, PhaseSpaceState

conservation = NoetherConservation(symmetry_group="time_translation")
conservation.add_conservation_law(
    name="energy",
    quantity_fn=lambda s: (s.p ** 2).sum() / 2,
    tolerance=1e-6
)

# Check conservation
is_conserved, violations = conservation.check_all(state_before, state_after)

# Conservation loss for training
loss = conservation.conservation_loss(state_before, state_after)
```

### Thermodynamic Dynamics (`thermodynamic.py`)

#### `FreeEnergy`
Variational free energy functional.

```python
from fishstick.dynamics import FreeEnergy

free_energy = FreeEnergy(
    likelihood_fn=log_likelihood,
    prior_fn=log_prior,
    beta=1.0
)

# Compute free energy
F = free_energy(q_mean, q_cov, n_samples=10)
```

#### `ThermodynamicGradientFlow`
Thermodynamic gradient flow optimizer.

```python
from fishstick.dynamics import ThermodynamicGradientFlow

optimizer = ThermodynamicGradientFlow(
    params=model.parameters(),
    lr=0.01,
    beta=1.0,
    temperature=1.0
)

# Training step
loss, work = optimizer.step(lambda: model.loss(x, y))

# Get free energy estimate
F = optimizer.get_free_energy_estimate()

# Check convergence
converged = optimizer.convergence_certificate()

# Efficiency
eta = optimizer.thermodynamic_efficiency()
```

#### `WassersteinGradientFlow`
Wasserstein gradient flow for probability measures.

```python
from fishstick.dynamics import WassersteinGradientFlow

wgf = WassersteinGradientFlow(
    dim=10,
    n_particles=100,
    lr=0.01
)

# Step
potential = wgf.gradient_flow_step(potential_fn)

# Sample
samples = wgf.sample(n=100)
```

#### `LandauerBound`
Landauer's principle for thermodynamic bounds.

```python
from fishstick.dynamics import LandauerBound

landauer = LandauerBound(temperature=300.0)

# Minimum energy for bit erasure
E_min = landauer.minimum_energy(bits_erased=1.0)

# Entropy reduction bound
E_bound = landauer.entropy_reduction_bound(S_initial, S_final)
```

## Mathematical Background

### Hamiltonian Mechanics
Hamilton's equations:
- dq/dt = ∂H/∂p
- dp/dt = -∂H/∂q

Symplectic integrators preserve the symplectic form ω = dq ∧ dp.

### Thermodynamic Gradient Flow
Learning as non-equilibrium thermodynamics:
- Work: W = ∫ ∇L · dθ
- Jarzynski equality: ⟨e^{-βW}⟩ = e^{-βΔF}
- Free energy: F[q] = E_q[log q] - E_q[log p(D,θ)]

## Dependencies

- `torch`: PyTorch tensors and autograd
- `numpy`: Numerical operations

## Usage Examples

### Hamiltonian Neural Network Training

```python
from fishstick.dynamics import HamiltonianNeuralNetwork
import torch

# Create HNN
hnn = HamiltonianNeuralNetwork(input_dim=2, hidden_dim=64)

# Generate training data from true dynamics
def true_hamiltonian(q, p):
    return 0.5 * (p**2 + q**2)  # Harmonic oscillator

# Training
optimizer = torch.optim.Adam(hnn.parameters(), lr=1e-3)

for epoch in range(1000):
    # Sample phase space points
    z = torch.randn(64, 4, requires_grad=True)
    
    # Compute true dynamics
    q, p = z[:, :2], z[:, 2:]
    H = true_hamiltonian(q, p)
    dHdq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
    dHdp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
    true_dzdt = torch.cat([dHdp, -dHdq], dim=1)
    
    # Predicted dynamics
    pred_dzdt = hnn(z)
    
    # Loss
    loss = ((pred_dzdt - true_dzdt) ** 2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Thermodynamic Training

```python
from fishstick.dynamics import ThermodynamicGradientFlow

# Setup
optimizer = ThermodynamicGradientFlow(
    model.parameters(),
    lr=0.01,
    temperature=1.0
)

for x, y in dataloader:
    loss, work = optimizer.step(lambda: model.loss(x, y))
    
    if optimizer.convergence_certificate():
        print("Converged!")
        break
```
