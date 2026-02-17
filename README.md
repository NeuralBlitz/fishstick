<img width="480" height="480" alt="image" src="https://github.com/user-attachments/assets/2369e054-b7bf-4ce9-acc2-8b2f3da78416" />



A mathematically rigorous, physically grounded AI framework synthesizing theoretical physics, formal mathematics, and advanced machine learning.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**fishstick** implements 6 unified theoretical frameworks (A-F) that combine:

- 🎯 **Theoretical Physics**: Symmetry, renormalization, variational principles, thermodynamics
- 📐 **Formal Mathematics**: Category theory, sheaf cohomology, differential geometry, type theory
- 🤖 **Advanced ML**: Equivariant deep learning, neuro-symbolic integration, formal verification

### Core Philosophy

This framework treats AI not as empirical engineering but as a branch of mathematical physics, where:
- Neural architectures are **morphisms in dagger compact closed categories**
- Training dynamics are **gradient flows on statistical manifolds**
- Attention mechanisms respect **sheaf cohomology constraints**
- Models satisfy **thermodynamic bounds** on computation

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/NeuralBlitz/fishstick.git
cd fishstick

# Install core dependencies
pip install torch numpy scipy pyyaml

# Install package
pip install -e .
```

### Full Installation (with all features)

```bash
# Install all optional dependencies
pip install torchdiffeq torch-geometric

# Or install with extras
pip install -e ".[full]"
```

### Dependencies

**Core:**
- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy ≥ 1.21
- SciPy ≥ 1.7

**Optional:**
- `torchdiffeq` - For Neural ODE solvers
- `torch-geometric` - For geometric graph neural networks

## Quick Start

### 1. Core Types and Manifolds

```python
import torch
from fishstick import (
    MetricTensor, SymplecticForm, PhaseSpaceState,
    StatisticalManifold, FisherInformationMetric
)

# Create statistical manifold
manifold = StatisticalManifold(dim=10)

# Fisher information metric
def log_prob(params):
    return (params**2).sum()

params = torch.randn(10, requires_grad=True)
metric = manifold.fisher_information(params, log_prob)

# Phase space for Hamiltonian dynamics
state = PhaseSpaceState(
    q=torch.randn(5),  # positions
    p=torch.randn(5)   # momenta
)
```

### 2. Categorical Structures

```python
from fishstick import (
    MonoidalCategory, Functor, NaturalTransformation,
    DaggerCategory, Lens
)

# Create monoidal category
cat = MonoidalCategory("NeuralCategory")

# Define objects and morphisms
from fishstick.categorical.category import Object

obj1 = Object(name="Input", shape=(784,))
obj2 = Object(name="Hidden", shape=(256,))

cat.add_object(obj1)
cat.add_object(obj2)

# Lens for bidirectional learning
lens = Lens(
    get=lambda x: x * 2,
    put=lambda s, a: s + a
)
```

### 3. Hamiltonian Neural Networks

```python
from fishstick import HamiltonianNeuralNetwork

# Energy-conserving neural network
hnn = HamiltonianNeuralNetwork(
    input_dim=10,
    hidden_dim=64,
    n_hidden=3
)

# Integrate dynamics
z0 = torch.randn(4, 20)  # batch of 4, phase space dim 20
trajectory = hnn.integrate(z0, n_steps=100, dt=0.01)

# trajectory shape: [101, 4, 20]
# Energy is conserved along trajectory
```

### 4. Sheaf-Optimized Attention

```python
from fishstick import SheafOptimizedAttention

# Attention with cohomological constraints
attn = SheafOptimizedAttention(
    embed_dim=256,
    num_heads=8,
    lambda_consistency=0.1
)

x = torch.randn(2, 100, 256)  # [batch, seq_len, embed_dim]

# Define open cover for local consistency
open_cover = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]

output, weights = attn(x, open_cover=open_cover)
```

## The 6 Frameworks

### A. UniIntelli - Categorical–Geometric–Thermodynamic Synthesis
```python
from fishstick.frameworks.uniintelli import create_uniintelli

model = create_uniintelli(
    input_dim=784,
    hidden_dim=256,
    output_dim=10
)
# 1.8M parameters
```

### B. HSCA - Holo-Symplectic Cognitive Architecture
```python
from fishstick.frameworks.hsca import create_hsca

model = create_hsca(input_dim=784, output_dim=10)
# 6.5M parameters - Energy-conserving Hamiltonian dynamics
```

### C. UIA - Unified Intelligence Architecture
```python
from fishstick.frameworks.uia import create_uia

model = create_uia(input_dim=784, output_dim=10)
# 1.7M parameters - CHNP + RG-AE + S-TF + DTL
```

### D. SCIF - Symplectic-Categorical Intelligence Framework
```python
from fishstick.frameworks.scif import create_scif

model = create_scif(input_dim=784, output_dim=10)
# 3.8M parameters - Fiber bundles + Hamiltonian dynamics
```

### E. UIF - Unified Intelligence Framework
```python
from fishstick.frameworks.uif import create_uif

model = create_uif(input_dim=784, output_dim=10)
# 367K parameters - 4-layer architecture
```

### F. UIS - Unified Intelligence Synthesis
```python
from fishstick.frameworks.uis import create_uis

model = create_uis(input_dim=784, output_dim=10)
# 861K parameters - Quantum-inspired + RG + Neuro-symbolic
```

## Advanced Features

### Neural ODEs

```python
from fishstick.neural_ode import NeuralODE, ODEFunction

# Define dynamics
odefunc = ODEFunction(dim=10, hidden_dim=64)

# Create Neural ODE with adaptive solver
node = NeuralODE(
    odefunc,
    t_span=(0.0, 1.0),
    method='dopri5',  # Dormand-Prince
    rtol=1e-5,
    atol=1e-6
)

z0 = torch.randn(4, 10)
z1 = node(z0)
```

### Geometric Graph Neural Networks

```python
from fishstick.graph import (
    EquivariantMessagePassing,
    GeometricGraphTransformer,
    MolecularGraphNetwork
)

# E(n)-equivariant message passing
layer = EquivariantMessagePassing(
    node_dim=64,
    edge_dim=0,
    hidden_dim=128
)

# Node features and 3D positions
x = torch.randn(100, 64)
pos = torch.randn(100, 3)
edge_index = torch.randint(0, 100, (2, 500))

x_out, pos_out = layer(x, pos, edge_index)
```

### Probabilistic / Bayesian Neural Networks

```python
from fishstick.probabilistic import (
    BayesianLinear,
    BayesianNeuralNetwork,
    MCDropout,
    DeepEnsemble
)

# Bayesian layer with variational inference
layer = BayesianLinear(784, 256, prior_sigma=1.0)
x = torch.randn(4, 784)
output = layer(x, sample=True)

# Full BNN
bnn = BayesianNeuralNetwork(
    input_dim=784,
    hidden_dims=[256, 128],
    output_dim=10
)

# Predict with uncertainty
mean, uncertainty = bnn.predict_with_uncertainty(x, n_samples=100)
```

### Normalizing Flows

```python
from fishstick.flows import RealNVP, Glow, MAF

# RealNVP coupling flows
flow = RealNVP(dim=8, n_coupling=8, hidden_dim=256)
x = torch.randn(100, 8)

# Density estimation
log_prob = flow.log_prob(x)

# Sampling
samples = flow.sample(1000)

# Glow with 1x1 convolutions
glow = Glow(dim=8, n_levels=3, n_steps_per_level=4)
```

### Equivariant Networks

```python
from fishstick.equivariant import (
    SE3EquivariantLayer,
    SE3Transformer,
    EquivariantMolecularEnergy
)

# SE(3)-equivariant layer
layer = SE3EquivariantLayer(
    in_features=32,
    out_features=32,
    hidden_dim=64
)

features = torch.randn(10, 32)
coords = torch.randn(10, 3)  # 3D positions
edge_index = torch.randint(0, 10, (2, 30))

f_out, c_out = layer(features, coords, edge_index)
# c_out is equivariant to rotations/reflections
```

### Causal Inference

```python
from fishstick.causal import (
    CausalGraph,
    StructuralCausalModel,
    CausalDiscovery
)

# Define causal DAG
import numpy as np
adjacency = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
])
graph = CausalGraph(n_nodes=3, adjacency=adjacency)

# Structural Causal Model
scm = StructuralCausalModel(graph, hidden_dim=64)

# Observational sampling
sample = scm.forward()

# Interventional: do(X=2.0)
intervention = {0: torch.tensor([[2.0]])}
sample_do = scm.do_calculus(intervention_node=0, 
                           value=torch.tensor([[2.0]]))

# Causal discovery from data
data = np.random.randn(1000, 3)
learned_graph = CausalDiscovery.pc_algorithm(data)
```

## Architecture

```
fishstick/
├── core/               # Fundamental types and manifolds
├── categorical/        # Category theory structures
├── geometric/          # Differential geometry
├── dynamics/           # Hamiltonian & thermodynamic
├── sheaf/             # Sheaf theory
├── rg/                # Renormalization group
├── verification/      # Formal verification
├── frameworks/        # 6 unified frameworks
├── neural_ode/        # Neural ODEs
├── graph/             # Geometric GNNs
├── probabilistic/     # Bayesian methods
├── flows/             # Normalizing flows
├── equivariant/       # Equivariant networks
└── causal/            # Causal inference
```

## Testing

Run the test suite:

```bash
# Core framework tests
python test_all.py

# Advanced features tests
python test_advanced.py

# Run all tests
python -m pytest test_all.py test_advanced.py -v
```

### Test Coverage

- **Core Framework**: 13/13 tests passed
- **Advanced Features**: 6/6 tests passed
- **Total**: 19/19 tests passed ✅

## Performance

All frameworks tested on synthetic data:

| Framework | Parameters | Forward Pass | Energy Cons. |
|-----------|------------|--------------|--------------|
| UniIntelli | 1.8M | ✓ | ✓ |
| HSCA | 6.5M | ✓ | ✓ (HNN) |
| UIA | 1.7M | ✓ | ✓ |
| SCIF | 3.8M | ✓ | ✓ |
| UIF | 367K | ✓ | ✓ |
| UIS | 861K | ✓ | ✓ |

## Documentation

For detailed mathematical documentation, see:

- `A.md` - UniIntelli: Categorical–Geometric–Thermodynamic Synthesis
- `B.md` - HSCA: Holo-Symplectic Cognitive Architecture  
- `C.md` - UIA: Unified Intelligence Architecture
- `D.md` - SCIF: Symplectic-Categorical Intelligence Framework
- `E.md` - UIF: Unified Intelligence Framework
- `F.md` - UIS: Unified Intelligence Synthesis

## Mathematical Foundations

### Key Theorems Implemented

1. **Natural Gradient = Geodesic Flow**: Theorem 2.1 - Natural gradient descent is the time-η flow of the gradient vector field w.r.t. Levi-Civita connection

2. **SOA Conserves Sheaf Cohomology**: Theorem 4.1 - Sheaf-Optimized Attention preserves δ¹(s) = 0 up to O(η²)

3. **TGF Convergence**: Theorem 5.1 - Thermodynamic Gradient Flow converges under non-equilibrium fluctuations

4. **No-Cloning in Learn**: Lemma 2.1 - No morphism Δ: A → A ⊗ A exists in the category of learning processes

## Contributing

We welcome contributions! Areas of interest:

- Implementing higher sheaf cohomology (H²) for mode connectivity
- Quantum-categorical neural networks
- Real-time formal verification with neural theorem proving
- Unified scaling laws from RG fixed points

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{fishstick_2026,
  title={fishstick: A Mathematically Rigorous AI Framework},
  author={[Your Name]},
  url={https://github.com/NeuralBlitz/fishstick},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

This framework synthesizes insights from:
- Statistical mechanics and thermodynamics
- Differential geometry and information geometry
- Category theory and type theory
- Symplectic geometry and Hamiltonian mechanics
- Sheaf theory and algebraic topology

The era of black-box AI is ending. The era of **principled intelligence** begins now.
