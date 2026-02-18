# TODO: Normalizing Flows Extensions for fishstick AI Framework

## Phase 1: Core Infrastructure
- [x] Create flows_ext directory structure
- [x] Create base flow module with common interfaces

## Phase 2: Neural Spline Flows
- [x] Implement rational quadratic spline (RQS) transform
- [x] Create NeuralSplineFlow class with coupling layers
- [x] Add learnable knot positions and derivatives
- [x] Implement 1D and 2D spline flows

## Phase 3: FFJORD Implementation
- [x] Create FFJORD continuous normalizing flow
- [x] Implement ODE solver (RK4, Euler)
- [x] Add trace estimation for Jacobian
- [x] Create FFJORD network (MLP with residual connections)

## Phase 4: Masked Autoregressive Flows
- [x] Implement MAF (Masked Autoregressive Flow)
- [x] Create MADE (Masked Autoencoder for Density Estimation)
- [x] Add inverse transform for sampling
- [x] Implement TAN (Transformer Autoregressive Network) variant

## Phase 5: Coupling Layer Variants
- [x] Create affine coupling layer
- [x] Implement additive coupling layer
- [x] Add neural spline coupling layer
- [x] Create conditional coupling layers

## Phase 6: Flow-based Density Estimation
- [x] Implement RealNVP-style density estimation
- [x] Create flow-based VAE integration
- [x] Add likelihood computation utilities
- [x] Implement invertible networks for density estimation

## Phase 7: Integration & Exports
- [x] Create comprehensive __init__.py
- [x] Add type hints and docstrings throughout
- [x] Verify Python syntax for all modules

## Module List:
1. neural_spline_flows.py - Neural spline flows with rational quadratic splines
2. ffjord.py - FFJORD continuous normalizing flows
3. masked_autoregressive_flow.py - MAF and MADE implementations
4. coupling_layers.py - Various coupling layer implementations
5. density_estimation.py - Flow-based density estimation models
6. __init__.py - Module exports
