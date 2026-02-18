"""
fishstick.physics_informed
==========================

Physics-Informed Machine Learning (PINN) tools for solving PDEs and
inverse problems with neural networks.

Provides:
- Physics-Informed Neural Networks (PINNs)
- PDE specification and autodiff operators
- Loss formulations for physics constraints
- Boundary and initial condition handling
- Conservation law enforcement
- PDE solvers for common equations

Based on: Physics-Informed Neural Networks (Raissi et al., 2019)
"""

from .pde_base import (
    PDE,
    PDEDescriptor,
    TimeDependentPDE,
    InversePDE,
    register_pde,
    list_registered_pdes,
)

from .autodiff import (
    grad,
    jacobian,
    hessian,
    divergence,
    laplacian,
    batch_jacobian,
    batch_hessian,
)

from .pinn import (
    PINN,
    PhysicsInformedNeuralNetwork,
    create_pinn,
    ForwardPINN,
    InversePINN,
)

from .pinn_layers import (
    FourierFeatures,
    SinusoidalRepresentation,
    PeriodicFeatures,
    WaveletFeatures,
    DomainAdaptiveLayer,
    ResidualBlockPINN,
)

from .domain import (
    CollocationPoints,
    DomainSampler,
    TemporalSampler,
    SpatioTemporalSampler,
    AdaptiveSampler,
    BoundarySampler,
)

from .loss import (
    PINNLoss,
    PhysicsLoss,
    DataLoss,
    BoundaryLoss,
    InitialLoss,
    CombinedLoss,
    SoftBoundaryLoss,
    PenaltyMethodLoss,
    AugmentedLagrangianLoss,
)

from .boundary import (
    BoundaryCondition,
    DirichletBC,
    NeumannBC,
    RobinBC,
    PeriodicBC,
    CauchyBC,
    BoundaryConditionHandler,
    enforce_dirichlet,
    enforce_neumann,
    compute_boundary_penalty,
)

from .initial import (
    InitialCondition,
    InitialConditionHandler,
    enforce_initial_condition,
    compute_initial_penalty,
)

from .solvers import (
    PDESolver,
    HeatEquationSolver,
    WaveEquationSolver,
    BurgersEquationSolver,
    PoissonSolver,
    NavierStokesSolver,
    SchrodingerSolver,
    AllenCahnSolver,
)

from .conservation import (
    ConservationLaw,
    MomentumConservation,
    EnergyConservation,
    MassConservation,
    enforce_conservation,
    ConservationPenalty,
    IntegralConservation,
)

from .utils import (
    compute_relative_error,
    compute_l2_error,
    compute_h1_error,
    compute_max_error,
    format_pinn_output,
    prepare_pinn_training,
    validate_pinn_solution,
)
