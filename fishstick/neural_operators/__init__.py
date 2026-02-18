from .fno import FNO1d, FNO2d, FNO3d, SpectralConv
from .deeponet import DeepONet, BranchNet, TrunkNet
from .pde import PdeNet, PodNet, NeuralGalerkin

__all__ = [
    "FNO1d",
    "FNO2d",
    "FNO3d",
    "SpectralConv",
    "DeepONet",
    "BranchNet",
    "TrunkNet",
    "PdeNet",
    "PodNet",
    "NeuralGalerkin",
]
