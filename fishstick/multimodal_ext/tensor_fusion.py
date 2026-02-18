"""
Tensor Fusion and Factorized Fusion for fishstick

This module provides tensor-based fusion mechanisms:
- Tensor Fusion Network (TFN)
- Factorized Bilinear Networks
- Multimodal Tensor Fusion
- Low-rank tensor fusion
"""

from typing import List, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TensorFusionNetwork(nn.Module):
    """Tensor Fusion Network for multi-modal learning."""

    def __init__(
        self,
        modality_dims: List[int],
        output_dim: int,
    ):
        super().__init__()
        self.modality_dims = modality_dims

        total_dim = 1
        for dim in modality_dims:
            total_dim *= dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, modalities: List[Tensor]) -> Tensor:
        if len(modalities) != len(self.modality_dims):
            raise ValueError(
                f"Expected {len(self.modality_dims)} modalities, got {len(modalities)}"
            )

        tensor_products = []
        for i, (mod, dim) in enumerate(zip(modalities, self.modality_dims)):
            if mod.size(-1) != dim:
                raise ValueError(
                    f"Modality {i} has dimension {mod.size(-1)}, expected {dim}"
                )
            tensor_products.append(mod.unsqueeze(-1))

        fused = tensor_products[0]
        for tensor_prod in tensor_products[1:]:
            fused = fused * tensor_prod

        fused = fused.view(fused.size(0), -1)
        return self.fusion(fused)


class FactorizedBilinearNetwork(nn.Module):
    """Factorized Bilinear Network for multi-modal fusion."""

    def __init__(
        self,
        modality1_dim: int,
        modality2_dim: int,
        output_dim: int,
        factor_dim: int = 64,
    ):
        super().__init__()
        self.factor_dim = factor_dim

        self.factor1 = nn.Linear(modality1_dim, factor_dim)
        self.factor2 = nn.Linear(modality2_dim, factor_dim)

        self.output = nn.Sequential(
            nn.Linear(factor_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        modality1: Tensor,
        modality2: Tensor,
    ) -> Tensor:
        f1 = self.factor1(modality1)
        f2 = self.factor2(modality2)

        fused = f1 * f2

        return self.output(fused)


class MultimodalFactorizedBilinear(nn.Module):
    """Multimodal factorized bilinear fusion for multiple modalities."""

    def __init__(
        self,
        modality_dims: List[int],
        output_dim: int,
        factor_dim: int = 64,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)

        self.factors = nn.ModuleList(
            [nn.Linear(dim, factor_dim) for dim in modality_dims]
        )

        self.output = nn.Sequential(
            nn.Linear(factor_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, modalities: List[Tensor]) -> Tensor:
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(modalities)}"
            )

        factors = [factor(mod) for factor, mod in zip(self.factors, modalities)]

        fused = factors[0]
        for factor in factors[1:]:
            fused = fused * factor

        return self.output(fused)


class LowRankTensorFusion(nn.Module):
    """Low-rank tensor fusion for efficient multi-modal learning."""

    def __init__(
        self,
        modality_dims: List[int],
        output_dim: int,
        rank: int = 16,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.rank = rank
        self.num_modalities = len(modality_dims)

        self.cores = nn.ModuleList(
            [nn.Parameter(torch.randn(dim, rank)) for dim in modality_dims]
        )

        self.output = nn.Sequential(
            nn.Linear(rank, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, modalities: List[Tensor]) -> Tensor:
        core_products = []
        for mod, core in zip(modalities, self.cores):
            projected = mod @ core
            core_products.append(projected)

        fused = core_products[0]
        for cp in core_products[1:]:
            fused = fused * cp

        return self.output(fused)


class HadamardFusion(nn.Module):
    """Hadamard (element-wise) product fusion."""

    def __init__(
        self,
        modality_dims: List[int],
        output_dim: int,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.projection = nn.Linear(max(modality_dims), output_dim)

    def forward(self, modalities: List[Tensor]) -> Tensor:
        aligned_modalities = []
        max_dim = max(self.modality_dims)

        for mod, dim in zip(modalities, self.modality_dims):
            if dim < max_dim:
                padding = torch.zeros(
                    *mod.shape[:-1], max_dim - dim, device=mod.device, dtype=mod.dtype
                )
                mod = torch.cat([mod, padding], dim=-1)
            aligned_modalities.append(mod)

        fused = aligned_modalities[0]
        for mod in aligned_modalities[1:]:
            fused = fused * mod

        return self.projection(fused)


class ConcatenationFusion(nn.Module):
    """Simple concatenation fusion."""

    def __init__(
        self,
        modality_dims: List[int],
        output_dim: int,
    ):
        super().__init__()
        total_dim = sum(modality_dims)

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, modalities: List[Tensor]) -> Tensor:
        concatenated = torch.cat(modalities, dim=-1)
        return self.fusion(concatenated)


class GMU(nn.Module):
    """Gated Multimodal Unit for multi-modal fusion."""

    def __init__(
        self,
        modality1_dim: int,
        modality2_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.mod1_transform = nn.Sequential(
            nn.Linear(modality1_dim, output_dim),
            nn.ReLU(),
        )
        self.mod2_transform = nn.Sequential(
            nn.Linear(modality2_dim, output_dim),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(modality1_dim + modality2_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        modality1: Tensor,
        modality2: Tensor,
    ) -> Tensor:
        h1 = self.mod1_transform(modality1)
        h2 = self.mod2_transform(modality2)

        gate = self.gate(torch.cat([modality1, modality2], dim=-1))

        fused = gate * h1 + (1 - gate) * h2
        return fused


class MultimodalBottleneck(nn.Module):
    """Multimodal bottleneck fusion."""

    def __init__(
        self,
        modality_dims: List[int],
        bottleneck_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, bottleneck_dim * 2),
                    nn.ReLU(),
                    nn.Linear(bottleneck_dim * 2, bottleneck_dim),
                )
                for dim in modality_dims
            ]
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim * len(modality_dims), bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, modalities: List[Tensor]) -> Tensor:
        encoded = []
        for encoder, mod in zip(self.encoders, modalities):
            encoded.append(encoder(mod))

        concatenated = torch.cat(encoded, dim=-1)
        return self.decoder(concatenated)


class MixFusion(nn.Module):
    """Mix of different fusion strategies."""

    def __init__(
        self,
        modality_dims: List[int],
        output_dim: int,
        fusion_types: List[str] = ["concatenation", "hadamard", "tensor"],
    ):
        super().__init__()
        self.fusion_types = fusion_types

        self.concatenation = ConcatenationFusion(modality_dims, output_dim)
        self.hadamard = HadamardFusion(modality_dims, output_dim)
        self.tensor = TensorFusionNetwork(modality_dims, output_dim)

        self.weight = nn.Parameter(torch.ones(len(fusion_types)))

    def forward(self, modalities: List[Tensor]) -> Tensor:
        fused_list = []

        if "concatenation" in self.fusion_types:
            fused_list.append(self.concatenation(modalities))
        if "hadamard" in self.fusion_types:
            fused_list.append(self.hadamard(modalities))
        if "tensor" in self.fusion_types:
            fused_list.append(self.tensor(modalities))

        weights = F.softmax(self.weight, dim=0)

        fused = sum(w * f for w, f in zip(weights, fused_list))
        return fused


def create_tensor_fusion(
    fusion_type: str = "tensor",
    modality_dims: List[int] = None,
    output_dim: int = 256,
    **kwargs,
) -> nn.Module:
    """Factory function to create tensor fusion modules."""
    if modality_dims is None:
        modality_dims = [512, 512]

    if fusion_type == "tensor":
        return TensorFusionNetwork(modality_dims, output_dim)
    elif fusion_type == "factorized":
        return FactorizedBilinearNetwork(
            modality_dims[0], modality_dims[1], output_dim, **kwargs
        )
    elif fusion_type == "multimodal_factorized":
        return MultimodalFactorizedBilinear(modality_dims, output_dim, **kwargs)
    elif fusion_type == "low_rank":
        return LowRankTensorFusion(modality_dims, output_dim, **kwargs)
    elif fusion_type == "hadamard":
        return HadamardFusion(modality_dims, output_dim)
    elif fusion_type == "concatenation":
        return ConcatenationFusion(modality_dims, output_dim)
    elif fusion_type == "gmu":
        return GMU(modality_dims[0], modality_dims[1], output_dim)
    elif fusion_type == "bottleneck":
        return MultimodalBottleneck(modality_dims, output_dim // 2, output_dim)
    elif fusion_type == "mix":
        return MixFusion(modality_dims, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
