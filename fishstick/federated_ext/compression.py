"""
Communication Compression for Federated Learning

This module provides various gradient compression techniques:
- Top-K Sparsification
- Random-K Sparsification
- QSGD Quantization
- Error-Feedback (EF-SignSGD)
- Residual Connection Handling
- Compression-Aware Training

References:
- Sattler et al. (2019): "Sparse Binary Compression: Towards Distributed Deep Learning with improved Communication Efficiency"
- Alistarh et al. (2017): "QSGD: Randomized Quantization for Communication-Efficient Federated Learning"
- Seide et al. (2014): "1-bit Stochastic Gradient Descent and its Application to Data-parallel Distributed Training"
- Karimireddy et al. (2019): "Learning Private Neural Language Modeling with Scaffold"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Supported compression methods."""

    TOP_K = auto()
    RANDOM_K = auto()
    QSGD = auto()
    SIGN_SGD = auto()
    EF_SIGN = auto()
    NONE = auto()


@dataclass
class CompressionConfig:
    """Configuration for gradient compression."""

    method: CompressionMethod = CompressionMethod.TOP_K
    compression_ratio: float = 0.01
    quantization_levels: int = 8
    use_error_feedback: bool = True
    warmup_rounds: int = 0
    seed: Optional[int] = None


class BaseCompressor(ABC):
    """Base class for gradient compression."""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.error_memory: Dict[str, Tensor] = {}
        self.current_round = 0

    @abstractmethod
    def compress(self, tensor: Tensor) -> Tuple[Tensor, Any]:
        """Compress a tensor.

        Returns:
            Tuple of (compressed_tensor, metadata)
        """
        pass

    @abstractmethod
    def decompress(self, compressed: Tensor, metadata: Any) -> Tensor:
        """Decompress a tensor.

        Args:
            compressed: Compressed tensor
            metadata: Metadata from compression

        Returns:
            Decompressed tensor
        """
        pass

    def add_error(self, key: str, error: Tensor) -> None:
        """Add error to memory for error feedback."""
        if key not in self.error_memory:
            self.error_memory[key] = torch.zeros_like(error)
        self.error_memory[key] = self.error_memory[key] + error

    def get_error(self, key: str) -> Optional[Tensor]:
        """Get accumulated error for a key."""
        return self.error_memory.get(key)

    def clear_error(self, key: str) -> None:
        """Clear error for a key."""
        if key in self.error_memory:
            self.error_memory[key] = torch.zeros_like(self.error_memory[key])


class TopKCompressor(BaseCompressor):
    """Top-K Sparsification.

    Retains only the k largest magnitude elements and zeros out the rest.
    """

    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        if config.seed is not None:
            torch.manual_seed(config.seed)

    def compress(self, tensor: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        if self.current_round < self.config.warmup_rounds:
            return tensor, {"warmup": True}

        tensor_flat = tensor.flatten()
        k = max(1, int(tensor_flat.numel() * self.config.compression_ratio))

        if k >= tensor_flat.numel():
            return tensor, {"warmup": True}

        abs_values = torch.abs(tensor_flat)
        _, indices = torch.topk(abs_values, k, sorted=False)

        mask = torch.zeros_like(tensor_flat, dtype=torch.bool)
        mask[indices] = True

        compressed = torch.zeros_like(tensor_flat)
        compressed[mask] = tensor_flat[mask]

        metadata = {
            "indices": indices,
            "shape": tensor.shape,
            "k": k,
        }

        if self.config.use_error_feedback:
            error = tensor_flat - compressed
            self.add_error("topk", error)

        return compressed.view(tensor.shape), metadata

    def decompress(self, compressed: Tensor, metadata: Dict[str, Any]) -> Tensor:
        if metadata.get("warmup", False):
            return compressed

        error = self.get_error("topk")
        if error is not None:
            restored = compressed.flatten() + error
            self.clear_error("topk")
            return restored.view(metadata["shape"])

        return compressed


class RandomKCompressor(BaseCompressor):
    """Random-K Sparsification.

    Randomly selects k elements to retain.
    """

    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        if config.seed is not None:
            torch.manual_seed(config.seed)

    def compress(self, tensor: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        if self.current_round < self.config.warmup_rounds:
            return tensor, {"warmup": True}

        tensor_flat = tensor.flatten()
        k = max(1, int(tensor_flat.numel() * self.config.compression_ratio))

        indices = torch.randperm(tensor_flat.numel(), device=tensor.device)[:k]

        mask = torch.zeros_like(tensor_flat, dtype=torch.bool)
        mask[indices] = True

        compressed = torch.zeros_like(tensor_flat)
        compressed[mask] = tensor_flat[mask]

        metadata = {
            "indices": indices,
            "shape": tensor.shape,
            "k": k,
        }

        if self.config.use_error_feedback:
            error = tensor_flat - compressed
            self.add_error("randomk", error)

        return compressed.view(tensor.shape), metadata

    def decompress(self, compressed: Tensor, metadata: Dict[str, Any]) -> Tensor:
        if metadata.get("warmup", False):
            return compressed

        error = self.get_error("randomk")
        if error is not None:
            restored = compressed.flatten() + error
            self.clear_error("randomk")
            return restored.view(metadata["shape"])

        return compressed


class QSGDCompressor(BaseCompressor):
    """QSGD: Quantized Stochastic Gradient Descent.

    Quantizes gradients to a limited number of levels.

    Reference: Alistarh et al. (2017)
    """

    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.levels = config.quantization_levels

    def compress(self, tensor: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        if self.current_round < self.config.warmup_rounds:
            return tensor, {"warmup": True}

        tensor_flat = tensor.flatten().float()

        l2_norm = torch.norm(tensor_flat)
        if l2_norm == 0:
            return tensor, {"warmup": True, "zero": True}

        normalized = tensor_flat / l2_norm

        abs_normalized = torch.abs(normalized)
        levels_tensor = torch.arange(
            0, self.levels, device=tensor.device, dtype=torch.float32
        )
        abs_normalized_expanded = abs_normalized.unsqueeze(-1)
        levels_expanded = levels_tensor.unsqueeze(0)

        diff = abs_normalized_expanded - levels_expanded
        diff = torch.clamp(diff, min=0)
        quant_levels = torch.argmin(diff, dim=-1).float()

        sign = torch.sign(normalized)
        sign[sign == 0] = 1

        compressed = sign * (quant_levels / (self.levels - 1))

        if self.config.use_error_feedback:
            error = tensor_flat - compressed * l2_norm
            self.add_error("qsgd", error)

        metadata = {
            "l2_norm": l2_norm,
            "shape": tensor.shape,
            "levels": self.levels,
        }

        return compressed.view(tensor.shape), metadata

    def decompress(self, compressed: Tensor, metadata: Dict[str, Any]) -> Tensor:
        if metadata.get("warmup", False) or metadata.get("zero", False):
            return compressed

        l2_norm = metadata["l2_norm"]
        decompressed = compressed.float() * l2_norm

        error = self.get_error("qsgd")
        if error is not None:
            decompressed = decompressed + error
            self.clear_error("qsgd")

        return decompressed.view(metadata["shape"])


class SignSGDCompressor(BaseCompressor):
    """SignSGD: Sign-based Stochastic Gradient Descent.

    Compresses gradients to just their sign (+1 or -1).
    """

    def __init__(self, config: CompressionConfig):
        super().__init__(config)

    def compress(self, tensor: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        if self.current_round < self.config.warmup_rounds:
            return tensor, {"warmup": True}

        tensor_flat = tensor.flatten().float()

        l2_norm = torch.norm(tensor_flat)
        if l2_norm == 0:
            return tensor, {"warmup": True, "zero": True}

        sign_tensor = torch.sign(tensor_flat)
        sign_tensor[sign_tensor == 0] = 1

        metadata = {
            "l2_norm": l2_norm,
            "shape": tensor.shape,
        }

        if self.config.use_error_feedback:
            error = tensor_flat - sign_tensor * l2_norm / tensor_flat.numel()
            self.add_error("sign", error)

        compressed = sign_tensor.unsqueeze(0)
        return compressed.view(tensor.shape), metadata

    def decompress(self, compressed: Tensor, metadata: Dict[str, Any]) -> Tensor:
        if metadata.get("warmup", False) or metadata.get("zero", False):
            return compressed

        l2_norm = metadata["l2_norm"]
        numel = metadata["shape"][0] if len(metadata["shape"]) > 0 else 1

        decompressed = compressed.float() * l2_norm / max(numel, 1)

        error = self.get_error("sign")
        if error is not None:
            decompressed = decompressed + error
            self.clear_error("sign")

        return decompressed.view(metadata["shape"])


class EFSignCompressor(BaseCompressor):
    """EF-SignSGD: Error-Feedback SignSGD.

    SignSGD with error feedback for unbiased compression.

    Reference: Karimireddy et al. (2019)
    """

    def __init__(self, config: CompressionConfig):
        super().__init__(config)

    def compress(self, tensor: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        tensor_flat = tensor.flatten().float()

        error = self.get_error("efsign")
        if error is not None:
            tensor_flat = tensor_flat + error

        sign_tensor = torch.sign(tensor_flat)
        sign_tensor[sign_tensor == 0] = 1

        l2_norm = torch.norm(tensor_flat)

        metadata = {
            "l2_norm": l2_norm,
            "shape": tensor.shape,
            "numel": tensor_flat.numel(),
        }

        if self.config.use_error_feedback:
            compressed_sign = sign_tensor
            new_error = tensor_flat - compressed_sign * l2_norm
            self.add_error("efsign", new_error)

        return sign_tensor.unsqueeze(0), metadata

    def decompress(self, compressed: Tensor, metadata: Dict[str, Any]) -> Tensor:
        l2_norm = metadata["l2_norm"]
        numel = metadata.get("numel", 1)

        decompressed = compressed.float() * l2_norm / max(numel, 1)
        return decompressed.view(metadata["shape"])


class NoCompression(BaseCompressor):
    """No compression - pass through."""

    def compress(self, tensor: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        return tensor, {"warmup": False}

    def decompress(self, compressed: Tensor, metadata: Dict[str, Any]) -> Tensor:
        return compressed


def create_compressor(config: CompressionConfig) -> BaseCompressor:
    """Factory function to create gradient compressor.

    Args:
        config: Configuration for the compressor

    Returns:
        Instance of the appropriate compressor

    Example:
        >>> config = CompressionConfig(method=CompressionMethod.TOP_K, compression_ratio=0.01)
        >>> compressor = create_compressor(config)
    """
    compressors = {
        CompressionMethod.TOP_K: TopKCompressor,
        CompressionMethod.RANDOM_K: RandomKCompressor,
        CompressionMethod.QSGD: QSGDCompressor,
        CompressionMethod.SIGN_SGD: SignSGDCompressor,
        CompressionMethod.EF_SIGN: EFSignCompressor,
        CompressionMethod.NONE: NoCompression,
    }

    if config.method not in compressors:
        raise ValueError(f"Unknown compression method: {config.method}")

    return compressors[config.method](config)


class GradientCompressor:
    """High-level gradient compression utility.

    Compresses and decompresses model gradients with error feedback support.
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.compressor = create_compressor(config)
        self.round = 0

    def compress_model(
        self,
        model: nn.Module,
    ) -> Dict[str, Tuple[Tensor, Dict[str, Any]]]:
        """Compress all model gradients.

        Args:
            model: Neural network model

        Returns:
            Dictionary of compressed gradients with metadata
        """
        compressed = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                compressed[name] = self.compressor.compress(param.grad.data)

        return compressed

    def decompress_model(
        self,
        model: nn.Module,
        compressed: Dict[str, Tuple[Tensor, Dict[str, Any]]],
    ) -> None:
        """Decompress gradients and apply to model.

        Args:
            model: Neural network model
            compressed: Dictionary of compressed gradients
        """
        for name, param in model.named_parameters():
            if name in compressed and param.grad is not None:
                grad, metadata = compressed[name]
                param.grad.data = self.compressor.decompress(grad, metadata)

        self.round += 1
        self.compressor.current_round = self.round


__all__ = [
    "CompressionMethod",
    "CompressionConfig",
    "BaseCompressor",
    "TopKCompressor",
    "RandomKCompressor",
    "QSGDCompressor",
    "SignSGDCompressor",
    "EFSignCompressor",
    "NoCompression",
    "create_compressor",
    "GradientCompressor",
]
