"""
Sparsity Patterns for Model Compression

Implements unstructured, block, and N:M sparsity patterns including 2:4 sparsity.
"""

from typing import Optional, List, Dict, Tuple, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math


class UnstructuredSparsity:
    """Unstructured sparsity pattern - random zeros distributed throughout tensor.

    Args:
        sparsity: Target sparsity ratio (0-1)
        random_init: Whether to initialize mask randomly
    """

    def __init__(self, sparsity: float = 0.5, random_init: bool = False):
        self.sparsity = sparsity
        self.random_init = random_init
        self.masks: Dict[str, Tensor] = {}

    def create_mask(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        """Create unstructured sparsity mask."""
        if self.random_init:
            mask = torch.rand(shape, device=device) > self.sparsity
        else:
            mask = torch.ones(shape, device=device, dtype=torch.bool)
        return mask

    def apply_to_model(self, model: nn.Module) -> Dict[str, float]:
        """Apply unstructured sparsity to model."""
        sparsity_stats = {}

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 1:
                threshold = torch.quantile(param.abs().flatten().float(), self.sparsity)
                mask = param.abs() > threshold
                self.masks[name] = mask
                param.data *= mask.float()

                actual_sparsity = 1.0 - mask.float().mean().item()
                sparsity_stats[name] = actual_sparsity

        return sparsity_stats

    def get_sparsity_ratio(self, tensor: Tensor) -> float:
        """Get sparsity ratio of a tensor."""
        return (tensor == 0).float().mean().item()

    def get_model_sparsity(self, model: nn.Module) -> float:
        """Get overall model sparsity."""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


class BlockSparsity:
    """Block sparsity pattern - zeros in contiguous blocks.

    More hardware-friendly than unstructured sparsity.

    Args:
        sparsity: Target sparsity ratio
        block_size: Size of sparsity blocks (block_size x block_size)
    """

    def __init__(self, sparsity: float = 0.5, block_size: int = 4):
        self.sparsity = sparsity
        self.block_size = block_size
        self.masks: Dict[str, Tensor] = {}
        self.block_importance: Dict[str, Tensor] = {}

    def compute_block_importance(self, tensor: Tensor) -> Tensor:
        """Compute importance score for each block."""
        if tensor.dim() == 2:
            return self._compute_2d_block_importance(tensor)
        elif tensor.dim() == 4:
            return self._compute_4d_block_importance(tensor)
        else:
            return self._compute_1d_block_importance(tensor)

    def _compute_2d_block_importance(self, tensor: Tensor) -> Tensor:
        """Compute block importance for 2D tensors."""
        rows, cols = tensor.shape
        block_rows = rows // self.block_size
        block_cols = cols // self.block_size

        padded = tensor[: block_rows * self.block_size, : block_cols * self.block_size]
        blocks = padded.view(block_rows, self.block_size, block_cols, self.block_size)
        blocks = blocks.permute(0, 2, 1, 3).contiguous()

        importance = blocks.abs().sum(dim=(2, 3))
        return importance

    def _compute_4d_block_importance(self, tensor: Tensor) -> Tensor:
        """Compute block importance for 4D tensors (Conv2d weights)."""
        out_ch, in_ch, kh, kw = tensor.shape

        if kh < self.block_size or kw < self.block_size:
            return tensor.abs().sum(dim=(2, 3))

        block_kh = kh // self.block_size
        block_kw = kw // self.block_size

        padded = tensor[
            :, :, : block_kh * self.block_size, : block_kw * self.block_size
        ]
        blocks = padded.view(
            out_ch, in_ch, block_kh, self.block_size, block_kw, self.block_size
        )
        blocks = blocks.permute(0, 2, 4, 1, 3, 5).contiguous()

        importance = blocks.abs().sum(dim=(3, 4, 5))
        return importance

    def _compute_1d_block_importance(self, tensor: Tensor) -> Tensor:
        """Compute block importance for 1D tensors."""
        flat = tensor.flatten()
        num_blocks = flat.numel() // self.block_size
        padded = flat[: num_blocks * self.block_size]
        blocks = padded.view(num_blocks, self.block_size)

        importance = blocks.abs().sum(dim=1)
        return importance

    def create_block_mask(self, tensor: Tensor) -> Tensor:
        """Create block sparsity mask for a tensor."""
        importance = self.compute_block_importance(tensor)

        threshold = torch.quantile(importance.flatten().float(), self.sparsity)
        block_mask = importance > threshold

        if tensor.dim() == 2:
            return self._expand_2d_block_mask(block_mask, tensor.shape)
        elif tensor.dim() == 4:
            return self._expand_4d_block_mask(block_mask, tensor.shape)
        else:
            return self._expand_1d_block_mask(block_mask, tensor.shape)

    def _expand_2d_block_mask(
        self, block_mask: Tensor, shape: Tuple[int, ...]
    ) -> Tensor:
        """Expand 2D block mask to full tensor mask."""
        block_rows, block_cols = block_mask.shape
        mask = torch.zeros(shape, dtype=torch.bool, device=block_mask.device)

        for i in range(block_rows):
            for j in range(block_cols):
                r_start = i * self.block_size
                r_end = min((i + 1) * self.block_size, shape[0])
                c_start = j * self.block_size
                c_end = min((j + 1) * self.block_size, shape[1])
                mask[r_start:r_end, c_start:c_end] = block_mask[i, j]

        return mask

    def _expand_4d_block_mask(
        self, block_mask: Tensor, shape: Tuple[int, ...]
    ) -> Tensor:
        """Expand 4D block mask to full tensor mask."""
        out_ch = shape[0]
        block_kh, block_kw, _ = block_mask.shape
        mask = torch.zeros(shape, dtype=torch.bool, device=block_mask.device)

        for o in range(out_ch):
            for i in range(block_kh):
                for j in range(block_kw):
                    kh_start = i * self.block_size
                    kh_end = min((i + 1) * self.block_size, shape[2])
                    kw_start = j * self.block_size
                    kw_end = min((j + 1) * self.block_size, shape[3])
                    mask[o, :, kh_start:kh_end, kw_start:kw_end] = block_mask[o, i, j]

        return mask

    def _expand_1d_block_mask(
        self, block_mask: Tensor, shape: Tuple[int, ...]
    ) -> Tensor:
        """Expand 1D block mask to full tensor mask."""
        mask = torch.zeros(shape, dtype=torch.bool, device=block_mask.device)
        num_blocks = block_mask.numel()

        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, shape[0])
            mask[start:end] = block_mask[i]

        return mask

    def apply_to_model(self, model: nn.Module) -> Dict[str, float]:
        """Apply block sparsity to model."""
        sparsity_stats = {}

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 1:
                mask = self.create_block_mask(param.data)
                self.masks[name] = mask
                param.data *= mask.float()

                actual_sparsity = 1.0 - mask.float().mean().item()
                sparsity_stats[name] = actual_sparsity

        return sparsity_stats


class NMSparsity:
    """N:M Sparsity pattern - N non-zero values per M consecutive elements.

    Hardware-optimized sparse pattern (e.g., 2:4 sparsity on NVIDIA Ampere).

    Args:
        n: Number of non-zero values per group
        m: Group size
    """

    def __init__(self, n: int = 2, m: int = 4):
        if n >= m:
            raise ValueError("n must be less than m")
        self.n = n
        self.m = m
        self.masks: Dict[str, Tensor] = {}

    @property
    def sparsity(self) -> float:
        """Get the effective sparsity ratio."""
        return 1.0 - self.n / self.m

    def create_n_m_mask(self, tensor: Tensor) -> Tensor:
        """Create N:M sparsity mask for a tensor."""
        original_shape = tensor.shape
        flat = tensor.flatten()
        num_elements = flat.numel()

        num_groups = num_elements // self.m
        if num_groups == 0:
            return torch.ones_like(flat, dtype=torch.bool)

        padded = flat[: num_groups * self.m]
        groups = padded.view(num_groups, self.m)

        _, top_indices = groups.abs().topk(self.n, dim=1)

        mask = torch.zeros(num_groups, self.m, dtype=torch.bool, device=tensor.device)
        mask.scatter_(1, top_indices, True)

        full_mask = torch.ones(num_elements, dtype=torch.bool, device=tensor.device)
        full_mask[: num_groups * self.m] = mask.flatten()

        return full_mask.view(original_shape)

    def apply_to_model(self, model: nn.Module) -> Dict[str, float]:
        """Apply N:M sparsity to model."""
        sparsity_stats = {}

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 1:
                mask = self.create_n_m_mask(param.data)
                self.masks[name] = mask
                param.data *= mask.float()

                actual_sparsity = 1.0 - mask.float().mean().item()
                sparsity_stats[name] = actual_sparsity

        return sparsity_stats

    def get_compatible_layers(self, model: nn.Module) -> List[str]:
        """Get layers compatible with N:M sparsity."""
        compatible = []

        for name, param in model.named_parameters():
            if "weight" in name and param.numel() >= self.m:
                compatible.append(name)

        return compatible


class TwoFourSparsity(NMSparsity):
    """2:4 Sparsity pattern - optimized for NVIDIA Ampere architecture.

    Maintains 2 non-zero values per 4 consecutive elements (50% sparsity).
    """

    def __init__(self):
        super().__init__(n=2, m=4)

    def get_sparse_matrix_tiles(self, tensor: Tensor) -> Tensor:
        """Get sparse matrix in tile format for hardware acceleration."""
        mask = self.create_n_m_mask(tensor)

        tile_rows = tensor.shape[0] // 4
        tile_cols = tensor.shape[1] // 4

        if tile_rows == 0 or tile_cols == 0:
            return tensor

        tiles = []
        for i in range(tile_rows):
            for j in range(tile_cols):
                tile = tensor[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4]
                tiles.append(tile)

        return torch.stack(tiles)


class VectorSparsity:
    """Vector sparsity pattern - entire rows/columns are zero.

    Args:
        sparsity: Target sparsity ratio
        direction: 'row', 'column', or 'both'
    """

    def __init__(self, sparsity: float = 0.5, direction: str = "row"):
        self.sparsity = sparsity
        self.direction = direction
        self.masks: Dict[str, Tensor] = {}

    def create_vector_mask(self, tensor: Tensor) -> Tensor:
        """Create vector sparsity mask."""
        if tensor.dim() < 2:
            return torch.ones_like(tensor, dtype=torch.bool)

        if self.direction == "row":
            return self._create_row_mask(tensor)
        elif self.direction == "column":
            return self._create_column_mask(tensor)
        else:
            row_mask = self._create_row_mask(tensor)
            col_mask = self._create_column_mask(tensor)
            return row_mask & col_mask

    def _create_row_mask(self, tensor: Tensor) -> Tensor:
        """Create row-wise sparsity mask."""
        row_norms = tensor.abs().sum(dim=1)
        threshold = torch.quantile(row_norms.float(), self.sparsity)
        keep_rows = row_norms > threshold

        mask = keep_rows.unsqueeze(1).expand_as(tensor)
        return mask

    def _create_column_mask(self, tensor: Tensor) -> Tensor:
        """Create column-wise sparsity mask."""
        col_norms = tensor.abs().sum(dim=0)
        threshold = torch.quantile(col_norms.float(), self.sparsity)
        keep_cols = col_norms > threshold

        mask = keep_cols.unsqueeze(0).expand_as(tensor)
        return mask

    def apply_to_model(self, model: nn.Module) -> Dict[str, float]:
        """Apply vector sparsity to model."""
        sparsity_stats = {}

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                mask = self.create_vector_mask(param.data)
                self.masks[name] = mask
                param.data *= mask.float()

                actual_sparsity = 1.0 - mask.float().mean().item()
                sparsity_stats[name] = actual_sparsity

        return sparsity_stats


class SemiStructuredSparsity:
    """Semi-structured sparsity combining block and vector patterns.

    Args:
        block_size: Size of local blocks
        sparsity: Target sparsity ratio
        pattern: 'checkerboard', 'striped', or 'random_block'
    """

    def __init__(
        self,
        block_size: int = 2,
        sparsity: float = 0.5,
        pattern: str = "checkerboard",
    ):
        self.block_size = block_size
        self.sparsity = sparsity
        self.pattern = pattern
        self.masks: Dict[str, Tensor] = {}

    def create_semi_structured_mask(self, tensor: Tensor) -> Tensor:
        """Create semi-structured sparsity mask."""
        if self.pattern == "checkerboard":
            return self._create_checkerboard_mask(tensor)
        elif self.pattern == "striped":
            return self._create_striped_mask(tensor)
        else:
            return self._create_random_block_mask(tensor)

    def _create_checkerboard_mask(self, tensor: Tensor) -> Tensor:
        """Create checkerboard pattern mask."""
        mask = torch.ones_like(tensor, dtype=torch.bool)

        rows, cols = tensor.shape[-2], tensor.shape[-1]

        for i in range(rows):
            for j in range(cols):
                block_i = i // self.block_size
                block_j = j // self.block_size

                if (block_i + block_j) % 2 == 0:
                    mask[..., i, j] = torch.rand(1) > self.sparsity

        return mask

    def _create_striped_mask(self, tensor: Tensor) -> Tensor:
        """Create striped pattern mask."""
        mask = torch.ones_like(tensor, dtype=torch.bool)

        rows = tensor.shape[-2]

        for i in range(rows):
            if i % self.block_size < self.block_size // 2:
                mask[..., i, :] = torch.rand(1) > self.sparsity

        return mask

    def _create_random_block_mask(self, tensor: Tensor) -> Tensor:
        """Create random block pattern mask."""
        mask = torch.ones_like(tensor, dtype=torch.bool)

        rows, cols = tensor.shape[-2], tensor.shape[-1]
        num_blocks_row = rows // self.block_size
        num_blocks_col = cols // self.block_size

        block_mask = torch.rand(num_blocks_row, num_blocks_col) > self.sparsity
        block_mask = block_mask.to(tensor.device)

        for i in range(num_blocks_row):
            for j in range(num_blocks_col):
                r_start, r_end = i * self.block_size, (i + 1) * self.block_size
                c_start, c_end = j * self.block_size, (j + 1) * self.block_size
                mask[..., r_start:r_end, c_start:c_end] = block_mask[i, j]

        return mask

    def apply_to_model(self, model: nn.Module) -> Dict[str, float]:
        """Apply semi-structured sparsity to model."""
        sparsity_stats = {}

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                mask = self.create_semi_structured_mask(param.data)
                self.masks[name] = mask
                param.data *= mask.float()

                actual_sparsity = 1.0 - mask.float().mean().item()
                sparsity_stats[name] = actual_sparsity

        return sparsity_stats


class SparsityScheduler:
    """Scheduler for gradual sparsity increase during training.

    Args:
        initial_sparsity: Starting sparsity
        final_sparsity: Target sparsity
        start_step: Step to start increasing sparsity
        end_step: Step to reach final sparsity
        schedule_type: 'linear', 'cubic', or 'exponential'
    """

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.9,
        start_step: int = 0,
        end_step: int = 10000,
        schedule_type: str = "cubic",
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.schedule_type = schedule_type
        self.current_step = 0
        self.current_sparsity = initial_sparsity

    def step(self) -> float:
        """Advance one step and return current sparsity."""
        self.current_step += 1
        self.current_sparsity = self.get_sparsity()
        return self.current_sparsity

    def get_sparsity(self) -> float:
        """Get current target sparsity."""
        if self.current_step < self.start_step:
            return self.initial_sparsity
        if self.current_step >= self.end_step:
            return self.final_sparsity

        progress = (self.current_step - self.start_step) / (
            self.end_step - self.start_step
        )

        if self.schedule_type == "linear":
            return (
                self.initial_sparsity
                + (self.final_sparsity - self.initial_sparsity) * progress
            )
        elif self.schedule_type == "cubic":
            return (
                self.final_sparsity
                + (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3
            )
        elif self.schedule_type == "exponential":
            return self.final_sparsity + (
                self.initial_sparsity - self.final_sparsity
            ) * math.exp(-3 * progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


__all__ = [
    "UnstructuredSparsity",
    "BlockSparsity",
    "NMSparsity",
    "TwoFourSparsity",
    "VectorSparsity",
    "SemiStructuredSparsity",
    "SparsityScheduler",
]
