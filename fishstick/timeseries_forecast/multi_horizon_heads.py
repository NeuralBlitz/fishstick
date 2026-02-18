"""
Multi-Horizon Prediction Heads for Time Series Forecasting.

Implements various strategies for multi-step forecasting:
- Direct: Independent predictions for each horizon
- Recursive: Autoregressive prediction
- DirRec: Hybrid direct + recursive
- Multi-Resolution: Multi-scale forecasting

Example:
    >>> from fishstick.timeseries_forecast import (
    ...     DirectMultiHorizonHead,
    ...     RecursiveMultiHorizonHead,
    ...     DirRecMultiHorizonHead,
    ...     MultiResolutionHead,
    ... )
"""

from typing import Optional, Tuple, List, Dict, Any, Callable
from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHorizonHead(ABC, nn.Module):
    """Abstract base class for multi-horizon prediction heads.

    Args:
        input_dim: Input feature dimension
        pred_len: Number of steps to predict
    """

    def __init__(self, input_dim: int, pred_len: int):
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D]

        Returns:
            Predictions [B, pred_len]
        """
        pass


class DirectMultiHorizonHead(MultiHorizonHead):
    """Direct multi-horizon prediction head.

    Makes independent predictions for each forecast horizon.
    O(pred_len) inference but accurate for each step.

    Args:
        input_dim: Input feature dimension
        pred_len: Number of steps to predict
        hidden_dim: Hidden layer dimension

    Example:
        >>> head = DirectMultiHorizonHead(
        ...     input_dim=256,
        ...     pred_len=48,
        ...     hidden_dim=128,
        ... )
        >>> x = torch.randn(32, 256)
        >>> output = head(x)
        >>> # output shape: [32, 48]
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__(input_dim, pred_len)

        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(pred_len)])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D]

        Returns:
            Predictions [B, pred_len]
        """
        h = self.shared(x)

        outputs = []
        for head in self.heads:
            outputs.append(head(h))

        return torch.cat(outputs, dim=-1)


class RecursiveMultiHorizonHead(MultiHorizonHead):
    """Recursive multi-horizon prediction head.

    Uses autoregressive prediction - output at step t becomes
    input for step t+1. More efficient but can accumulate errors.

    Args:
        input_dim: Input feature dimension
        pred_len: Number of steps to predict
        hidden_dim: Hidden layer dimension
        n_layers: Number of LSTM layers

    Example:
        >>> head = RecursiveMultiHorizonHead(
        ...     input_dim=256,
        ...     pred_len=48,
        ...     hidden_dim=128,
        ... )
        >>> x = torch.randn(32, 256)
        >>> output = head(x)
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        hidden_dim: int = 128,
        n_layers: int = 1,
    ):
        super().__init__(input_dim, pred_len)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0,
        )

        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D]

        Returns:
            Predictions [B, pred_len]
        """
        B = x.shape[0]

        h = self.encoder(x)
        h = h.unsqueeze(1).expand(-1, self.pred_len, -1)

        outputs = []
        hidden = None

        for t in range(self.pred_len):
            if t == 0:
                dec_input = torch.zeros(B, 1, 1, device=x.device)
            else:
                dec_input = outputs[-1].unsqueeze(1)

            dec_out, hidden = self.decoder(dec_input, hidden)

            pred = self.projection(dec_out.squeeze(1))
            outputs.append(pred)

        return torch.stack(outputs, dim=1)


class DirRecMultiHorizonHead(MultiHorizonHead):
    """DirRec (Direct + Recursive) hybrid prediction head.

    Combines direct and recursive approaches: makes direct predictions
    but uses them recursively as inputs for subsequent steps.

    Args:
        input_dim: Input feature dimension
        pred_len: Number of steps to predict
        hidden_dim: Hidden layer dimension

    Example:
        >>> head = DirRecMultiHorizonHead(
        ...     input_dim=256,
        ...     pred_len=48,
        ...     hidden_dim=128,
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        hidden_dim: int = 128,
    ):
        super().__init__(input_dim, pred_len)

        self.shared = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D]

        Returns:
            Predictions [B, pred_len]
        """
        B = x.shape[0]

        outputs = []
        current_input = x

        for t in range(self.pred_len):
            if t == 0:
                prev_pred = torch.zeros(B, 1, device=x.device)
            else:
                prev_pred = outputs[-1].unsqueeze(-1)

            h = torch.cat([current_input, prev_pred], dim=-1)
            h = self.shared(h)
            pred = self.output(h)

            outputs.append(pred.squeeze(-1))

        return torch.stack(outputs, dim=1)


class MultiResolutionHead(nn.Module):
    """Multi-resolution prediction head.

    Predicts at multiple temporal resolutions simultaneously,
    capturing both short-term and long-term patterns.

    Args:
        input_dim: Input feature dimension
        pred_len: Base prediction length
        resolutions: List of resolution factors

    Example:
        >>> head = MultiResolutionHead(
        ...     input_dim=256,
        ...     pred_len=48,
        ...     resolutions=[1, 2, 4],
        ... )
        >>> x = torch.randn(32, 256)
        >>> outputs = head(x)
        >>> # outputs[0]: [32, 48], outputs[1]: [32, 24], outputs[2]: [12, 12]
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        resolutions: Optional[List[int]] = None,
    ):
        super().__init__()

        if resolutions is None:
            resolutions = [1, 2, 4]

        self.pred_len = pred_len
        self.resolutions = resolutions

        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
        )

        self.heads = nn.ModuleDict()
        for res in resolutions:
            head_pred_len = pred_len // res
            self.heads[str(res)] = nn.Linear(input_dim // 2, head_pred_len)

    def forward(
        self,
        x: Tensor,
        return_dict: bool = True,
    ) -> Union[List[Tensor], Dict[str, Tensor]]:
        """Forward pass.

        Args:
            x: Input features [B, D]
            return_dict: Whether to return dict format

        Returns:
            List or dict of predictions at each resolution
        """
        h = self.shared(x)

        outputs = {}
        for res in self.resolutions:
            outputs[str(res)] = self.heads[str(res)](h)

        if return_dict:
            return outputs
        return list(outputs.values())

    def get_full_resolution(
        self,
        outputs: Dict[str, Tensor],
    ) -> Tensor:
        """Upsample lower resolution predictions to full resolution.

        Args:
            outputs: Dictionary of resolution outputs

        Returns:
            Full resolution predictions [B, pred_len]
        """
        preds = []
        for res in self.resolutions:
            pred = outputs[str(res)]
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=self.pred_len,
                mode="linear",
                align_corners=False,
            )
            preds.append(pred.squeeze(1))

        return torch.stack(preds).mean(dim=0)


class AttentionMultiHorizonHead(nn.Module):
    """Attention-based multi-horizon prediction head.

    Uses attention to dynamically weight different horizon predictions.

    Args:
        input_dim: Input feature dimension
        pred_len: Number of steps to predict
        hidden_dim: Hidden dimension
        n_heads: Number of attention heads

    Example:
        >>> head = AttentionMultiHorizonHead(
        ...     input_dim=256,
        ...     pred_len=48,
        ...     hidden_dim=128,
        ...     n_heads=4,
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        self.pred_len = pred_len

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(pred_len, hidden_dim)
        self.value = nn.Linear(pred_len, hidden_dim)

        self.horizon_embedding = nn.Embedding(pred_len, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D]

        Returns:
            Predictions [B, pred_len]
        """
        B = x.shape[0]

        q = self.query(x).unsqueeze(1)

        horizon_idx = torch.arange(self.pred_len, device=x.device)
        k = self.horizon_embedding(horizon_idx).unsqueeze(0).expand(B, -1, -1)
        v = self.horizon_embedding(horizon_idx).unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.attention(q, k, v)
        attn_out = attn_out.squeeze(1)

        output = self.output(attn_out)

        return output


class ResidualMultiHorizonHead(nn.Module):
    """Residual multi-horizon prediction head.

    Uses residual connections to improve long-horizon predictions.

    Args:
        input_dim: Input feature dimension
        pred_len: Number of steps to predict
        hidden_dim: Hidden dimension
        n_blocks: Number of residual blocks

    Example:
        >>> head = ResidualMultiHorizonHead(
        ...     input_dim=256,
        ...     pred_len=48,
        ...     hidden_dim=128,
        ...     n_blocks=3,
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        hidden_dim: int = 128,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.pred_len = pred_len

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [self._make_block(hidden_dim) for _ in range(n_blocks)]
        )

        self.output = nn.Linear(hidden_dim, pred_len)

    def _make_block(self, dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D]

        Returns:
            Predictions [B, pred_len]
        """
        h = self.input_proj(x)

        for block in self.blocks:
            h = h + block(h)

        return self.output(h)


class StrategySelector(nn.Module):
    """Learns to select best prediction strategy.

    Can be used to combine multiple multi-horizon strategies.

    Args:
        input_dim: Input feature dimension
        n_strategies: Number of strategies to select from

    Example:
        >>> selector = StrategySelector(
        ...     input_dim=256,
        ...     n_strategies=4,
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        n_strategies: int = 4,
    ):
        super().__init__()
        self.n_strategies = n_strategies

        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, n_strategies),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Select strategy weights.

        Args:
            x: Input features [B, D]

        Returns:
            Strategy weights [B, n_strategies]
        """
        return F.softmax(self.network(x), dim=-1)


class MultiHorizonForecaster(nn.Module):
    """Complete multi-horizon forecaster with multiple strategy options.

    Args:
        encoder: Backbone encoder
        head_type: Type of prediction head
        pred_len: Prediction horizon
        head_kwargs: Additional kwargs for head

    Example:
        >>> encoder = nn.Linear(7, 256)
        >>> model = MultiHorizonForecaster(
        ...     encoder=encoder,
        ...     head_type='direct',
        ...     pred_len=48,
        ... )
    """

    HEAD_TYPES = {
        "direct": DirectMultiHorizonHead,
        "recursive": RecursiveMultiHorizonHead,
        "dirrec": DirRecMultiHorizonHead,
        "attention": AttentionMultiHorizonHead,
        "residual": ResidualMultiHorizonHead,
    }

    def __init__(
        self,
        encoder: nn.Module,
        head_type: str = "direct",
        pred_len: int = 48,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.head_type = head_type
        self.pred_len = pred_len

        if head_kwargs is None:
            head_kwargs = {}

        head_cls = self.HEAD_TYPES.get(head_type.lower())
        if head_cls is None:
            raise ValueError(f"Unknown head type: {head_type}")

        self.head = head_cls(
            input_dim=head_kwargs.get("input_dim", 256),
            pred_len=pred_len,
            **head_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            Predictions [B, pred_len]
        """
        features = self.encoder(x)
        if isinstance(features, tuple):
            features = features[0]

        if features.dim() == 3:
            features = features[:, -1, :]

        return self.head(features)


def create_multihorizon_head(
    head_type: str,
    input_dim: int,
    pred_len: int,
    **kwargs,
) -> MultiHorizonHead:
    """Factory function to create multi-horizon prediction heads.

    Args:
        head_type: Type of head ('direct', 'recursive', 'dirrec', etc.)
        input_dim: Input dimension
        pred_len: Prediction length
        **kwargs: Additional arguments

    Returns:
        Initialized prediction head

    Example:
        >>> head = create_multihorizon_head('direct', 256, 48)
    """
    heads = {
        "direct": DirectMultiHorizonHead,
        "recursive": RecursiveMultiHorizonHead,
        "dirrec": DirRecMultiHorizonHead,
        "attention": AttentionMultiHorizonHead,
        "residual": ResidualMultiHorizonHead,
    }

    if head_type not in heads:
        raise ValueError(f"Unknown head type: {head_type}")

    return heads[head_type](
        input_dim=input_dim,
        pred_len=pred_len,
        **kwargs,
    )
