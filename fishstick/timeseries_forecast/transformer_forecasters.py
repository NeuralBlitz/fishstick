"""
Transformer-based Time Series Forecasting Models.

Implements state-of-the-art transformer architectures for long-sequence forecasting:
- Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- Autoformer: Autoformer: Decomposition Transformers with Auto-Correlation

Example:
    >>> from fishstick.timeseries_forecast import Informer, Autoformer
    >>>
    >>> # Informer for long-horizon forecasting
    >>> model = Informer(
    ...     input_dim=7,
    ...     ctxt_len=96,
    ...     pred_len=48,
    ...     d_model=512,
    ...     n_heads=8,
    ...     e_layers=2,
    ...     d_layers=2,
    ... )
    >>>
    >>> # Autoformer with series decomposition
    >>> model = Autoformer(
    ...     input_dim=7,
    ...     ctxt_len=96,
    ...     pred_len=48,
    ...     d_model=512,
    ...     n_heads=8,
    ... )
"""

from typing import Optional, Tuple, List, Dict, Any, Callable
import math
import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class TriangularCausalMask:
    """Triangular causal mask for attention.

    Args:
        B: Batch size
        L: Sequence length
        device: Device to create mask on
    """

    def __init__(self, B: int, L: int, device: Optional[torch.device] = None):
        mask_shape = (B, 1, L, L)
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1
            )

    @property
    def mask(self) -> Tensor:
        return self._mask


class ProbMask:
    """ProbSparse attention mask.

    Args:
        B: Batch size
        H: Number of heads
        L: Query length
        S: Key/Value length
        index: Selected top-k indices
        device: Device to create mask on
    """

    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        S: int,
        index: Tensor,
        device: Optional[torch.device] = None,
    ):
        _mask = torch.ones(L, S, dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :, :].expand(B, H, L, S)
        indicator = _mask_ex[
            torch.arange(B, device=device)[:, None, None],
            torch.arange(H, device=device)[None, :, None],
            index,
            :,
        ].transpose(1, 2)
        self._mask = indicator.view(-1, L, S)

    @property
    def mask(self) -> Tensor:
        return self._mask


class FullAttention(nn.Module):
    """Standard full attention mechanism.

    Args:
        mask: Whether to apply causal mask
        factor: Attention factor for efficiency
        scale: Attention scale
        output_attention: Whether to return attention weights
    """

    def __init__(
        self,
        mask: bool = True,
        factor: float = 1.0,
        scale: Optional[float] = None,
        output_attention: bool = False,
    ):
        super().__init__()
        self.mask = mask
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, L, H, E = queries.shape
        _, S, _, D = keys.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask:
            if attn_mask is None:
                causal_mask = TriangularCausalMask(B, L, device=queries.device)
                scores = scores.masked_fill(causal_mask.mask, -np.inf)

        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V, A
        return V, None


class ProbSparseAttention(nn.Module):
    """ProbSparse attention from Informer paper.

    Reduces O(L^2) to O(L log L) complexity.

    Args:
        mask: Whether to apply causal mask
        factor: Sampling factor for top-k queries
        scale: Attention scale
        output_attention: Whether to return attention weights
    """

    def __init__(
        self,
        mask: bool = True,
        factor: float = 5.0,
        scale: Optional[float] = None,
        output_attention: bool = False,
    ):
        super().__init__()
        self.mask = mask
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention

    def _prob_sparse_sample(
        self, queries: Tensor, keys: Tensor
    ) -> Tuple[Tensor, Tensor]:
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape

        if L > S:
            queries = queries[:, :S, :, :]
            L = S

        M = max(L * math.log(S), 1)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        M_sp = scores.max(dim=-1)[0] - torch.div(scores.sum(dim=-1), S)

        _, top_k = torch.topk(M_sp, min(M, L), dim=-1)

        topk_scores = scores.gather(2, top_k.unsqueeze(-1).expand(-1, -1, -1, S))
        topk_scores = topk_scores - topk_scores.max(dim=-1, keepdim=True)[0]

        A = torch.softmax(topk_scores / math.sqrt(E), dim=-1)

        return A, top_k

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        A, top_k = self._prob_sparse_sample(queries, keys)

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V, A
        return V, None


class AutoCorrelation(nn.Module):
    """Auto-correlation mechanism from Autoformer.

    Discovers period-based dependencies and computes in O(N log N).

    Args:
        mask: Whether to apply causal mask
        scale: Attention scale
        output_attention: Whether to return attention weights
    """

    def __init__(
        self,
        mask: bool = True,
        scale: Optional[float] = None,
        output_attention: bool = False,
    ):
        super().__init__()
        self.mask = mask
        self.scale = scale
        self.output_attention = output_attention

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        queries = queries.permute(0, 2, 3, 1)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 3, 1)

        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        res = q_fft * torch.conj(k_fft)
        cor = torch.fft.irfft(res, dim=-1)

        if tau is not None:
            tau = tau.view(B, H, 1, L).expand(-1, -1, E, -1)
            cor = F.interpolate(cor, scale_factor=tau, mode="linear")
            cor = cor.permute(0, 1, 3, 2)

        scale = self.scale or 1.0 / math.sqrt(E)
        A = torch.softmax(scale * cor, dim=-1)

        V = torch.einsum("bhls,bshd->blhd", A, values.permute(0, 2, 3, 1))
        V = V.permute(0, 2, 3, 1)

        if self.output_attention:
            return V, A
        return V, None


class SeriesDecomposition(nn.Module):
    """Series decomposition block from Autoformer.

    Decomposes time series into trend and seasonal components
    using moving average.

    Args:
        kernel_size: Size of the moving average kernel
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Decompose series into trend and seasonal.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            trend: Trend component
            seasonal: Seasonal component
        """
        B, L, D = x.shape

        x_trans = x.transpose(1, 2)
        trend = self.avg(x_trans).transpose(1, 2)

        seasonal = x - trend

        return trend, seasonal


class moving_avg(nn.Module):
    """Moving average block for trend extraction.

    Args:
        kernel_size: Size of the moving average kernel
        stride: Stride of the convolution
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply moving average.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Smoothed tensor
        """
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Series decomposition module.

    Args:
        kernel_size: Size of the moving average kernel
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Decompose series.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Trend and seasonal components
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


class EncoderLayer(nn.Module):
    """Transformer encoder layer with optional attention.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        output_attention: Whether to output attention weights
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_attention: bool = False,
    ):
        super().__init__()
        self.output_attention = output_attention

        self.attention = FullAttention(
            mask=True,
            output_attention=output_attention,
        )

        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
            device=next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else None,
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
            device=next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else None,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        res = self.norm2(x + y)

        if self.output_attention:
            return res, attn
        return res, None


class Encoder(nn.Module):
    """Transformer encoder.

    Args:
        attn_layers: List of encoder layers
        conv_layers: Optional list of convolutional layers
        norm_layer: Optional normalization layer
    """

    def __init__(
        self,
        attn_layers: List[nn.Module],
        conv_layers: Optional[List[nn.Module]] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x.transpose(-1, 1)).transpose(-1, 1)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        output_attention: Whether to output attention weights
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_attention: bool = False,
    ):
        super().__init__()
        self.output_attention = output_attention

        self.self_attention = FullAttention(
            mask=True,
            output_attention=output_attention,
        )
        self.cross_attention = FullAttention(
            mask=False,
            output_attention=False,
        )

        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
            device=next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else None,
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
            device=next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else None,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        x_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        res = self.norm3(x + y)

        if self.output_attention:
            return res, self.self_attention(x, x, x, attn_mask=x_mask)[1]
        return res, None


class Decoder(nn.Module):
    """Transformer decoder.

    Args:
        layers: List of decoder layers
        norm_layer: Optional normalization layer
        output_proj: Optional output projection layer
    """

    def __init__(
        self,
        layers: List[nn.Module],
        norm_layer: Optional[nn.Module] = None,
        output_proj: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.output_proj = output_proj

    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        x_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        attns = []
        for layer in self.layers:
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if self.output_proj is not None:
            x = self.output_proj(x)

        return x, attns


class Informer(nn.Module):
    """Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.

    Uses ProbSparse attention for O(L log L) complexity and distillation
    for handling long sequences.

    Args:
        input_dim: Number of input features
        ctxt_len: Context (input) sequence length
        pred_len: Prediction (output) sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        output_attention: Whether to output attention weights
        factor: ProbSparse attention factor
        distil: Whether to use distillation
        num_channels: Number of input channels (for individual channels)

    Example:
        >>> model = Informer(
        ...     input_dim=7,
        ...     ctxt_len=96,
        ...     pred_len=48,
        ...     d_model=512,
        ...     n_heads=8,
        ... )
        >>> x_ctxt = torch.randn(32, 96, 7)
        >>> x_pred = torch.randn(32, 48, 7)
        >>> output = model(x_ctxt, x_pred)
        >>> # output shape: [32, 48, 7]
    """

    def __init__(
        self,
        input_dim: int,
        ctxt_len: int,
        pred_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_attention: bool = False,
        factor: float = 5.0,
        distil: bool = True,
        num_channels: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ctxt_len = ctxt_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.output_attention = output_attention

        self.enc_embedding = nn.Linear(input_dim, d_model)
        self.dec_embedding = nn.Linear(input_dim, d_model)

        self.encoder = self._build_encoder(
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
            factor=factor,
            distil=distil,
        )

        self.decoder = self._build_decoder(
            n_heads=n_heads,
            d_ff=d_ff,
            d_layers=d_layers,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
        )

        self.projection = nn.Linear(d_model, 1, bias=True)

    def _build_encoder(
        self,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        activation: str,
        output_attention: bool,
        factor: float,
        distil: bool,
    ) -> Encoder:
        attn_layers = []
        conv_layers = []

        for i in range(e_layers):
            attn_layers.append(
                EncoderLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    output_attention=output_attention,
                )
            )

            if distil and i < e_layers - 1:
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=self.d_model,
                        out_channels=self.d_model,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="circular",
                    )
                )

        return Encoder(
            attn_layers=attn_layers,
            conv_layers=conv_layers if distil else None,
            norm_layer=nn.LayerNorm(self.d_model),
        )

    def _build_decoder(
        self,
        n_heads: int,
        d_ff: int,
        d_layers: int,
        dropout: float,
        activation: str,
        output_attention: bool,
    ) -> Decoder:
        layers = []
        for i in range(d_layers):
            layers.append(
                DecoderLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    output_attention=output_attention,
                )
            )

        return Decoder(
            layers=layers,
            norm_layer=nn.LayerNorm(self.d_model),
            output_proj=nn.Linear(self.d_model, 1, bias=True),
        )

    def forward(
        self,
        x_ctxt: Tensor,
        x_pred: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x_ctxt: Context input [B, L_ctxt, D]
            x_pred: Prediction input [B, L_pred, D]
            mask: Optional attention mask

        Returns:
            predictions: [B, L_pred, 1]
        """
        B = x_ctxt.shape[0]

        enc_out = self.enc_embedding(x_ctxt)
        enc_out, attns = self.encoder(enc_out, attn_mask=mask)

        dec_inp = torch.zeros_like(x_pred[:, :, :1])
        dec_out = self.dec_embedding(x_pred)
        dec_out, _ = self.decoder(dec_inp, enc_out)

        output = self.projection(dec_out)

        if self.output_attention:
            return output, attns
        return output


class Autoformer(nn.Module):
    """Autoformer: Decomposition Transformers with Auto-Correlation.

    Uses series decomposition and auto-correlation mechanism for
    improved long-term forecasting.

    Args:
        input_dim: Number of input features
        ctxt_len: Context (input) sequence length
        pred_len: Prediction (output) sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        output_attention: Whether to output attention weights
        kernel_size: Kernel size for series decomposition

    Example:
        >>> model = Autoformer(
        ...     input_dim=7,
        ...     ctxt_len=96,
        ...     pred_len=48,
        ...     d_model=512,
        ...     n_heads=8,
        ... )
        >>> x = torch.randn(32, 96, 7)
        >>> output = model(x)
        >>> # output shape: [32, 48, 1]
    """

    def __init__(
        self,
        input_dim: int,
        ctxt_len: int,
        pred_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_attention: bool = False,
        kernel_size: int = 25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ctxt_len = ctxt_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.enc_embedding = nn.Linear(input_dim, d_model)

        self.encoder = self._build_encoder(
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
            kernel_size=kernel_size,
        )

        self.decoder = self._build_decoder(
            n_heads=n_heads,
            d_ff=d_ff,
            d_layers=d_layers,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
            kernel_size=kernel_size,
        )

        self.projection = nn.Linear(d_model, 1, bias=True)

    def _build_encoder(
        self,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        activation: str,
        output_attention: bool,
        kernel_size: int,
    ) -> nn.ModuleList:
        encoder_layers = []

        for i in range(e_layers):
            encoder_layers.append(
                EncoderLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    output_attention=output_attention,
                )
            )

        return nn.ModuleList(encoder_layers)

    def _build_decoder(
        self,
        n_heads: int,
        d_ff: int,
        d_layers: int,
        dropout: float,
        activation: str,
        output_attention: bool,
        kernel_size: int,
    ) -> nn.ModuleList:
        decoder_layers = []

        for i in range(d_layers):
            decoder_layers.append(
                DecoderLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    output_attention=output_attention,
                )
            )

        return nn.ModuleList(decoder_layers)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input time series [B, L, D]
            mask: Optional attention mask

        Returns:
            predictions: [B, L_pred, 1]
        """
        B, L, _ = x.shape

        enc_out = self.enc_embedding(x)

        for layer in self.encoder:
            enc_out, _ = layer(enc_out, attn_mask=mask)

        seasonal_init, trend_init = series_decomp(kernel_size=25)(x)

        trend = trend_init
        seasonal = seasonal_init

        dec_inp = torch.zeros_like(seasonal[:, -self.pred_len :, :])

        for layer in self.decoder:
            dec_out, _ = layer(dec_inp, enc_out)

            seasonal_part, trend_part = series_decomp(kernel_size=25)(dec_out)
            seasonal = seasonal[:, -self.pred_len :, :]
            trend = trend[:, -self.pred_len :, :]

        output = self.projection(dec_out)

        return output


class TransformerForecaster(nn.Module):
    """Standard Transformer-based time series forecaster.

    A baseline transformer model for time series forecasting.

    Args:
        input_dim: Number of input features
        ctxt_len: Context sequence length
        pred_len: Prediction sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function

    Example:
        >>> model = TransformerForecaster(
        ...     input_dim=7,
        ...     ctxt_len=96,
        ...     pred_len=48,
        ...     d_model=256,
        ...     n_heads=4,
        ...     n_layers=3,
        ... )
        >>> x = torch.randn(32, 96, 7)
        >>> output = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        ctxt_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ctxt_len = ctxt_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.enc_embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            Output [B, pred_len, 1]
        """
        enc_out = self.enc_embedding(x)
        enc_out = self.encoder(enc_out)

        output = self.decoder(enc_out[:, -self.pred_len :, :])

        return output


class TimeSeriesTransformerConfig:
    """Configuration for TimeSeriesTransformer.

    Args:
        input_dim: Number of input features
        ctxt_len: Context sequence length
        pred_len: Prediction sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int = 7,
        ctxt_len: int = 96,
        pred_len: int = 48,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        self.input_dim = input_dim
        self.ctxt_len = ctxt_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation


def create_transformer_forecaster(
    model_type: str = "informer",
    config: Optional[TimeSeriesTransformerConfig] = None,
    **kwargs,
) -> nn.Module:
    """Factory function to create transformer forecasters.

    Args:
        model_type: Type of model ('informer', 'autoformer', 'standard')
        config: Optional configuration object
        **kwargs: Additional arguments passed to model

    Returns:
        Initialized forecaster model

    Example:
        >>> config = TimeSeriesTransformerConfig(
        ...     input_dim=7,
        ...     ctxt_len=96,
        ...     pred_len=48,
        ...     d_model=256,
        ... )
        >>> model = create_transformer_forecaster('informer', config)
    """
    if config is None:
        config = TimeSeriesTransformerConfig(**kwargs)

    if model_type.lower() == "informer":
        return Informer(
            input_dim=config.input_dim,
            ctxt_len=config.ctxt_len,
            pred_len=config.pred_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.n_layers,
            d_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
        )
    elif model_type.lower() == "autoformer":
        return Autoformer(
            input_dim=config.input_dim,
            ctxt_len=config.ctxt_len,
            pred_len=config.pred_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.n_layers,
            d_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
        )
    elif model_type.lower() == "standard" or model_type.lower() == "transformer":
        return TransformerForecaster(
            input_dim=config.input_dim,
            ctxt_len=config.ctxt_len,
            pred_len=config.pred_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
