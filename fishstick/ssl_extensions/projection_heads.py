"""
SSL Projection Heads

Advanced projection heads for self-supervised learning:
- MLP projection heads
- Transformer projection heads
- Multi-layer projection heads
- Cosine similarity projection heads
- Memory banks
"""

from typing import Optional, Tuple, Dict, Any, List, Callable
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from fishstick.ssl_extensions.base import (
    L2Normalize,
    DropPath,
    PositionalEmbedding2D,
)


class MLProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output projection dimension
        num_layers: Number of layers
        use_bn: Whether to use batch normalization
        bias_last: Whether to use bias in last layer
        activation: Activation function
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 256,
        num_layers: int = 3,
        use_bn: bool = True,
        bias_last: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        act_fn = self._get_activation(activation)
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act_fn)
            
        layers.append(nn.Linear(dims[-2], dims[-1], bias=bias_last))
        
        self.projection = nn.Sequential(*layers)
        
    def _get_activation(self, name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "gelu":
            return nn.GELU()
        elif name == "swish" or name == "silu":
            return nn.SiLU(inplace=True)
        elif name == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
            
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class TransformerProjectionHead(nn.Module):
    """Transformer-based projection head.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
            dropout=dropout,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layernorm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        
        x = self.transformer(x)
        
        x = self.layernorm(x)
        
        return x.squeeze(1)


class MultiLayerProjectionHead(nn.Module):
    """Multi-layer projection head with skip connections.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions
        output_dim: Output dimension
        use_bn: Whether to use batch normalization
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_bn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                    
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class CosineProjectionHead(nn.Module):
    """Cosine similarity projection head.
    
    Projects features to a space where similarity is computed via cosine similarity.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (default: same as input for direct cosine)
        use_normalize: Whether to normalize output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: Optional[int] = None,
        use_normalize: bool = True,
    ):
        super().__init__()
        
        self.output_dim = output_dim or input_dim
        self.use_normalize = use_normalize
        
        if output_dim is not None and output_dim != input_dim:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.projector = nn.Identity()
            
    def forward(self, x: Tensor) -> Tensor:
        x = self.projector(x)
        
        if self.use_normalize:
            x = F.normalize(x, dim=-1)
            
        return x


class NonLinearProjectionHead(nn.Module):
    """Non-linear projection head with customizable architecture.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        use_bn: Whether to use batch normalization
        num_layers: Number of layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 256,
        use_bn: bool = True,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                
            in_dim = out_dim
            
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class BYOLProjectionHead(nn.Module):
    """Projection head for BYOL-style methods.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        use_bn: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 4096,
        output_dim: int = 256,
        use_bn: bool = True,
    ):
        super().__init__()
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
            
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        ])
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class SimSiamProjectionHead(nn.Module):
    """Projection head for SimSiam.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of layers
        use_bn: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        num_layers: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()
        
        layers = []
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
            elif not use_bn:
                layers.append(nn.BatchNorm1d(out_dim, affine=False))
                
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class SimSiamPredictorHead(nn.Module):
    """Predictor head for SimSiam.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.BatchNorm1d(out_dim, affine=False))
                
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(x)


class SwAVProjectionHead(nn.Module):
    """Projection head for SwAV.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        n_prototypes: Number of prototypes
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 256,
        n_prototypes: int = 3000,
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.prototypes = nn.Linear(output_dim, n_prototypes, bias=False)
        
    def forward(self, x: Tensor, return_prototypes: bool = True) -> Tensor:
        x = self.projection(x)
        
        if return_prototypes:
            x = self.prototypes(x)
            
        return x


class TemporalProjectionHead(nn.Module):
    """Temporal projection head for video/audio sequences.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of layers
        temporal_aggregation: How to aggregate temporal features
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 2,
        temporal_aggregation: str = "mean",
    ):
        super().__init__()
        self.temporal_aggregation = temporal_aggregation
        
        layers = []
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.extend([
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                ])
                
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            if self.temporal_aggregation == "mean":
                x = x.mean(dim=1)
            elif self.temporal_aggregation == "max":
                x = x.max(dim=1)[0]
            elif self.temporal_aggregation == "last":
                x = x[:, -1]
                
        return self.projection(x)


class MemoryBankProjection(nn.Module):
    """Projection head with integrated memory bank.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        bank_size: Size of memory bank
        temperature: Temperature for similarity
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        bank_size: int = 4096,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.register_buffer("memory_bank", torch.randn(bank_size, output_dim))
        self.memory_bank = F.normalize(self.memory_bank, dim=-1)
        
        self.bank_ptr = 0
        self.bank_size = bank_size
        self.temperature = temperature
        
    def forward(self, x: Tensor, update_bank: bool = True) -> Tuple[Tensor, Tensor]:
        projected = self.projection(x)
        projected = F.normalize(projected, dim=-1)
        
        if update_bank and self.training:
            self._update_bank(projected.detach())
            
        similarities = projected @ self.memory_bank.T / self.temperature
        
        return projected, similarities
        
    def _update_bank(self, features: Tensor):
        batch_size = features.shape[0]
        
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size <= self.bank_size:
            self.memory_bank[ptr : ptr + batch_size] = features
        else:
            remaining = self.bank_size - ptr
            self.memory_bank[ptr:] = features[:remaining]
            self.memory_bank[:batch_size - remaining] = features[remaining:]
            
        self.bank_ptr = (ptr + batch_size) % self.bank_size


class ProjectionHeadEnsemble(nn.Module):
    """Ensemble of multiple projection heads.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_heads: Number of projection heads
        head_type: Type of projection head
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_heads: int = 4,
        head_type: str = "mlp",
    ):
        super().__init__()
        
        self.num_heads = num_heads
        
        for i in range(num_heads):
            if head_type == "mlp":
                head = MLProjectionHead(
                    input_dim, hidden_dim, output_dim, num_layers=3
                )
            elif head_type == "nonlinear":
                head = NonLinearProjectionHead(
                    input_dim, hidden_dim, output_dim, num_layers=2
                )
            else:
                head = MLProjectionHead(
                    input_dim, hidden_dim, output_dim, num_layers=3
                )
                
            self.add_module(f"head_{i}", head)
            
    def forward(self, x: Tensor, head_idx: Optional[int] = None) -> List[Tensor]:
        if head_idx is not None:
            return [getattr(self, f"head_{head_idx}")(x)]
            
        outputs = []
        for i in range(self.num_heads):
            head = getattr(self, f"head_{i}")
            outputs.append(head(x))
            
        return outputs


class LinearProjectionHead(nn.Module):
    """Simple linear projection head.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class IdentityProjectionHead(nn.Module):
    """Identity projection (no transformation).
    
    Args:
        normalize: Whether to normalize output
    """
    
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        
    def forward(self, x: Tensor) -> Tensor:
        if self.normalize:
            return F.normalize(x, dim=-1)
        return x
