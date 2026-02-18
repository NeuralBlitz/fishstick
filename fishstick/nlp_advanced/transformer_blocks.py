import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(
            device
        )

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class ALiBiAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        is_sliding: bool = False,
        window_size: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_sliding = is_sliding
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        self._init_alibi_slopes(num_heads)

    def _init_alibi_slopes(self, num_heads: int):
        def get_alibi_slopes(num_heads: int):
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            base = torch.tensor(
                2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32
            )
            powers = torch.arange(1, closest_power_of_2 + 1, dtype=torch.float32)
            slopes = torch.pow(base, powers)

            if num_heads % closest_power_of_2 != 0:
                extra_slopes = torch.linspace(
                    2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3) - 1)),
                    2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
                    num_heads - closest_power_of_2,
                    dtype=torch.float32,
                )
                slopes = torch.cat([slopes, extra_slopes])
            return slopes

        slopes = get_alibi_slopes(num_heads)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        alibi = self.alibi_slopes.view(-1, 1, 1) * relative_positions.abs().unsqueeze(0)
        alibi = alibi.masked_fill(relative_positions < 0, -alibi)
        return alibi

    def _get_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        mask = torch.ones(seq_len, seq_len, device=device)
        mask = mask.masked_fill(
            torch.abs(relative_positions) >= self.window_size, float("-inf")
        )
        return mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        alibi_bias = self._get_alibi_bias(N, x.device)
        attn = attn + alibi_bias.unsqueeze(0)

        if self.is_sliding:
            sliding_mask = self._get_sliding_window_mask(N, x.device)
            attn = attn + sliding_mask.unsqueeze(0)

        if attention_mask is not None:
            attn = attn + attention_mask

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FlashAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)

        try:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=is_causal,
                dropout_p=self.dropout if self.training else 0.0,
            )
        except:
            attn = self._manual_attention(q, k, v, attention_mask, is_causal)

        attn = attn.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        B, H, N, D = q.shape

        attn = torch.matmul(q, k.transpose(-2, -1)) * (D**-0.5)

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = True,
        dropouts: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.glu = GatedLinearUnit(dim, hidden_dim, bias)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropouts)

        if dim != hidden_dim:
            self.skip = nn.Linear(dim, dim, bias=bias)
        else:
            self.skip = None

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        if self.skip is not None and skip is not None:
            residual = self.skip(skip)

        x = self.fc1(x)
        x = self.glu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return self.norm(x + residual)


class SwitchTransformerRouting(nn.Module):
    def __init__(
        self, dim: int, num_experts: int = 8, top_k: int = 2, bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts, bias=bias)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 4, bias=bias),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim, bias=bias),
                )
                for _ in range(num_experts)
            ]
        )

        self.noise_std = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        flat_x = x.view(-1, C)

        logits = self.gate(flat_x)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        top_k_logits = top_k_logits - torch.logsumexp(
            top_k_logits, dim=-1, keepdim=True
        )

        flat_x_expanded = flat_x.unsqueeze(1).expand(-1, self.top_k, -1)
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, C)

        expert_inputs = torch.gather(flat_x_expanded, 1, top_k_indices_expanded)

        expert_outputs = torch.zeros(
            B * N, self.top_k, C, device=x.device, dtype=x.dtype
        )

        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_inputs_i = expert_inputs[mask].view(-1, C)
                expert_outputs[mask] = expert(expert_inputs_i).view(-1, self.top_k, C)

        expert_outputs = expert_outputs.sum(dim=1)

        return expert_outputs.view(B, N, C)


class SwitchTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, bias=bias, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.switch = SwitchTransformerRouting(dim, num_experts, top_k, bias)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = (
            x
            + self.attn(
                self.norm1(x),
                self.norm1(x),
                self.norm1(x),
                key_padding_mask=key_padding_mask,
            )[0]
        )

        x = x + self.switch(self.norm2(x))

        x = x + self.mlp(self.norm3(x))

        return x


class AdvancedTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_flash_attention: bool = True,
        use_rotary: bool = True,
        use_alibi: bool = False,
        use_gated_residual: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        sliding_window: int = 256,
    ):
        super().__init__()

        self.use_rotary = use_rotary
        self.use_alibi = use_alibi

        self.norm1 = nn.LayerNorm(dim)

        if use_flash_attention:
            self.attn = FlashAttention(dim, num_heads, bias, dropout)
        elif use_alibi:
            self.attn = ALiBiAttention(
                dim, num_heads, bias, is_sliding=True, window_size=sliding_window
            )
        else:
            self.attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, bias=bias, batch_first=True
            )

        if use_rotary:
            self.rotary = RotaryPositionEmbedding(dim // num_heads)

        self.norm2 = nn.LayerNorm(dim)

        if use_gated_residual:
            self.mlp = GatedResidualNetwork(dim, dropout=dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim, bias=bias),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_rotary and isinstance(self.attn, FlashAttention):
            B, N, C = x.shape
            qkv = self.attn.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, -1)

        x = x + self.attn(self.norm1(x), attention_mask, is_causal)
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        use_flash_attention: bool = True,
        use_rotary: bool = True,
        use_alibi: bool = False,
        use_gated_residual: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        num_experts: Optional[int] = None,
        top_k: int = 2,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if num_experts is not None and i % 2 == 1:
                layer = SwitchTransformerBlock(
                    dim, num_heads, num_experts, top_k, dropout, bias
                )
            else:
                layer = AdvancedTransformerBlock(
                    dim,
                    num_heads,
                    use_flash_attention,
                    use_rotary,
                    use_alibi,
                    use_gated_residual,
                    dropout,
                    bias,
                )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask, is_causal, key_padding_mask)

        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
