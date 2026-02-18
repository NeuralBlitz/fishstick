"""
Position Encoding Module

Advanced positional encoding implementations including RoPE, ALiBi, and relative position.
"""

from fishstick.nlp_extensions.position_encoding.rope import (
    RotaryPositionalEmbedding,
    RotaryEmbedding,
    LinearRoPE,
    YaRNScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    MultiHeadRotaryAttention,
)
from fishstick.nlp_extensions.position_encoding.alibi import (
    ALiBiAttention,
    ALiBiPositionalEmbedding,
    ALiBiMultiHeadAttention,
    AliBiFlashAttention,
    build_alibi_bias,
)
from fishstick.nlp_extensions.position_encoding.relative_position import (
    RelativePositionBias,
    SinusoidalRelativePositionBias,
    RelativePositionMultiHeadAttention,
    T5RelativePositionBias,
    ShawRelativePosition,
    RelativePositionKeyValue,
    relative_position_bucket,
)

__all__ = [
    "RotaryPositionalEmbedding",
    "RotaryEmbedding",
    "LinearRoPE",
    "YaRNScalingRotaryEmbedding",
    "apply_rotary_pos_emb",
    "MultiHeadRotaryAttention",
    "ALiBiAttention",
    "ALiBiPositionalEmbedding",
    "ALiBiMultiHeadAttention",
    "AliBiFlashAttention",
    "build_alibi_bias",
    "RelativePositionBias",
    "SinusoidalRelativePositionBias",
    "RelativePositionMultiHeadAttention",
    "T5RelativePositionBias",
    "ShawRelativePosition",
    "RelativePositionKeyValue",
    "relative_position_bucket",
]
