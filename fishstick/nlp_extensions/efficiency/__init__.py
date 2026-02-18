"""
Efficiency Module

Model efficiency techniques including gradient checkpointing, sparse attention, and flash attention.
"""

from fishstick.nlp_extensions.efficiency.gradient_checkpointing import (
    GradientCheckpointing,
    CheckpointWrapper,
    ModularGradientCheckpointing,
    checkpoint_wrapper,
    create_gradient_checkpointing_function,
    MixedPrecisionCheckpointing,
    SelectiveCheckpointing,
    RecomputationScheduler,
)
from fishstick.nlp_extensions.efficiency.sparse_attention import (
    SparseAttention,
    SlidingWindowAttention,
    BlockSparseAttention,
    DilatedAttention,
    GlobalLocalAttention,
    RandomAttention,
    SparseMultiHeadAttention,
    BigBirdAttention,
)
from fishstick.nlp_extensions.efficiency.flash_attention import (
    FlashAttention,
    FlashAttentionV2,
    MemoryEfficientAttention,
    FlashMultiHeadAttention,
    FlashAttentionWithPdrop,
    is_flash_attn_available,
    create_flash_attention,
)

__all__ = [
    "GradientCheckpointing",
    "CheckpointWrapper",
    "ModularGradientCheckpointing",
    "checkpoint_wrapper",
    "create_gradient_checkpointing_function",
    "MixedPrecisionCheckpointing",
    "SelectiveCheckpointing",
    "RecomputationScheduler",
    "SparseAttention",
    "SlidingWindowAttention",
    "BlockSparseAttention",
    "DilatedAttention",
    "GlobalLocalAttention",
    "RandomAttention",
    "SparseMultiHeadAttention",
    "BigBirdAttention",
    "FlashAttention",
    "FlashAttentionV2",
    "MemoryEfficientAttention",
    "FlashMultiHeadAttention",
    "FlashAttentionWithPdrop",
    "is_flash_attn_available",
    "create_flash_attention",
]
