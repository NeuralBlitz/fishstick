"""
Training Utilities for fishstick

Comprehensive training utility modules including mixed precision training,
gradient clipping, learning rate warmup, early stopping, and checkpoint management.
"""

from fishstick.training_utils.mixed_precision import (
    MixedPrecisionManager,
    AMPTrainer,
    FP16Manager,
    BF16Manager,
    GradScaler,
    DynamicLossScaler,
)
from fishstick.training_utils.gradient_clipping import (
    GradientClipper,
    NormClipper,
    ValueClipper,
    AdaptiveClipper,
    GradientClipperFactory,
    clip_gradients,
)
from fishstick.training_utils.lr_warmup import (
    LRWarmup,
    LinearWarmup,
    ExponentialWarmup,
    CosineWarmup,
    PolynomialWarmup,
    ConstantWarmup,
    WarmupScheduler,
    create_warmup_scheduler,
)
from fishstick.training_utils.early_stopping import (
    EarlyStopping,
    PatienceEarlyStopping,
    DeltaEarlyStopping,
    BestEarlyStopping,
    EarlyStoppingWithRecovery,
    CompositeEarlyStopping,
    EarlyStoppingCallback,
)
from fishstick.training_utils.checkpoint import (
    CheckpointManager,
    Checkpoint,
    CheckpointStrategy,
    PeriodicCheckpointManager,
    BestCheckpointManager,
    load_checkpoint,
    save_checkpoint,
    load_pretrained,
)

__all__ = [
    # Mixed Precision
    "MixedPrecisionManager",
    "AMPTrainer",
    "FP16Manager",
    "BF16Manager",
    "GradScaler",
    "DynamicLossScaler",
    # Gradient Clipping
    "GradientClipper",
    "NormClipper",
    "ValueClipper",
    "AdaptiveClipper",
    "GradientClipperFactory",
    "clip_gradients",
    # LR Warmup
    "LRWarmup",
    "LinearWarmup",
    "ExponentialWarmup",
    "CosineWarmup",
    "PolynomialWarmup",
    "ConstantWarmup",
    "WarmupScheduler",
    "create_warmup_scheduler",
    # Early Stopping
    "EarlyStopping",
    "PatienceEarlyStopping",
    "DeltaEarlyStopping",
    "BestEarlyStopping",
    "EarlyStoppingWithRecovery",
    "CompositeEarlyStopping",
    "EarlyStoppingCallback",
    # Checkpoint
    "CheckpointManager",
    "Checkpoint",
    "CheckpointStrategy",
    "PeriodicCheckpointManager",
    "BestCheckpointManager",
    "load_checkpoint",
    "save_checkpoint",
    "load_pretrained",
]
