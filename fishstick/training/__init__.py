"""
fishstick Training Module

Comprehensive training utilities including callbacks, schedulers, metrics, and training loops.
"""

from fishstick.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    GradientClipping,
    ProgressBar,
    TensorBoardCallback,
    WandBCallback,
    MetricsLogger,
    ParameterMonitor,
    EpochTimer,
)
from fishstick.training.schedulers import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    WarmupScheduler,
    CyclicCosineScheduler,
    PolynomialDecay,
    ExponentialWarmup,
)
from fishstick.training.loop import Trainer, train_model
from fishstick.training.metrics import (
    Metric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUCROC,
    ConfusionMatrix,
    MetricTracker,
)
from fishstick.training.distributed import (
    DistributedTrainer,
    GradientAccumulator,
    ExponentialMovingAverage,
    MixedPrecisionTrainer,
    average_gradients,
)

__all__ = [
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "GradientClipping",
    "ProgressBar",
    "TensorBoardCallback",
    "WandBCallback",
    "MetricsLogger",
    "ParameterMonitor",
    "EpochTimer",
    # Schedulers
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "WarmupScheduler",
    "CyclicCosineScheduler",
    "PolynomialDecay",
    "ExponentialWarmup",
    # Training
    "Trainer",
    "train_model",
    # Metrics
    "Metric",
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "AUCROC",
    "ConfusionMatrix",
    "MetricTracker",
    # Distributed
    "DistributedTrainer",
    "GradientAccumulator",
    "ExponentialMovingAverage",
    "MixedPrecisionTrainer",
    "average_gradients",
]
