# Training Utilities

Training utilities, callbacks, and learning rate schedulers.

## Installation

```bash
pip install fishstick[training]
```

## Overview

The `training` module provides utilities for training neural networks including callbacks, schedulers, and distributed training support.

## Usage

```python
from fishstick.training import Trainer, train_model
from fishstick.training import EarlyStopping, ModelCheckpoint

# Training loop
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    callbacks=[
        EarlyStopping(patience=5),
        ModelCheckpoint("best.pt", mode="max")
    ]
)
trainer.train(epochs=100)

# Or functional API
train_model(model, train_loader, optimizer)
```

## Callbacks

| Callback | Description |
|----------|-------------|
| `EarlyStopping` | Stop training when metric stops improving |
| `ModelCheckpoint` | Save model checkpoints |
| `LearningRateScheduler` | Learning rate scheduling |

## Schedulers

| Scheduler | Description |
|-----------|-------------|
| `CosineAnnealingWarmRestarts` | Cosine annealing with warm restarts |
| `OneCycleLR` | One cycle learning rate |
| `WarmupScheduler` | Learning rate warmup |

## Trainers

| Trainer | Description |
|---------|-------------|
| `Trainer` | Base trainer |
| `DistributedTrainer` | Distributed training |
| `MixedPrecisionTrainer` | Mixed precision training |
| `EMA` | Exponential moving average |

## Metrics

| Metric | Description |
|--------|-------------|
| `Accuracy` | Classification accuracy |
| `Precision` | Precision |
| `Recall` | Recall |
| `F1Score` | F1 score |
| `AUCROC` | AUC-ROC |

## Examples

See `examples/training/` for complete examples.
