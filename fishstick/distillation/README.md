# Knowledge Distillation

Model compression via knowledge distillation.

## Installation

```bash
pip install fishstick[distillation]
```

## Overview

The `distillation` module provides knowledge distillation techniques for compressing large models into smaller ones.

## Usage

```python
from fishstick.distillation import DistillationLoss, KnowledgeDistillationLoss, FeatureDistillation

# Basic distillation
kd_loss = KnowledgeDistillationLoss(
    teacher_model=teacher,
    student_model=student,
    temperature=4.0,
    alpha=0.5
)
loss = kd_loss(student_logits, teacher_logits, labels)

# Feature-based distillation
feature_distill = FeatureDistillation(
    teacher_hidden=[256, 128],
    student_hidden=[128, 64]
)
```

## Losses

| Loss | Description |
|------|-------------|
| `DistillationLoss` | Base distillation loss |
| `KnowledgeDistillationLoss` | Standard KD loss |
| `TemperatureScaledLoss` | Temperature-scaled cross entropy |
| `FeatureDistillation` | Feature matching loss |
| `RelationDistillation` | Relation-based distillation |
| `AttentionTransfer` | Attention transfer |

## Methods

| Method | Description |
|--------|-------------|
| `TakeKD` | Take knowledge distillation |
| `DeepMutualLearning` | Deep mutual learning |

## Examples

See `examples/distillation/` for complete examples.
