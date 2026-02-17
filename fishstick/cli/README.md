# cli - Command Line Interface Module

## Overview

The `cli` module provides a comprehensive command-line interface for fishstick, enabling training, evaluation, model management, and serving without writing Python code.

## Purpose and Scope

This module enables:
- Model training from command line
- Model evaluation and benchmarking
- Project initialization
- Model serving via REST API
- Pretrained model downloading

## Available Commands

### `fishstick train`
Train a model on a dataset.

```bash
fishstick train --model uniintelli --dataset mnist --epochs 10

# With all options
fishstick train \
    --model uniintelli \
    --dataset mnist \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --project my-project \
    --experiment exp-001 \
    --tracker tensorboard \
    --save model.pt
```

**Arguments:**
- `--model`: Model architecture (uniintelli, hsca, uia, etc.)
- `--dataset`: Dataset name
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--lr`: Learning rate
- `--project`: Project name for tracking
- `--experiment`: Experiment name
- `--tracker`: Tracking backend (tensorboard, wandb, mlflow)
- `--save`: Path to save trained model

### `fishstick eval`
Evaluate a trained model.

```bash
fishstick eval model.pt --dataset test

# Output:
# accuracy: 92.00%
# precision: 91.00%
# recall: 93.00%
# f1: 92.00%
```

### `fishstick download-model`
Download a pretrained model.

```bash
fishstick download-model gpt2 --output ./models/
fishstick download-model bert-base-uncased
```

### `fishstick list-models`
List available models.

```bash
fishstick list-models

# Frameworks:
#   uniintelli - Categorical-Geometric-Thermodynamic
#   hsca - Holo-Symplectic Cognitive Architecture
#   ...
# Components:
#   hamiltonian - Hamiltonian Neural Networks
#   sheaf - Sheaf-Optimized Attention
#   ...
```

### `fishstick serve`
Serve model via REST API.

```bash
fishstick serve --model model.pt --port 8000

# API available at http://localhost:8000
# Endpoints:
#   POST /predict - Get predictions
#   GET /health - Health check
#   GET /info - Model info
```

### `fishstick init`
Initialize a new project.

```bash
fishstick init my-project

# Creates:
#   my-project/
#   ├── src/
#   ├── data/
#   ├── models/
#   ├── configs/
#   ├── notebooks/
#   ├── experiments/
#   ├── tests/
#   ├── configs/default.yaml
#   ├── train.py
#   └── README.md
```

### `fishstick demo`
Run interactive demo.

```bash
fishstick demo

# Demonstrates:
# 1. Hamiltonian Neural Network
# 2. UniIntelli framework
# 3. Energy conservation
```

## Usage Examples

### Complete Training Pipeline

```bash
# Initialize project
fishstick init my-ml-project
cd my-ml-project

# Train model
fishstick train \
    --model uniintelli \
    --dataset mnist \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.0005 \
    --save models/best.pt

# Evaluate
fishstick eval models/best.pt --dataset test

# Serve
fishstick serve --model models/best.pt --port 8000
```

### Integration with Experiment Tracking

```bash
# Using Weights & Biases
fishstick train \
    --model uniintelli \
    --dataset mnist \
    --epochs 100 \
    --tracker wandb \
    --project my-project

# Using MLflow
fishstick train \
    --model uniintelli \
    --dataset mnist \
    --epochs 100 \
    --tracker mlflow \
    --project my-project
```

### REST API Usage After Serving

```bash
# Start server
fishstick serve --model model.pt --port 8000 &

# Make predictions
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"data": [[1.0, 2.0, 3.0]], "return_probabilities": true}'

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/info
```

## Dependencies

- `argparse`: Command-line parsing
- `torch`: Model operations
- `fastapi`: API serving (optional)
- `uvicorn`: ASGI server (optional)
- `transformers`: Pretrained models (optional)

## Programmatic Usage

```python
from fishstick.cli import train_command, eval_command

# Programmatically train
args = argparse.Namespace(
    model="uniintelli",
    dataset="mnist",
    epochs=10,
    batch_size=32,
    lr=0.001
)
train_command(args)
```
