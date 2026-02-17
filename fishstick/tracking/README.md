# Tracking Module

## Overview

Experiment tracking module with unified interface for popular ML tracking tools: Weights & Biases, MLflow, TensorBoard, and Neptune. Provides consistent API for logging metrics, parameters, artifacts, and models.

## Purpose and Scope

- Unified tracking interface across backends
- Multi-backend support simultaneously
- Automatic metric and parameter logging
- Model checkpointing and versioning
- System metrics monitoring

## Key Classes and Functions

### BaseTracker

Abstract base class defining the tracking interface.

```python
from fishstick.tracking import BaseTracker

class CustomTracker(BaseTracker):
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: int = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def log_model(self, model, name: str) -> None: ...
    def finish(self) -> None: ...
```

### WandbTracker

Weights & Biases integration.

```python
from fishstick.tracking import WandbTracker

tracker = WandbTracker(
    project_name="my_project",
    experiment_name="experiment_1",
    api_key="your-api-key"
)

tracker.log_params({"lr": 0.001, "batch_size": 32})
tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
tracker.log_model(model, "best_model")
tracker.finish()
```

### MLflowTracker

MLflow integration for local and server-based tracking.

```python
from fishstick.tracking import MLflowTracker

tracker = MLflowTracker(
    project_name="my_project",
    tracking_uri="http://localhost:5000"
)

tracker.log_params({"epochs": 50, "optimizer": "adam"})
tracker.log_metrics({"train_loss": 0.3, "val_loss": 0.5})
tracker.finish()
```

### TensorBoardTracker

TensorBoard integration for local visualization.

```python
from fishstick.tracking import TensorBoardTracker

tracker = TensorBoardTracker(
    project_name="my_project",
    log_dir="./runs"
)

tracker.log_params({"hidden_dim": 256})
tracker.log_metrics({"loss": 0.5}, step=100)
tracker.log_histogram("weights", model.fc1.weight)
tracker.finish()

# Launch dashboard
tracker.launch_dashboard(port=6006)
```

### MultiTracker

Track experiments across multiple backends simultaneously.

```python
from fishstick.tracking import MultiTracker

tracker = MultiTracker(
    project_name="my_project",
    experiment_name="exp_1",
    trackers=["wandb", "mlflow", "tensorboard"]
)

# Logs to all backends at once
tracker.log_params({"lr": 0.001})
tracker.log_metrics({"loss": 0.5}, step=100)
tracker.finish()
```

### ExperimentLogger

High-level logger with automatic metric tracking.

```python
from fishstick.tracking import ExperimentLogger, create_tracker

tracker = create_tracker("my_project", backend="wandb")
logger = ExperimentLogger(tracker, log_interval=10)

# Automatic step counting and periodic logging
for batch in dataloader:
    loss = train_step(batch)
    logger.log_training_step({"loss": loss}, model=model)

# Epoch logging
logger.log_epoch_end(epoch=1, metrics={"val_loss": 0.3}, model=model)

# System metrics
logger.log_system_metrics()

# Summary
print(logger.get_training_summary())
logger.finish()
```

### Convenience Functions

```python
from fishstick.tracking import create_tracker, create_multi_tracker

# Single backend
tracker = create_tracker("project", backend="wandb")

# Multiple backends
multi = create_multi_tracker(
    "project",
    trackers=["wandb", "tensorboard"],
    wandb_api_key="key",
    tensorboard_dir="./runs"
)
```

## Feature Comparison

| Feature | W&B | MLflow | TensorBoard |
|---------|-----|--------|-------------|
| Real-time viz | ✓ | ✓ | ✓ |
| Hyperparameter sweeps | ✓ | ✓ | ✗ |
| Model registry | ✓ | ✓ | ✗ |
| Offline support | Limited | ✓ | ✓ |
| Collaboration | ✓ | Limited | ✗ |
| Artifacts | ✓ | ✓ | Limited |

## Dependencies

- `torch` - Model logging
- `pathlib` - Path handling
- `datetime` - Timestamp generation

**Optional:**
- `wandb` - W&B backend
- `mlflow` - MLflow backend
- `tensorboard` - TensorBoard backend
- `psutil` - System metrics

## Usage Examples

### Complete Training Pipeline

```python
from fishstick.tracking import create_tracker, ExperimentLogger

def train(model, dataloader, config):
    tracker = create_tracker(
        project_name="fishstick_experiments",
        experiment_name=config["name"],
        backend="wandb"
    )
    
    logger = ExperimentLogger(tracker, log_interval=10)
    logger.log_params(config)
    
    for epoch in range(config["epochs"]):
        for batch in dataloader:
            loss = train_step(model, batch)
            logger.log_training_step({"loss": loss})
        
        val_metrics = validate(model, val_loader)
        logger.log_epoch_end(epoch, val_metrics, model)
        logger.log_system_metrics()
    
    logger.finish()
```

### Multi-Backend Tracking

```python
from fishstick.tracking import MultiTracker

tracker = MultiTracker(
    project_name="research",
    trackers=["wandb", "tensorboard"]
)

# Log to both W&B and TensorBoard
tracker.log_params(config)
tracker.log_metrics({"accuracy": 0.95})
```

### Model Versioning with W&B

```python
from fishstick.tracking import WandbTracker

tracker = WandbTracker("model_registry")
tracker.log_model(model, "production_model_v1")

# Log with metadata
tracker.log_artifact("model.pt", artifact_path="models")
```
