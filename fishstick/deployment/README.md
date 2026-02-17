# deployment - Production Deployment Module

## Overview

The `deployment` module provides production deployment utilities including Docker containerization, model registry, A/B testing, and monitoring.

## Purpose and Scope

This module enables:
- Docker containerization for model serving
- Model versioning and registry
- A/B testing for model deployment
- Health monitoring and metrics
- Cloud deployment helpers

## Key Classes and Functions

### Docker Deployment

#### `DockerBuilder`
Build Docker containers for model serving.

```python
from fishstick.deployment import DockerBuilder

builder = DockerBuilder(base_image="python:3.11-slim")

# Generate Dockerfile
dockerfile = builder.generate_dockerfile(
    model_path="model.pt",
    port=8000,
    requirements=["torch", "fastapi"]
)

# Build image
tag = builder.build(
    tag="fishstick-model:v1",
    model_path="model.pt",
    context_dir=".",
    push=False
)
```

### Model Registry

#### `ModelRegistry`
Model versioning and tracking.

```python
from fishstick.deployment import ModelRegistry

registry = ModelRegistry("./model_registry")

# Register model
uri = registry.register(
    model_name="classifier",
    version="v1.0.0",
    model_path="model.pt",
    metadata={"accuracy": 0.95, "dataset": "mnist"}
)

# Get model
path = registry.get_model("classifier", version="v1.0.0")

# List models
models = registry.list_models()
```

### A/B Testing

#### `ABTestManager`
Manage A/B testing for model deployment.

```python
from fishstick.deployment import ABTestManager

manager = ABTestManager()

# Create test
test_id = manager.create_test(
    test_name="model_comparison",
    model_a="model_v1.pt",
    model_b="model_v2.pt",
    traffic_split=0.5
)

# Route request
model_path = manager.route_request(test_id, user_id="user123")

# End test
manager.end_test(test_id, winner="b")
```

### Monitoring

#### `HealthMonitor`
Monitor model health and performance.

```python
from fishstick.deployment import HealthMonitor

monitor = HealthMonitor("./monitoring/metrics.json")

# Log predictions
monitor.log_prediction(
    model_name="classifier",
    latency=12.5,
    input_shape=(1, 784),
    output_shape=(1, 10)
)

# Get statistics
stats = monitor.get_statistics()

# Check health
health = monitor.check_health(latency_threshold_ms=100)
```

### Convenience Functions

```python
from fishstick.deployment import (
    deploy_docker, register_model, check_model_health
)

# Quick Docker deploy
container_id = deploy_docker("model.pt", "fishstick:v1", port=8000)

# Quick model registration
uri = register_model("model", "v1.0", "model.pt")

# Quick health check
health = check_model_health()
```

## Dependencies

- `docker` (optional): Docker SDK
- `torch`: Model operations
- Standard library: json, pathlib, datetime

## Usage Examples

### Complete Deployment Pipeline

```python
from fishstick.deployment import (
    DockerBuilder, ModelRegistry, ABTestManager, HealthMonitor
)

# 1. Register model
registry = ModelRegistry()
registry.register("classifier", "v2.0.0", "model.pt")

# 2. Build Docker image
builder = DockerBuilder()
builder.build("classifier:v2.0.0", "model.pt", push=True)

# 3. Deploy and monitor
monitor = HealthMonitor()
# ... run model ...

# 4. A/B test new version
ab = ABTestManager()
test_id = ab.create_test("v2_vs_v3", "model_v2.pt", "model_v3.pt", 0.2)
```
