# config - Configuration Management Module

## Overview

The `config` module provides YAML/JSON configuration management with environment variable interpolation, validation, and hierarchical configuration support.

## Purpose and Scope

This module enables:
- YAML and JSON configuration loading
- Environment variable interpolation
- Multi-environment configurations (dev/staging/prod)
- Hierarchical configuration merging
- Type-safe configuration access

## Key Classes and Functions

### Configuration Classes

#### `Config`
Configuration manager with dict-like and attribute access.

```python
from fishstick.config import Config

config = Config({
    "model": {
        "name": "uniintelli",
        "hidden_dim": 256
    },
    "training": {
        "epochs": 100,
        "lr": 0.001
    }
})

# Access methods
name = config.model.name         # Dot notation
epochs = config["training"]["epochs"]  # Bracket notation
lr = config.get("training.lr", default=0.01)  # With default

# Convert to dict
config_dict = config.to_dict()
```

### Loading Functions

#### `load_yaml`
Load YAML config with environment variable interpolation.

```yaml
# config.yaml
project:
  name: my-project
  
model:
  name: uniintelli
  input_dim: 784
  
training:
  epochs: 100
  lr: 0.001
  
tracking:
  backend: tensorboard
  project_name: ${PROJECT_NAME}  # Environment variable
```

```python
from fishstick.config import load_yaml

import os
os.environ["PROJECT_NAME"] = "my-experiment"

config = load_yaml("config.yaml")
print(config.tracking.project_name)  # "my-experiment"
```

#### `load_json`
Load JSON configuration.

```python
from fishstick.config import load_json

config = load_json("config.json")
```

### Configuration Manager

#### `ConfigManager`
Manage multiple configurations for different environments.

```python
from fishstick.config import ConfigManager

manager = ConfigManager(config_dir="configs")

# Load base config
config = manager.load("default")

# Load with environment override
config = manager.load("default", env="production")

# The above loads:
# - configs/default.yaml (base)
# - configs/default.production.yaml (overrides)
```

### Default Configuration

```python
from fishstick.config import ConfigManager

manager = ConfigManager()
default = manager.get_default_config()

# Contains:
# project.name, project.version
# model.name, model.input_dim, model.output_dim, model.hidden_dim
# training.epochs, training.batch_size, training.lr, ...
# tracking.backend, tracking.project_name, tracking.log_dir
```

### Convenience Functions

#### `create_default_config`
Create a default configuration file.

```python
from fishstick.config import create_default_config

create_default_config("config.yaml")
# Creates config.yaml with UniIntelli defaults
```

## Dependencies

- `pyyaml`: YAML parsing
- `python-dotenv` (optional): Environment loading

## Usage Examples

### Basic Configuration

```python
from fishstick.config import load_yaml, Config

# Load config
config = load_yaml("configs/training.yaml")

# Use in training
model = create_model(
    input_dim=config.model.input_dim,
    hidden_dim=config.model.hidden_dim
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.training.lr
)

for epoch in range(config.training.epochs):
    train(model, dataloader, optimizer)
```

### Multi-Environment Setup

```
configs/
├── default.yaml          # Base configuration
├── default.dev.yaml      # Development overrides
├── default.staging.yaml  # Staging overrides
└── default.prod.yaml     # Production overrides
```

```python
from fishstick.config import ConfigManager
import os

manager = ConfigManager("configs")
env = os.getenv("ENV", "dev")

config = manager.load("default", env=env)
```

### Environment Variables

```yaml
# config.yaml
database:
  host: ${DB_HOST}
  port: ${DB_PORT:-5432}  # With default
  
model:
  checkpoint: ${CHECKPOINT_PATH}
```

```bash
export DB_HOST=localhost
export CHECKPOINT_PATH=/models/best.pt
```

```python
# ${DB_HOST} will be replaced with "localhost"
# ${DB_PORT:-5432} uses default 5432 if not set
config = load_yaml("config.yaml")
```
