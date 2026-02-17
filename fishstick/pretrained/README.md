# Pretrained Models Module

## Overview

Pretrained model weights, registry, and loading utilities for the fishstick framework. Provides a unified interface for loading and managing pretrained models from various sources.

## Purpose and Scope

- Centralized model registry for pretrained weights
- Weight initialization and adaptation utilities
- Model loading from local and remote sources
- Version control for model weights

## Key Classes and Functions

### ModelRegistry

Central registry for managing pretrained models.

```python
from fishstick.pretrained import ModelRegistry, list_models, load_pretrained

# List available models
models = list_models()
print(models)  # ['resnet50', 'bert-base', 'gpt2', ...]

# Get model info
info = get_model_info('resnet50')
print(info['num_params'], info['input_size'])

# Load pretrained model
model = load_pretrained('resnet50', pretrained=True)
```

### Weights

Weight container with initialization and adaptation utilities.

```python
from fishstick.pretrained import Weights

# Load weights from checkpoint
weights = Weights.from_checkpoint('model.pt')

# Initialize new model with pretrained weights
weights.initialize(model)

# Adapt weights for fine-tuning
weights.adapt(target_model, adaptation_type='lora')
```

### load_pretrained

Factory function for loading pretrained models.

```python
from fishstick.pretrained import load_pretrained

# Load with default configuration
model = load_pretrained('bert-base-uncased')

# Load with custom configuration
model = load_pretrained(
    'gpt2',
    config={'n_layer': 12, 'n_head': 12},
    strict=False
)
```

## Model Registry Features

- **Versioning**: Track model versions and checksums
- **Caching**: Local caching of downloaded weights
- **Validation**: Checksum verification for weights
- **Lazy Loading**: Load weights on demand

## Supported Model Types

| Category | Examples |
|----------|----------|
| Vision | ResNet, ViT, ConvNeXt |
| Language | BERT, GPT, LLaMA |
| Multimodal | CLIP, BLIP |
| Audio | Whisper, Wav2Vec |

## Dependencies

- `torch` - PyTorch for model definitions
- `hashlib` - Checksum computation
- `json` - Model metadata handling
- `pathlib` - File path utilities

## Usage Examples

### Basic Model Loading

```python
from fishstick.pretrained import load_pretrained

# Load pretrained classifier
model = load_pretrained('resnet50', num_classes=10)
model.eval()

# Run inference
output = model(input_tensor)
```

### Fine-tuning with Pretrained Weights

```python
from fishstick.pretrained import load_pretrained, Weights

# Load base model
model = load_pretrained('bert-base')

# Load custom weights
weights = Weights.from_checkpoint('fine_tuned.pt')
weights.load_into(model, strict=False)
```

### Model Registry Operations

```python
from fishstick.pretrained import ModelRegistry

# Create custom registry
registry = ModelRegistry()

# Register custom model
registry.register(
    name='my-model',
    model_class=MyModel,
    weights_url='https://example.com/weights.pt',
    config={'hidden_dim': 256}
)

# List all registered models
all_models = registry.list_all()
```
