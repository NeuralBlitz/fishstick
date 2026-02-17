# 🥫 fishstick Sauce

**The Secret Recipe** - Pre-configured, ready-to-train models for all 6 fishstick frameworks.

Just like fishsticks need sauce, your AI needs these models! 🐟

## Quick Start

```python
from fishstick.sauce import create_model

# Grab a bottle and start training immediately
model = create_model('uniintelli_base')
# Model is ready to train - just add your data!
```

## 🥫 Available Sauces (Models)

> *"Just like tartar sauce completes fishsticks, these models complete your AI!"*

### Bottle Sizes

Each sauce comes in 3 bottle sizes:

Each framework comes in 3 sizes:

| Size | Parameters | Best For |
|------|------------|----------|
| **Small** | 150K - 1.5M | Fast prototyping, limited compute |
| **Base** | 1.7M - 6.5M | Balanced performance |
| **Large** | 7M+ | Maximum accuracy |

### Framework A: UniIntelli (Categorical-Geometric-Thermodynamic)

```python
# Small: 500K params - Fast training
model = create_model('uniintelli_small')

# Base: 1.8M params - Balanced (RECOMMENDED)
model = create_model('uniintelli_base')

# Large: 7M params - Best accuracy
model = create_model('uniintelli_large')
```

### Framework B: HSCA (Holo-Symplectic Cognitive Architecture)

```python
# Energy-conserving Hamiltonian dynamics
model = create_model('hsca_small')   # 1.3M params
model = create_model('hsca_base')    # 6.5M params
```

### Framework C: UIA (Unified Intelligence Architecture)

```python
# CHNP + RG-AE + S-TF + DTL
model = create_model('uia_small')    # 800K params
model = create_model('uia_base')     # 1.7M params
```

### Framework D: SCIF (Symplectic-Categorical Intelligence)

```python
# Fiber bundle geometry
model = create_model('scif_small')   # 1.5M params
model = create_model('scif_base')    # 3.8M params
```

### Framework E: UIF (Unified Intelligence Framework)

```python
# 4-layer stack: Category → Geometry → Dynamics → Verification
model = create_model('uif_small')    # 150K params
model = create_model('uif_base')     # 367K params
```

### Framework F: UIS (Unified Intelligence Synthesis)

```python
# Quantum-inspired + RG + Neuro-symbolic
model = create_model('uis_small')    # 400K params
model = create_model('uis_base')     # 861K params
```

## Task-Specific Models

### Image Classification

```python
from fishstick.sauce import TaskModels

# For MNIST (10 classes)
model = TaskModels.image_classifier(num_classes=10)

# For CIFAR-100 (100 classes)
model = TaskModels.image_classifier(
    num_classes=100,
    framework='uniintelli',
    size='large'
)

# For ImageNet (1000 classes)
model = TaskModels.image_classifier(
    num_classes=1000,
    framework='hsca',
    size='base'
)
```

### Text Classification

```python
# Sentiment analysis (2 classes)
model = TaskModels.text_classifier(
    num_classes=2,
    vocab_size=10000,
    framework='uia'
)

# Multi-class text classification
model = TaskModels.text_classifier(
    num_classes=20,
    vocab_size=50000,
    framework='uis',
    size='large'
)
```

### Regression

```python
# House price prediction (1 output)
model = TaskModels.regression_model(
    output_dim=1,
    input_dim=50,
    framework='scif'
)

# Multi-output regression
model = TaskModels.regression_model(
    output_dim=5,
    input_dim=100,
    framework='uif',
    size='small'
)
```

### Time Series Forecasting

```python
# Predict next 24 hours
model = TaskModels.time_series_forecaster(
    forecast_horizon=24,
    input_dim=10,  # 10 features
    framework='uis'
)
```

## Custom Model Building

### Using the Fluent Builder API

```python
from fishstick.sauce import build_model

# Build custom model step by step
model = (build_model()
    .with_framework('uniintelli')      # Choose framework
    .with_input_dim(784)               # Input size
    .with_output_dim(10)               # Output classes
    .with_hidden_dim(512)              # Hidden layer size
    .with_layers(6)                    # Number of layers
    .build())
```

### Manual Configuration

```python
from fishstick.sauce import create_model

# Override any default parameter
model = create_model(
    'uniintelli_base',
    input_dim=1000,      # Custom input size
    output_dim=100,      # Custom output classes
    hidden_dim=512       # Custom hidden dimension
)
```

## Complete Training Example

```python
import torch
from fishstick.sauce import create_model
from fishstick.tracking import create_tracker

# 1. Create model
model = create_model('uniintelli_base')
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# 2. Setup experiment tracking
tracker = create_tracker(
    project_name="my_project",
    experiment_name="experiment_1",
    backend="wandb"  # or "tensorboard", "mlflow"
)

# 3. Log hyperparameters
tracker.log_params({
    "model": "uniintelli_base",
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 100
})

# 4. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    
    # Your training code here
    # for batch in dataloader:
    #     optimizer.zero_grad()
    #     outputs = model(batch['input'])
    #     loss = criterion(outputs, batch['target'])
    #     loss.backward()
    #     optimizer.step()
    
    # Log metrics
    tracker.log_metrics({
        "train/loss": 0.5,  # Your actual loss
        "train/accuracy": 0.95  # Your actual accuracy
    }, step=epoch)

# 5. Save trained model
torch.save(model.state_dict(), "models/my_trained_model.pt")
tracker.finish()
```

## Transfer Learning

### Load Pretrained Model

```python
from fishstick.sauce import TransferLearning

# Load a model you previously trained
model = TransferLearning.load_pretrained(
    model_name='uniintelli_base',
    checkpoint_path='models/my_trained_model.pt'
)
```

### Freeze Early Layers

```python
# Freeze first 3 layers for fine-tuning
model = TransferLearning.freeze_layers(model, num_layers=3)

# Only last few layers will train
```

### Replace Final Layer

```python
# Adapt model to new task with different number of classes
model = TransferLearning.replace_head(model, new_output_dim=50)
# Now model outputs 50 classes instead of original
```

## Listing Available Models

```python
from fishstick.sauce import list_available_models

# Get all available model configurations
models = list_available_models()

for name, description in models.items():
    print(f"{name:20s} - {description}")
```

Output:
```
uniintelli_small     - UniIntelli-Small: 500K params, fast training
uniintelli_base      - UniIntelli-Base: 1.8M params, balanced performance
uniintelli_large     - UniIntelli-Large: 7M params, best accuracy
hsca_small           - HSCA-Small: 2M params, energy-conserving
hsca_base            - HSCA-Base: 6.5M params, symplectic dynamics
uia_small            - UIA-Small: 800K params, categorical-Hamiltonian
...
```

## Model Selection Guide

### Choose Your Framework

| Framework | Best For | Key Feature |
|-----------|----------|-------------|
| **UniIntelli** | General purpose | Category + Geometry + Thermodynamics |
| **HSCA** | Physics-aware tasks | Energy-conserving Hamiltonian dynamics |
| **UIA** | Complex reasoning | CHNP + RG-AE + Sheaf + Verification |
| **SCIF** | Structured data | Fiber bundle geometry |
| **UIF** | Resource-constrained | Lightweight 4-layer stack |
| **UIS** | Multi-modal tasks | Quantum-inspired + Neuro-symbolic |

### Choose Your Size

- **Small**: Fast iteration, limited compute, prototyping
- **Base**: Production use, balanced speed/accuracy (RECOMMENDED)
- **Large**: Maximum accuracy when compute is available

## Advanced Usage

### Custom Configuration

```python
from fishstick.sauce import ModelZoo

# Access raw configurations
config = ModelZoo.CONFIGS['uniintelli_base']
print(config)
# {'input_dim': 784, 'output_dim': 10, 'hidden_dim': 256, 'n_layers': 4}

# Create with modifications
model = ModelZoo.create_model('uniintelli_base', n_layers=8)
```

### Batch Model Creation

```python
# Compare multiple models
models = {
    'uniintelli': create_model('uniintelli_base'),
    'hsca': create_model('hsca_base'),
    'uia': create_model('uia_base'),
}

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} parameters")
```

## Tips for Training

1. **Start Small**: Use `*_small` models for prototyping, then scale up
2. **Track Experiments**: Always use tracking to compare models
3. **Monitor Energy**: For HSCA models, check energy conservation
4. **Use Appropriate Framework**: Match framework to your data type
5. **Transfer Learning**: Start with pretrained models when available

## Examples by Use Case

### Computer Vision
```python
# Image classification
model = TaskModels.image_classifier(num_classes=10, framework='uniintelli')

# With fishstick vision module
from fishstick.vision import VisionTransformer
model = VisionTransformer(img_size=224, num_classes=1000)
```

### Natural Language Processing
```python
# Text classification
model = TaskModels.text_classifier(
    num_classes=2,
    vocab_size=10000,
    framework='uia',
    size='base'
)
```

### Time Series
```python
# Forecasting
model = TaskModels.time_series_forecaster(
    forecast_horizon=24,
    input_dim=10,
    framework='uis'
)
```

### Scientific Computing
```python
# Physics-informed (use HSCA for energy conservation)
model = create_model('hsca_base', input_dim=100, output_dim=50)
```

## Integration with Other fishstick Modules

```python
# Create model
from fishstick.sauce import create_model
model = create_model('uniintelli_base')

# Track training
from fishstick.tracking import create_tracker
tracker = create_tracker('my_project')

# Optimize for deployment
from fishstick.compression import optimize_model
model = optimize_model(model, quantize=True)

# Deploy
from fishstick.deployment import deploy_docker
deploy_docker('model.pt', 'myapp:v1')
```

## Troubleshooting

**Model too big for GPU?**
```python
# Use smaller model
model = create_model('uniintelli_small')

# Or quantize
from fishstick.compression import quantize_model
model = quantize_model(model)
```

**Need different input/output sizes?**
```python
# Just specify when creating
model = create_model('uniintelli_base', input_dim=1000, output_dim=100)
```

**Want to compare frameworks?**
```python
frameworks = ['uniintelli', 'hsca', 'uia', 'scif', 'uif', 'uis']
models = [create_model(f'{fw}_base') for fw in frameworks]
```

## Next Steps

1. **Choose a model** from the zoo above
2. **Prepare your data** (see `fishstick.data` module)
3. **Train with tracking** (see `fishstick.tracking` module)
4. **Optimize** (see `fishstick.compression` module)
5. **Deploy** (see `fishstick.deployment` module)

## See Also

- [Main README](../README.md) - Overview of fishstick
- [Framework Documentation](../A.md) - Detailed framework docs (A-F)
- [Vision Module](../fishstick/vision/) - Computer vision models
- [Tracking Module](../fishstick/tracking/) - Experiment tracking
- [Compression Module](../fishstick/compression/) - Model optimization

---

**Ready to train?** Just pick a model above and go! 🚀