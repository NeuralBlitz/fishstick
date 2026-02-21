# Visualization Module

Comprehensive visualization tools for models, training progress, data analysis, and real-time dashboards.

## Overview

This module provides tools to visualize neural networks, training metrics, data distributions, and create interactive dashboards for monitoring deep learning experiments.

### Components

| Component | Description |
|-----------|-------------|
| `TrainingVisualizer` | Visualize training curves and metrics |
| `ModelVisualizer` | Neural network architecture visualization |
| `DataVisualizer` | Data distribution and sample visualization |
| `TrainingDashboard` | Real-time interactive dashboard |

## Quick Start

### Training Visualization

```python
from fishstick.visualization import TrainingVisualizer
import numpy as np

# Initialize visualizer
viz = TrainingVisualizer(
    metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy'],
    title="Training Progress"
)

# Log metrics each epoch
for epoch in range(100):
    loss = np.random.uniform(0.1, 0.5)
    accuracy = 0.5 + epoch * 0.005 + np.random.uniform(-0.01, 0.01)
    val_loss = loss + np.random.uniform(0, 0.1)
    val_accuracy = accuracy - 0.05
    
    viz.log_epoch(epoch, {
        'loss': loss,
        'accuracy': accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })

viz.show()
```

### Model Architecture Visualization

```python
from fishstick.visualization import ModelVisualizer
import torch.nn as nn

# Define a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Visualize architecture
model_viz = ModelVisualizer(model)
model_viz.plot_network()
model_viz.save("architecture.png")
```

### Data Visualization

```python
from fishstick.visualization import DataVisualizer

# Create data visualizer
data_viz = DataVisualizer()

# Visualize dataset statistics
data_viz.plot_class_distribution(dataset)
data_viz.plot_feature_histograms(dataset)
data_viz.plot_samples(dataset, num_samples=16)
```

### Interactive Dashboard

```python
from fishstick.visualization import (
    TrainingDashboard,
    create_interactive_dashboard
)

# Quick dashboard
dashboard = create_interactive_dashboard(
    port=8080,
    title="Experiment Dashboard"
)

# Or create custom dashboard
dashboard = TrainingDashboard(
    title="My Training Run",
    metrics=['loss', 'accuracy'],
    refresh_rate=1.0
)

# Add custom plots
dashboard.add_plot(
    name="learning_rate",
    xlabel="Epoch",
    ylabel="LR",
    plot_type="line"
)

# Register callback for training
from fishstick.visualization import DashboardCallback

class MyCallback(DashboardCallback):
    def on_epoch_end(self, epoch, logs):
        dashboard.update({
            'loss': logs['loss'],
            'accuracy': logs['accuracy']
        })

# Run with dashboard
trainer.fit(train_data, callbacks=[MyCallback()])
dashboard.serve()
```

## API Reference

### TrainingVisualizer

```python
TrainingVisualizer(
    metrics: List[str],           # Metrics to track
    title: str = "Training",      # Plot title
    save_path: str = None,        # Save path for plots
    show: bool = True             # Display immediately
)
```

| Method | Description |
|--------|-------------|
| `log_epoch(epoch, metrics)` | Log metrics for an epoch |
| `log_batch(batch, metrics)` | Log metrics for a batch |
| `plot()` | Generate plot |
| `show()` | Display plot |
| `save(path)` | Save plot to file |

### ModelVisualizer

```python
ModelVisualizer(
    model: nn.Module,             # PyTorch model
    input_shape: tuple = None,    # Input shape for tracing
    save_path: str = None
)
```

| Method | Description |
|--------|-------------|
| `plot_network()` | Visualize network architecture |
| `plot_layer_outputs(layer_idx, input_data)` | Visualize layer outputs |
| `plot_attention(attention_weights)` | Visualize attention maps |
| `save(path)` | Save visualization |

### DataVisualizer

```python
DataVisualizer(
    dataset: Dataset = None,
    num_classes: int = None
)
```

| Method | Description |
|--------|-------------|
| `plot_samples(n=16)` | Plot sample images |
| `plot_class_distribution()` | Plot class distribution |
| `plot_feature_histograms()` | Plot feature histograms |
| `plot_tsne(features, labels)` | Plot t-SNE embedding |
| `plot_confusion_matrix(cm)` | Plot confusion matrix |

### TrainingDashboard

```python
TrainingDashboard(
    title: str,
    metrics: List[str],
    port: int = 8050,
    refresh_rate: float = 1.0
)
```

| Method | Description |
|--------|-------------|
| `update(metrics)` | Update dashboard with new metrics |
| `add_plot(name, xlabel, ylabel, plot_type)` | Add custom plot |
| `serve()` | Start dashboard server |
| `stop()` | Stop dashboard server |

### Quick Plotting Functions

```python
# Quick loss plotting
quick_plot_loss(
    train_losses: List[float],
    val_losses: List[float] = None,
    save_path: str = None
)

# Quick metrics plotting
quick_plot_metrics(
    metrics: Dict[str, List[float]],
    save_path: str = None
)
```

## Advanced Usage

### Custom Layer Visualization

```python
from fishstick.visualization import LayerVisualizer, AttentionVisualizer

# Visualize specific layers
layer_viz = LayerVisualizer(model)
layer_viz.visualize_conv_filters(layer_name="features.0")
layer_viz.visualize_activations(input_data)

# Visualize attention
attn_viz = AttentionVisualizer()
attn_viz.plot_attention_weights(attention_matrix)
attn_viz.plot_attention_head(head_idx=0, attention_weights)
```

### Real-Time Plotting

```python
from fishstick.visualization import RealTimePlot

plot = RealTimePlot(
    title="Live Training",
    xlabel="Step",
    ylabel="Loss"
)

for step in range(1000):
    loss = compute_loss()
    plot.update(step, loss)
    time.sleep(0.1)
```

### Prediction Visualization

```python
from fishstick.visualization import PredictionVisualizer

pred_viz = PredictionVisualizer(model)
pred_viz.plot_predictions(
    images=test_images,
    labels=test_labels,
    predictions=predictions,
    class_names=class_names
)
```

## Installation Requirements

```bash
pip install matplotlib seaborn plotly
```

For interactive dashboards:
```bash
pip install dash plotly
```
