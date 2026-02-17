# Fishstick Interactive Training Visualization Dashboard

A comprehensive, interactive dashboard for visualizing neural network training in real-time. Supports both matplotlib (static) and plotly (interactive) backends, works in Jupyter notebooks, standalone scripts, and as a web server.

## Features

### 1. **TrainingDashboard** - Real-time Training Metrics
- Live plotting of loss, accuracy, learning rate
- Automatic integration with fishstick Trainer via callbacks
- Support for multiple plot types: line charts, scatter plots
- Export to PNG, HTML, and JSON formats

### 2. **LayerVisualizer** - Deep Network Inspection
- Activation visualization for any layer
- Weight distribution histograms
- Gradient flow analysis
- Feature map visualization for CNNs
- Overlay visualizations

### 3. **AttentionVisualizer** - Transformer Analysis
- Attention heatmaps for any layer/head
- Attention rollout visualization
- Multi-head attention comparison
- Specialized BERT visualization support

### 4. **PredictionVisualizer** - Model Evaluation
- Confusion matrices (normalized and raw)
- ROC curves with AUC scores
- Precision-Recall curves
- Misclassification analysis with image overlays

### 5. **RealTimePlot** - Non-blocking Updates
- Update plots without blocking training
- Jupyter notebook support
- Standalone script support
- Dual backend: matplotlib and plotly

### 6. **DashboardServer** - Web Interface
- Flask-based web dashboard
- WebSocket support for real-time updates
- Automatic browser refresh
- REST API for metrics

## Quick Start

### Basic Usage with Trainer

```python
from fishstick.training.advanced import Trainer
from fishstick.visualization.dashboard import TrainingDashboard, DashboardCallback

# Create dashboard
dashboard = TrainingDashboard(
    save_dir="visualizations",
    real_time=True,
    backend='plotly'
)

# Create callback
callback = DashboardCallback(dashboard)

# Create trainer with callback
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    callbacks=[callback]
)

# Train - visualizations update automatically
history = trainer.fit(train_loader, val_loader, epochs=100)
```

### Layer Visualization

```python
from fishstick.visualization.dashboard import LayerVisualizer

# Create visualizer
viz = LayerVisualizer(model, save_dir="visualizations/layers")

# Register hooks
viz.register_hooks()

# Forward pass
output = model(input_data)

# Visualize activations
viz.plot_activations('conv1', max_channels=16)

# Backward pass
loss.backward()

# Visualize gradients
viz.plot_gradient_flow()
viz.plot_weight_distribution()

# Clean up
viz.remove_hooks()
```

### Attention Visualization

```python
from fishstick.visualization.dashboard import AttentionVisualizer

# Create visualizer
viz = AttentionVisualizer(model)

# Set tokens (for NLP models)
viz.set_tokens(['[CLS]', 'The', 'cat', ...])

# Forward pass with attention
outputs = model(input_ids, output_attentions=True)
attentions = torch.stack(outputs.attentions)

# Visualize
viz.plot_attention_heatmap(attentions, layer_idx=0, head_idx=0)
viz.plot_attention_rollout(attentions)
viz.plot_multi_head_comparison(attentions, layer_idx=0)
```

### Prediction Analysis

```python
from fishstick.visualization.dashboard import PredictionVisualizer

# Create visualizer
viz = PredictionVisualizer(
    class_names=['cat', 'dog', 'bird'],
    save_dir="visualizations/predictions"
)

# Update with predictions
viz.update(predictions, targets, probabilities)

# Generate plots
viz.plot_confusion_matrix()
viz.plot_roc_curve()
viz.plot_pr_curve()
viz.analyze_misclassifications(data_loader, model)
```

### Real-time Plotting

```python
from fishstick.visualization.dashboard import RealTimePlot

# Create plot
plot = RealTimePlot(backend='plotly', title="Training Progress")

# During training
for epoch in range(epochs):
    loss = train_epoch()
    plot.append('loss', epoch, loss)
    plot.update()  # Non-blocking update

# Save
plot.save('training.html')
```

### Web Dashboard

```python
from fishstick.visualization.dashboard import DashboardServer

# Create server
server = DashboardServer(host='localhost', port=5000)

# Start server (non-blocking)
server.start()

# During training
for epoch in range(epochs):
    loss = train_epoch()
    server.update_metrics(epoch=epoch, loss=loss)

# Stop server
server.stop()
```

### Quick Convenience Functions

```python
from fishstick.visualization.dashboard import (
    quick_plot_loss,
    quick_plot_metrics,
    create_interactive_dashboard
)

# Quick loss plot
quick_plot_loss(train_loss, val_loss, save_path='loss.png')

# Quick metrics plot
metrics = {'accuracy': acc_history, 'f1': f1_history}
quick_plot_metrics(metrics, save_path='metrics.png')

# Interactive HTML dashboard
create_interactive_dashboard(history, save_path='dashboard.html')
```

## Backend Options

### Matplotlib Backend
- Static plots suitable for publications
- Works everywhere
- Fast rendering

```python
dashboard = TrainingDashboard(backend='matplotlib')
```

### Plotly Backend
- Interactive plots with zoom, pan, hover
- Web-friendly HTML output
- Better for exploration

```python
dashboard = TrainingDashboard(backend='plotly')
```

## Jupyter Notebook Support

```python
from fishstick.visualization.dashboard import TrainingDashboard

# Dashboard auto-detects Jupyter and displays inline
dashboard = TrainingDashboard(real_time=True)

# Plots update automatically in the notebook
```

## Complete Examples

See `examples/dashboard_demo.py` for comprehensive examples including:
- Basic dashboard with Trainer integration
- Layer visualization for CNNs
- Attention visualization for BERT
- Prediction analysis
- Real-time plotting
- Web dashboard

Run examples:
```bash
python examples/dashboard_demo.py
```

## API Reference

### TrainingDashboard

```python
TrainingDashboard(
    save_dir: str = "visualizations/dashboard",
    real_time: bool = True,
    backend: str = 'matplotlib',
    update_interval: int = 1,
    metrics_to_plot: Optional[List[str]] = None,
    interactive: bool = False
)
```

**Methods:**
- `update(epoch, logs)` - Update with new metrics
- `plot_all_metrics(save_path)` - Plot all metrics
- `save_history(filepath)` - Save to JSON
- `load_history(filepath)` - Load from JSON
- `get_summary()` - Get statistics summary
- `print_summary()` - Print formatted summary

### LayerVisualizer

```python
LayerVisualizer(
    model: nn.Module,
    save_dir: str = "visualizations/layers"
)
```

**Methods:**
- `register_hooks(layer_names)` - Register forward/backward hooks
- `remove_hooks()` - Remove all hooks
- `plot_activations(layer_name, max_channels, save_path, interactive)`
- `plot_weight_distribution(max_layers, save_path, interactive)`
- `plot_gradient_flow(save_path, interactive)`
- `plot_feature_map_grid(layer_name, input_data, save_path)`

### AttentionVisualizer

```python
AttentionVisualizer(
    model: nn.Module,
    save_dir: str = "visualizations/attention"
)
```

**Methods:**
- `set_tokens(tokens)` - Set token labels
- `plot_attention_heatmap(attention_weights, layer_idx, head_idx, save_path, interactive)`
- `plot_attention_rollout(attention_weights, discard_ratio, save_path, interactive)`
- `plot_multi_head_comparison(attention_weights, layer_idx, max_heads, save_path)`
- `visualize_bert_attention(input_ids, tokenizer, save_path)`

### PredictionVisualizer

```python
PredictionVisualizer(
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    save_dir: str = "visualizations/predictions"
)
```

**Methods:**
- `update(predictions, targets, probabilities)` - Add predictions
- `plot_confusion_matrix(normalize, save_path, interactive)`
- `plot_roc_curve(save_path, interactive)`
- `plot_pr_curve(save_path, interactive)`
- `analyze_misclassifications(data_loader, model, device, max_samples, save_path)`

### RealTimePlot

```python
RealTimePlot(
    backend: str = 'matplotlib',
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Training Visualization"
)
```

**Methods:**
- `add_line(name, x, y, **style)` - Add line series
- `add_scatter(name, x, y, **style)` - Add scatter series
- `append(name, x, y)` - Append point to series
- `update()` - Update plot (non-blocking)
- `clear()` - Clear all data
- `save(filepath)` - Save to file
- `show()` - Show plot (blocking for matplotlib)

### DashboardServer

```python
DashboardServer(
    host: str = 'localhost',
    port: int = 5000,
    save_dir: str = "visualizations/server"
)
```

**Methods:**
- `start(blocking=False)` - Start server
- `stop()` - Stop server
- `update_metrics(**metrics)` - Broadcast metrics
- `update_plots(plot_data)` - Broadcast plot data

## Dependencies

Core dependencies (already in fishstick):
- torch
- numpy
- matplotlib

Optional dependencies:
- `plotly` - For interactive plots
- `flask`, `flask-socketio` - For web dashboard
- `transformers` - For BERT attention visualization
- `sklearn` - For metrics (confusion matrix, ROC, etc.)

Install optional dependencies:
```bash
pip install plotly flask flask-socketio transformers scikit-learn
```

## Output Structure

```
visualizations/
├── dashboard/
│   ├── all_metrics.png
│   └── training_history.json
├── layers/
│   ├── conv1_activations.png
│   ├── weight_distributions.png
│   └── gradient_flow.png
├── attention/
│   ├── attention_heatmap.png
│   └── multi_head_layer0.png
└── predictions/
    ├── confusion_matrix.png
    ├── roc_curves.html
    └── pr_curves.html
```

## Tips

1. **Memory Usage**: For large models, limit `max_channels` and `max_layers` in visualizations
2. **Performance**: Disable `real_time` for faster training, enable for debugging
3. **File Formats**: Use `.png` for static, `.html` for interactive plots
4. **Jupyter**: Dashboard auto-detects Jupyter environment and displays inline
5. **Web Dashboard**: Great for monitoring training on remote servers
