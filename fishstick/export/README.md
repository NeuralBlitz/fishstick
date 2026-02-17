# Model Export

Export models to various formats including ONNX, TensorFlow, and TensorRT.

## Installation

```bash
pip install fishstick[export]
```

## Overview

The `export` module provides utilities for converting PyTorch models to various deployment-friendly formats.

## Usage

```python
from fishstick.export import export_to_onnx, optimize_onnx, convert_to_tensorflow

# Export to ONNX
export_to_onnx(model, input_sample, "model.onnx")

# Optimize ONNX model
optimize_onnx("model.onnx", "model_optimized.onnx")

# Convert to TensorFlow
convert_to_tensorflow("model.onnx", "model_savedmodel/")
```

## Functions

| Function | Description |
|----------|-------------|
| `export_to_onnx` | Export PyTorch model to ONNX format |
| `optimize_onnx` | Optimize ONNX model for inference |
| `convert_to_tensorflow` | Convert ONNX to TensorFlow SavedModel |
| `convert_to_tensorrt` | Convert to TensorRT engine |

## Examples

See `examples/export/` for complete examples.
