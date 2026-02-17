# compression - Model Compression Module

## Overview

The `compression` module provides tools for model compression and optimization including pruning, quantization, knowledge distillation, and export utilities.

## Purpose and Scope

This module enables:
- Weight pruning (magnitude-based, structured)
- Model quantization (INT8, FP16)
- Knowledge distillation for model compression
- ONNX export for cross-platform deployment
- TorchScript compilation for production

## Key Classes and Functions

### Pruning

#### `Pruner`
Model pruning for removing redundant parameters.

```python
from fishstick.compression import Pruner
import torch.nn as nn

pruner = Pruner(model, pruning_type="magnitude")

# Magnitude-based pruning
pruned_model = pruner.prune_magnitude(amount=0.3)  # Remove 30% of weights

# Structured pruning
pruned_model = pruner.prune_structured(amount=0.3, dim=0)

# Check sparsity
sparsity = pruner.get_sparsity(pruned_model)
print(f"Model is {sparsity:.1%} sparse")
```

### Quantization

#### `Quantizer`
Model quantization for efficient inference.

```python
from fishstick.compression import Quantizer

quantizer = Quantizer(model)

# Dynamic quantization
quantized = quantizer.quantize_dynamic(dtype=torch.qint8)

# Static quantization (requires calibration)
quantized = quantizer.quantize_static(calibration_data)

# FP16 conversion
fp16_model = quantizer.to_fp16()
```

### Knowledge Distillation

#### `KnowledgeDistiller`
Train smaller student models from larger teachers.

```python
from fishstick.compression import KnowledgeDistiller

distiller = KnowledgeDistiller(
    teacher=teacher_model,
    student=student_model,
    temperature=4.0,
    alpha=0.7
)

# Training step
loss = distiller.train_step(x, y, optimizer)
```

### Export Utilities

#### `ONNXExporter`
Export models to ONNX format.

```python
from fishstick.compression import ONNXExporter, export_onnx

exporter = ONNXExporter(model)
exporter.export(
    output_path="model.onnx",
    input_shape=(1, 3, 224, 224),
    opset_version=11
)

# Convenience function
export_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

#### `TorchScriptCompiler`
Compile models to TorchScript.

```python
from fishstick.compression import TorchScriptCompiler

compiler = TorchScriptCompiler(model)
scripted = compiler.compile_script(example_input)
compiler.save(scripted, "model.pt")
```

### Model Optimizer

#### `ModelOptimizer`
High-level optimization pipeline.

```python
from fishstick.compression import ModelOptimizer

optimizer = ModelOptimizer(model)
optimized = optimizer.optimize(
    prune_amount=0.3,
    quantize=True,
    compile_torchscript=False,
    calibration_data=calibration_data
)

# Benchmark
results = optimizer.benchmark(optimized, input_tensor, num_runs=100)
print(f"Throughput: {results['throughput']:.1f} inf/s")
```

### Convenience Functions

```python
from fishstick.compression import prune_model, quantize_model, optimize_model

# Quick pruning
pruned = prune_model(model, amount=0.3)

# Quick quantization
quantized = quantize_model(model, calibration_data=data)

# Full optimization
optimized = optimize_model(model, prune_amount=0.2, quantize=True)
```

## Dependencies

- `torch`: PyTorch for model operations
- `numpy`: Numerical operations
- Optional: `onnx` for ONNX export

## Usage Examples

### Complete Compression Pipeline

```python
import torch
from fishstick.compression import (
    Pruner, Quantizer, KnowledgeDistiller,
    ONNXExporter, ModelOptimizer
)

# Load model
model = MyLargeModel()

# Step 1: Prune
pruner = Pruner(model)
pruned = pruner.prune_magnitude(amount=0.4)
print(f"Sparsity: {pruner.get_sparsity(pruned):.1%}")

# Step 2: Quantize
quantizer = Quantizer(pruned)
quantized = quantizer.quantize_dynamic()

# Step 3: Export
exporter = ONNXExporter(quantized)
exporter.export("compressed_model.onnx", input_shape=(1, 784))
```

### Knowledge Distillation Training

```python
from fishstick.compression import KnowledgeDistiller
import torch.nn as nn

# Teacher (large) and student (small)
teacher = LargeModel()
student = SmallModel()

# Load pretrained teacher
teacher.load_state_dict(torch.load("teacher.pt"))
teacher.eval()

# Distillation
distiller = KnowledgeDistiller(
    teacher=teacher,
    student=student,
    temperature=4.0,
    alpha=0.7
)

optimizer = torch.optim.Adam(student.parameters())

for x, y in train_loader:
    loss = distiller.train_step(x, y, optimizer)
```
