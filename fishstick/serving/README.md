# Model Serving

Model deployment and serving infrastructure.

## Installation

```bash
pip install fishstick[serving]
```

## Overview

The `serving` module provides utilities for deploying and serving machine learning models with support for various backends.

## Usage

```python
from fishstick.serving import TorchServeWrapper, TritonInferenceServer, FastAPIServer

# TorchServe
exporter = TorchScriptExporter()
exported = exporter.export(model, input_sample)
wrapper = TorchServeWrapper(exported)
wrapper.start()

# Triton
triton = TritonInferenceServer(model_path="models/")
triton.deploy()

# FastAPI
app = FastAPIServer(model=model)
app.add_endpoint("/predict", predict_fn)
app.run()
```

## Exporters

| Exporter | Description |
|----------|-------------|
| `TorchScriptExporter` | Export to TorchScript |
| `ONNXExporter` | Export to ONNX |
| `TensorRTExporter` | Export to TensorRT |

## Backends

| Backend | Description |
|---------|-------------|
| `TorchServeWrapper` | TorchServe integration |
| `TritonInferenceServer` | NVIDIA Triton inference server |
| `FastAPIServer` | FastAPI-based serving |

## Optimization

| Class | Description |
|-------|-------------|
| `ModelQuantizer` | Post-training quantization |
| `ModelPruner` | Model pruning |
| `BatchOptimizer` | Dynamic batching |

## Monitoring

| Class | Description |
|-------|-------------|
| `LatencyMonitor` | Monitor inference latency |
| `ThroughputMonitor` | Monitor throughput |
| `MemoryMonitor` | Monitor memory usage |

## Examples

See `examples/serving/` for complete examples.
