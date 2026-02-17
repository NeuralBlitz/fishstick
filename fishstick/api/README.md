# api - REST API Module

## Overview

The `api` module provides a FastAPI-based REST API for serving fishstick models. It includes endpoints for predictions, health checks, model information, and batch processing.

## Purpose and Scope

This module enables:
- Model serving via REST API
- Health monitoring and status checks
- Batch prediction processing
- Integration with production deployment systems

## Key Classes and Functions

### Request/Response Models

#### `PredictionRequest`
Request model for predictions.

```python
from fishstick.api import PredictionRequest

request = PredictionRequest(
    data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Batch of inputs
    return_probabilities=True
)
```

#### `PredictionResponse`
Response model for predictions.

```python
from fishstick.api import PredictionResponse

# Contains:
# - predictions: List[int]
# - probabilities: Optional[List[List[float]]]
# - model_name: str
# - inference_time_ms: float
```

#### `ModelInfo`
Model information response.

```python
from fishstick.api import ModelInfo

info = ModelInfo(
    name="fishstick-model",
    version="0.1.0",
    input_shape=[784],
    output_shape=[10],
    num_parameters=1000000,
    device="cuda"
)
```

#### `HealthResponse`
Health check response for monitoring.

### Main Functions

#### `create_app`
Create FastAPI application for model serving.

```python
from fishstick.api import create_app
from fishstick.frameworks.uniintelli import create_uniintelli

# Option 1: Pass pre-loaded model
model = create_uniintelli(input_dim=784, output_dim=10)
app = create_app(model=model)

# Option 2: Load from path
app = create_app(model_path="model.pt")
```

## API Endpoints

### `GET /`
Root endpoint returning API information.

### `GET /health`
Health check endpoint returning model status.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### `GET /info`
Returns detailed model information.

```json
{
  "name": "fishstick-model",
  "version": "0.1.0",
  "input_shape": [784],
  "output_shape": [10],
  "num_parameters": 1000000,
  "device": "cuda"
}
```

### `POST /predict`
Make predictions on input data.

**Request:**
```json
{
  "data": [[1.0, 2.0, ...], [3.0, 4.0, ...]],
  "return_probabilities": true
}
```

**Response:**
```json
{
  "predictions": [3, 7],
  "probabilities": [[0.1, 0.2, ...], [0.05, 0.1, ...]],
  "model_name": "fishstick-model",
  "inference_time_ms": 12.5
}
```

### `POST /predict/batch`
Process multiple prediction requests in a single call.

## Dependencies

- `fastapi`: Web framework
- `pydantic`: Data validation
- `torch`: PyTorch for model inference
- `uvicorn`: ASGI server (for running)

## Usage Examples

### Starting the Server

```bash
# Using CLI
fishstick serve --model model.pt --port 8000

# Or programmatically
import uvicorn
from fishstick.api import create_app

app = create_app(model_path="model.pt")
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Making Predictions

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "data": [[1.0] * 784],
        "return_probabilities": True
    }
)
result = response.json()
print(result["predictions"])  # [3]
print(result["inference_time_ms"])  # 5.2

# Health check
health = requests.get("http://localhost:8000/health")
print(health.json())  # {"status": "healthy", ...}
```

### Custom Model Serving

```python
from fishstick.api import create_app
from fishstick.frameworks import create_uniintelli
import uvicorn

# Create and train model
model = create_uniintelli(input_dim=784, output_dim=10)
# ... training code ...

# Create and run API
app = create_app(model=model)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Interactive Documentation

FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Production Deployment

For production, consider:
1. Using Gunicorn with Uvicorn workers
2. Adding authentication middleware
3. Implementing rate limiting
4. Setting up monitoring and logging
5. Using a reverse proxy (nginx)
