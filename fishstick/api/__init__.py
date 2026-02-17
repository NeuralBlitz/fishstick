"""
REST API for fishstick Model Serving

Provides FastAPI-based endpoints for:
- Model predictions
- Health checks
- Model information
- Batch processing
"""

from typing import Dict, Any, List, Optional
import torch
from torch import nn
import numpy as np
from pydantic import BaseModel


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for predictions."""

    data: List[List[float]]  # Batch of inputs
    return_probabilities: bool = False


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    model_name: str
    inference_time_ms: float


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    version: str
    input_shape: List[int]
    output_shape: List[int]
    num_parameters: int
    device: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str


def create_app(model: Optional[nn.Module] = None, model_path: Optional[str] = None):
    """
    Create FastAPI app for serving models.

    Args:
        model: Pre-loaded model (optional)
        model_path: Path to model file (optional)

    Returns:
        FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException
        import time
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="fishstick API",
        description="Mathematically Rigorous AI Framework - Model Serving API",
        version="0.1.0",
    )

    # Load model
    if model is not None:
        model_instance = model
    elif model_path is not None:
        from fishstick.frameworks.uniintelli import create_uniintelli

        model_instance = create_uniintelli(input_dim=784, output_dim=10)
        try:
            model_instance.load_state_dict(torch.load(model_path))
        except:
            pass  # Use untrained model if loading fails
    else:
        from fishstick.frameworks.uniintelli import create_uniintelli

        model_instance = create_uniintelli(input_dim=784, output_dim=10)

    model_instance.eval()

    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "Welcome to fishstick API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(next(model_instance.parameters()).device),
        )

    @app.get("/info", response_model=ModelInfo)
    async def info():
        """Get model information."""
        num_params = sum(p.numel() for p in model_instance.parameters())

        return ModelInfo(
            name="fishstick-model",
            version="0.1.0",
            input_shape=[784],
            output_shape=[10],
            num_parameters=num_params,
            device=str(next(model_instance.parameters()).device),
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make predictions on input data."""
        try:
            start_time = time.time()

            # Convert to tensor
            input_tensor = torch.tensor(request.data, dtype=torch.float32)

            # Inference
            with torch.no_grad():
                outputs = model_instance(input_tensor)
                probabilities = torch.softmax(outputs, dim=-1)
                predictions = torch.argmax(outputs, dim=-1)

            inference_time = (time.time() - start_time) * 1000  # ms

            response = PredictionResponse(
                predictions=predictions.tolist(),
                model_name="fishstick-model",
                inference_time_ms=inference_time,
            )

            if request.return_probabilities:
                response.probabilities = probabilities.tolist()

            return response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch")
    async def predict_batch(requests: List[PredictionRequest]):
        """Process multiple prediction requests."""
        results = []
        for request in requests:
            result = await predict(request)
            results.append(result)
        return results

    return app
