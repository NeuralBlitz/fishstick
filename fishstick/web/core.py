"""
fishstick/web/core.py
===================
Comprehensive web module for deploying and serving fishstick models.
Supports FastAPI, Flask, Streamlit, Gradio, React, Vue, and deployment utilities.
"""

from __future__ import annotations

import os
import json
import asyncio
import inspect
import tempfile
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union,
    get_type_hints, Awaitable, Tuple, Protocol, runtime_checkable
)
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

import numpy as np

# Web framework imports with availability checks
try:
    from fastapi import FastAPI, APIRouter, Request, Response, HTTPException, status, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field, create_model
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    from flask import Flask, request, jsonify, Blueprint, Response as FlaskResponse
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

try:
    import gradio as gr
    from gradio.components import Component
    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


T = TypeVar('T')
ModelType = TypeVar('ModelType')


# =============================================================================
# Base Types and Protocols
# =============================================================================

class ModelInput(BaseModel):
    """Base model for API inputs."""
    inputs: Union[List[Any], Dict[str, Any]] = Field(..., description="Model inputs")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Inference parameters")


class ModelOutput(BaseModel):
    """Base model for API outputs."""
    predictions: Union[List[Any], Dict[str, Any]] = Field(..., description="Model predictions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    models_loaded: List[str]


class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    version: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    parameters: Dict[str, Any]


@runtime_checkable
class Predictable(Protocol):
    """Protocol for models that can make predictions."""
    def predict(self, inputs: Any) -> Any: ...
    def predict_batch(self, inputs: List[Any]) -> List[Any]: ...


@runtime_checkable
class AsyncPredictable(Protocol):
    """Protocol for models with async predictions."""
    async def predict_async(self, inputs: Any) -> Any: ...
    async def predict_batch_async(self, inputs: List[Any]) -> List[Any]: ...


# =============================================================================
# FastAPI Implementation
# =============================================================================

if _FASTAPI_AVAILABLE:
    
    @dataclass
    class FastAPIApp:
        """FastAPI application wrapper for fishstick models."""
        app: FastAPI
        models: Dict[str, Any] = field(default_factory=dict)
        routers: Dict[str, APIRouter] = field(default_factory=dict)
        
        def __post_init__(self):
            self.app.state.models = self.models
        
        def add_model(self, name: str, model: Any, endpoint_prefix: str = "/api/v1") -> APIRouter:
            """Add a model to the FastAPI app."""
            self.models[name] = model
            router = add_model_endpoint(self.app, model, name, endpoint_prefix)
            self.routers[name] = router
            return router
        
        def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
            """Run the FastAPI application."""
            uvicorn.run(self.app, host=host, port=port, **kwargs)
        
        async def health_check(self) -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                version="0.1.0",
                models_loaded=list(self.models.keys())
            )
    
    
    class ModelInferenceEndpoint:
        """Endpoint handler for model inference."""
        
        def __init__(self, model: Any, name: str, batch_size: int = 32):
            self.model = model
            self.name = name
            self.batch_size = batch_size
            self.request_count = 0
            self.error_count = 0
        
        async def predict(self, input_data: ModelInput) -> ModelOutput:
            """Run inference on input data."""
            import time
            start_time = time.time()
            
            try:
                self.request_count += 1
                
                # Check if model has async methods
                if isinstance(self.model, AsyncPredictable):
                    if isinstance(input_data.inputs, list) and len(input_data.inputs) > 1:
                        predictions = await self.model.predict_batch_async(input_data.inputs)
                    else:
                        predictions = await self.model.predict_async(input_data.inputs)
                else:
                    # Run in thread pool for sync models
                    loop = asyncio.get_event_loop()
                    if isinstance(input_data.inputs, list) and len(input_data.inputs) > 1:
                        predictions = await loop.run_in_executor(
                            None, self.model.predict_batch, input_data.inputs
                        )
                    else:
                        predictions = await loop.run_in_executor(
                            None, self.model.predict, input_data.inputs
                        )
                
                processing_time = (time.time() - start_time) * 1000
                
                return ModelOutput(
                    predictions=predictions,
                    metadata={
                        "model_name": self.name,
                        "batch_size": len(input_data.inputs) if isinstance(input_data.inputs, list) else 1,
                        "total_requests": self.request_count
                    },
                    processing_time_ms=processing_time
                )
            except Exception as e:
                self.error_count += 1
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Inference error: {str(e)}"
                )
        
        async def predict_stream(self, input_data: ModelInput):
            """Stream predictions for large inputs."""
            inputs = input_data.inputs
            if not isinstance(inputs, list):
                inputs = [inputs]
            
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i:i + self.batch_size]
                
                if isinstance(self.model, AsyncPredictable):
                    predictions = await self.model.predict_batch_async(batch)
                else:
                    loop = asyncio.get_event_loop()
                    predictions = await loop.run_in_executor(None, self.model.predict_batch, batch)
                
                yield json.dumps({
                    "batch_index": i // self.batch_size,
                    "predictions": predictions
                }) + "\n"
        
        def get_info(self) -> ModelInfo:
            """Get model information."""
            # Extract schema from model if available
            input_schema = {}
            output_schema = {}
            
            if hasattr(self.model, 'input_schema'):
                input_schema = self.model.input_schema
            if hasattr(self.model, 'output_schema'):
                output_schema = self.model.output_schema
            
            return ModelInfo(
                name=self.name,
                version=getattr(self.model, 'version', '1.0.0'),
                description=getattr(self.model, 'description', f'Model {self.name}'),
                input_schema=input_schema,
                output_schema=output_schema,
                parameters=getattr(self.model, 'parameters', {})
            )
    
    
    def create_fastapi_app(
        title: str = "Fishstick API",
        description: str = "Fishstick Model Serving API",
        version: str = "1.0.0",
        enable_cors: bool = True,
        enable_gzip: bool = True
    ) -> FastAPIApp:
        """Create a new FastAPI application for fishstick models."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan handler."""
            # Startup
            yield
            # Shutdown
        
        app = FastAPI(
            title=title,
            description=description,
            version=version,
            lifespan=lifespan
        )
        
        if enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        if enable_gzip:
            app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Add health check endpoint
        @app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                version=version,
                models_loaded=[]
            )
        
        # Add docs redirect
        @app.get("/")
        async def root():
            return {"message": "Fishstick API", "docs": "/docs"}
        
        return FastAPIApp(app=app)
    
    
    def add_model_endpoint(
        app: FastAPI,
        model: Any,
        model_name: str,
        prefix: str = "/api/v1",
        tags: Optional[List[str]] = None
    ) -> APIRouter:
        """Add a model endpoint to a FastAPI app."""
        
        if tags is None:
            tags = [model_name]
        
        router = APIRouter(prefix=f"{prefix}/{model_name}", tags=tags)
        endpoint = ModelInferenceEndpoint(model, model_name)
        
        @router.post("/predict", response_model=ModelOutput)
        async def predict(input_data: ModelInput):
            """Run model prediction."""
            return await endpoint.predict(input_data)
        
        @router.post("/predict/stream")
        async def predict_stream(input_data: ModelInput):
            """Stream model predictions."""
            return StreamingResponse(
                endpoint.predict_stream(input_data),
                media_type="application/x-ndjson"
            )
        
        @router.get("/info", response_model=ModelInfo)
        async def model_info():
            """Get model information."""
            return endpoint.get_info()
        
        @router.get("/health")
        async def model_health():
            """Check model health."""
            return {
                "status": "healthy",
                "model": model_name,
                "requests": endpoint.request_count,
                "errors": endpoint.error_count
            }
        
        # Batch prediction endpoint
        @router.post("/predict/batch")
        async def predict_batch(input_data: ModelInput):
            """Run batch prediction."""
            return await endpoint.predict(input_data)
        
        app.include_router(router)
        return router


# =============================================================================
# Flask Implementation
# =============================================================================

if _FLASK_AVAILABLE:
    
    @dataclass
    class FlaskApp:
        """Flask application wrapper for fishstick models."""
        app: Flask
        models: Dict[str, Any] = field(default_factory=dict)
        blueprints: Dict[str, Blueprint] = field(default_factory=dict)
        
        def add_model(self, name: str, model: Any, url_prefix: str = "/api/v1") -> Blueprint:
            """Add a model to the Flask app."""
            self.models[name] = model
            blueprint = ModelAPI(model, name).create_blueprint(url_prefix)
            self.app.register_blueprint(blueprint)
            self.blueprints[name] = blueprint
            return blueprint
        
        def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False, **kwargs) -> None:
            """Run the Flask application."""
            self.app.run(host=host, port=port, debug=debug, **kwargs)
        
        def health_check(self) -> Dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "0.1.0",
                "models_loaded": list(self.models.keys())
            }
    
    
    class PredictionEndpoint:
        """Flask endpoint for model predictions."""
        
        def __init__(self, model: Any, name: str):
            self.model = model
            self.name = name
            self.request_count = 0
            self.error_count = 0
        
        def predict(self) -> FlaskResponse:
            """Handle prediction request."""
            import time
            start_time = time.time()
            
            try:
                data = request.get_json()
                if not data or 'inputs' not in data:
                    return jsonify({"error": "Missing 'inputs' in request"}), 400
                
                inputs = data['inputs']
                self.request_count += 1
                
                # Run prediction
                if isinstance(inputs, list) and len(inputs) > 1:
                    if hasattr(self.model, 'predict_batch'):
                        predictions = self.model.predict_batch(inputs)
                    else:
                        predictions = [self.model.predict(inp) for inp in inputs]
                else:
                    predictions = self.model.predict(inputs)
                
                processing_time = (time.time() - start_time) * 1000
                
                return jsonify({
                    "predictions": predictions,
                    "metadata": {
                        "model_name": self.name,
                        "processing_time_ms": processing_time,
                        "total_requests": self.request_count
                    }
                })
                
            except Exception as e:
                self.error_count += 1
                return jsonify({"error": str(e)}), 500
        
        def predict_batch(self) -> FlaskResponse:
            """Handle batch prediction request."""
            import time
            start_time = time.time()
            
            try:
                data = request.get_json()
                if not data or 'inputs' not in data:
                    return jsonify({"error": "Missing 'inputs' in request"}), 400
                
                inputs = data['inputs']
                batch_size = data.get('batch_size', 32)
                
                # Process in batches
                all_predictions = []
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i + batch_size]
                    if hasattr(self.model, 'predict_batch'):
                        batch_preds = self.model.predict_batch(batch)
                    else:
                        batch_preds = [self.model.predict(inp) for inp in batch]
                    all_predictions.extend(batch_preds)
                
                processing_time = (time.time() - start_time) * 1000
                
                return jsonify({
                    "predictions": all_predictions,
                    "metadata": {
                        "model_name": self.name,
                        "batch_size": batch_size,
                        "processing_time_ms": processing_time
                    }
                })
                
            except Exception as e:
                self.error_count += 1
                return jsonify({"error": str(e)}), 500
        
        def health(self) -> FlaskResponse:
            """Health check."""
            return jsonify({
                "status": "healthy",
                "model": self.name,
                "requests": self.request_count,
                "errors": self.error_count
            })
        
        def info(self) -> FlaskResponse:
            """Get model info."""
            return jsonify({
                "name": self.name,
                "version": getattr(self.model, 'version', '1.0.0'),
                "description": getattr(self.model, 'description', f'Model {self.name}'),
                "parameters": getattr(self.model, 'parameters', {})
            })
    
    
    class ModelAPI:
        """Complete model API for Flask."""
        
        def __init__(self, model: Any, model_name: str):
            self.model = model
            self.name = model_name
            self.endpoint = PredictionEndpoint(model, model_name)
        
        def create_blueprint(self, url_prefix: str = "/api/v1") -> Blueprint:
            """Create a Flask blueprint for the model."""
            bp = Blueprint(self.name, __name__, url_prefix=f"{url_prefix}/{self.name}")
            
            @bp.route("/predict", methods=["POST"])
            def predict():
                return self.endpoint.predict()
            
            @bp.route("/predict/batch", methods=["POST"])
            def predict_batch():
                return self.endpoint.predict_batch()
            
            @bp.route("/health", methods=["GET"])
            def health():
                return self.endpoint.health()
            
            @bp.route("/info", methods=["GET"])
            def info():
                return self.endpoint.info()
            
            return bp
        
        def register(self, app: Flask, url_prefix: str = "/api/v1") -> None:
            """Register the API with a Flask app."""
            bp = self.create_blueprint(url_prefix)
            app.register_blueprint(bp)
    
    
    def create_flask_app(
        name: str = "fishstick",
        enable_cors: bool = True
    ) -> FlaskApp:
        """Create a new Flask application for fishstick models."""
        app = Flask(name)
        
        if enable_cors:
            CORS(app)
        
        @app.route("/")
        def index():
            return jsonify({"message": "Fishstick Flask API", "status": "running"})
        
        @app.route("/health")
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "0.1.0"
            })
        
        return FlaskApp(app=app)


# =============================================================================
# Streamlit Implementation
# =============================================================================

if _STREAMLIT_AVAILABLE:
    
    @dataclass
    class StreamlitApp:
        """Streamlit application wrapper for fishstick models."""
        title: str = "Fishstick Model Dashboard"
        models: Dict[str, Any] = field(default_factory=dict)
        dashboards: Dict[str, 'ModelDashboard'] = field(default_factory=dict)
        
        def __post_init__(self):
            st.set_page_config(
                page_title=self.title,
                page_icon="🐟",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        
        def add_model(self, name: str, model: Any, dashboard: Optional['ModelDashboard'] = None) -> None:
            """Add a model to the Streamlit app."""
            self.models[name] = model
            if dashboard is None:
                dashboard = ModelDashboard(name, model)
            self.dashboards[name] = dashboard
        
        def run(self) -> None:
            """Run the Streamlit application."""
            st.title(f"🐟 {self.title}")
            
            # Sidebar navigation
            st.sidebar.title("Navigation")
            page = st.sidebar.radio(
                "Select Page",
                ["Home"] + list(self.models.keys()) + ["Settings"]
            )
            
            if page == "Home":
                self._render_home()
            elif page == "Settings":
                self._render_settings()
            elif page in self.dashboards:
                self.dashboards[page].render()
        
        def _render_home(self) -> None:
            """Render home page."""
            st.header("Welcome to Fishstick Dashboard")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Models Loaded", len(self.models))
            with col2:
                st.metric("Total Predictions", sum(
                    getattr(d.model, 'prediction_count', 0) 
                    for d in self.dashboards.values()
                ))
            with col3:
                st.metric("Status", "Active")
            
            st.subheader("Loaded Models")
            for name, model in self.models.items():
                with st.expander(f"📊 {name}"):
                    st.write(f"**Version:** {getattr(model, 'version', 'N/A')}")
                    st.write(f"**Description:** {getattr(model, 'description', 'N/A')}")
        
        def _render_settings(self) -> None:
            """Render settings page."""
            st.header("Settings")
            
            st.subheader("Theme")
            theme = st.selectbox("Color Theme", ["Light", "Dark"])
            
            st.subheader("Performance")
            st.checkbox("Enable caching", value=True)
            st.slider("Max batch size", 1, 128, 32)
            
            st.subheader("API Configuration")
            st.text_input("API Endpoint", value="http://localhost:8000")
    
    
    class ModelDashboard:
        """Dashboard for a single model."""
        
        def __init__(self, name: str, model: Any):
            self.name = name
            self.model = model
            self.prediction_history: List[Dict[str, Any]] = []
            self.visualizer = Visualization()
        
        def render(self) -> None:
            """Render the model dashboard."""
            st.header(f"📊 {self.name}")
            
            # Model info
            with st.expander("Model Information", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Version:** {getattr(self.model, 'version', 'N/A')}")
                    st.write(f"**Description:** {getattr(self.model, 'description', 'N/A')}")
                with col2:
                    st.write(f"**Type:** {type(self.model).__name__}")
                    st.write(f"**Parameters:** {getattr(self.model, 'num_parameters', 'N/A')}")
            
            # Tabs for different functionalities
            tabs = st.tabs(["Inference", "Batch Processing", "Visualization", "History"])
            
            with tabs[0]:
                self._render_inference_tab()
            
            with tabs[1]:
                self._render_batch_tab()
            
            with tabs[2]:
                self._render_visualization_tab()
            
            with tabs[3]:
                self._render_history_tab()
        
        def _render_inference_tab(self) -> None:
            """Render single inference tab."""
            st.subheader("Single Prediction")
            
            # Input method selection
            input_method = st.radio("Input Method", ["Upload File", "Manual Input", "Sample Data"])
            
            if input_method == "Upload File":
                uploaded_file = st.file_uploader("Upload input file", type=["json", "csv", "txt"])
                if uploaded_file is not None:
                    try:
                        import pandas as pd
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                            st.write("Preview:")
                            st.dataframe(df.head())
                            input_data = df.to_dict('records')
                        else:
                            content = uploaded_file.read().decode('utf-8')
                            input_data = json.loads(content)
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                        input_data = None
            elif input_method == "Manual Input":
                input_json = st.text_area("Enter JSON input", value='{"input": "value"}')
                try:
                    input_data = json.loads(input_json)
                except json.JSONDecodeError:
                    st.error("Invalid JSON")
                    input_data = None
            else:  # Sample Data
                input_data = {"sample": "data"}
                st.write("Using sample data:", input_data)
            
            if st.button("Run Prediction") and input_data is not None:
                with st.spinner("Running inference..."):
                    import time
                    start_time = time.time()
                    
                    try:
                        prediction = self.model.predict(input_data)
                        processing_time = (time.time() - start_time) * 1000
                        
                        # Store in history
                        self.prediction_history.append({
                            "timestamp": datetime.now(),
                            "input": input_data,
                            "prediction": prediction,
                            "time_ms": processing_time
                        })
                        
                        st.success(f"Prediction completed in {processing_time:.2f}ms")
                        st.json(prediction)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        
        def _render_batch_tab(self) -> None:
            """Render batch processing tab."""
            st.subheader("Batch Prediction")
            
            uploaded_file = st.file_uploader("Upload batch data (CSV/JSON)", type=["csv", "json"])
            batch_size = st.slider("Batch Size", 1, 100, 10)
            
            if uploaded_file and st.button("Run Batch Prediction"):
                with st.progress(0) as progress_bar:
                    try:
                        import pandas as pd
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_json(uploaded_file)
                        
                        inputs = df.to_dict('records')
                        total = len(inputs)
                        results = []
                        
                        for i in range(0, total, batch_size):
                            batch = inputs[i:i + batch_size]
                            if hasattr(self.model, 'predict_batch'):
                                batch_results = self.model.predict_batch(batch)
                            else:
                                batch_results = [self.model.predict(inp) for inp in batch]
                            results.extend(batch_results)
                            
                            progress = min((i + batch_size) / total, 1.0)
                            progress_bar.progress(progress)
                        
                        st.success(f"Processed {total} items")
                        
                        # Show results
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Batch processing failed: {e}")
        
        def _render_visualization_tab(self) -> None:
            """Render visualization tab."""
            st.subheader("Model Visualizations")
            
            if len(self.prediction_history) == 0:
                st.info("Run some predictions to see visualizations")
                return
            
            # Performance over time
            self.visualizer.plot_prediction_times(self.prediction_history)
            
            # Prediction distribution
            if len(self.prediction_history) > 1:
                self.visualizer.plot_prediction_distribution(self.prediction_history)
        
        def _render_history_tab(self) -> None:
            """Render history tab."""
            st.subheader("Prediction History")
            
            if len(self.prediction_history) == 0:
                st.info("No predictions yet")
                return
            
            for i, record in enumerate(reversed(self.prediction_history[-20:])):
                with st.expander(f"Prediction {len(self.prediction_history) - i} - {record['timestamp'].strftime('%H:%M:%S')}"):
                    st.write(f"**Time:** {record['time_ms']:.2f}ms")
                    st.write("**Input:**")
                    st.json(record['input'])
                    st.write("**Output:**")
                    st.json(record['prediction'])
    
    
    class Visualization:
        """Visualization utilities for Streamlit dashboards."""
        
        def __init__(self):
            self.colors = px.colors.qualitative.Set2
        
        def plot_prediction_times(self, history: List[Dict[str, Any]]) -> None:
            """Plot prediction times over time."""
            if len(history) < 2:
                return
            
            times = [h['time_ms'] for h in history]
            indices = list(range(len(times)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=indices,
                y=times,
                mode='lines+markers',
                name='Processing Time',
                line=dict(color=self.colors[0], width=2)
            ))
            
            fig.update_layout(
                title="Prediction Latency Over Time",
                xaxis_title="Prediction #",
                yaxis_title="Time (ms)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        def plot_prediction_distribution(self, history: List[Dict[str, Any]]) -> None:
            """Plot distribution of predictions."""
            times = [h['time_ms'] for h in history]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=times,
                nbinsx=20,
                name='Distribution',
                marker_color=self.colors[1]
            ))
            
            fig.update_layout(
                title="Prediction Time Distribution",
                xaxis_title="Time (ms)",
                yaxis_title="Count",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None) -> None:
            """Plot confusion matrix."""
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="True", color="Count"),
                x=labels if labels else [f"Class {i}" for i in range(cm.shape[1])],
                y=labels if labels else [f"Class {i}" for i in range(cm.shape[0])],
                color_continuous_scale="Blues"
            )
            
            fig.update_layout(title="Confusion Matrix", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        def plot_feature_importance(self, features: List[str], importance: List[float]) -> None:
            """Plot feature importance."""
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker_color=self.colors[2]
            ))
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc: float) -> None:
            """Plot ROC curve."""
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc:.3f})',
                line=dict(color=self.colors[0], width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', dash='dash')
            ))
            
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    
    def create_streamlit_app(
        title: str = "Fishstick Dashboard",
        models: Optional[Dict[str, Any]] = None
    ) -> StreamlitApp:
        """Create a new Streamlit application."""
        app = StreamlitApp(title=title)
        
        if models:
            for name, model in models.items():
                app.add_model(name, model)
        
        return app


# =============================================================================
# Gradio Implementation
# =============================================================================

if _GRADIO_AVAILABLE:
    
    @dataclass
    class GradioInterface:
        """Gradio interface wrapper for fishstick models."""
        interfaces: Dict[str, gr.Blocks] = field(default_factory=dict)
        demos: Dict[str, 'ModelDemo'] = field(default_factory=dict)
        
        def add_model(
            self,
            name: str,
            model: Any,
            inputs: Union[str, Component, List[Component]] = "text",
            outputs: Union[str, Component, List[Component]] = "text",
            title: Optional[str] = None,
            description: Optional[str] = None
        ) -> gr.Blocks:
            """Add a model as a Gradio interface."""
            
            demo = ModelDemo(model, name, inputs, outputs, title, description)
            interface = demo.create_interface()
            
            self.demos[name] = demo
            self.interfaces[name] = interface
            
            return interface
        
        def create_multi_model_interface(self, title: str = "Fishstick Models") -> gr.Blocks:
            """Create a unified interface for multiple models."""
            
            with gr.Blocks(title=title) as demo:
                gr.Markdown(f"# {title}")
                gr.Markdown("Select a model from the tabs below to interact with it.")
                
                if len(self.interfaces) > 0:
                    with gr.Tabs():
                        for name, interface in self.interfaces.items():
                            with gr.TabItem(name):
                                # Render the interface components
                                interface.render()
                else:
                    gr.Markdown("No models loaded yet.")
            
            return demo
        
        def launch(
            self,
            model_name: Optional[str] = None,
            share: bool = False,
            server_name: str = "0.0.0.0",
            server_port: int = 7860,
            **kwargs
        ) -> None:
            """Launch a Gradio interface."""
            
            if model_name and model_name in self.interfaces:
                self.interfaces[model_name].launch(
                    share=share,
                    server_name=server_name,
                    server_port=server_port,
                    **kwargs
                )
            elif len(self.interfaces) == 1:
                list(self.interfaces.values())[0].launch(
                    share=share,
                    server_name=server_name,
                    server_port=server_port,
                    **kwargs
                )
            else:
                # Launch multi-model interface
                multi = self.create_multi_model_interface()
                multi.launch(
                    share=share,
                    server_name=server_name,
                    server_port=server_port,
                    **kwargs
                )
    
    
    class ModelDemo:
        """Gradio demo for a single model."""
        
        def __init__(
            self,
            model: Any,
            name: str,
            inputs: Union[str, Component, List[Component]] = "text",
            outputs: Union[str, Component, List[Component]] = "text",
            title: Optional[str] = None,
            description: Optional[str] = None
        ):
            self.model = model
            self.name = name
            self.inputs = inputs
            self.outputs = outputs
            self.title = title or f"{name} Demo"
            self.description = description or f"Interactive demo for {name}"
            self.examples: List[List[Any]] = []
        
        def create_interface(self) -> gr.Blocks:
            """Create the Gradio interface."""
            
            with gr.Blocks(title=self.title) as demo:
                gr.Markdown(f"# {self.title}")
                gr.Markdown(self.description)
                
                with gr.Row():
                    with gr.Column():
                        input_components = self._create_inputs()
                        submit_btn = gr.Button("Predict", variant="primary")
                        clear_btn = gr.Button("Clear")
                    
                    with gr.Column():
                        output_components = self._create_outputs()
                
                # Add examples if available
                if self.examples:
                    gr.Examples(examples=self.examples, inputs=input_components)
                
                # Add model info
                with gr.Accordion("Model Information", open=False):
                    gr.Markdown(f"**Name:** {self.name}")
                    gr.Markdown(f"**Type:** {type(self.model).__name__}")
                    gr.Markdown(f"**Version:** {getattr(self.model, 'version', 'N/A')}")
                
                # Bind events
                submit_btn.click(
                    fn=self._predict,
                    inputs=input_components,
                    outputs=output_components
                )
                clear_btn.click(
                    fn=self._clear,
                    inputs=None,
                    outputs=input_components + output_components
                )
            
            return demo
        
        def _create_inputs(self) -> Union[Component, List[Component]]:
            """Create input components."""
            if isinstance(self.inputs, str):
                if self.inputs == "text":
                    return gr.Textbox(label="Input", lines=3)
                elif self.inputs == "image":
                    return gr.Image(label="Input Image")
                elif self.inputs == "audio":
                    return gr.Audio(label="Input Audio")
                elif self.inputs == "file":
                    return gr.File(label="Input File")
                else:
                    return gr.Textbox(label="Input")
            return self.inputs
        
        def _create_outputs(self) -> Union[Component, List[Component]]:
            """Create output components."""
            if isinstance(self.outputs, str):
                if self.outputs == "text":
                    return gr.Textbox(label="Output", lines=5)
                elif self.outputs == "image":
                    return gr.Image(label="Output Image")
                elif self.outputs == "label":
                    return gr.Label(label="Prediction")
                elif self.outputs == "json":
                    return gr.JSON(label="Output")
                elif self.outputs == "dataframe":
                    return gr.Dataframe(label="Results")
                else:
                    return gr.Textbox(label="Output")
            return self.outputs
        
        def _predict(self, *args):
            """Run prediction."""
            try:
                # Handle single vs multiple inputs
                if len(args) == 1:
                    result = self.model.predict(args[0])
                else:
                    result = self.model.predict(list(args))
                
                # Format output based on type
                if isinstance(self.outputs, str) and self.outputs == "label":
                    if isinstance(result, dict):
                        return result
                    return {"prediction": result}
                
                return result
            except Exception as e:
                return f"Error: {str(e)}"
        
        def _clear(self):
            """Clear inputs and outputs."""
            return [None] * (len(self._create_inputs()) + len(self._create_outputs()))
        
        def add_examples(self, examples: List[List[Any]]) -> None:
            """Add example inputs."""
            self.examples = examples
        
        def launch(self, **kwargs) -> None:
            """Launch the demo."""
            demo = self.create_interface()
            demo.launch(**kwargs)
    
    
    def create_gradio_interface(
        models: Optional[Dict[str, Any]] = None
    ) -> GradioInterface:
        """Create a new Gradio interface manager."""
        interface = GradioInterface()
        
        if models:
            for name, model in models.items():
                interface.add_model(name, model)
        
        return interface


# =============================================================================
# React Implementation
# =============================================================================

@dataclass
class ReactComponent:
    """React component generator for fishstick models."""
    name: str
    props: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    children: List['ReactComponent'] = field(default_factory=list)
    
    def to_jsx(self, indent: int = 0) -> str:
        """Convert to JSX code."""
        spaces = "  " * indent
        
        # Build props string
        props_str = ""
        for key, value in self.props.items():
            if isinstance(value, str):
                props_str += f' {key}="{value}"'
            elif isinstance(value, bool):
                if value:
                    props_str += f' {key}'
            else:
                props_str += f' {key}={{{value}}}'
        
        if len(self.children) == 0:
            return f"{spaces}<{self.name}{props_str} />"
        
        children_str = "\n".join(child.to_jsx(indent + 1) for child in self.children)
        return f"{spaces}<{self.name}{props_str}>\n{children_str}\n{spaces}</{self.name}>"


class ModelFrontend:
    """React frontend generator for model UIs."""
    
    def __init__(self, model_name: str, model_info: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_info = model_info or {}
        self.components: List[ReactComponent] = []
        self.api_client: Optional[APIClient] = None
    
    def add_component(self, component: ReactComponent) -> None:
        """Add a component to the frontend."""
        self.components.append(component)
    
    def set_api_client(self, client: APIClient) -> None:
        """Set the API client."""
        self.api_client = client
    
    def generate_app(self) -> str:
        """Generate complete React app code."""
        
        code = f'''import React, {{ useState, useEffect }} from 'react';
import axios from 'axios';
import {{ Container, Row, Col, Card, Button, Form, Alert, Spinner }} from 'react-bootstrap';

// API Client
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({{
  baseURL: API_BASE_URL,
  headers: {{ 'Content-Type': 'application/json' }}
}});

// Model Component
const {self.model_name}Predictor = () => {{
  const [input, setInput] = useState('');
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {{
    fetchModelInfo();
  }}, []);

  const fetchModelInfo = async () => {{
    try {{
      const response = await api.get('/api/v1/{self.model_name}/info');
      setModelInfo(response.data);
    }} catch (err) {{
      console.error('Failed to fetch model info:', err);
    }}
  }};

  const handlePredict = async () => {{
    setLoading(true);
    setError(null);
    
    try {{
      const response = await api.post('/api/v1/{self.model_name}/predict', {{
        inputs: JSON.parse(input)
      }});
      setOutput(response.data);
    }} catch (err) {{
      setError(err.response?.data?.detail || err.message);
    }} finally {{
      setLoading(false);
    }}
  }};

  return (
    <Container className="mt-4">
      <Row>
        <Col>
          <Card>
            <Card.Header>
              <h2>{self.model_name} Model</h2>
            </Card.Header>
            <Card.Body>
              {{modelInfo && (
                <Alert variant="info">
                  <strong>Version:</strong> {{modelInfo.version}}<br/>
                  <strong>Description:</strong> {{modelInfo.description}}
                </Alert>
              )}}
              
              <Form.Group className="mb-3">
                <Form.Label>Input (JSON)</Form.Label>
                <Form.Control
                  as="textarea"
                  rows={6}
                  value={{input}}
                  onChange={{(e) => setInput(e.target.value)}}
                  placeholder='{{"key": "value"}}'
                />
              </Form.Group>

              <Button
                variant="primary"
                onClick={{handlePredict}}
                disabled={{loading}}
              >
                {{loading ? <Spinner animation="border" size="sm" /> : 'Predict'}}
              </Button>

              {{error && (
                <Alert variant="danger" className="mt-3">
                  {{error}}
                </Alert>
              )}}

              {{output && (
                <Card className="mt-3">
                  <Card.Header>Output</Card.Header>
                  <Card.Body>
                    <pre>{{JSON.stringify(output, null, 2)}}</pre>
                  </Card.Body>
                </Card>
              )}}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}};

// Dashboard Component
const ModelDashboard = () => {{
  const [models, setModels] = useState([]);
  const [health, setHealth] = useState(null);

  useEffect(() => {{
    fetchHealth();
  }}, []);

  const fetchHealth = async () => {{
    try {{
      const response = await api.get('/health');
      setHealth(response.data);
      setModels(response.data.models_loaded || []);
    }} catch (err) {{
      console.error('Health check failed:', err);
    }}
  }};

  return (
    <Container className="mt-4">
      <h1>Fishstick Model Dashboard</h1>
      
      {{health && (
        <Alert variant="success">
          Status: {{health.status}} | Version: {{health.version}}
        </Alert>
      )}}

      <Row className="mt-4">
        {{models.map((model) => (
          <Col key={{model}} md={{4}} className="mb-3">
            <Card>
              <Card.Body>
                <Card.Title>{{model}}</Card.Title>
                <Button variant="primary" href={`/{{model}}`}>
                  Open
                </Button>
              </Card.Body>
            </Card>
          </Col>
        ))}}
      </Row>
    </Container>
  );
}};

// Main App
function App() {{
  return (
    <div className="App">
      <ModelDashboard />
    </div>
  );
}}

export default App;
'''
        return code
    
    def generate_package_json(self) -> str:
        """Generate package.json for the React app."""
        
        package = {
            "name": f"{self.model_name.lower()}-frontend",
            "version": "1.0.0",
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.20.0",
                "react-bootstrap": "^2.9.0",
                "bootstrap": "^5.3.0",
                "axios": "^1.6.0",
                "react-scripts": "5.0.1"
            },
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "test": "react-scripts test",
                "eject": "react-scripts eject"
            },
            "eslintConfig": {
                "extends": ["react-app"]
            },
            "browserslist": {
                "production": [">0.2%", "not dead", "not op_mini all"],
                "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
            }
        }
        
        return json.dumps(package, indent=2)
    
    def write_to_directory(self, output_dir: str) -> None:
        """Write all frontend files to a directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write package.json
        with open(output_path / "package.json", "w") as f:
            f.write(self.generate_package_json())
        
        # Write src directory
        src_dir = output_path / "src"
        src_dir.mkdir(exist_ok=True)
        
        with open(src_dir / "App.js", "w") as f:
            f.write(self.generate_app())
        
        with open(src_dir / "index.js", "w") as f:
            f.write('''import React from 'react';
import ReactDOM from 'react-dom/client';
import 'bootstrap/dist/css/bootstrap.min.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
''')
        
        with open(src_dir / "index.css", "w") as f:
            f.write('''body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
''')
        
        # Write public directory
        public_dir = output_path / "public"
        public_dir.mkdir(exist_ok=True)
        
        with open(public_dir / "index.html", "w") as f:
            f.write(f'''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Fishstick Model Frontend - {self.model_name}" />
    <title>{self.model_name} - Fishstick</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
''')


class APIClient:
    """React API client generator."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.endpoints: Dict[str, Dict[str, str]] = {}
    
    def add_endpoint(self, name: str, path: str, method: str = "POST") -> None:
        """Add an API endpoint."""
        self.endpoints[name] = {"path": path, "method": method}
    
    def generate_client(self) -> str:
        """Generate JavaScript API client code."""
        
        endpoints_str = "\n".join([
            f"  {name}: {{ path: '{info['path']}', method: '{info['method']}' }},"
            for name, info in self.endpoints.items()
        ])
        
        code = f'''import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '{self.base_url}';

const api = axios.create({{
  baseURL: API_BASE_URL,
  headers: {{
    'Content-Type': 'application/json'
  }}
}});

// Request interceptor for auth
api.interceptors.request.use(
  (config) => {{
    const token = localStorage.getItem('auth_token');
    if (token) {{
      config.headers.Authorization = `Bearer ${{token}}`;
    }}
    return config;
  }},
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {{
    if (error.response?.status === 401) {{
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }}
    return Promise.reject(error);
  }}
);

// Endpoints
const ENDPOINTS = {{
{endpoints_str}
}};

// API Client class
class FishstickAPI {{
  constructor(baseURL = API_BASE_URL) {{
    this.client = axios.create({{
      baseURL,
      headers: {{ 'Content-Type': 'application/json' }}
    }});
  }}

  async predict(modelName, inputs, parameters = null) {{
    const response = await this.client.post(`/api/v1/${{modelName}}/predict`, {{
      inputs,
      parameters
    }});
    return response.data;
  }}

  async predictBatch(modelName, inputs, batchSize = 32) {{
    const response = await this.client.post(`/api/v1/${{modelName}}/predict/batch`, {{
      inputs,
      batch_size: batchSize
    }});
    return response.data;
  }}

  async getModelInfo(modelName) {{
    const response = await this.client.get(`/api/v1/${{modelName}}/info`);
    return response.data;
  }}

  async healthCheck() {{
    const response = await this.client.get('/health');
    return response.data;
  }}

  async getModels() {{
    const health = await this.healthCheck();
    return health.models_loaded || [];
  }}

  async streamPredict(modelName, inputs) {{
    const response = await fetch(`${{API_BASE_URL}}/api/v1/${{modelName}}/predict/stream`, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ inputs }})
    }});

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    return {{
      async *[Symbol.asyncIterator]() {{
        while (true) {{
          const {{ done, value }} = await reader.read();
          if (done) break;
          yield JSON.parse(decoder.decode(value));
        }}
      }}
    }};
  }}
}}

export const fishstickAPI = new FishstickAPI();
export default api;
export {{ ENDPOINTS }};
'''
        return code


def create_react_component(
    name: str,
    props: Optional[Dict[str, Any]] = None,
    children: Optional[List[ReactComponent]] = None
) -> ReactComponent:
    """Create a React component."""
    return ReactComponent(
        name=name,
        props=props or {},
        children=children or []
    )


# =============================================================================
# Vue Implementation
# =============================================================================

@dataclass
class VueComponent:
    """Vue component generator for fishstick models."""
    name: str
    template: str = ""
    script: str = ""
    style: str = ""
    props: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    methods: Dict[str, str] = field(default_factory=dict)
    computed: Dict[str, str] = field(default_factory=dict)
    children: List['VueComponent'] = field(default_factory=list)
    
    def to_sfc(self) -> str:
        """Convert to Vue Single File Component."""
        
        # Build props
        props_list = []
        for key, value in self.props.items():
            if isinstance(value, dict):
                props_list.append(f"    {key}: {{")
                for k, v in value.items():
                    props_list.append(f"      {k}: {json.dumps(v)},")
                props_list.append("    }")
            else:
                props_list.append(f"    {key}: {{")
                props_list.append(f"      type: {value},")
                props_list.append("      required: true")
                props_list.append("    }")
        
        props_str = "\n".join(props_list)
        
        # Build data
        data_str = ",\n    ".join([f"{k}: {json.dumps(v)}" for k, v in self.data.items()])
        
        # Build methods
        methods_str = "\n\n    ".join([
            f"{name}() {{\n      {body}\n    }}"
            for name, body in self.methods.items()
        ])
        
        # Build computed
        computed_str = "\n\n    ".join([
            f"{name}() {{\n      return {body};\n    }}"
            for name, body in self.computed.items()
        ])
        
        code = f'''<template>
{self.template}
</template>

<script>
export default {{
  name: '{self.name}',
  
  props: {{
{props_str}
  }},
  
  data() {{
    return {{
      {data_str}
    }};
  }},
  
  computed: {{
    {computed_str}
  }},
  
  methods: {{
    {methods_str}
  }},
  
  mounted() {{
    console.log('{self.name} component mounted');
  }}
}};
</script>

<style scoped>
{self.style}
</style>
'''
        return code


class ModelDashboardVue:
    """Vue dashboard generator for model UIs."""
    
    def __init__(self, model_name: str, model_info: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_info = model_info or {}
        self.components: List[VueComponent] = []
    
    def create_predictor_component(self) -> VueComponent:
        """Create a prediction component."""
        
        template = '''
  <div class="model-predictor">
    <h2>{{ title }}</h2>
    
    <div v-if="modelInfo" class="info-alert">
      <strong>Version:</strong> {{ modelInfo.version }}<br>
      <strong>Description:</strong> {{ modelInfo.description }}
    </div>
    
    <div class="input-section">
      <label>Input (JSON):</label>
      <textarea
        v-model="input"
        rows="6"
        placeholder='{"key": "value"}'
      ></textarea>
    </div>
    
    <button 
      @click="predict" 
      :disabled="loading"
      class="predict-btn"
    >
      <span v-if="loading">Loading...</span>
      <span v-else>Predict</span>
    </button>
    
    <div v-if="error" class="error-alert">
      {{ error }}
    </div>
    
    <div v-if="output" class="output-section">
      <h3>Output</h3>
      <pre>{{ JSON.stringify(output, null, 2) }}</pre>
    </div>
  </div>
'''
        
        script_body = {
            "title": self.model_name,
            "input": "",
            "output": None,
            "loading": False,
            "error": None,
            "modelInfo": None,
            "apiBaseUrl": "http://localhost:8000"
        }
        
        methods = {
            "async predict": '''
      this.loading = true;
      this.error = null;
      
      try {
        const response = await fetch(`${this.apiBaseUrl}/api/v1/''' + self.model_name + '''/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ inputs: JSON.parse(this.input) })
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        this.output = await response.json();
      } catch (err) {
        this.error = err.message;
      } finally {
        this.loading = false;
      }
    ''',
            "async fetchModelInfo": '''
      try {
        const response = await fetch(`${this.apiBaseUrl}/api/v1/''' + self.model_name + '''/info`);
        this.modelInfo = await response.json();
      } catch (err) {
        console.error('Failed to fetch model info:', err);
      }
    '''
        }
        
        style = '''
.model-predictor {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.info-alert {
  background: #e3f2fd;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.input-section {
  margin-bottom: 20px;
}

.input-section label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.input-section textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-family: monospace;
}

.predict-btn {
  background: #007bff;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.predict-btn:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.error-alert {
  background: #f8d7da;
  color: #721c24;
  padding: 15px;
  border-radius: 4px;
  margin-top: 20px;
}

.output-section {
  margin-top: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 4px;
}

.output-section pre {
  background: #fff;
  padding: 10px;
  border-radius: 4px;
  overflow-x: auto;
}
'''
        
        return VueComponent(
            name=f"{self.model_name}Predictor",
            template=template,
            data=script_body,
            methods=methods,
            style=style
        )
    
    def create_dashboard_component(self) -> VueComponent:
        """Create a dashboard component."""
        
        template = '''
  <div class="dashboard">
    <h1>Fishstick Model Dashboard</h1>
    
    <div v-if="health" class="status-bar">
      Status: <span :class="health.status">{{ health.status }}</span> | 
      Version: {{ health.version }}
    </div>
    
    <div class="models-grid">
      <div 
        v-for="model in models" 
        :key="model"
        class="model-card"
        @click="selectModel(model)"
      >
        <h3>{{ model }}</h3>
        <button>Open</button>
      </div>
    </div>
    
    <div v-if="selectedModel" class="selected-model">
      <component :is="selectedModel + 'Predictor'" />
    </div>
  </div>
'''
        
        data = {
            "models": [],
            "health": None,
            "selectedModel": None,
            "apiBaseUrl": "http://localhost:8000"
        }
        
        methods = {
            "async fetchHealth": '''
      try {
        const response = await fetch(`${this.apiBaseUrl}/health`);
        const data = await response.json();
        this.health = data;
        this.models = data.models_loaded || [];
      } catch (err) {
        console.error('Health check failed:', err);
      }
    ''',
            "selectModel": '''
      this.selectedModel = model;
    '''
        }
        
        computed = {
            "statusClass": "this.health ? this.health.status : 'unknown'"
        }
        
        style = '''
.dashboard {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.status-bar {
  background: #d4edda;
  padding: 10px 15px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.status-bar .healthy {
  color: #155724;
  font-weight: bold;
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.model-card {
  border: 1px solid #ddd;
  padding: 20px;
  border-radius: 8px;
  cursor: pointer;
  transition: box-shadow 0.2s;
}

.model-card:hover {
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.model-card button {
  background: #28a745;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
}
'''
        
        return VueComponent(
            name="ModelDashboard",
            template=template,
            data=data,
            methods=methods,
            computed=computed,
            style=style
        )
    
    def generate_app_vue(self) -> str:
        """Generate main App.vue."""
        
        predictor = self.create_predictor_component()
        dashboard = self.create_dashboard_component()
        
        code = f'''<!-- App.vue -->
<template>
  <div id="app">
    <nav class="navbar">
      <div class="nav-brand">Fishstick</div>
      <div class="nav-links">
        <router-link to="/">Dashboard</router-link>
        <router-link to="/predict">Predict</router-link>
      </div>
    </nav>
    
    <router-view />
  </div>
</template>

<script>
import ModelDashboard from './components/ModelDashboard.vue';
import {self.model_name}Predictor from './components/{self.model_name}Predictor.vue';

export default {{
  name: 'App',
  components: {{
    ModelDashboard,
    {self.model_name}Predictor
  }}
}};
</script>

<style>
* {{
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f5f5f5;
}}

.navbar {{
  background: #2c3e50;
  color: white;
  padding: 15px 30px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}

.nav-brand {{
  font-size: 24px;
  font-weight: bold;
}}

.nav-links a {{
  color: white;
  text-decoration: none;
  margin-left: 20px;
  opacity: 0.8;
}}

.nav-links a:hover,
.nav-links a.router-link-active {{
  opacity: 1;
}}
</style>
'''
        return code
    
    def write_to_directory(self, output_dir: str) -> None:
        """Write all Vue files to a directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create components directory
        components_dir = output_path / "src" / "components"
        components_dir.mkdir(parents=True, exist_ok=True)
        
        # Write predictor component
        predictor = self.create_predictor_component()
        with open(components_dir / f"{self.model_name}Predictor.vue", "w") as f:
            f.write(predictor.to_sfc())
        
        # Write dashboard component
        dashboard = self.create_dashboard_component()
        with open(components_dir / "ModelDashboard.vue", "w") as f:
            f.write(dashboard.to_sfc())
        
        # Write App.vue
        src_dir = output_path / "src"
        with open(src_dir / "App.vue", "w") as f:
            f.write(self.generate_app_vue())
        
        # Write main.js
        with open(src_dir / "main.js", "w") as f:
            f.write('''import { createApp } from 'vue';
import { createRouter, createWebHistory } from 'vue-router';
import App from './App.vue';
import ModelDashboard from './components/ModelDashboard.vue';

const routes = [
  { path: '/', component: ModelDashboard },
  { path: '/predict', component: () => import('./components/''' + self.model_name + '''Predictor.vue') }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

const app = createApp(App);
app.use(router);
app.mount('#app');
''')
        
        # Write package.json
        with open(output_path / "package.json", "w") as f:
            package = {
                "name": f"{self.model_name.lower()}-vue-dashboard",
                "version": "1.0.0",
                "scripts": {
                    "serve": "vue-cli-service serve",
                    "build": "vue-cli-service build",
                    "lint": "vue-cli-service lint"
                },
                "dependencies": {
                    "core-js": "^3.8.3",
                    "vue": "^3.2.13",
                    "vue-router": "^4.0.3"
                },
                "devDependencies": {
                    "@vue/cli-plugin-eslint": "~5.0.0",
                    "@vue/cli-plugin-router": "~5.0.0",
                    "@vue/cli-service": "~5.0.0",
                    "eslint": "^7.32.0",
                    "eslint-plugin-vue": "^8.0.3"
                }
            }
            json.dump(package, f, indent=2)


def create_vue_component(
    name: str,
    template: str = "",
    script: str = "",
    style: str = "",
    props: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None
) -> VueComponent:
    """Create a Vue component."""
    return VueComponent(
        name=name,
        template=template,
        script=script,
        style=style,
        props=props or {},
        data=data or {}
    )


# =============================================================================
# Deployment Implementation
# =============================================================================

class Dockerize:
    """Docker deployment utilities."""
    
    def __init__(self, app_name: str = "fishstick-app"):
        self.app_name = app_name
        self.base_image = "python:3.10-slim"
        self.port = 8000
        self.dependencies: List[str] = [
            "fastapi",
            "uvicorn",
            "flask",
            "gunicorn",
            "numpy",
            "torch",
            "transformers"
        ]
    
    def generate_dockerfile(
        self,
        framework: str = "fastapi",
        app_file: str = "app.py",
        requirements_file: Optional[str] = None
    ) -> str:
        """Generate Dockerfile."""
        
        dockerfile = f'''FROM {{{self.base_image}}}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
'''
        if requirements_file:
            dockerfile += f"COPY {requirements_file} .\nRUN pip install --no-cache-dir -r {requirements_file}\n\n"
        else:
            deps = " ".join(self.dependencies)
            dockerfile += f"RUN pip install --no-cache-dir {deps}\n\n"
        
        dockerfile += f'''# Copy application code
COPY . .

# Expose port
EXPOSE {{{self.port}}}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{{{self.port}}}/health || exit 1

# Run application
'''
        
        if framework == "fastapi":
            dockerfile += f'CMD ["uvicorn", "{app_file.replace(".py", "")}:app", "--host", "0.0.0.0", "--port", "{self.port}"]\n'
        elif framework == "flask":
            dockerfile += f'CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:{self.port}", "{app_file.replace(".py", "")}:app"]\n'
        else:
            dockerfile += f'CMD ["python", "{app_file}"]\n'
        
        return dockerfile
    
    def generate_docker_compose(
        self,
        services: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate docker-compose.yml."""
        
        compose = {
            "version": "3.8",
            "services": services or {
                "api": {
                    "build": ".",
                    "ports": [f"{self.port}:{self.port}"],
                    "environment": [
                        "ENVIRONMENT=production",
                        "LOG_LEVEL=info"
                    ],
                    "volumes": [
                        "./models:/app/models:ro"
                    ],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", f"http://localhost:{self.port}/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    },
                    "restart": "unless-stopped"
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"],
                    "restart": "unless-stopped"
                }
            },
            "volumes": {
                "redis_data": {}
            }
        }
        
        import yaml
        return yaml.dump(compose, default_flow_style=False)
    
    def generate_nginx_config(self) -> str:
        """Generate nginx configuration."""
        
        config = f'''upstream api_servers {{
    server api:{{{self.port}}};
}}

server {{
    listen 80;
    server_name localhost;

    location / {{
        proxy_pass http://api_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }}

    location /health {{
        access_log off;
        proxy_pass http://api_servers/health;
    }}
}}
'''
        return config
    
    def build_image(self, tag: Optional[str] = None, path: str = ".") -> str:
        """Build Docker image."""
        tag = tag or self.app_name
        cmd = f"docker build -t {tag} {path}"
        return cmd
    
    def write_configs(self, output_dir: str) -> None:
        """Write all Docker configurations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write Dockerfile
        with open(output_path / "Dockerfile", "w") as f:
            f.write(self.generate_dockerfile())
        
        # Write docker-compose.yml
        with open(output_path / "docker-compose.yml", "w") as f:
            f.write(self.generate_docker_compose())
        
        # Write nginx config
        nginx_dir = output_path / "nginx"
        nginx_dir.mkdir(exist_ok=True)
        with open(nginx_dir / "default.conf", "w") as f:
            f.write(self.generate_nginx_config())
        
        # Write .dockerignore
        with open(output_path / ".dockerignore", "w") as f:
            f.write('''__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.git
.gitignore
.env
.venv
venv
env
.pytest_cache
.coverage
htmlcov
.tox
.nox
.mypy_cache
.dmypy.json
*.log
.DS_Store
.idea
.vscode
node_modules
*.md
''')


class KubernetesDeploy:
    """Kubernetes deployment utilities."""
    
    def __init__(self, app_name: str = "fishstick-app"):
        self.app_name = app_name
        self.namespace = "default"
        self.image = f"{app_name}:latest"
        self.replicas = 3
        self.port = 8000
    
    def generate_deployment(self) -> str:
        """Generate Kubernetes deployment YAML."""
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.app_name,
                "namespace": self.namespace,
                "labels": {"app": self.app_name}
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {
                    "matchLabels": {"app": self.app_name}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": self.app_name}
                    },
                    "spec": {
                        "containers": [{
                            "name": self.app_name,
                            "image": self.image,
                            "ports": [{"containerPort": self.port}],
                            "resources": {
                                "requests": {
                                    "memory": "256Mi",
                                    "cpu": "250m"
                                },
                                "limits": {
                                    "memory": "512Mi",
                                    "cpu": "500m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": self.port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": self.port
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            },
                            "env": [
                                {"name": "ENVIRONMENT", "value": "production"},
                                {"name": "LOG_LEVEL", "value": "info"}
                            ]
                        }]
                    }
                }
            }
        }
        
        import yaml
        return yaml.dump(deployment, default_flow_style=False)
    
    def generate_service(self, service_type: str = "ClusterIP") -> str:
        """Generate Kubernetes service YAML."""
        
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.app_name}-service",
                "namespace": self.namespace
            },
            "spec": {
                "type": service_type,
                "selector": {"app": self.app_name},
                "ports": [{
                    "port": 80,
                    "targetPort": self.port,
                    "protocol": "TCP"
                }]
            }
        }
        
        import yaml
        return yaml.dump(service, default_flow_style=False)
    
    def generate_ingress(self, host: str = "api.example.com") -> str:
        """Generate Kubernetes ingress YAML."""
        
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.app_name}-ingress",
                "namespace": self.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true"
                }
            },
            "spec": {
                "tls": [{
                    "hosts": [host],
                    "secretName": f"{self.app_name}-tls"
                }],
                "rules": [{
                    "host": host,
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"{self.app_name}-service",
                                    "port": {"number": 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        import yaml
        return yaml.dump(ingress, default_flow_style=False)
    
    def generate_hpa(self) -> str:
        """Generate HorizontalPodAutoscaler YAML."""
        
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.app_name}-hpa",
                "namespace": self.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.app_name
                },
                "minReplicas": 2,
                "maxReplicas": 10,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 70
                        }
                    }
                }]
            }
        }
        
        import yaml
        return yaml.dump(hpa, default_flow_style=False)
    
    def generate_configmap(self, config: Dict[str, str]) -> str:
        """Generate ConfigMap YAML."""
        
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.app_name}-config",
                "namespace": self.namespace
            },
            "data": config
        }
        
        import yaml
        return yaml.dump(configmap, default_flow_style=False)
    
    def write_configs(self, output_dir: str) -> None:
        """Write all Kubernetes configurations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        k8s_dir = output_path / "k8s"
        k8s_dir.mkdir(exist_ok=True)
        
        # Write deployment
        with open(k8s_dir / "deployment.yaml", "w") as f:
            f.write(self.generate_deployment())
        
        # Write service
        with open(k8s_dir / "service.yaml", "w") as f:
            f.write(self.generate_service())
        
        # Write ingress
        with open(k8s_dir / "ingress.yaml", "w") as f:
            f.write(self.generate_ingress())
        
        # Write HPA
        with open(k8s_dir / "hpa.yaml", "w") as f:
            f.write(self.generate_hpa())
        
        # Write kustomization
        with open(k8s_dir / "kustomization.yaml", "w") as f:
            f.write(f'''apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - ingress.yaml
  - hpa.yaml

namespace: {self.namespace}

images:
  - name: {self.app_name}
    newTag: latest
''')


class CloudDeploy:
    """Cloud deployment utilities (AWS, GCP, Azure)."""
    
    def __init__(self, app_name: str = "fishstick-app"):
        self.app_name = app_name
        self.region = "us-east-1"
    
    def generate_aws_ecs_task(self) -> str:
        """Generate AWS ECS task definition."""
        
        task = {
            "family": self.app_name,
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "512",
            "memory": "1024",
            "executionRoleArn": f"arn:aws:iam::ACCOUNT_ID:role/{self.app_name}-execution-role",
            "containerDefinitions": [{
                "name": self.app_name,
                "image": f"{self.app_name}:latest",
                "portMappings": [{
                    "containerPort": 8000,
                    "protocol": "tcp"
                }],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": f"/ecs/{self.app_name}",
                        "awslogs-region": self.region,
                        "awslogs-stream-prefix": "ecs"
                    }
                },
                "environment": [
                    {"name": "ENVIRONMENT", "value": "production"}
                ]
            }]
        }
        
        return json.dumps(task, indent=2)
    
    def generate_terraform_aws(self) -> str:
        """Generate Terraform configuration for AWS."""
        
        tf = f'''
provider "aws" {{
  region = "{self.region}"
}}

resource "aws_ecs_cluster" "main" {{
  name = "{self.app_name}-cluster"
}}

resource "aws_ecs_service" "main" {{
  name            = "{self.app_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {{
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }}

  load_balancer {{
    target_group_arn = aws_lb_target_group.main.arn
    container_name   = "{self.app_name}"
    container_port   = 8000
  }}
}}

resource "aws_lb" "main" {{
  name               = "{self.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb.id]
  subnets            = aws_subnet.public[*].id
}}
'''
        return tf
    
    def generate_gcp_cloud_run(self) -> str:
        """Generate Google Cloud Run service YAML."""
        
        service = {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {
                "name": self.app_name,
                "annotations": {
                    "run.googleapis.com/ingress": "all"
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/minScale": "1",
                            "autoscaling.knative.dev/maxScale": "100"
                        }
                    },
                    "spec": {
                        "containerConcurrency": 80,
                        "timeoutSeconds": 300,
                        "containers": [{
                            "image": f"gcr.io/PROJECT_ID/{self.app_name}:latest",
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "limits": {
                                    "cpu": "1000m",
                                    "memory": "512Mi"
                                }
                            },
                            "env": [
                                {"name": "ENVIRONMENT", "value": "production"}
                            ]
                        }]
                    }
                }
            }
        }
        
        import yaml
        return yaml.dump(service, default_flow_style=False)
    
    def generate_azure_container_instance(self) -> str:
        """Generate Azure Container Instance ARM template."""
        
        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "resources": [{
                "type": "Microsoft.ContainerInstance/containerGroups",
                "apiVersion": "2021-03-01",
                "name": self.app_name,
                "location": "[resourceGroup().location]",
                "properties": {
                    "containers": [{
                        "name": self.app_name,
                        "properties": {
                            "image": f"{self.app_name}:latest",
                            "resources": {
                                "requests": {
                                    "cpu": 1,
                                    "memoryInGB": 1
                                }
                            },
                            "ports": [{"port": 8000}]
                        }
                    }],
                    "osType": "Linux",
                    "ipAddress": {
                        "type": "Public",
                        "ports": [{
                            "protocol": "TCP",
                            "port": 8000
                        }]
                    }
                }
            }]
        }
        
        return json.dumps(template, indent=2)


class ServerlessDeploy:
    """Serverless deployment utilities."""
    
    def __init__(self, app_name: str = "fishstick-app"):
        self.app_name = app_name
        self.runtime = "python3.10"
        self.memory = 512
        self.timeout = 30
    
    def generate_serverless_framework(self, provider: str = "aws") -> str:
        """Generate serverless.yml for Serverless Framework."""
        
        config = {
            "service": self.app_name,
            "frameworkVersion": "3",
            "provider": {
                "name": provider,
                "runtime": self.runtime,
                "memorySize": self.memory,
                "timeout": self.timeout,
                "environment": {
                    "STAGE": "${{opt:stage, 'dev'}}"
                }
            },
            "functions": {
                "predict": {
                    "handler": "handler.predict",
                    "events": [{
                        "http": {
                            "path": "predict",
                            "method": "post",
                            "cors": True
                        }
                    }]
                },
                "health": {
                    "handler": "handler.health",
                    "events": [{
                        "http": {
                            "path": "health",
                            "method": "get",
                            "cors": True
                        }
                    }]
                }
            },
            "plugins": ["serverless-python-requirements"],
            "custom": {
                "pythonRequirements": {
                    "dockerizePip": True,
                    "slim": True
                }
            }
        }
        
        import yaml
        return yaml.dump(config, default_flow_style=False)
    
    def generate_aws_lambda_handler(self) -> str:
        """Generate AWS Lambda handler."""
        
        code = '''import json
import os
from mangum import Mangum

# Import your FastAPI/Flask app
from app import app

# Wrap with Mangum for Lambda
handler = Mangum(app, lifespan="off")

def predict(event, context):
    """Lambda handler for predictions."""
    return handler(event, context)

def health(event, context):
    """Lambda handler for health checks."""
    return {
        "statusCode": 200,
        "body": json.dumps({
            "status": "healthy",
            "version": "1.0.0"
        })
    }
'''
        return code
    
    def generate_aws_sam_template(self) -> str:
        """Generate AWS SAM template."""
        
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Transform": "AWS::Serverless-2016-10-31",
            "Description": f"{self.app_name} Serverless API",
            "Globals": {
                "Function": {
                    "Timeout": self.timeout,
                    "MemorySize": self.memory,
                    "Runtime": self.runtime
                }
            },
            "Resources": {
                "ApiFunction": {
                    "Type": "AWS::Serverless::Function",
                    "Properties": {
                        "CodeUri": ".",
                        "Handler": "app.handler",
                        "Events": {
                            "ApiEvent": {
                                "Type": "Api",
                                "Properties": {
                                    "Path": "/{proxy+}",
                                    "Method": "ANY"
                                }
                            }
                        }
                    }
                }
            },
            "Outputs": {
                "ApiUrl": {
                    "Description": "API Gateway endpoint URL",
                    "Value": "!Sub 'https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/'"
                }
            }
        }
        
        return json.dumps(template, indent=2)
    
    def generate_azure_functions(self) -> str:
        """Generate Azure Functions configuration."""
        
        function_json = {
            "scriptFile": "__init__.py",
            "bindings": [
                {
                    "authLevel": "anonymous",
                    "type": "httpTrigger",
                    "direction": "in",
                    "name": "req",
                    "methods": ["get", "post"]
                },
                {
                    "type": "http",
                    "direction": "out",
                    "name": "$return"
                }
            ]
        }
        
        return json.dumps(function_json, indent=2)
    
    def generate_gcp_cloud_function(self) -> str:
        """Generate Google Cloud Function code."""
        
        code = '''import functions_framework
from flask import Flask, request, jsonify

# Import your app
from app import app as flask_app

@functions_framework.http
def predict(request):
    """HTTP Cloud Function entry point."""
    with flask_app.test_client() as client:
        response = client.post('/predict', 
                             json=request.get_json())
        return response.get_json(), response.status_code

@functions_framework.http
def health(request):
    """Health check endpoint."""
    return {{
        "status": "healthy",
        "version": "1.0.0"
    }}, 200
'''
        return code
    
    def write_configs(self, output_dir: str, provider: str = "aws") -> None:
        """Write all serverless configurations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write serverless.yml
        with open(output_path / "serverless.yml", "w") as f:
            f.write(self.generate_serverless_framework(provider))
        
        # Write handler
        with open(output_path / "handler.py", "w") as f:
            f.write(self.generate_aws_lambda_handler())
        
        # Write SAM template
        if provider == "aws":
            with open(output_path / "template.yaml", "w") as f:
                f.write(self.generate_aws_sam_template())


# =============================================================================
# Utility Functions
# =============================================================================

def web_app(
    framework: str = "fastapi",
    models: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Union[FastAPIApp, FlaskApp, StreamlitApp, GradioInterface, None]:
    """
    Create a web application for serving fishstick models.
    
    Args:
        framework: Framework to use ('fastapi', 'flask', 'streamlit', 'gradio')
        models: Dictionary of models to serve
        **kwargs: Additional arguments for the specific framework
    
    Returns:
        Web application instance
    
    Example:
        >>> models = {"classifier": my_model}
        >>> app = web_app("fastapi", models)
        >>> app.run()
    """
    framework = framework.lower()
    
    if framework == "fastapi":
        if not _FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")
        app = create_fastapi_app(**kwargs)
        if models:
            for name, model in models.items():
                app.add_model(name, model)
        return app
    
    elif framework == "flask":
        if not _FLASK_AVAILABLE:
            raise ImportError("Flask not installed. Install with: pip install flask flask-cors")
        app = create_flask_app(**kwargs)
        if models:
            for name, model in models.items():
                app.add_model(name, model)
        return app
    
    elif framework == "streamlit":
        if not _STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not installed. Install with: pip install streamlit")
        app = create_streamlit_app(**kwargs)
        if models:
            for name, model in models.items():
                app.add_model(name, model)
        return app
    
    elif framework == "gradio":
        if not _GRADIO_AVAILABLE:
            raise ImportError("Gradio not installed. Install with: pip install gradio")
        interface = create_gradio_interface()
        if models:
            for name, model in models.items():
                interface.add_model(name, model)
        return interface
    
    else:
        raise ValueError(f"Unknown framework: {framework}. Choose from: fastapi, flask, streamlit, gradio")


def create_api(
    model: Any,
    name: str,
    framework: str = "fastapi",
    **kwargs
) -> Union[APIRouter, Blueprint, None]:
    """
    Create an API endpoint for a model.
    
    Args:
        model: The model to serve
        name: Name of the model endpoint
        framework: Framework to use ('fastapi' or 'flask')
        **kwargs: Additional configuration
    
    Returns:
        API router or blueprint
    
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> router = create_api(my_model, "classifier", "fastapi")
        >>> app.include_router(router)
    """
    framework = framework.lower()
    
    if framework == "fastapi":
        if not _FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed")
        app = kwargs.get("app") or FastAPI()
        return add_model_endpoint(app, model, name)
    
    elif framework == "flask":
        if not _FLASK_AVAILABLE:
            raise ImportError("Flask not installed")
        return ModelAPI(model, name).create_blueprint()
    
    else:
        raise ValueError(f"Unknown framework: {framework}")


def create_dashboard(
    model: Any,
    name: str,
    framework: str = "streamlit",
    **kwargs
) -> Union[ModelDashboard, ModelDashboardVue, ModelDemo, None]:
    """
    Create a dashboard for a model.
    
    Args:
        model: The model to visualize
        name: Name of the model
        framework: Framework to use ('streamlit', 'vue', 'gradio')
        **kwargs: Additional configuration
    
    Returns:
        Dashboard instance
    
    Example:
        >>> dashboard = create_dashboard(my_model, "classifier", "streamlit")
        >>> dashboard.render()
    """
    framework = framework.lower()
    
    if framework == "streamlit":
        if not _STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not installed")
        return ModelDashboard(name, model)
    
    elif framework == "vue":
        return ModelDashboardVue(name, model_info=kwargs.get("model_info"))
    
    elif framework == "gradio":
        if not _GRADIO_AVAILABLE:
            raise ImportError("Gradio not installed")
        return ModelDemo(model, name)
    
    else:
        raise ValueError(f"Unknown framework: {framework}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # FastAPI
    "FastAPIApp",
    "create_fastapi_app",
    "add_model_endpoint",
    "ModelInferenceEndpoint",
    
    # Flask
    "FlaskApp",
    "create_flask_app",
    "ModelAPI",
    "PredictionEndpoint",
    
    # Streamlit
    "StreamlitApp",
    "create_streamlit_app",
    "ModelDashboard",
    "Visualization",
    
    # Gradio
    "GradioInterface",
    "create_gradio_interface",
    "ModelDemo",
    
    # React
    "ReactComponent",
    "create_react_component",
    "ModelFrontend",
    "APIClient",
    
    # Vue
    "VueComponent",
    "create_vue_component",
    "ModelDashboardVue",
    
    # Deployment
    "Dockerize",
    "KubernetesDeploy",
    "CloudDeploy",
    "ServerlessDeploy",
    
    # Utilities
    "web_app",
    "create_api",
    "create_dashboard",
    
    # Types
    "ModelInput",
    "ModelOutput",
    "HealthResponse",
    "ModelInfo",
    "Predictable",
    "AsyncPredictable",
]

# Availability flags
__all__.extend([
    "_FASTAPI_AVAILABLE",
    "_FLASK_AVAILABLE",
    "_STREAMLIT_AVAILABLE",
    "_GRADIO_AVAILABLE",
    "_TORCH_AVAILABLE",
])
