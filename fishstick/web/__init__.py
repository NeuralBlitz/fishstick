"""
fishstick/web/__init__.py
========================
Web deployment module for fishstick models.

This module provides comprehensive web deployment capabilities including:
- FastAPI and Flask APIs
- Streamlit and Gradio dashboards
- React and Vue frontend generators
- Docker, Kubernetes, and cloud deployment tools
"""

from fishstick.web.core import (
    # FastAPI
    FastAPIApp,
    create_fastapi_app,
    add_model_endpoint,
    ModelInferenceEndpoint,
    # Flask
    FlaskApp,
    create_flask_app,
    ModelAPI,
    PredictionEndpoint,
    # Streamlit
    StreamlitApp,
    create_streamlit_app,
    ModelDashboard,
    Visualization,
    # Gradio
    GradioInterface,
    create_gradio_interface,
    ModelDemo,
    # React
    ReactComponent,
    create_react_component,
    ModelFrontend,
    APIClient,
    # Vue
    VueComponent,
    create_vue_component,
    ModelDashboardVue,
    # Deployment
    Dockerize,
    KubernetesDeploy,
    CloudDeploy,
    ServerlessDeploy,
    # Utilities
    web_app,
    create_api,
    create_dashboard,
    # Types
    ModelInput,
    ModelOutput,
    HealthResponse,
    ModelInfo,
    Predictable,
    AsyncPredictable,
)

__version__ = "1.0.0"

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
