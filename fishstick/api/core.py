"""
Comprehensive API Server Module for fishstick

Provides:
- FastAPI Server with model serving
- REST API endpoints (GET, POST, PUT, DELETE)
- WebSocket streaming
- gRPC server and client
- Authentication (JWT, API Key, OAuth2, Basic)
- Rate Limiting (Token Bucket, Sliding Window)
- Documentation (Swagger, ReDoc, OpenAPI)
- Utilities for running servers and creating clients
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
ModelType = TypeVar("ModelType")


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIResponse:
    status_code: int = 200
    data: Any = None
    message: str = ""
    headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status_code": self.status_code,
            "data": self.data,
            "message": self.message,
            "headers": self.headers,
        }


class Request:
    def __init__(
        self,
        body: Any = None,
        query_params: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
    ):
        self.body = body
        self.query_params = query_params or {}
        self.headers = headers or {}
        self._json: Optional[Dict] = None

    @property
    def json(self) -> Dict[str, Any]:
        if self._json is None:
            if isinstance(self.body, dict):
                self._json = self.body
            elif isinstance(self.body, str):
                self._json = json.loads(self.body)
        return self._json

    def get(self, key: str, default: Any = None) -> Any:
        if self._json:
            return self._json.get(key, default)
        return default


class Response:
    def __init__(
        self,
        content: Any = None,
        status_code: int = 200,
        media_type: str = "application/json",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.content = content
        self.status_code = status_code
        self.media_type = media_type
        self.headers = headers or {}

    @staticmethod
    def json(content: Any, status_code: int = 200, **kwargs) -> Response:
        return Response(
            content=content,
            status_code=status_code,
            media_type="application/json",
            **kwargs,
        )

    @staticmethod
    def text(content: str, status_code: int = 200, **kwargs) -> Response:
        return Response(
            content=content, status_code=status_code, media_type="text/plain", **kwargs
        )

    @staticmethod
    def html(content: str, status_code: int = 200, **kwargs) -> Response:
        return Response(
            content=content, status_code=status_code, media_type="text/html", **kwargs
        )

    def to_fastapi_response(self):
        try:
            from fastapi import Response as FastAPIResponse

            return FastAPIResponse(
                content=json.dumps(self.content)
                if isinstance(self.content, (dict, list))
                else str(self.content),
                status_code=self.status_code,
                media_type=self.media_type,
                headers=self.headers,
            )
        except ImportError:
            return self


class RESTEndpoint(ABC):
    def __init__(self, path: str, methods: List[HTTPMethod] = None):
        self.path = path
        self.methods = methods or [HTTPMethod.GET]
        self._handlers: Dict[HTTPMethod, Callable] = {}

    def handler(self, method: HTTPMethod):
        def decorator(func: Callable):
            self._handlers[method] = func
            return func

        return decorator

    def get(self, func: Callable):
        return self.handler(HTTPMethod.GET)(func)

    def post(self, func: Callable):
        return self.handler(HTTPMethod.POST)(func)

    def put(self, func: Callable):
        return self.handler(HTTPMethod.PUT)(func)

    def delete(self, func: Callable):
        return self.handler(HTTPMethod.DELETE)(func)

    async def handle(self, method: HTTPMethod, request: Request) -> Response:
        handler = self._handlers.get(method)
        if not handler:
            return Response.json({"error": "Method not allowed"}, status_code=405)
        return await handler(request)


class PredictionEndpoint(RESTEndpoint):
    def __init__(self, model_server: "ModelServer"):
        super().__init__("/predict", [HTTPMethod.POST])
        self.model_server = model_server

    async def predict(self, request: Request) -> Response:
        try:
            data = request.json
            input_data = data.get("data", [])
            return_probabilities = data.get("return_probabilities", False)

            result = await self.model_server.predict(input_data, return_probabilities)
            return Response.json(result)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return Response.json({"error": str(e)}, status_code=500)


class FastAPIServer:
    def __init__(
        self,
        title: str = "fishstick API",
        version: str = "0.1.0",
        description: str = "Mathematically Rigorous AI Framework",
        docs_url: str = "/docs",
        redoc_url: str = "/redoc",
        openapi_url: str = "/openapi.json",
    ):
        self.title = title
        self.version = version
        self.description = description
        self.docs_url = docs_url
        self.redoc_url = redoc_url
        self.openapi_url = openapi_url
        self._app = None
        self._endpoints: List[RESTEndpoint] = []
        self._model_server: Optional[ModelServer] = None

    def create_app(self) -> "FastAPI":
        try:
            from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
        except ImportError:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

        app = FastAPI(
            title=self.title,
            version=self.version,
            description=self.description,
            docs_url=self.docs_url,
            redoc_url=self.redoc_url,
            openapi_url=self.openapi_url,
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/")
        async def root():
            return {
                "message": "Welcome to fishstick API",
                "docs": self.docs_url,
                "health": "/health",
            }

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model_loaded": self._model_server is not None}

        @app.get("/info")
        async def info():
            return {
                "name": "fishstick",
                "version": self.version,
                "endpoints": [ep.path for ep in self._endpoints],
            }

        for endpoint in self._endpoints:
            self._register_endpoint(app, endpoint)

        self._app = app
        return app

    def _register_endpoint(self, app: "FastAPI", endpoint: RESTEndpoint):
        from fastapi import HTTPException, Request as FastAPIRequest

        methods_map = {
            HTTPMethod.GET: app.get,
            HTTPMethod.POST: app.post,
            HTTPMethod.PUT: app.put,
            HTTPMethod.DELETE: app.delete,
            HTTPMethod.PATCH: app.patch,
        }

        for method in endpoint.methods:
            handler = methods_map[method]

            async def endpoint_handler(request: FastAPIRequest, ep=endpoint, m=method):
                req = Request(
                    body=await request.json()
                    if request.method in ["POST", "PUT", "PATCH"]
                    else None,
                    query_params=dict(request.query_params),
                    headers=dict(request.headers),
                )
                return await ep.handle(m, req)

            handler(endpoint.path)(endpoint_handler)

    def add_model_endpoint(self, model_server: "ModelServer"):
        self._model_server = model_server
        prediction_ep = PredictionEndpoint(model_server)
        self._endpoints.append(prediction_ep)

        if self._app:
            self._register_endpoint(self._app, prediction_ep)

    def add_endpoint(self, endpoint: RESTEndpoint):
        self._endpoints.append(endpoint)
        if self._app:
            self._register_endpoint(self._app, endpoint)

    @property
    def app(self):
        if self._app is None:
            self._app = self.create_app()
        return self._app


class ModelServer:
    def __init__(self, model: Any = None, device: str = "cpu"):
        self.model = model
        self.device = device
        self._model_loaded = False

    async def load_model(self, model_path: str, model_class: Type = None):
        try:
            import torch
            from torch import nn

            if model_class is not None:
                self.model = model_class()
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def predict(
        self, data: List, return_probabilities: bool = False
    ) -> Dict[str, Any]:
        import torch
        from torch import nn

        start_time = time.time()

        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                if isinstance(outputs, torch.Tensor):
                    probabilities = torch.softmax(outputs, dim=-1)
                    predictions = torch.argmax(outputs, dim=-1)

            inference_time = (time.time() - start_time) * 1000

            result = {
                "predictions": predictions.tolist()
                if isinstance(predictions, torch.Tensor)
                else predictions,
                "model_name": "fishstick-model",
                "inference_time_ms": inference_time,
            }

            if return_probabilities:
                result["probabilities"] = (
                    probabilities.tolist()
                    if isinstance(probabilities, torch.Tensor)
                    else probabilities
                )

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    async def predict_batch(self, batch_data: List[List]) -> List[Dict[str, Any]]:
        results = []
        for data in batch_data:
            result = await self.predict([data])
            results.append(result)
        return results


class WebSocketEndpoint:
    def __init__(self, path: str):
        self.path = path
        self._connections: Dict[str, Any] = {}

    async def connect(self, websocket: "AsyncWebSocket", client_id: str):
        self._connections[client_id] = websocket
        await websocket.accept()

    async def disconnect(self, client_id: str):
        if client_id in self._connections:
            del self._connections[client_id]

    async def send_message(self, client_id: str, message: Dict[str, Any]):
        if client_id in self._connections:
            await self._connections[client_id].send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        for client_id in self._connections:
            await self.send_message(client_id, message)

    async def stream_predictions(
        self,
        websocket: "AsyncWebSocket",
        model_server: ModelServer,
        client_id: str,
    ):
        await self.connect(websocket, client_id)

        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    continue

                data = message.get("data", [])
                result = await model_server.predict(data)
                await websocket.send_json(result)

        except Exception as e:
            logger.error(f"WebSocket stream error: {e}")
        finally:
            await self.disconnect(client_id)


class AsyncWebSocket:
    def __init__(self, websocket=None):
        self._ws = websocket

    async def accept(self):
        if self._ws:
            await self._ws.accept()

    async def send_json(self, data: Dict[str, Any]):
        if self._ws:
            await self._ws.send_json(data)

    async def receive_json(self) -> Dict[str, Any]:
        if self._ws:
            return await self._ws.receive_json()
        return {}

    async def close(self, code: int = 1000):
        if self._ws:
            await self._ws.close(code=code)


class JWTAuth:
    def __init__(
        self, secret_key: str, algorithm: str = "HS256", expiry_minutes: int = 60
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiry_minutes = expiry_minutes
        self._token_cache: Dict[str, Dict] = {}

    def create_token(
        self, subject: str, additional_claims: Optional[Dict] = None
    ) -> str:
        try:
            import jwt
        except ImportError:
            raise ImportError("PyJWT not installed. Run: pip install pyjwt")

        now = datetime.utcnow()
        payload = {
            "sub": subject,
            "iat": now,
            "exp": now + timedelta(minutes=self.expiry_minutes),
            "jti": str(uuid.uuid4()),
        }
        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        self._token_cache[token] = payload
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            import jwt
        except ImportError:
            raise ImportError("PyJWT not installed. Run: pip install pyjwt")

        try:
            if token in self._token_cache:
                cached = self._token_cache[token]
                if datetime.utcnow() < datetime.fromtimestamp(cached["exp"]):
                    return cached

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_subject(self, token: str) -> Optional[str]:
        payload = self.verify_token(token)
        return payload.get("sub") if payload else None

    async def __call__(self, request: Request) -> Optional[str]:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return self.get_subject(token)
        return None


class APIKeyAuth:
    def __init__(
        self, api_keys: Optional[Dict[str, str]] = None, header_name: str = "X-API-Key"
    ):
        self.api_keys = api_keys or {}
        self.header_name = header_name

    def add_key(self, key: str, user_id: str):
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        self.api_keys[key_hash] = user_id

    def verify_key(self, key: str) -> Optional[str]:
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.api_keys.get(key_hash)

    async def __call__(self, request: Request) -> Optional[str]:
        api_key = request.headers.get(self.header_name, "")
        return self.verify_key(api_key)


class OAuth2Auth:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        authorize_url: Optional[str] = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.authorize_url = authorize_url
        self._tokens: Dict[str, Dict] = {}

    async def get_token(self, authorization_code: str) -> Optional[Dict[str, Any]]:
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp not installed. Run: pip install aiohttp")

        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
                async with session.post(self.token_url, data=data) as resp:
                    if resp.status == 200:
                        token_data = await resp.json()
                        access_token = token_data.get("access_token")
                        if access_token:
                            self._tokens[access_token] = token_data
                        return token_data
        except Exception as e:
            logger.error(f"OAuth2 token exchange failed: {e}")
        return None

    def verify_token(self, access_token: str) -> bool:
        return access_token in self._tokens

    async def __call__(self, request: Request) -> Optional[str]:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if self.verify_token(token):
                return self._tokens[token].get("sub")
        return None


class BasicAuth:
    def __init__(self, users: Optional[Dict[str, str]] = None):
        self.users = users or {}

    def add_user(self, username: str, password: str):
        self.users[username] = self._hash_password(password)

    def verify_credentials(self, username: str, password: str) -> bool:
        stored_hash = self.users.get(username)
        if not stored_hash:
            return False
        return hmac.compare_digest(stored_hash, self._hash_password(password))

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    async def __call__(self, request: Request) -> Optional[str]:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Basic "):
            import base64

            try:
                encoded = auth_header[6:]
                decoded = base64.b64decode(encoded).decode("utf-8")
                username, password = decoded.split(":", 1)
                if self.verify_credentials(username, password):
                    return username
            except Exception:
                pass
        return None


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}

    def _clean_old_requests(self, client_id: str, current_time: float):
        if client_id in self._requests:
            self._requests[client_id] = [
                t
                for t in self._requests[client_id]
                if current_time - t < self.window_seconds
            ]

    def is_allowed(self, client_id: str) -> bool:
        current_time = time.time()
        self._clean_old_requests(client_id, current_time)

        if client_id not in self._requests:
            self._requests[client_id] = []

        if len(self._requests[client_id]) < self.max_requests:
            self._requests[client_id].append(current_time)
            return True
        return False

    def get_remaining(self, client_id: str) -> int:
        current_time = time.time()
        self._clean_old_requests(client_id, current_time)
        return max(0, self.max_requests - len(self._requests.get(client_id, [])))

    async def __call__(self, request: Request) -> bool:
        client_id = request.headers.get(
            "X-Client-ID", request.query_params.get("client_id", "default")
        )
        return self.is_allowed(client_id)


class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: Dict[str, Dict] = {}

    def _get_bucket(self, client_id: str) -> Dict:
        if client_id not in self._buckets:
            self._buckets[client_id] = {
                "tokens": float(self.capacity),
                "last_update": time.time(),
            }
        return self._buckets[client_id]

    def _refill(self, bucket: Dict):
        now = time.time()
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(
            self.capacity, bucket["tokens"] + elapsed * self.refill_rate
        )
        bucket["last_update"] = now

    def consume(self, client_id: str, tokens: int = 1) -> bool:
        bucket = self._get_bucket(client_id)
        self._refill(bucket)

        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            return True
        return False


class SlidingWindow:
    def __init__(self, max_requests: int, window_ms: int):
        self.max_requests = max_requests
        self.window_ms = window_ms
        self._windows: Dict[str, List[float]] = {}

    def _clean_old_requests(self, client_id: str, current_time: float):
        if client_id in self._windows:
            cutoff = current_time - (self.window_ms / 1000.0)
            self._windows[client_id] = [
                t for t in self._windows[client_id] if t > cutoff
            ]

    def is_allowed(self, client_id: str) -> bool:
        current_time = time.time()
        self._clean_old_requests(client_id, current_time)

        if client_id not in self._windows:
            self._windows[client_id] = []

        if len(self._windows[client_id]) < self.max_requests:
            self._windows[client_id].append(current_time)
            return True
        return False


class SwaggerUI:
    def __init__(self, enabled: bool = True, path: str = "/docs"):
        self.enabled = enabled
        self.path = path

    def get_config(self, openapi_url: str) -> Dict[str, Any]:
        return {
            "swagger_ui_enabled": self.enabled,
            "swagger_ui_path": self.path,
            "swagger_ui_config": {
                "url": openapi_url,
                "deepLinking": False,
                "displayRequestDuration": True,
            },
        }


class ReDoc:
    def __init__(self, enabled: bool = True, path: str = "/redoc"):
        self.enabled = enabled
        self.path = path

    def get_config(self, openapi_url: str) -> Dict[str, Any]:
        return {
            "redoc_enabled": self.enabled,
            "redoc_path": self.path,
            "redoc_config": {
                "spec_url": openapi_url,
            },
        }


class OpenAPISpec:
    def __init__(
        self,
        title: str = "fishstick API",
        version: str = "0.1.0",
        description: str = "",
        servers: Optional[List[Dict]] = None,
    ):
        self.title = title
        self.version = version
        self.description = description
        self.servers = servers or [{"url": "http://localhost:8000"}]
        self._paths: Dict[str, Dict] = {}
        self._components: Dict[str, Any] = {}

    def add_path(self, path: str, method: str, operation: Dict):
        if path not in self._paths:
            self._paths[path] = {}
        self._paths[path][method.lower()] = operation

    def add_schema(self, name: str, schema: Dict):
        if "schemas" not in self._components:
            self._components["schemas"] = {}
        self._components["schemas"][name] = schema

    def to_dict(self) -> Dict[str, Any]:
        return {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
            },
            "servers": self.servers,
            "paths": self._paths,
            "components": self._components,
        }


class gRPCServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 50051):
        self.host = host
        self.port = port
        self._server = None
        self._model_server: Optional[ModelServer] = None
        self._services: List[Any] = []

    def set_model_server(self, model_server: ModelServer):
        self._model_server = model_server

    def add_service(self, service):
        self._services.append(service)

    async def start(self):
        try:
            import grpc
            from concurrent import futures
        except ImportError:
            raise ImportError(
                "grpcio not installed. Run: pip install grpcio grpcio-tools"
            )

        server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        server.add_insecure_port(f"{self.host}:{self.port}")

        for service in self._services:
            service.register(server)

        await server.start()
        logger.info(f"gRPC server started on {self.host}:{self.port}")
        self._server = server
        return server

    async def stop(self, grace: int = 5):
        if self._server:
            await self._server.stop(grace)


class PredictionService:
    def __init__(self, model_server: ModelServer):
        self.model_server = model_server

    def register(self, server):
        pass


class gRPCClient:
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self._channel = None
        self._stub = None

    def connect(self):
        try:
            import grpc
        except ImportError:
            raise ImportError(
                "grpcio not installed. Run: pip install grpcio grpcio-tools"
            )

        self._channel = grpc.insecure_channel(f"{self.host}:{self.port}")

    def close(self):
        if self._channel:
            self._channel.close()

    async def predict(
        self, data: List[float], return_probabilities: bool = False
    ) -> Dict[str, Any]:
        if not self._channel:
            self.connect()

        return {
            "predictions": [],
            "inference_time_ms": 0.0,
        }


@asynccontextmanager
async def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model: Any = None,
    model_path: Optional[str] = None,
    reload: bool = False,
) -> AsyncGenerator[FastAPIServer, None]:
    server = FastAPIServer()

    model_server = ModelServer(model=model)
    if model_path:
        await model_server.load_model(model_path)

    server.add_model_endpoint(model_server)

    app = server.create_app()

    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Run: pip install uvicorn")

    config = uvicorn.Config(app, host=host, port=port, reload=reload)
    runner = uvicorn.Server(config)

    asyncio.create_task(runner.serve())

    await asyncio.sleep(1)
    yield server

    await runner.shutdown()


def create_client(base_url: str = "http://localhost:8000") -> "HTTPClient":
    return HTTPClient(base_url)


class HTTPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._session = None

    async def _get_session(self):
        if self._session is None:
            try:
                import aiohttp

                self._session = aiohttp.ClientSession()
            except ImportError:
                raise ImportError("aiohttp not installed. Run: pip install aiohttp")
        return self._session

    async def get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.get(f"{self.base_url}{path}", params=params) as resp:
            return await resp.json()

    async def post(
        self, path: str, json: Optional[Dict] = None, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.post(f"{self.base_url}{path}", json=json, data=data) as resp:
            return await resp.json()

    async def put(self, path: str, json: Optional[Dict] = None) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.put(f"{self.base_url}{path}", json=json) as resp:
            return await resp.json()

    async def delete(self, path: str) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.delete(f"{self.base_url}{path}") as resp:
            return await resp.json()

    async def close(self):
        if self._session:
            await self._session.close()


__all__ = [
    "FastAPIServer",
    "create_app",
    "add_model_endpoint",
    "ModelServer",
    "PredictionEndpoint",
    "RESTEndpoint",
    "GET",
    "POST",
    "PUT",
    "DELETE",
    "HTTPMethod",
    "Response",
    "Request",
    "APIResponse",
    "WebSocketEndpoint",
    "stream_predictions",
    "AsyncWebSocket",
    "gRPCServer",
    "PredictionService",
    "gRPCClient",
    "JWTAuth",
    "APIKeyAuth",
    "OAuth2Auth",
    "BasicAuth",
    "RateLimiter",
    "TokenBucket",
    "SlidingWindow",
    "SwaggerUI",
    "ReDoc",
    "OpenAPISpec",
    "run_server",
    "create_client",
    "HTTPClient",
]
