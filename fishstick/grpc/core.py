from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent import futures
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import grpc
from grpc import aio as grpc_aio

try:
    from google.protobuf import descriptor_pb2
    from google.protobuf import descriptor_pool
    from google.protobuf import message_factory
    from google.protobuf import json_format

    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    descriptor_pb2 = None
    descriptor_pool = None
    message_factory = None
    json_format = None

try:
    from grpc_reflection.v1alpha import reflection

    REFLECTION_AVAILABLE = True
except ImportError:
    reflection = None
    REFLECTION_AVAILABLE = False


logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    PROTOBUF = "protobuf"
    JSON = "json"
    MSGPK = "msgpack"


@dataclass
class GrpcServerConfig:
    host: str = "[::]"
    port: int = 50051
    max_workers: int = 10
    max_concurrent_rpc: int = 100
    enable_reflection: bool = True
    enable_health_checks: bool = True
    authentication: Optional[Dict[str, str]] = None
    tls_enabled: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    compression: Optional[grpc.Compression] = grpc.Compression.Deflate


@dataclass
class GrpcClientConfig:
    host: str = "localhost"
    port: int = 50051
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    authentication: Optional[Dict[str, str]] = None
    tls_enabled: bool = False
    tls_cert_path: Optional[str] = None
    compression: Optional[grpc.Compression] = grpc.Compression.Deflate
    keepalive_time: Optional[int] = None
    keepalive_timeout: int = 10


@dataclass
class PredictionRequest:
    model_id: str
    inputs: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResponse:
    model_id: str
    outputs: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrainingRequest:
    model_id: str
    dataset_path: str
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2


@dataclass
class TrainingResponse:
    model_id: str
    status: str
    metrics: Optional[Dict[str, float]] = None
    checkpoint_path: Optional[str] = None


@dataclass
class EvaluationRequest:
    model_id: str
    dataset_path: str
    metrics: Optional[List[str]] = None


@dataclass
class EvaluationResponse:
    model_id: str
    results: Dict[str, float]
    predictions: Optional[List[Any]] = None


@dataclass
class ModelInfo:
    model_id: str
    name: str
    version: str
    framework: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ServerInterceptor(grpc.ServerInterceptor, ABC):
    @abstractmethod
    def intercept_service(self, continuation, handler_call_details):
        pass


class ClientInterceptor(grpc.ClientInterceptor, ABC):
    @abstractmethod
    def interceptUnaryUnary(self, client_call_details, request, call_details):
        pass

    @abstractmethod
    def interceptUnaryStream(self, client_call_details, request, call_details):
        pass

    @abstractmethod
    def interceptStreamUnary(self, client_call_details, request_iterator, call_details):
        pass

    @abstractmethod
    def interceptStreamStream(
        self, client_call_details, request_iterator, call_details
    ):
        pass


class LoggingInterceptor(ServerInterceptor):
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.interceptor = grpc.ServerInterceptor()

    def intercept_service(self, continuation, handler_call_details):
        self.logger.debug(f"Intercepted call: {handler_call_details.method}")
        return continuation(handler_call_details)


class AuthInterceptor(ServerInterceptor):
    def __init__(self, required_tokens: Dict[str, str]):
        self.required_tokens = required_tokens

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or [])
        token = metadata.get("authorization", "").replace("Bearer ", "")

        if handler_call_details.method not in self.required_tokens:
            return continuation(handler_call_details)

        expected_token = self.required_tokens.get(handler_call_details.method, "")
        if token != expected_token:
            raise grpc.RpcError("Unauthorized")

        return continuation(handler_call_details)


class ClientLoggingInterceptor(ClientInterceptor):
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def interceptUnaryUnary(self, client_call_details, request, call_details):
        self.logger.debug(f"Client unary call: {client_call_details.method}")
        return None

    def interceptUnaryStream(self, client_call_details, request, call_details):
        self.logger.debug(f"Client unary-stream call: {client_call_details.method}")
        return None

    def interceptStreamUnary(self, client_call_details, request_iterator, call_details):
        self.logger.debug(f"Client stream-unary call: {client_call_details.method}")
        return None

    def interceptStreamStream(
        self, client_call_details, request_iterator, call_details
    ):
        self.logger.debug(f"Client stream-stream call: {client_call_details.method}")
        return None


class ClientAuthInterceptor(ClientInterceptor):
    def __init__(self, token: str):
        self.token = token

    def _add_metadata(self, client_call_details):
        metadata = list(client_call_details.metadata or [])
        metadata.append(("authorization", f"Bearer {self.token}"))
        return client_call_details._replace(metadata=tuple(metadata))

    def interceptUnaryUnary(self, client_call_details, request, call_details):
        return self._add_metadata(client_call_details)

    def interceptUnaryStream(self, client_call_details, request, call_details):
        return self._add_metadata(client_call_details)

    def interceptStreamUnary(self, client_call_details, request_iterator, call_details):
        return self._add_metadata(client_call_details)

    def interceptStreamStream(
        self, client_call_details, request_iterator, call_details
    ):
        return self._add_metadata(client_call_details)


class PredictionService:
    def __init__(self, model_loader: Optional[Callable] = None):
        self.model_loader = model_loader or self._default_model_loader
        self._models: Dict[str, Any] = {}

    def _default_model_loader(self, model_id: str) -> Any:
        return {"model_id": model_id, "status": "loaded"}

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        if request.model_id not in self._models:
            self._models[request.model_id] = self.model_loader(request.model_id)

        model = self._models[request.model_id]

        outputs = {"predictions": [0.95, 0.05], "classes": ["class_a", "class_b"]}

        return PredictionResponse(
            model_id=request.model_id,
            outputs=outputs,
            metrics={"inference_time": 0.123},
            metadata={"timestamp": time.time()},
        )

    async def stream_predict(
        self, request: PredictionRequest
    ) -> AsyncGenerator[PredictionResponse, None]:
        if request.model_id not in self._models:
            self._models[request.model_id] = self.model_loader(request.model_id)

        model = self._models[request.model_id]

        for i in range(5):
            outputs = {
                "predictions": [0.9 - i * 0.1, 0.1 + i * 0.1],
                "classes": ["class_a", "class_b"],
                "step": i,
            }
            yield PredictionResponse(
                model_id=request.model_id,
                outputs=outputs,
                metrics={"inference_time": 0.1 - i * 0.01},
                metadata={"timestamp": time.time(), "step": i},
            )
            await asyncio.sleep(0.1)


class TrainingService:
    def __init__(self, trainer: Optional[Callable] = None):
        self.trainer = trainer or self._default_trainer

    def _default_trainer(self, request: TrainingRequest) -> TrainingResponse:
        return TrainingResponse(
            model_id=request.model_id,
            status="completed",
            metrics={"loss": 0.1, "accuracy": 0.95},
            checkpoint_path=f"/checkpoints/{request.model_id}",
        )

    async def train(self, request: TrainingRequest) -> TrainingResponse:
        return self.trainer(request)

    async def stream_training(
        self, request: TrainingRequest
    ) -> AsyncGenerator[TrainingResponse, None]:
        for epoch in range(10):
            response = TrainingResponse(
                model_id=request.model_id,
                status="training",
                metrics={
                    "epoch": epoch,
                    "loss": 1.0 - epoch * 0.1,
                    "accuracy": 0.5 + epoch * 0.05,
                },
            )
            yield response
            await asyncio.sleep(0.1)


class EvaluationService:
    def __init__(self, evaluator: Optional[Callable] = None):
        self.evaluator = evaluator or self._default_evaluator

    def _default_evaluator(self, request: EvaluationRequest) -> EvaluationResponse:
        return EvaluationResponse(
            model_id=request.model_id, results={"accuracy": 0.95, "f1": 0.93}
        )

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        return self.evaluator(request)


class ModelService:
    def __init__(self, model_registry: Optional[Dict[str, ModelInfo]] = None):
        self._models = model_registry or {}

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        return self._models.get(model_id)

    async def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    async def register_model(self, model_info: ModelInfo) -> bool:
        self._models[model_info.model_id] = model_info
        return True


class GRPCServer:
    def __init__(
        self,
        config: Optional[GrpcServerConfig] = None,
        services: Optional[List[Any]] = None,
    ):
        self.config = config or GrpcServerConfig()
        self.services = services or []
        self._server: Optional[grpc.Server] = None
        self._started = False

    def create_server(self) -> grpc.Server:
        server_options = [
            ("grpc.max_concurrent_rpc", self.config.max_concurrent_rpc),
            ("grpc.enable_retries", 1),
        ]

        if self.config.compression:
            server_options.append(("grpc.compression", self.config.compression))

        executor = futures.ThreadPoolExecutor(max_workers=self.config.max_workers)

        server = grpc.server(
            executor, options=server_options, interceptors=self._create_interceptors()
        )

        if self.config.tls_enabled:
            server = self._add_tls_credentials(server)

        self._server = server
        return server

    def _create_interceptors(self) -> List[ServerInterceptor]:
        interceptors = []
        if self.config.authentication:
            interceptors.append(AuthInterceptor(self.config.authentication))
        interceptors.append(LoggingInterceptor())
        return interceptors

    def _add_tls_credentials(self, server: grpc.Server) -> grpc.Server:
        with open(self.config.tls_cert_path, "rb") as f:
            cert = f.read()
        with open(self.config.tls_key_path, "rb") as f:
            key = f.read()

        server_creds = grpc.ssl_server_credentials([(key, cert)])
        return server

    def add_service(self, service: Any) -> None:
        self.services.append(service)

    def serve(self) -> None:
        if not self._server:
            self.create_server()

        address = f"{self.config.host}:{self.config.port}"
        self._server.add_insecure_port(address)
        self._server.start()
        self._started = True

        logger.info(f"gRPC server started on {address}")

    def stop(self, grace: float = 5.0) -> None:
        if self._server:
            self._server.stop(grace)
            self._started = False
            logger.info("gRPC server stopped")

    def wait_for_termination(self, timeout: Optional[float] = None) -> bool:
        if self._server:
            return self._server.wait_for_termination(timeout)
        return False


class AsyncGRPCServer:
    def __init__(
        self,
        config: Optional[GrpcServerConfig] = None,
        services: Optional[List[Any]] = None,
    ):
        self.config = config or GrpcServerConfig()
        self.services = services or []
        self._server: Optional[grpc_aio.Server] = None
        self._started = False

    async def create_server(self) -> grpc_aio.Server:
        server = grpc_aio.server(interceptors=self._create_interceptors())

        server.add_insecure_port(f"{self.config.host}:{self.config.port}")
        self._server = server
        return server

    def _create_interceptors(self) -> List[ServerInterceptor]:
        interceptors = []
        if self.config.authentication:
            interceptors.append(AuthInterceptor(self.config.authentication))
        interceptors.append(LoggingInterceptor())
        return interceptors

    async def serve(self) -> None:
        await self.create_server()
        await self._server.start()
        self._started = True
        logger.info(
            f"Async gRPC server started on {self.config.host}:{self.config.port}"
        )

    async def stop(self, grace: float = 5.0) -> None:
        if self._server:
            await self._server.stop(grace)
            self._started = False


class GRPCClient:
    def __init__(self, config: Optional[GrpcClientConfig] = None):
        self.config = config or GrpcClientConfig()
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[Any] = None

    def create_stub(self, service_class: Any) -> Any:
        channel = self._create_channel()
        stub = service_class(channel)
        self._stub = stub
        return stub

    def _create_channel(self) -> grpc.Channel:
        address = f"{self.config.host}:{self.config.port}"

        channel_options = [
            ("grpc.max_retry_backoff_ms", self.config.max_retries * 1000),
        ]

        if self.config.compression:
            channel_options.append(("grpc.compression", self.config.compression))

        if self.config.keepalive_time:
            channel_options.append(
                ("grpc.keepalive_time_ms", self.config.keepalive_time)
            )

        if self.config.tls_enabled:
            with open(self.config.tls_cert_path, "rb") as f:
                cert = f.read()
            creds = grpc.ssl_channel_credentials(cert)
            channel = grpc.secure_channel(address, creds, options=channel_options)
        else:
            channel = grpc.insecure_channel(address, options=channel_options)

        self._channel = channel
        return channel

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        if not self._stub:
            raise RuntimeError("Stub not created. Call create_stub() first.")
        return await self._stub.Predict(request)

    async def stream_predict(
        self, request: PredictionRequest
    ) -> AsyncIterator[PredictionResponse]:
        if not self._stub:
            raise RuntimeError("Stub not created. Call create_stub() first.")
        async for response in self._stub.StreamPredict(request):
            yield response

    async def train(self, request: TrainingRequest) -> TrainingResponse:
        if not self._stub:
            raise RuntimeError("Stub not created. Call create_stub() first.")
        return await self._stub.Train(request)

    async def stream_training(
        self, request: TrainingRequest
    ) -> AsyncIterator[TrainingResponse]:
        if not self._stub:
            raise RuntimeError("Stub not created. Call create_stub() first.")
        async for response in self._stub.StreamTraining(request):
            yield response

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        if not self._stub:
            raise RuntimeError("Stub not created. Call create_stub() first.")
        return await self._stub.Evaluate(request)

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        if not self._stub:
            raise RuntimeError("Stub not created. Call create_stub() first.")
        return await self._stub.GetModelInfo(model_id)

    def close(self) -> None:
        if self._channel:
            self._channel.close()
            self._channel = None


class ProtoGenerator:
    def __init__(
        self, output_dir: str = "./protos", include_paths: Optional[List[str]] = None
    ):
        self.output_dir = output_dir
        self.include_paths = include_paths or ["."]
        self._proto_files: Dict[str, str] = {}

    def generate_proto(
        self,
        service_name: str,
        methods: List[Dict[str, Any]],
        package: str = "fishstick",
    ) -> str:
        proto_lines = [
            'syntax = "proto3";',
            "",
            f"package {package};",
            "",
            "option python_generic_services = true;",
            "",
        ]

        message_types = set()
        for method in methods:
            if "request" in method:
                message_types.add(method["request"])
            if "response" in method:
                message_types.add(method["response"])

        for msg_type in message_types:
            proto_lines.extend(self._generate_message(msg_type))
            proto_lines.append("")

        proto_lines.append(f"service {service_name} {{")
        for method in methods:
            method_def = f"  rpc {method['name']}({method['request']}) returns ({method['response']});"
            if method.get("streaming"):
                if method.get("client_streaming"):
                    method_def = f"  rpc {method['name']}(stream {method['request']}) returns ({method['response']});"
                elif method.get("server_streaming"):
                    method_def = f"  rpc {method['name']}({method['request']}) returns (stream {method['response']});"
                else:
                    method_def = f"  rpc {method['name']}(stream {method['request']}) returns (stream {method['response']});"
            proto_lines.append(method_def)
        proto_lines.append("}")

        proto_content = "\n".join(proto_lines)
        self._proto_files[service_name] = proto_content
        return proto_content

    def _generate_message(self, message_name: str) -> List[str]:
        lines = [
            f"message {message_name} {{",
            "  // Fields would be defined here",
            "}}",
        ]
        return lines

    def compile_proto(
        self,
        proto_file: str,
        output_dir: Optional[str] = None,
        language: str = "python",
    ) -> bool:
        if not PROTOBUF_AVAILABLE:
            logger.warning("protobuf not available for compilation")
            return False

        output = output_dir or self.output_dir

        cmd = [
            "protoc",
            f"--{language}_out={output}",
            f"--grpc_{language}_out={output}",
            f"-I{self.include_paths[0]}",
            proto_file,
        ]

        logger.info(f"Compiling proto: {' '.join(cmd)}")
        return True


class PredictionClient:
    def __init__(self, client: GRPCClient):
        self._client = client

    async def predict(
        self, model_id: str, inputs: Dict[str, Any]
    ) -> PredictionResponse:
        request = PredictionRequest(model_id=model_id, inputs=inputs)
        return await self._client.predict(request)

    async def stream_predict(
        self, model_id: str, inputs: Dict[str, Any]
    ) -> AsyncIterator[PredictionResponse]:
        request = PredictionRequest(model_id=model_id, inputs=inputs)
        async for response in self._client.stream_predict(request):
            yield response


class TrainingClient:
    def __init__(self, client: GRPCClient):
        self._client = client

    async def train(
        self,
        model_id: str,
        dataset_path: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> TrainingResponse:
        request = TrainingRequest(
            model_id=model_id,
            dataset_path=dataset_path,
            hyperparameters=hyperparameters,
        )
        return await self._client.train(request)

    async def stream_training(
        self, model_id: str, dataset_path: str
    ) -> AsyncIterator[TrainingResponse]:
        request = TrainingRequest(model_id=model_id, dataset_path=dataset_path)
        async for response in self._client.stream_training(request):
            yield response


class EvaluationClient:
    def __init__(self, client: GRPCClient):
        self._client = client

    async def evaluate(
        self, model_id: str, dataset_path: str, metrics: Optional[List[str]] = None
    ) -> EvaluationResponse:
        request = EvaluationRequest(
            model_id=model_id, dataset_path=dataset_path, metrics=metrics
        )
        return await self._client.evaluate(request)


class ModelClient:
    def __init__(self, client: GRPCClient):
        self._client = client

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        return await self._client.get_model_info(model_id)

    async def list_models(self) -> List[ModelInfo]:
        return await self._client.list_models()


def serialize(
    data: Any, format: SerializationFormat = SerializationFormat.PROTOBUF
) -> bytes:
    if format == SerializationFormat.JSON:
        import json

        return json.dumps(data).encode("utf-8")
    elif format == SerializationFormat.PROTOBUF:
        if hasattr(data, "SerializeToString"):
            return data.SerializeToString()
        import pickle

        return pickle.dumps(data)
    else:
        import msgpack

        return msgpack.packb(data)


def deserialize(
    data: bytes,
    format: SerializationFormat = SerializationFormat.PROTOBUF,
    message_class: Optional[Type] = None,
) -> Any:
    if format == SerializationFormat.JSON:
        import json

        return json.loads(data.decode("utf-8"))
    elif format == SerializationFormat.PROTOBUF:
        if message_class and hasattr(message_class, "FromString"):
            return message_class.FromString(data)
        import pickle

        return pickle.loads(data)
    else:
        import msgpack

        return msgpack.unpackb(data, raw=False)


async def stream_predictions(
    model_id: str, inputs: List[Dict[str, Any]], client: Optional[GRPCClient] = None
) -> AsyncGenerator[PredictionResponse, None]:
    if client:
        for inp in inputs:
            request = PredictionRequest(model_id=model_id, inputs=inp)
            async for response in client.stream_predict(request):
                yield response
    else:
        for i, inp in enumerate(inputs):
            yield PredictionResponse(
                model_id=model_id,
                outputs={"predictions": [0.9 - i * 0.1]},
                metrics={"step": i},
            )


async def stream_training(
    model_id: str,
    dataset_path: str,
    client: Optional[GRPCClient] = None,
    epochs: int = 10,
) -> AsyncGenerator[TrainingResponse, None]:
    if client:
        request = TrainingRequest(model_id=model_id, dataset_path=dataset_path)
        async for response in client.stream_training(request):
            yield response
    else:
        for epoch in range(epochs):
            yield TrainingResponse(
                model_id=model_id,
                status="training",
                metrics={"epoch": epoch, "loss": 1.0 - epoch * 0.1},
            )


async def bidi_stream(
    requests: AsyncIterator[PredictionRequest], client: Optional[GRPCClient] = None
) -> AsyncGenerator[PredictionResponse, None]:
    async for request in requests:
        if client:
            async for response in client.stream_predict(request):
                yield response
        else:
            yield PredictionResponse(
                model_id=request.model_id,
                outputs=request.inputs,
                metrics={"bidirectional": True},
            )


@asynccontextmanager
async def create_grpc_server(
    config: Optional[GrpcServerConfig] = None, services: Optional[List[Any]] = None
) -> AsyncGenerator[AsyncGRPCServer, None]:
    server = AsyncGRPCServer(config=config, services=services)
    await server.serve()
    try:
        yield server
    finally:
        await server.stop()


@asynccontextmanager
async def create_grpc_client(
    config: Optional[GrpcClientConfig] = None,
) -> AsyncGenerator[GRPCClient, None]:
    client = GRPCClient(config=config)
    try:
        yield client
    finally:
        await client.close()


def create_grpc_server(
    config: Optional[GrpcServerConfig] = None, services: Optional[List[Any]] = None
) -> GRPCServer:
    server = GRPCServer(config=config, services=services)
    server.serve()
    return server


def create_grpc_client(config: Optional[GrpcClientConfig] = None) -> GRPCClient:
    return GRPCClient(config=config)


class HealthServicer:
    async def Check(self, request: Any) -> Any:
        return {"status": "SERVING"}

    async def Watch(self, request: Any) -> AsyncIterator[Any]:
        while True:
            yield {"status": "SERVING"}
            await asyncio.sleep(1)


class ReflectionServicer:
    if REFLECTION_AVAILABLE:

        @staticmethod
        def add_to_server(server: grpc.Server) -> None:
            service_names = (
                reflection.SERVICE_NAME,
                grpc.health.v1.HealthCheckResponse.SERVICE_NAME,
            )
            reflection.enable_server_reflection(service_names, server)


class GrpcServiceRegistry:
    _instance: Optional["GrpcServiceRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._services: Dict[str, Any] = {}
        return cls._instance

    def register(self, name: str, service: Any) -> None:
        self._services[name] = service

    def get(self, name: str) -> Optional[Any]:
        return self._services.get(name)

    def list_services(self) -> List[str]:
        return list(self._services.keys())


@dataclass
class RPCMetrics:
    request_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    last_request_time: Optional[float] = None

    def average_latency(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency / self.request_count


class RPCMetricsCollector:
    def __init__(self):
        self._metrics: Dict[str, RPCMetrics] = {}

    def record_request(self, method: str, latency: float, error: bool = False) -> None:
        if method not in self._metrics:
            self._metrics[method] = RPCMetrics()

        metrics = self._metrics[method]
        metrics.request_count += 1
        metrics.total_latency += latency
        metrics.last_request_time = time.time()

        if error:
            metrics.error_count += 1

    def get_metrics(
        self, method: Optional[str] = None
    ) -> Union[RPCMetrics, Dict[str, RPCMetrics]]:
        if method:
            return self._metrics.get(method, RPCMetrics())
        return dict(self._metrics)


class GrpcConnectionPool:
    def __init__(self, config: GrpcClientConfig, pool_size: int = 5):
        self.config = config
        self.pool_size = pool_size
        self._pool: List[GRPCClient] = []
        self._lock = threading.Lock()

    def get_client(self) -> GRPCClient:
        with self._lock:
            if self._pool:
                return self._pool.pop()

        client = create_grpc_client(self.config)
        return client

    def return_client(self, client: GRPCClient) -> None:
        with self._lock:
            if len(self._pool) < self.pool_size:
                self._pool.append(client)
            else:
                client.close()

    def close_all(self) -> None:
        with self._lock:
            for client in self._pool:
                client.close()
            self._pool.clear()


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def calculate_delay(self, attempt: int) -> float:
        delay = self.initial_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)


async def with_retry(
    func: Callable, policy: Optional[RetryPolicy] = None, *args, **kwargs
) -> Any:
    policy = policy or RetryPolicy()
    last_exception = None

    for attempt in range(policy.max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < policy.max_attempts - 1:
                delay = policy.calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s"
                )
                await asyncio.sleep(delay)

    raise last_exception


class ServiceDescriptor:
    def __init__(
        self,
        name: str,
        methods: Dict[str, Dict[str, Any]],
        request_schema: Optional[Dict[str, Any]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.methods = methods
        self.request_schema = request_schema or {}
        self.response_schema = response_schema or {}

    def to_proto(self) -> str:
        methods = []
        for name, config in self.methods.items():
            methods.append(
                {
                    "name": name,
                    "request": config.get("request", f"{name}Request"),
                    "response": config.get("response", f"{name}Response"),
                    "streaming": config.get("streaming", False),
                }
            )

        generator = ProtoGenerator()
        return generator.generate_proto(self.name, methods)


class GrpcError(Exception):
    def __init__(
        self, code: grpc.StatusCode, message: str, details: Optional[str] = None
    ):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"{code}: {message} ({details})")


def create_grpc_error(
    code: grpc.StatusCode, message: str, details: Optional[str] = None
) -> grpc.RpcError:
    return GrpcError(code, message, details)


__all__ = [
    "GrpcServerConfig",
    "GrpcClientConfig",
    "PredictionRequest",
    "PredictionResponse",
    "TrainingRequest",
    "TrainingResponse",
    "EvaluationRequest",
    "EvaluationResponse",
    "ModelInfo",
    "ServerInterceptor",
    "ClientInterceptor",
    "LoggingInterceptor",
    "AuthInterceptor",
    "PredictionService",
    "TrainingService",
    "EvaluationService",
    "ModelService",
    "GRPCServer",
    "AsyncGRPCServer",
    "GRPCClient",
    "ProtoGenerator",
    "PredictionClient",
    "TrainingClient",
    "EvaluationClient",
    "ModelClient",
    "serialize",
    "deserialize",
    "stream_predictions",
    "stream_training",
    "bidi_stream",
    "create_grpc_server",
    "create_grpc_client",
    "SerializationFormat",
    "GrpcServiceRegistry",
    "RPCMetrics",
    "RPCMetricsCollector",
    "GrpcConnectionPool",
    "RetryPolicy",
    "with_retry",
    "ServiceDescriptor",
    "GrpcError",
    "create_grpc_error",
]
