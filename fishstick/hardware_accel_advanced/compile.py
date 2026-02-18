from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
import torch
import torch.nn as nn
from pathlib import Path


class CompilationBackend(Enum):
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    DYNAMO = "dynamo"
    INDUCTOR = "inductor"


@dataclass
class CompilationConfig:
    backend: CompilationBackend = CompilationBackend.DYNAMO
    mode: str = "default"
    fullgraph: bool = True
    dynamic: bool = False
    backend: str = "inductor"
    tracing_mode: str = "symbolic"
    optimization_level: int = 3
    enable_profiling: bool = False


class TorchScriptCompiler:
    def __init__(self, optimize: bool = True):
        self.optimize = optimize

    def compile(
        self,
        model: nn.Module,
        example_inputs: Any,
        method_name: str = "forward",
    ) -> torch.jit.ScriptModule:
        model.eval()
        if isinstance(example_inputs, Tensor):
            example_inputs = (example_inputs,)

        if self.optimize:
            traced = torch.jit.trace(model, example_inputs)
            scripted = traced
        else:
            scripted = torch.jit.script(model, example_inputs)

        if self.optimize:
            scripted = torch.jit.optimize_for_inference(scripted)

        return scripted

    def save(self, model: torch.jit.ScriptModule, path: str) -> None:
        model.save(path)

    def load(self, path: str) -> torch.jit.ScriptModule:
        return torch.jit.load(path)


class ONNXExporter:
    def __init__(
        self,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]
        self.dynamic_axes = dynamic_axes or {}

    def export(
        self,
        model: nn.Module,
        example_inputs: Any,
        output_path: str,
        opset_version: int = 14,
        **kwargs: Any,
    ) -> None:
        model.eval()
        if isinstance(example_inputs, (list, tuple)):
            pass
        else:
            example_inputs = (example_inputs,)

        torch.onnx.export(
            model,
            example_inputs,
            output_path,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            opset_version=opset_version,
            **kwargs,
        )

    def export_with_runtime(
        self,
        model: nn.Module,
        example_inputs: Any,
        output_path: str,
    ) -> Any:
        import onnxruntime as ort

        self.export(model, example_inputs, output_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(output_path, sess_options)
        return session


class TensorRTCompiler:
    def __init__(
        self,
        max_workspace_size: int = 1 << 30,
        fp16: bool = False,
        int8: bool = False,
        strict: bool = False,
    ):
        self.max_workspace_size = max_workspace_size
        self.fp16 = fp16
        self.int8 = int8
        self.strict = strict

    def compile(
        self,
        model: nn.Module,
        example_inputs: Any,
        min_input_shape: Optional[Tuple[int, ...]] = None,
        opt_input_shape: Optional[Tuple[int, ...]] = None,
        max_input_shape: Optional[Tuple[int, ...]] = None,
    ) -> Any:
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            config = builder.create_builder_config()

            config.max_workspace_size = self.max_workspace_size
            if self.fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if self.int8:
                config.set_flag(trt.BuilderFlag.INT8)

            if isinstance(example_inputs, (list, tuple)):
                example_inputs = example_inputs[0]

            profile = builder.create_optimization_profile()
            input_shape = tuple(example_inputs.shape)
            profile.set_shape("input", input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)

            parser = trt.OnnxParser(network, logger)
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                onnx_path = f.name

            torch.onnx.export(model, example_inputs, onnx_path, verbose=False)
            with open(onnx_path, "rb") as f:
                parser.parse(f.read())

            engine = builder.build_serialized_network(network, config)
            if engine is None:
                raise RuntimeError("TensorRT engine build failed")

            runtime = trt.Runtime(logger)
            return runtime.deserialize_cuda_engine(engine)

        except ImportError:
            raise ImportError(
                "TensorRT not installed. Install with: pip install tensorrt"
            )

    def create_runtime(self) -> Any:
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            return trt.Runtime(logger)
        except ImportError:
            raise ImportError("TensorRT not installed")


def compile_model(
    model: nn.Module,
    config: CompilationConfig,
    example_inputs: Any,
) -> nn.Module:
    model.eval()

    if config.backend == CompilationBackend.DYNAMO:
        if hasattr(torch, "compile"):
            model = torch.compile(
                model,
                mode=config.mode,
                fullgraph=config.fullgraph,
                dynamic=config.dynamic,
                backend=config.backend,
            )
        else:
            raise RuntimeError("torch.compile not available in this PyTorch version")

    elif config.backend == CompilationBackend.TORCHSCRIPT:
        compiler = TorchScriptCompiler()
        model = compiler.compile(model, example_inputs)

    elif config.backend == CompilationBackend.ONNX:
        raise ValueError("Use ONNXExporter for ONNX export")

    elif config.backend == CompilationBackend.TENSORRT:
        raise ValueError("Use TensorRTCompiler for TensorRT compilation")

    return model


def optimize_for_inference(
    model: nn.Module,
    example_inputs: Any,
    strategies: Optional[List[str]] = None,
) -> nn.Module:
    if strategies is None:
        strategies = ["jit", "channels_last", "cudnn_benchmark"]

    model.eval()

    for strategy in strategies:
        if strategy == "jit":
            if hasattr(torch.jit, "optimize_for_inference"):
                model = torch.jit.optimize_for_inference(model)

        elif strategy == "channels_last":
            model = model.to(memory_format=torch.channels_last)

        elif strategy == "cudnn_benchmark":
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True

        elif strategy == "amp":
            model = model.half()

        elif strategy == "inference_mode":
            with torch.inference_mode():
                pass

    return model


class ModelOptimizer:
    def __init__(self, model: nn.Module):
        self.model = model

    def optimize(
        self,
        pass_config: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        if pass_config is None:
            pass_config = {}

        for module in self.model.modules():
            if hasattr(module, "inplace"):
                module.inplace = pass_config.get("inplace", True)

        return self.model

    def fold_batchnorm(self) -> nn.Module:
        from torch.nn.utils import fuse_conv_bn_weights

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Conv2d) and hasattr(module, "bn"):
                fused = fuse_conv_bn_weights(
                    module.weight,
                    module.bias,
                    module.bn.running_mean,
                    module.bn.running_var,
                    module.bn.eps,
                    module.bn.weight,
                    module.bn.bias,
                )
                module.weight.data = fused[0]
                module.bias.data = fused[1]
                del module.bn
        return self.model

    def remove_dropout(self) -> nn.Module:
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.p = 0
        return self.model


def export_to_onnx(
    model: nn.Module,
    example_inputs: Any,
    output_path: str,
    **kwargs: Any,
) -> None:
    exporter = ONNXExporter()
    exporter.export(model, example_inputs, output_path, **kwargs)


def trace_and_compile(
    model: nn.Module,
    example_inputs: Any,
    backend: str = "inductor",
    mode: str = "default",
) -> nn.Module:
    if hasattr(torch, "compile"):
        return torch.compile(model, backend=backend, mode=mode)
    else:
        traced = torch.jit.trace(model, example_inputs)
        return torch.jit.freeze(traced)
