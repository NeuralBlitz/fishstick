from typing import Optional, Dict, Any, List, Tuple, Callable, Union, Literal
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.jit import ScriptModule, script_method, ScriptFunction


class TorchScriptCompiler:
    @staticmethod
    def trace(
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
        optimize_for_inference: bool = True,
        strict: bool = True,
    ) -> ScriptModule:
        if isinstance(example_inputs, Tensor):
            example_inputs = (example_inputs,)

        traced = torch.jit.trace(model, example_inputs, strict=strict)

        if optimize_for_inference:
            traced = torch.jit.optimize_for_inference(traced)

        return traced

    @staticmethod
    def script(
        model: nn.Module,
        optimize_for_inference: bool = True,
    ) -> ScriptModule:
        scripted = torch.jit.script(model)

        if optimize_for_inference:
            scripted = torch.jit.optimize_for_inference(scripted)

        return scripted

    @staticmethod
    def freeze(model: ScriptModule) -> ScriptModule:
        return torch.jit.freeze(model)

    @staticmethod
    def serialize(
        model: ScriptModule,
        path: Union[str, Path],
        use_fake_quant: bool = False,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if use_fake_quant:
            from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx

            model = convert_fx(model)

        torch.jit.save(model, str(path))

    @staticmethod
    def deserialize(path: Union[str, Path]) -> ScriptModule:
        return torch.jit.load(str(path))


class ONNXExporter:
    def __init__(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        self.model = model
        self.example_inputs = example_inputs
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]
        self.dynamic_axes = dynamic_axes or {}

    def export(
        self,
        path: Union[str, Path],
        opset_version: int = 14,
        verbose: bool = False,
        do_constant_folding: bool = True,
        training: Literal["training", "eval"] = "eval",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        torch.onnx.export(
            self.model,
            self.example_inputs,
            str(path),
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            opset_version=opset_version,
            verbose=verbose,
            do_constant_folding=do_constant_folding,
            training=training,
        )

    def validate(self, path: Union[str, Path]) -> bool:
        try:
            import onnx

            onnx.load(str(path))
            return True
        except Exception:
            return False


class TensorRTCompiler:
    def __init__(
        self,
        onnx_path: Union[str, Path],
        fp16: bool = False,
        int8: bool = False,
        workspace_size: int = 1 << 30,
        max_batch_size: int = 32,
    ):
        self.onnx_path = Path(onnx_path)
        self.fp16 = fp16
        self.int8 = int8
        self.workspace_size = workspace_size
        self.max_batch_size = max_batch_size
        self._engine = None

    def build_engine(self) -> Any:
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)

            with open(self.onnx_path, "rb") as f:
                parser.parse(f.read())

            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.workspace_size
            )

            if self.fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if self.int8:
                config.set_flag(trt.BuilderFlag.INT8)

            self._engine = builder.build_serialized_network(network, config)

            return self._engine

        except ImportError:
            raise ImportError(
                "TensorRT not installed. Install with: pip install tensorrt"
            )

    def save_engine(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._engine is None:
            self.build_engine()

        with open(path, "wb") as f:
            f.write(self._engine)

    def load_engine(self, path: Union[str, Path]) -> Any:
        import tensorrt as trt

        path = Path(path)
        with open(path, "rb") as f:
            self._engine = f.read()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        return runtime.deserialize_cuda_engine(self._engine)


@dataclass
class CompilationResult:
    model: nn.Module
    compilation_time: float
    method: str
    output_path: Optional[Path]
    file_size: Optional[int]
    inference_time_ms: Optional[float]


class ModelCompiler:
    SUPPORTED_METHODS = ["torchscript", "onnx", "tensorrt", "trace", "script"]

    def __init__(self, model: nn.Module):
        self.model = model
        self._compiled = {}

    def compile(
        self,
        method: str,
        output_path: Optional[Union[str, Path]] = None,
        example_inputs: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
        **kwargs,
    ) -> CompilationResult:
        method = method.lower()

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: {method}. Supported: {self.SUPPORTED_METHODS}"
            )

        import time

        start_time = time.perf_counter()

        compiled_model = None
        file_size = None
        output_path = Path(output_path) if output_path else None

        if method == "torchscript":
            if example_inputs is None:
                raise ValueError("example_inputs required for torchscript")
            compiled_model = TorchScriptCompiler.trace(self.model, example_inputs)

        elif method == "trace":
            if example_inputs is None:
                raise ValueError("example_inputs required for trace")
            compiled_model = TorchScriptCompiler.trace(self.model, example_inputs)

        elif method == "script":
            compiled_model = TorchScriptCompiler.script(self.model)

        elif method == "onnx":
            if example_inputs is None:
                raise ValueError("example_inputs required for onnx")

            exporter = ONNXExporter(
                self.model,
                example_inputs,
                input_names=kwargs.get("input_names"),
                output_names=kwargs.get("output_names"),
                dynamic_axes=kwargs.get("dynamic_axes"),
            )

            if output_path:
                exporter.export(
                    output_path,
                    opset_version=kwargs.get("opset_version", 14),
                )
                file_size = output_path.stat().st_size

            compiled_model = self.model

        elif method == "tensorrt":
            if example_inputs is None:
                raise ValueError("example_inputs required for tensorrt")

            onnx_path = (
                output_path.with_suffix(".onnx") if output_path else Path("model.onnx")
            )

            exporter = ONNXExporter(
                self.model,
                example_inputs,
                input_names=kwargs.get("input_names"),
                output_names=kwargs.get("output_names"),
                dynamic_axes=kwargs.get("dynamic_axes"),
            )
            exporter.export(onnx_path)

            trt_compiler = TensorRTCompiler(
                onnx_path,
                fp16=kwargs.get("fp16", False),
                int8=kwargs.get("int8", False),
            )

            if output_path:
                trt_compiler.save_engine(output_path)
                file_size = output_path.stat().st_size

            compiled_model = self.model

        compilation_time = time.perf_counter() - start_time

        if output_path and method in ["torchscript", "trace", "script"]:
            TorchScriptCompiler.serialize(compiled_model, output_path)
            file_size = output_path.stat().st_size

        self._compiled[method] = compiled_model

        return CompilationResult(
            model=compiled_model,
            compilation_time=compilation_time,
            method=method,
            output_path=output_path,
            file_size=file_size,
            inference_time_ms=None,
        )

    def benchmark(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
        warmup: int = 10,
        runs: int = 100,
    ) -> float:
        if isinstance(example_inputs, Tensor):
            example_inputs = (example_inputs,)

        model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(*example_inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            times = []
            for _ in range(runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                import time

                start = time.perf_counter()

                _ = model(*example_inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)

        return sum(times) / len(times)


def compile_model(
    model: nn.Module,
    method: str,
    output_path: Optional[Union[str, Path]] = None,
    example_inputs: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
    **kwargs,
) -> CompilationResult:
    compiler = ModelCompiler(model)
    return compiler.compile(method, output_path, example_inputs, **kwargs)


def optimize_for_inference(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    for module in model.modules():
        if hasattr(module, "inference_mode"):
            module.inference_mode()
        elif hasattr(module, "eval"):
            module.eval()

    if hasattr(torch.jit, "optimize_for_inference"):
        try:
            model = torch.jit.optimize_for_inference(model)
        except Exception:
            pass

    return model


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        "param_size_mb": param_size / (1024**2),
        "buffer_size_mb": buffer_size / (1024**2),
        "total_size_mb": total_size / (1024**2),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_buffers": sum(b.numel() for b in model.buffers()),
    }


def fuse_modules(model: nn.Module) -> nn.Module:
    from torch.nn.utils import fuse_conv_bn_eval

    for m in model.modules():
        if isinstance(m, nn.Sequential):
            modules = list(m.children())
            for i in range(len(modules) - 1):
                if isinstance(modules[i], nn.Conv2d) and isinstance(
                    modules[i + 1], nn.BatchNorm2d
                ):
                    try:
                        fuse_conv_bn_eval(modules[i], modules[i + 1])
                    except Exception:
                        pass

    return model
