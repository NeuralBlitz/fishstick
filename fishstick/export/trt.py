"""
TensorRT Conversion Utilities
"""

from typing import Dict, Any, Optional
import torch
from torch import nn


def convert_to_tensorrt(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    max_batch_size: int = 1,
) -> None:
    """
    Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT engine
        fp16: Whether to use FP16 precision
        max_batch_size: Maximum batch size
    """
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            (1, 3, 224, 224),
            (max_batch_size, 3, 224, 224),
            (max_batch_size, 3, 224, 224),
        )
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)

        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        print(f"TensorRT engine saved to {output_path}")

    except ImportError:
        print("TensorRT not installed. Skipping conversion.")


def infer_tensorrt(
    engine_path: str,
    input_data: torch.Tensor,
) -> torch.Tensor:
    """
    Run inference with TensorRT engine.

    Args:
        engine_path: Path to TensorRT engine
        input_data: Input tensor

    Returns:
        Output tensor
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Allocate device memory
        d_input = cuda.mem_alloc(input_data.numpy().nbytes)
        d_output = cuda.mem_alloc(input_data.numpy().nbytes)  # Assumes same size

        # Copy input to device
        cuda.memcpy_htod(d_input, input_data.numpy())

        # Run inference
        context.execute_v2([int(d_input), int(d_output)])

        # Copy output to host
        output = torch.empty_like(input_data)
        cuda.memcpy_dtoh(output.numpy(), d_output)

        return output

    except ImportError:
        print("TensorRT or pycuda not installed.")
        return input_data
