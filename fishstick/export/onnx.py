"""
ONNX Export Utilities
"""

from typing import Optional, Dict, Any
import torch
from torch import nn
from pathlib import Path


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: tuple,
    opset_version: int = 14,
    dynamic_axes: Optional[Dict] = None,
    verbose: bool = False,
) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path to save the ONNX model
        input_shape: Input tensor shape (excluding batch)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable dimensions
        verbose: Whether to print verbose output
    """
    model.eval()
    dummy_input = torch.randn(1, *input_shape)

    if dynamic_axes is None:
        dynamic_axes = {}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )
    print(f"Model exported to {output_path}")


def optimize_onnx(
    input_path: str,
    output_path: str,
    level: int = 3,
) -> None:
    """
    Optimize an ONNX model.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
        level: Optimization level (1-3)
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer

        onnx_model = onnx.load(input_path)
        optimized_model = optimizer.optimize_model(
            input_path,
            optimization_level=level,
        )
        onnx.save(optimized_model, output_path)
        print(f"Optimized model saved to {output_path}")
    except ImportError:
        print("onnxruntime not installed. Skipping optimization.")


def validate_onnx(model_path: str) -> Dict[str, Any]:
    """
    Validate an ONNX model.

    Args:
        model_path: Path to ONNX model

    Returns:
        Validation results
    """
    try:
        import onnx

        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        return {
            "valid": True,
            "graph": model.graph.name,
            "inputs": [i.name for i in model.graph.input],
            "outputs": [o.name for o in model.graph.output],
        }
    except ImportError:
        return {"valid": False, "error": "onnx not installed"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def infer_onnx_shape(model_path: str, input_data: torch.Tensor) -> Dict:
    """
    Infer output shapes from an ONNX model.

    Args:
        model_path: Path to ONNX model
        input_data: Sample input tensor

    Returns:
        Output shapes
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_data.numpy()})

        return {"outputs": [o.shape for o in output]}
    except ImportError:
        return {"error": "onnxruntime not installed"}
