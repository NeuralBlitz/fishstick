"""
TensorFlow Conversion Utilities
"""

from typing import Dict, Any, Optional
import torch
from torch import nn


def convert_to_tensorflow(
    model: nn.Module,
    input_shape: tuple,
    output_path: str,
) -> None:
    """
    Convert PyTorch model to TensorFlow SavedModel format.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        output_path: Path to save TensorFlow model
    """
    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare

        import tempfile

        temp_onnx = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)

        dummy_input = torch.randn(1, *input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            temp_onnx.name,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

        onnx_model = onnx.load(temp_onnx.name)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(output_path)

        print(f"Model exported to TensorFlow: {output_path}")

    except ImportError as e:
        print(f"Missing dependencies: {e}")


def convert_to_tflite(
    saved_model_path: str,
    output_path: str,
    quantization: str = "none",
) -> None:
    """
    Convert TensorFlow SavedModel to TFLite.

    Args:
        saved_model_path: Path to SavedModel
        output_path: Path to save TFLite model
        quantization: Quantization type ("none", "dynamic", "float16")
    """
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

        if quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        print(f"TFLite model saved to {output_path}")

    except ImportError:
        print("TensorFlow not installed. Skipping TFLite conversion.")


def compare_pytorch_tf(
    pytorch_model: nn.Module,
    tf_model_path: str,
    input_shape: tuple,
    num_samples: int = 10,
) -> Dict[str, Any]:
    """
    Compare PyTorch and TensorFlow model outputs.

    Args:
        pytorch_model: PyTorch model
        tf_model_path: Path to TensorFlow SavedModel
        input_shape: Input shape
        num_samples: Number of test samples

    Returns:
        Comparison results
    """
    try:
        import tensorflow as tf
        import numpy as np

        pytorch_model.eval()
        tf_model = tf.saved_model.load(tf_model_path)

        differences = []

        for _ in range(num_samples):
            dummy_input = torch.randn(1, *input_shape)

            with torch.no_grad():
                pytorch_out = pytorch_model(dummy_input).numpy()

            tf_out = tf_model(dummy_input.numpy())["output"].numpy()

            diff = np.abs(pytorch_out - tf_out).mean()
            differences.append(diff)

        return {
            "mean_difference": np.mean(differences),
            "max_difference": np.max(differences),
            "success": True,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
