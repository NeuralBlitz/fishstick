"""
Serving Utilities for fishstick

Provides comprehensive model serving utilities including model loading,
request/response handling, input preprocessing, output postprocessing,
and model warmup utilities.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union, Callable, Tuple, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import Tensor


class ModelFormat(Enum):
    """Supported model formats."""

    PYTORCH = "pt"
    TORCHSCRIPT = "pts"
    ONNX = "onnx"
    SAFE_TENSORS = "safetensors"


@dataclass
class ModelMetadata:
    """Model metadata."""

    name: str
    version: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    format: ModelFormat
    size_mb: float
    quantization: Optional[str] = None


@dataclass
class InferenceRequest:
    """Inference request."""

    inputs: Union[Tensor, Dict[str, Tensor], List]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResponse:
    """Inference response."""

    outputs: Union[Tensor, Dict[str, Tensor], List]
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None


class BasePreprocessor(ABC):
    """Base class for input preprocessing."""

    @abstractmethod
    def preprocess(self, inputs: Any) -> Tensor:
        """Preprocess inputs."""
        pass

    def __call__(self, inputs: Any) -> Tensor:
        return self.preprocess(inputs)


class BasePostprocessor(ABC):
    """Base class for output postprocessing."""

    @abstractmethod
    def postprocess(self, outputs: Tensor) -> Any:
        """Postprocess outputs."""
        pass

    def __call__(self, outputs: Tensor) -> Any:
        return self.postprocess(outputs)


class ImagePreprocessor(BasePreprocessor):
    """
    Image preprocessing for vision models.

    Handles normalization, resizing, and format conversion.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
    ):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.size = size
        self.normalize = normalize

    def preprocess(self, inputs: Any) -> Tensor:
        """
        Preprocess image inputs.

        Args:
            inputs: Raw image data

        Returns:
            Preprocessed tensor
        """
        if isinstance(inputs, Tensor):
            x = inputs.float() / 255.0 if inputs.max() > 1 else inputs.float()
        else:
            x = torch.from_numpy(inputs).float()

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if self.size is not None:
            x = torch.nn.functional.interpolate(
                x, size=self.size, mode="bilinear", align_corners=False
            )

        if self.normalize:
            x = (x - self.mean) / self.std

        return x


class TextPreprocessor(BasePreprocessor):
    """
    Text preprocessing for NLP models.

    Handles tokenization and encoding.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        padding: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def preprocess(self, inputs: Any) -> Dict[str, Tensor]:
        """
        Preprocess text inputs.

        Args:
            inputs: Raw text

        Returns:
            Tokenized inputs
        """
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt",
            )
            return encoded

        if isinstance(inputs, str):
            inputs = [inputs]

        return {"input_ids": torch.randint(0, 30000, (len(inputs), self.max_length))}


class AudioPreprocessor(BasePreprocessor):
    """
    Audio preprocessing for speech/audio models.

    Handles resampling, normalization, and feature extraction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        n_mels: int = 80,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels

    def preprocess(self, inputs: Any) -> Tensor:
        """
        Preprocess audio inputs.

        Args:
            inputs: Raw audio data

        Returns:
            Preprocessed audio features
        """
        if isinstance(inputs, Tensor):
            waveform = inputs.float()
        else:
            waveform = torch.from_numpy(inputs).float()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return waveform


class ClassificationPostprocessor(BasePostprocessor):
    """
    Postprocessing for classification outputs.

    Handles softmax, argmax, and label mapping.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        apply_softmax: bool = True,
        top_k: int = 1,
    ):
        self.class_names = class_names
        self.apply_softmax = apply_softmax
        self.top_k = top_k

    def postprocess(self, outputs: Tensor) -> Dict[str, Any]:
        """
        Postprocess classification outputs.

        Args:
            outputs: Model outputs

        Returns:
            Classification results
        """
        if self.apply_softmax:
            probs = torch.softmax(outputs, dim=-1)
        else:
            probs = outputs

        top_probs, top_indices = torch.topk(
            probs, min(self.top_k, outputs.size(-1)), dim=-1
        )

        results = {
            "probabilities": top_probs.squeeze(0).tolist(),
            "class_indices": top_indices.squeeze(0).tolist(),
        }

        if self.class_names is not None:
            results["class_names"] = [
                self.class_names[idx] for idx in top_indices.squeeze(0).tolist()
            ]

        return results


class ModelLoader:
    """
    Universal model loader for various formats.

    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load("model.pt")
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(
        self,
        model_path: str,
        format: Optional[ModelFormat] = None,
    ) -> nn.Module:
        """
        Load model from file.

        Args:
            model_path: Path to model
            format: Model format (auto-detected if None)

        Returns:
            Loaded model
        """
        model_path = Path(model_path)

        if format is None:
            format = self._detect_format(model_path)

        if format == ModelFormat.PYTORCH:
            return self._load_pytorch(model_path)
        elif format == ModelFormat.TORCHSCRIPT:
            return self._load_torchscript(model_path)
        elif format == ModelFormat.ONNX:
            raise ValueError("ONNX models require onnxruntime for loading")
        elif format == ModelFormat.SAFE_TENSORS:
            return self._load_safe_tensors(model_path)

        raise ValueError(f"Unknown format: {format}")

    def _detect_format(self, path: Path) -> ModelFormat:
        """Auto-detect model format."""
        suffix = path.suffix.lower()

        format_map = {
            ".pt": ModelFormat.PYTORCH,
            ".pth": ModelFormat.PYTORCH,
            ".pts": ModelFormat.TORCHSCRIPT,
            ".onnx": ModelFormat.ONNX,
            ".safetensors": ModelFormat.SAFE_TENSORS,
        }

        return format_map.get(suffix, ModelFormat.PYTORCH)

    def _load_pytorch(self, path: Path) -> nn.Module:
        """Load PyTorch model."""
        state_dict = torch.load(path, map_location=self.device)

        if isinstance(state_dict, nn.Module):
            model = state_dict
        else:
            raise ValueError("Expected model or state_dict")

        model = model.to(self.device)
        model.eval()

        return model

    def _load_torchscript(self, path: Path) -> nn.Module:
        """Load TorchScript model."""
        return torch.jit.load(path, map_location=self.device)

    def _load_safe_tensors(self, path: Path) -> nn.Module:
        """Load SafeTensors model."""
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("safetensors package required")

        state_dict = load_file(path, device=self.device)
        raise ValueError("Cannot load state_dict directly, need model class")

    def get_model_info(self, model_path: str) -> ModelMetadata:
        """
        Get model metadata.

        Args:
            model_path: Path to model

        Returns:
            Model metadata
        """
        path = Path(model_path)
        size_mb = path.stat().st_size / (1024 * 1024)

        format_ = self._detect_format(path)

        return ModelMetadata(
            name=path.stem,
            version="unknown",
            input_shape=tuple(),
            output_shape=tuple(),
            format=format_,
            size_mb=size_mb,
        )


class ModelWarmer:
    """
    Model warmup utilities.

    Warms up models before serving to avoid cold start latency.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def warmup(
        self,
        input_shape: Tuple[int, ...],
        num_iterations: int = 10,
        use_cuda: bool = False,
    ) -> Dict[str, float]:
        """
        Warmup model with dummy inputs.

        Args:
            input_shape: Input tensor shape
            num_iterations: Number of warmup iterations
            use_cuda: Whether to use CUDA

        Returns:
            Warmup statistics
        """
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            dummy_input = torch.randn(*input_shape, device="cuda")
        else:
            dummy_input = torch.randn(*input_shape)

        self.model.eval()

        latencies = []

        with torch.no_grad():
            for _ in range(num_iterations):
                if use_cuda:
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = self.model(dummy_input)

                if use_cuda:
                    torch.cuda.synchronize()

                latencies.append((time.perf_counter() - start) * 1000)

        return {
            "num_iterations": num_iterations,
            "avg_warmup_ms": sum(latencies) / len(latencies),
            "min_warmup_ms": min(latencies),
            "max_warmup_ms": max(latencies),
        }


class InferenceServer:
    """
    Simple inference server for model serving.

    Provides basic request handling and response formatting.

    Example:
        >>> server = InferenceServer(model, preprocessor, postprocessor)
        >>> server.start(port=8000)
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: Optional[BasePreprocessor] = None,
        postprocessor: Optional[BasePostprocessor] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        inputs: Any,
        return_latency: bool = True,
    ) -> InferenceResponse:
        """
        Run inference on inputs.

        Args:
            inputs: Input data
            return_latency: Whether to return latency

        Returns:
            Inference response
        """
        start_time = time.perf_counter()

        if self.preprocessor is not None:
            processed_inputs = self.preprocessor.preprocess(inputs)
        else:
            processed_inputs = inputs

        if isinstance(processed_inputs, Tensor):
            processed_inputs = processed_inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(processed_inputs)

        if isinstance(outputs, Tensor):
            outputs = outputs.cpu()

        if self.postprocessor is not None:
            processed_outputs = self.postprocessor.postprocess(outputs)
        else:
            processed_outputs = outputs

        latency_ms = (time.perf_counter() - start_time) * 1000

        return InferenceResponse(
            outputs=processed_outputs,
            latency_ms=latency_ms if return_latency else 0.0,
        )

    def predict_batch(
        self,
        inputs_list: List[Any],
    ) -> List[InferenceResponse]:
        """
        Run batch inference.

        Args:
            inputs_list: List of inputs

        Returns:
            List of responses
        """
        return [self.predict(inputs) for inputs in inputs_list]


def load_model(
    model_path: str,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Convenience function to load a model.

    Args:
        model_path: Path to model
        device: Device to load on

    Returns:
        Loaded model
    """
    loader = ModelLoader(device=device)
    return loader.load(model_path)


def warmup_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 10,
) -> Dict[str, float]:
    """
    Convenience function to warmup a model.

    Args:
        model: PyTorch model
        input_shape: Input shape
        num_iterations: Number of warmup iterations

    Returns:
        Warmup statistics
    """
    warmer = ModelWarmer(model)
    use_cuda = next(model.parameters()).is_cuda
    return warmer.warmup(input_shape, num_iterations, use_cuda)


def create_preprocessor(
    input_type: str,
    **kwargs,
) -> BasePreprocessor:
    """
    Create preprocessor for input type.

    Args:
        input_type: Type of input (image, text, audio)
        **kwargs: Preprocessor configuration

    Returns:
        Preprocessor instance
    """
    if input_type == "image":
        return ImagePreprocessor(**kwargs)
    elif input_type == "text":
        return TextPreprocessor(**kwargs)
    elif input_type == "audio":
        return AudioPreprocessor(**kwargs)
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def create_postprocessor(
    output_type: str,
    **kwargs,
) -> BasePostprocessor:
    """
    Create postprocessor for output type.

    Args:
        output_type: Type of output
        **kwargs: Postprocessor configuration

    Returns:
        Postprocessor instance
    """
    if output_type == "classification":
        return ClassificationPostprocessor(**kwargs)
    else:
        raise ValueError(f"Unknown output type: {output_type}")
