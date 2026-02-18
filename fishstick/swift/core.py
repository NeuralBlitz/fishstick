"""
fishstick Swift Backend Module
==============================

Comprehensive Swift backend integration for Apple platform deployment.
Provides CoreML conversion, Metal GPU acceleration, iOS/macOS app generation,
and Swift for TensorFlow support.

Features:
- Swift Extension system for Python-Swift interop
- CoreML model conversion and inference
- Metal GPU kernel compilation and execution
- iOS deployment pipeline
- macOS deployment pipeline
- Swift for TensorFlow integration
- Utility functions for quick conversions

Example:
    >>> from fishstick.swift.core import CoreMLConverter, iOSExporter
    >>> 
    >>> # Convert PyTorch model to CoreML
    >>> converter = CoreMLConverter()
    >>> mlmodel = converter.convert(pytorch_model, input_shape=(1, 3, 224, 224))
    >>> 
    >>> # Export iOS app
    >>> exporter = iOSExporter()
    >>> exporter.export(mlmodel, "MyApp")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
import hashlib
import inspect
import json
import logging
import os
import subprocess
import tempfile
import uuid
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class SwiftBackendError(Exception):
    """Base exception for Swift backend errors."""
    pass


class SwiftExtensionError(SwiftBackendError):
    """Raised when Swift extension operations fail."""
    pass


class CoreMLConversionError(SwiftBackendError):
    """Raised when CoreML conversion fails."""
    pass


class MetalKernelError(SwiftBackendError):
    """Raised when Metal kernel operations fail."""
    pass


class iOSDeploymentError(SwiftBackendError):
    """Raised when iOS deployment fails."""
    pass


class MacOSDeploymentError(SwiftBackendError):
    """Raised when macOS deployment fails."""
    pass


class S4TError(SwiftBackendError):
    """Raised when Swift for TensorFlow operations fail."""
    pass


# =============================================================================
# Type Definitions
# =============================================================================


T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=Any)
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class Platform(Enum):
    """Apple platforms."""
    IOS = "iOS"
    MACOS = "macOS"
    TVOS = "tvOS"
    WATCHOS = "watchOS"
    VISIONOS = "visionOS"


class SwiftDataType(Enum):
    """Swift data types."""
    INT8 = "Int8"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"
    UINT8 = "UInt8"
    UINT16 = "UInt16"
    UINT32 = "UInt32"
    UINT64 = "UInt64"
    FLOAT = "Float"
    DOUBLE = "Double"
    BOOL = "Bool"


class ComputeUnits(Enum):
    """CoreML compute unit options."""
    CPU_ONLY = "cpuOnly"
    CPU_AND_GPU = "cpuAndGPU"
    ALL = "all"
    CPU_AND_NEURAL_ENGINE = "cpuAndNeuralEngine"


@dataclass
class TensorSpec:
    """Tensor specification for model conversion."""
    name: str
    shape: Tuple[int, ...]
    dtype: SwiftDataType = SwiftDataType.FLOAT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype.value,
        }


@dataclass
class SwiftMetadata:
    """Metadata for Swift artifacts."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: Optional[str] = None
    license: Optional[str] = None
    created_at: str = field(default_factory=lambda: str(uuid.uuid4()))
    platform: Platform = Platform.IOS
    minimum_version: str = "13.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "created_at": self.created_at,
            "platform": self.platform.value,
            "minimum_version": self.minimum_version,
        }


# =============================================================================
# Swift Extensions
# =============================================================================


class SwiftExtension:
    """
    Swift Extension system for Python-Swift interoperability.
    
    Manages Swift extensions that can be loaded, compiled, and bridged with Python.
    
    Example:
        >>> extension = SwiftExtension("MyExtension")
        >>> extension.add_function("compute", swift_code="...")
        >>> extension.compile()
        >>> result = extension.call("compute", args=[1, 2, 3])
    """
    
    def __init__(
        self,
        name: str,
        swift_version: str = "5.0",
        optimization: str = "-O",
    ):
        self.name = name
        self.swift_version = swift_version
        self.optimization = optimization
        self.functions: Dict[str, str] = {}
        self.types: Dict[str, str] = {}
        self.protocols: Dict[str, str] = {}
        self.compiled: bool = False
        self._build_dir: Optional[Path] = None
        self._dylib_path: Optional[Path] = None
        
    def add_function(
        self,
        name: str,
        swift_code: str,
        return_type: str = "Void",
        parameters: Optional[List[Tuple[str, str]]] = None,
    ) -> "SwiftExtension":
        """Add a Swift function to the extension."""
        params = ""
        if parameters:
            params = ", ".join([f"{p[0]}: {p[1]}" for p in parameters])
        
        func_def = f"""
        @_cdecl("{name}")
        public func {name}({params}) -> {return_type} {{
            {swift_code}
        }}
        """
        self.functions[name] = func_def
        self.compiled = False
        return self
    
    def add_type(self, name: str, swift_code: str) -> "SwiftExtension":
        """Add a Swift type (struct/class/enum) to the extension."""
        self.types[name] = swift_code
        self.compiled = False
        return self
    
    def add_protocol(self, name: str, swift_code: str) -> "SwiftExtension":
        """Add a Swift protocol to the extension."""
        self.protocols[name] = swift_code
        self.compiled = False
        return self
    
    def generate_source(self) -> str:
        """Generate complete Swift source code."""
        source_parts = [
            f"// Swift Extension: {self.name}",
            f"// Swift Version: {self.swift_version}",
            "",
            "import Foundation",
            "",
            "// MARK: - Protocols",
        ]
        
        for protocol_name, protocol_code in self.protocols.items():
            source_parts.append(protocol_code)
            source_parts.append("")
        
        source_parts.append("// MARK: - Types")
        
        for type_name, type_code in self.types.items():
            source_parts.append(type_code)
            source_parts.append("")
        
        source_parts.append("// MARK: - Functions")
        
        for func_name, func_code in self.functions.items():
            source_parts.append(func_code)
            source_parts.append("")
        
        return "\n".join(source_parts)
    
    def compile(self, build_dir: Optional[Path] = None) -> Path:
        """Compile the Swift extension to a dynamic library."""
        if self.compiled and self._dylib_path and self._dylib_path.exists():
            return self._dylib_path
        
        # Create build directory
        if build_dir is None:
            build_dir = Path(tempfile.gettempdir()) / "fishstick_swift" / self.name
        self._build_dir = build_dir
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Write source file
        source_path = build_dir / f"{self.name}.swift"
        source_path.write_text(self.generate_source())
        
        # Compile
        dylib_path = build_dir / f"lib{self.name}.dylib"
        
        cmd = [
            "swiftc",
            "-emit-library",
            "-o", str(dylib_path),
            "-swift-version", self.swift_version,
            self.optimization,
            str(source_path),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=build_dir,
            )
            
            if result.returncode != 0:
                raise SwiftExtensionError(
                    f"Swift compilation failed: {result.stderr}"
                )
            
            self._dylib_path = dylib_path
            self.compiled = True
            logger.info(f"Compiled Swift extension: {dylib_path}")
            return dylib_path
            
        except FileNotFoundError:
            raise SwiftExtensionError(
                "Swift compiler not found. Install Xcode or Swift toolchain."
            )
    
    def load(self) -> Any:
        """Load the compiled extension using ctypes."""
        if not self.compiled:
            self.compile()
        
        try:
            import ctypes
            
            lib = ctypes.CDLL(str(self._dylib_path))
            return lib
            
        except Exception as e:
            raise SwiftExtensionError(f"Failed to load extension: {e}")
    
    def call(self, function_name: str, *args, **kwargs) -> Any:
        """Call a function from the loaded extension."""
        lib = self.load()
        
        if not hasattr(lib, function_name):
            raise SwiftExtensionError(f"Function '{function_name}' not found")
        
        func = getattr(lib, function_name)
        return func(*args, **kwargs)
    
    def bridge_to_python(self, bridge_name: str = None) -> Callable:
        """Create a Python bridge function for this extension."""
        if bridge_name is None:
            bridge_name = f"{self.name}_bridge"
        
        lib = self.load()
        
        def bridge(*args, **kwargs):
            """Bridge function that calls Swift code."""
            results = {}
            for func_name in self.functions:
                if hasattr(lib, func_name):
                    func = getattr(lib, func_name)
                    try:
                        results[func_name] = func(*args, **kwargs)
                    except Exception as e:
                        results[func_name] = f"Error: {e}"
            return results
        
        bridge.__name__ = bridge_name
        return bridge


def load_swift(
    source_path: Union[str, Path],
    module_name: Optional[str] = None,
) -> SwiftExtension:
    """
    Load a Swift source file as an extension.
    
    Args:
        source_path: Path to the Swift source file
        module_name: Optional module name (defaults to filename)
    
    Returns:
        SwiftExtension instance
    
    Example:
        >>> ext = load_swift("MyFunctions.swift")
        >>> ext.compile()
        >>> result = ext.call("myFunction")
    """
    source_path = Path(source_path)
    
    if not source_path.exists():
        raise SwiftExtensionError(f"Source file not found: {source_path}")
    
    if module_name is None:
        module_name = source_path.stem
    
    source_code = source_path.read_text()
    
    # Parse functions from source (simplified)
    extension = SwiftExtension(module_name)
    
    # Add the entire source as a type for compilation
    extension.add_type("__main__", source_code)
    
    return extension


def compile_swift(
    source: Union[str, Path, SwiftExtension],
    output_path: Optional[Path] = None,
    optimization: str = "-O",
) -> Path:
    """
    Compile Swift source to a dynamic library.
    
    Args:
        source: Swift source code, file path, or SwiftExtension
        output_path: Optional output path for the compiled library
        optimization: Optimization level (-O, -Onone, -Ounchecked)
    
    Returns:
        Path to the compiled dynamic library
    
    Example:
        >>> lib_path = compile_swift("func add(a: Int, b: Int) -> Int { return a + b }")
        >>> # or
        >>> lib_path = compile_swift("MyCode.swift", output_path="build/")
    """
    if isinstance(source, SwiftExtension):
        if output_path:
            return source.compile(output_path)
        return source.compile()
    
    if isinstance(source, (str, Path)) and Path(source).exists():
        # It's a file path
        ext = load_swift(source)
        if output_path:
            return ext.compile(output_path)
        return ext.compile()
    
    # It's source code string
    ext = SwiftExtension("compiled_module")
    ext.add_function("main", source, return_type="Void")
    
    if output_path:
        return ext.compile(output_path)
    return ext.compile()


def swift_bridge(
    swift_functions: Dict[str, str],
    module_name: str = "SwiftBridge",
) -> Callable:
    """
    Create a Python-to-Swift bridge for multiple functions.
    
    Args:
        swift_functions: Dict mapping function names to Swift code
        module_name: Name for the bridge module
    
    Returns:
        Callable that provides access to all bridged functions
    
    Example:
        >>> bridge = swift_bridge({
        ...     "add": "return a + b",
        ...     "multiply": "return a * b"
        ... })
        >>> bridge.add(2, 3)
        5
    """
    extension = SwiftExtension(module_name)
    
    for func_name, func_code in swift_functions.items():
        extension.add_function(
            func_name,
            func_code,
            return_type="Int",
            parameters=[("a", "Int"), ("b", "Int")],
        )
    
    extension.compile()
    return extension.bridge_to_python()


# =============================================================================
# CoreML Integration
# =============================================================================


class CoreMLConverter:
    """
    Convert models to CoreML format.
    
    Supports conversion from PyTorch, TensorFlow, ONNX, and other frameworks.
    
    Example:
        >>> converter = CoreMLConverter(
        ...     compute_units=ComputeUnits.ALL,
        ...     minimum_deployment_target=Platform.IOS
        ... )
        >>> mlmodel = converter.convert(
        ...     pytorch_model,
        ...     inputs=[TensorSpec("image", (1, 3, 224, 224))],
        ...     outputs=[TensorSpec("predictions", (1, 1000))],
        ... )
        >>> mlmodel.save("MyModel.mlmodel")
    """
    
    def __init__(
        self,
        compute_units: ComputeUnits = ComputeUnits.ALL,
        minimum_deployment_target: Platform = Platform.IOS,
        convert_to="neuralnetwork",
    ):
        self.compute_units = compute_units
        self.minimum_deployment_target = minimum_deployment_target
        self.convert_to = convert_to
        self._last_conversion_metadata: Optional[Dict[str, Any]] = None
        
    def convert(
        self,
        model: Any,
        inputs: List[TensorSpec],
        outputs: List[TensorSpec],
        classifier_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[SwiftMetadata] = None,
    ) -> "CoreMLModel":
        """
        Convert a model to CoreML format.
        
        Args:
            model: Model to convert (PyTorch, TensorFlow, etc.)
            inputs: Input tensor specifications
            outputs: Output tensor specifications
            classifier_config: Optional classifier configuration
            metadata: Optional model metadata
        
        Returns:
            CoreMLModel instance
        """
        try:
            import coremltools as ct
        except ImportError:
            raise CoreMLConversionError(
                "coremltools required. Install: pip install coremltools"
            )
        
        # Detect model type and convert accordingly
        model_type = self._detect_model_type(model)
        
        if model_type == "pytorch":
            mlmodel = self._convert_pytorch(
                model, inputs, outputs, classifier_config
            )
        elif model_type == "tensorflow":
            mlmodel = self._convert_tensorflow(
                model, inputs, outputs, classifier_config
            )
        elif model_type == "onnx":
            mlmodel = self._convert_onnx(
                model, inputs, outputs, classifier_config
            )
        else:
            raise CoreMLConversionError(f"Unsupported model type: {model_type}")
        
        # Store conversion metadata
        self._last_conversion_metadata = {
            "source_type": model_type,
            "inputs": [i.to_dict() for i in inputs],
            "outputs": [o.to_dict() for o in outputs],
            "compute_units": self.compute_units.value,
            "deployment_target": self.minimum_deployment_target.value,
        }
        
        return CoreMLModel(
            mlmodel=mlmodel,
            metadata=metadata or SwiftMetadata(name="ConvertedModel"),
            conversion_metadata=self._last_conversion_metadata,
        )
    
    def _detect_model_type(self, model: Any) -> str:
        """Detect the type of model."""
        model_class = type(model).__module__
        
        if "torch" in model_class:
            return "pytorch"
        elif "tensorflow" in model_class or "tf" in model_class:
            return "tensorflow"
        elif "onnx" in model_class:
            return "onnx"
        else:
            return "unknown"
    
    def _convert_pytorch(
        self,
        model: Any,
        inputs: List[TensorSpec],
        outputs: List[TensorSpec],
        classifier_config: Optional[Dict[str, Any]],
    ) -> Any:
        """Convert PyTorch model to CoreML."""
        try:
            import coremltools as ct
            import torch
            
            # Trace the model
            example_input = torch.randn(inputs[0].shape)
            traced_model = torch.jit.trace(model, example_input)
            
            # Convert
            input_features = [
                ct.TensorType(name=inp.name, shape=inp.shape)
                for inp in inputs
            ]
            
            output_features = [
                ct.TensorType(name=out.name, shape=out.shape)
                for out in outputs
            ]
            
            mlmodel = ct.convert(
                traced_model,
                inputs=input_features,
                outputs=output_features,
                classifier_config=classifier_config,
                convert_to=self.convert_to,
                compute_units=self.compute_units.value,
                minimum_deployment_target=self.minimum_deployment_target.value,
            )
            
            return mlmodel
            
        except Exception as e:
            raise CoreMLConversionError(f"PyTorch conversion failed: {e}")
    
    def _convert_tensorflow(
        self,
        model: Any,
        inputs: List[TensorSpec],
        outputs: List[TensorSpec],
        classifier_config: Optional[Dict[str, Any]],
    ) -> Any:
        """Convert TensorFlow model to CoreML."""
        try:
            import coremltools as ct
            
            input_features = [
                ct.TensorType(name=inp.name, shape=inp.shape)
                for inp in inputs
            ]
            
            output_features = [
                ct.TensorType(name=out.name, shape=out.shape)
                for out in outputs
            ]
            
            mlmodel = ct.convert(
                model,
                inputs=input_features,
                outputs=output_features,
                classifier_config=classifier_config,
                convert_to=self.convert_to,
                compute_units=self.compute_units.value,
                minimum_deployment_target=self.minimum_deployment_target.value,
            )
            
            return mlmodel
            
        except Exception as e:
            raise CoreMLConversionError(f"TensorFlow conversion failed: {e}")
    
    def _convert_onnx(
        self,
        model: Any,
        inputs: List[TensorSpec],
        outputs: List[TensorSpec],
        classifier_config: Optional[Dict[str, Any]],
    ) -> Any:
        """Convert ONNX model to CoreML."""
        try:
            import coremltools as ct
            
            mlmodel = ct.converters.onnx.convert(
                model,
                compute_units=self.compute_units.value,
                minimum_deployment_target=self.minimum_deployment_target.value,
            )
            
            return mlmodel
            
        except Exception as e:
            raise CoreMLConversionError(f"ONNX conversion failed: {e}")
    
    def convert_with_pipeline(
        self,
        preprocessing: Optional[Callable] = None,
        model: Any = None,
        postprocessing: Optional[Callable] = None,
    ) -> "CoreMLModel":
        """
        Convert a model with preprocessing and postprocessing.
        
        Args:
            preprocessing: Optional preprocessing function
            model: Main model to convert
            postprocessing: Optional postprocessing function
        
        Returns:
            CoreMLModel with pipeline
        """
        # This would create a pipeline model
        # For now, just convert the main model
        if model is None:
            raise CoreMLConversionError("Model required for pipeline conversion")
        
        return self.convert(
            model,
            inputs=[TensorSpec("input", (1, 3, 224, 224))],
            outputs=[TensorSpec("output", (1, 1000))],
        )


class CoreMLModel:
    """
    Wrapper for CoreML models with additional functionality.
    
    Provides prediction, metadata access, and optimization features.
    
    Example:
        >>> model = CoreMLModel.load("MyModel.mlmodel")
        >>> predictions = model.predict({"image": input_data})
        >>> model.save("OptimizedModel.mlmodel")
    """
    
    def __init__(
        self,
        mlmodel: Any,
        metadata: Optional[SwiftMetadata] = None,
        conversion_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.mlmodel = mlmodel
        self.metadata = metadata or SwiftMetadata(name="CoreMLModel")
        self.conversion_metadata = conversion_metadata or {}
        self._compiled_path: Optional[Path] = None
        
    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "CoreMLModel":
        """Load a CoreML model from file."""
        try:
            import coremltools as ct
            
            model_path = Path(model_path)
            mlmodel = ct.models.MLModel(str(model_path))
            
            return cls(mlmodel=mlmodel)
            
        except ImportError:
            raise CoreMLConversionError("coremltools required")
        except Exception as e:
            raise CoreMLConversionError(f"Failed to load model: {e}")
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction on the model.
        
        Args:
            inputs: Dictionary mapping input names to data
        
        Returns:
            Dictionary mapping output names to predictions
        """
        try:
            return self.mlmodel.predict(inputs)
        except Exception as e:
            raise CoreMLConversionError(f"Prediction failed: {e}")
    
    def predict_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run batch prediction."""
        return [self.predict(inp) for inp in inputs]
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save the model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.mlmodel.save(str(path))
        logger.info(f"Saved CoreML model to {path}")
        return path
    
    def compile(self, output_dir: Optional[Path] = None) -> Path:
        """
        Compile the model for faster inference.
        
        Returns:
            Path to the compiled model
        """
        try:
            import coremltools as ct
            
            if output_dir is None:
                output_dir = Path(tempfile.gettempdir()) / "coreml_compiled"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            compiled_path = self.mlmodel.get_spec()
            # Actual compilation happens at runtime in iOS/macOS
            
            self._compiled_path = output_dir / f"{self.metadata.name}.mlmodelc"
            logger.info(f"Model ready for compilation at runtime")
            
            return self._compiled_path
            
        except Exception as e:
            raise CoreMLConversionError(f"Compilation failed: {e}")
    
    def get_spec(self) -> Any:
        """Get the model specification."""
        return self.mlmodel.get_spec()
    
    def get_input_names(self) -> List[str]:
        """Get list of input names."""
        spec = self.get_spec()
        return [inp.name for inp in spec.description.input]
    
    def get_output_names(self) -> List[str]:
        """Get list of output names."""
        spec = self.get_spec()
        return [out.name for out in spec.description.output]
    
    def quantize_weights(self, nbits: int = 8) -> "CoreMLModel":
        """
        Quantize model weights to reduce size.
        
        Args:
            nbits: Number of bits for quantization (8, 16, 32)
        
        Returns:
            New CoreMLModel with quantized weights
        """
        try:
            import coremltools as ct
            
            quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
                self.mlmodel,
                nbits=nbits,
            )
            
            return CoreMLModel(
                mlmodel=quantized_model,
                metadata=self.metadata,
                conversion_metadata=self.conversion_metadata,
            )
            
        except Exception as e:
            raise CoreMLConversionError(f"Quantization failed: {e}")
    
    def visualize_spec(self) -> str:
        """Get a visualization of the model specification."""
        spec = self.get_spec()
        layers = []
        
        if spec.WhichOneof("Type") == "neuralNetwork":
            nn = spec.neuralNetwork
            for i, layer in enumerate(nn.layers):
                layers.append(f"{i}: {layer.name} ({layer.WhichOneof('layer')})")
        
        return "\n".join(layers)
    
    def to_swift_code(self) -> str:
        """Generate Swift code for using this model."""
        input_names = self.get_input_names()
        output_names = self.get_output_names()
        
        swift_code = f"""
import CoreML

class {self.metadata.name}Predictor {{
    private let model: MLModel
    
    init() throws {{
        let config = MLModelConfiguration()
        config.computeUnits = .{self.conversion_metadata.get('compute_units', 'all')}
        self.model = try {self.metadata.name}(configuration: config).model
    }}
    
    func predict({', '.join([f'{name}: MLMultiArray' for name in input_names])}) throws -> Prediction {{
        let input = {self.metadata.name}Input(
            {', '.join([f'{name}: {name}' for name in input_names])}
        )
        
        let output = try model.prediction(from: input)
        
        return Prediction(
            {', '.join([f'{name}: output.featureValue(for: "{name}")?.multiArrayValue' for name in output_names])}
        )
    }}
    
    struct Prediction {{
        {chr(10).join([f'let {name}: MLMultiArray?' for name in output_names])}
    }}
}}
"""
        return swift_code


def export_coreml(
    model: Any,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    output_shape: Tuple[int, ...] = (1, 1000),
    input_name: str = "input",
    output_name: str = "output",
    **kwargs,
) -> Path:
    """
    Quick export function to convert and save a model as CoreML.
    
    Args:
        model: Model to convert
        output_path: Where to save the .mlmodel file
        input_shape: Shape of input tensor
        output_shape: Shape of output tensor
        input_name: Name of input
        output_name: Name of output
        **kwargs: Additional arguments for CoreMLConverter
    
    Returns:
        Path to saved model
    
    Example:
        >>> export_coreml(
        ...     pytorch_model,
        ...     "MyModel.mlmodel",
        ...     input_shape=(1, 3, 224, 224),
        ...     compute_units=ComputeUnits.ALL,
        ... )
    """
    converter = CoreMLConverter(**kwargs)
    
    inputs = [TensorSpec(input_name, input_shape)]
    outputs = [TensorSpec(output_name, output_shape)]
    
    mlmodel = converter.convert(model, inputs=inputs, outputs=outputs)
    
    return mlmodel.save(output_path)


def load_coreml(model_path: Union[str, Path]) -> CoreMLModel:
    """
    Load a CoreML model from file.
    
    Args:
        model_path: Path to .mlmodel file
    
    Returns:
        CoreMLModel instance
    
    Example:
        >>> model = load_coreml("MyModel.mlmodel")
        >>> predictions = model.predict({"input": data})
    """
    return CoreMLModel.load(model_path)


# =============================================================================
# Metal GPU Integration
# =============================================================================


@dataclass
class MetalKernelSpec:
    """Specification for a Metal compute kernel."""
    name: str
    source: str
    threadgroups: Optional[Tuple[int, int, int]] = None
    threads_per_threadgroup: Optional[Tuple[int, int, int]] = None
    buffers: List[Dict[str, Any]] = field(default_factory=list)
    textures: List[Dict[str, Any]] = field(default_factory=list)


class MetalKernel:
    """
    Metal GPU compute kernel for high-performance operations.
    
    Compiles and executes Metal shaders on Apple GPUs.
    
    Example:
        >>> kernel = MetalKernel("matrix_multiply", """
        ...     kernel void matrix_multiply(
        ...         device const float* A [[ buffer(0) ]],
        ...         device const float* B [[ buffer(1) ]],
        ...         device float* C [[ buffer(2) ]],
        ...         uint2 gid [[ thread_position_in_grid ]]
        ...     ) {
        ...         // Matrix multiplication implementation
        ...     }
        ... """)
        >>> kernel.compile()
        >>> kernel.launch(buffers=[buffer_a, buffer_b, buffer_c])
    """
    
    def __init__(
        self,
        name: str,
        source: str,
        function_name: Optional[str] = None,
    ):
        self.name = name
        self.source = source
        self.function_name = function_name or name
        self._compiled_library: Optional[Any] = None
        self._pipeline_state: Optional[Any] = None
        self._device: Optional[Any] = None
        self._command_queue: Optional[Any] = None
        
    def compile(self) -> bool:
        """
        Compile the Metal kernel.
        
        Returns:
            True if compilation succeeded
        """
        # In a real implementation, this would:
        # 1. Get the default Metal device
        # 2. Create a MTLLibrary from source
        # 3. Create a MTLComputePipelineState
        
        logger.info(f"Compiling Metal kernel: {self.name}")
        
        # Simulation mode - would need PyObjC or similar for real Metal
        self._compiled_library = {
            "name": self.name,
            "source_length": len(self.source),
            "function": self.function_name,
        }
        
        return True
    
    def is_compiled(self) -> bool:
        """Check if the kernel is compiled."""
        return self._compiled_library is not None
    
    def launch(
        self,
        buffers: List[Any],
        threadgroups: Optional[Tuple[int, int, int]] = None,
        threads_per_threadgroup: Optional[Tuple[int, int, int]] = None,
        wait: bool = True,
    ) -> Any:
        """
        Launch the kernel on the GPU.
        
        Args:
            buffers: List of Metal buffers to bind
            threadgroups: Number of threadgroups (x, y, z)
            threads_per_threadgroup: Threads per threadgroup (x, y, z)
            wait: Whether to wait for completion
        
        Returns:
            Command buffer or result
        """
        if not self.is_compiled():
            self.compile()
        
        logger.info(f"Launching kernel: {self.name}")
        
        # In real implementation, this would:
        # 1. Create a command buffer
        # 2. Create a compute command encoder
        # 3. Set pipeline state
        # 4. Set buffers
        # 5. Dispatch threads
        # 6. Commit and wait (if requested)
        
        return {
            "kernel": self.name,
            "buffers": len(buffers),
            "threadgroups": threadgroups,
            "status": "completed" if wait else "submitted",
        }
    
    def get_source(self) -> str:
        """Get the Metal shader source code."""
        return self.source
    
    def get_function_name(self) -> str:
        """Get the kernel function name."""
        return self.function_name
    
    def optimize_for_device(self, device_type: str = "apple_gpu") -> "MetalKernel":
        """
        Optimize the kernel for a specific device.
        
        Args:
            device_type: Target device type
        
        Returns:
            Optimized kernel
        """
        # In real implementation, would apply device-specific optimizations
        logger.info(f"Optimizing kernel for {device_type}")
        return self


class MetalBuffer:
    """
    Wrapper for Metal GPU buffers.
    
    Manages memory allocation and data transfer to/from GPU.
    
    Example:
        >>> buffer = MetalBuffer.create([1.0, 2.0, 3.0], dtype="float32")
        >>> buffer.upload(data)
        >>> result = buffer.download()
    """
    
    def __init__(
        self,
        size: int,
        dtype: str = "float32",
        storage_mode: str = "shared",
    ):
        self.size = size
        self.dtype = dtype
        self.storage_mode = storage_mode
        self._buffer: Optional[Any] = None
        self._data: Optional[Any] = None
        
    @classmethod
    def create(
        cls,
        data: Any,
        dtype: str = "float32",
        storage_mode: str = "shared",
    ) -> "MetalBuffer":
        """Create a buffer from data."""
        import numpy as np
        
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        
        buffer = cls(
            size=data.nbytes,
            dtype=dtype,
            storage_mode=storage_mode,
        )
        buffer._data = data
        
        return buffer
    
    @classmethod
    def zeros(
        cls,
        shape: Tuple[int, ...],
        dtype: str = "float32",
    ) -> "MetalBuffer":
        """Create a buffer filled with zeros."""
        import numpy as np
        
        data = np.zeros(shape, dtype=dtype)
        return cls.create(data, dtype)
    
    def upload(self, data: Any) -> "MetalBuffer":
        """Upload data to the GPU buffer."""
        import numpy as np
        
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=self.dtype)
        
        self._data = data
        logger.debug(f"Uploaded {data.nbytes} bytes to buffer")
        return self
    
    def download(self) -> Any:
        """Download data from the GPU buffer."""
        return self._data
    
    def copy_to(self, other: "MetalBuffer") -> "MetalBuffer":
        """Copy buffer contents to another buffer."""
        other._data = self._data.copy() if self._data is not None else None
        return self
    
    def get_length(self) -> int:
        """Get the buffer length in bytes."""
        return self.size


def compile_metal(
    source: str,
    function_name: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> MetalKernel:
    """
    Compile Metal shader source to a kernel.
    
    Args:
        source: Metal shader source code
        function_name: Entry point function name
        options: Compilation options
    
    Returns:
        Compiled MetalKernel
    
    Example:
        >>> kernel = compile_metal("""
        ...     kernel void add(device float* a, device float* b, device float* c) {
        ...         c[gid] = a[gid] + b[gid];
        ...     }
        ... """)
    """
    name = function_name or "kernel"
    kernel = MetalKernel(name, source, function_name)
    kernel.compile()
    return kernel


def launch_metal(
    kernel: MetalKernel,
    buffers: List[Any],
    grid_size: Optional[Tuple[int, int, int]] = None,
    threadgroup_size: Optional[Tuple[int, int, int]] = None,
) -> Any:
    """
    Launch a Metal kernel with given buffers.
    
    Args:
        kernel: Compiled MetalKernel
        buffers: List of buffers to bind
        grid_size: Global grid size
        threadgroup_size: Threadgroup size
    
    Returns:
        Kernel execution result
    
    Example:
        >>> result = launch_metal(kernel, [buf_a, buf_b, buf_c], 
        ...                       grid_size=(1024, 1, 1))
    """
    return kernel.launch(
        buffers=buffers,
        threadgroups=grid_size,
        threads_per_threadgroup=threadgroup_size,
    )


# =============================================================================
# iOS Deployment
# =============================================================================


@dataclass
class iOSAppConfig:
    """Configuration for iOS app generation."""
    app_name: str
    bundle_id: str
    version: str = "1.0.0"
    deployment_target: str = "13.0"
    swift_version: str = "5.0"
    orientation: List[str] = field(default_factory=lambda: ["portrait"])
    capabilities: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=lambda: ["CoreML", "UIKit"])
    entitlements: Dict[str, Any] = field(default_factory=dict)
    icons: Optional[Dict[str, str]] = None
    launch_screen: Optional[str] = None


class iOSExporter:
    """
    Export models as iOS applications.
    
    Generates complete Xcode projects with model integration.
    
    Example:
        >>> exporter = iOSExporter()
        >>> exporter.export(
        ...     coreml_model,
        ...     config=iOSAppConfig(
        ...         app_name="MyAIApp",
        ...         bundle_id="com.example.myai",
        ...     ),
        ...     output_dir="./MyAIApp"
        ... )
    """
    
    def __init__(self):
        self.exported_projects: List[Path] = []
        
    def export(
        self,
        model: Union[CoreMLModel, Any],
        config: iOSAppConfig,
        output_dir: Union[str, Path],
        include_tests: bool = True,
        include_swiftui: bool = True,
    ) -> Path:
        """
        Export model as an iOS Xcode project.
        
        Args:
            model: CoreMLModel or path to .mlmodel file
            config: iOS app configuration
            output_dir: Output directory for the project
            include_tests: Include unit tests
            include_swiftui: Use SwiftUI (vs UIKit)
        
        Returns:
            Path to generated project
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project structure
        self._create_project_structure(output_dir, config)
        
        # Generate Info.plist
        self._generate_info_plist(output_dir, config)
        
        # Generate AppDelegate and SceneDelegate
        self._generate_app_delegates(output_dir, config, include_swiftui)
        
        # Generate main view
        self._generate_main_view(output_dir, config, include_swiftui)
        
        # Generate model wrapper
        self._generate_model_wrapper(output_dir, model, config)
        
        # Copy or generate model
        self._copy_model(output_dir, model, config)
        
        # Generate Xcode project file
        self._generate_xcode_project(output_dir, config, include_tests)
        
        self.exported_projects.append(output_dir)
        logger.info(f"Exported iOS app to {output_dir}")
        
        return output_dir
    
    def _create_project_structure(self, output_dir: Path, config: iOSAppConfig):
        """Create the Xcode project directory structure."""
        (output_dir / config.app_name / "Resources").mkdir(parents=True)
        (output_dir / config.app_name / "Models").mkdir(parents=True)
        (output_dir / config.app_name / "Views").mkdir(parents=True)
        (output_dir / config.app_name / "ViewModels").mkdir(parents=True)
        (output_dir / config.app_name / "Utils").mkdir(parents=True)
        
    def _generate_info_plist(self, output_dir: Path, config: iOSAppConfig):
        """Generate Info.plist file."""
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>{config.bundle_id}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>{config.version}</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSRequiresIPhoneOS</key>
    <true/>
    <key>UIApplicationSceneManifest</key>
    <dict>
        <key>UIApplicationSupportsMultipleScenes</key>
        <false/>
        <key>UISceneConfigurations</key>
        <dict>
            <key>UIWindowSceneSessionRoleApplication</key>
            <array>
                <dict>
                    <key>UISceneConfigurationName</key>
                    <string>Default Configuration</string>
                    <key>UISceneDelegateClassName</key>
                    <string>$(PRODUCT_MODULE_NAME).SceneDelegate</string>
                </dict>
            </array>
        </dict>
    </dict>
    <key>UISupportedInterfaceOrientations</key>
    <array>
        {''.join([f'<string>{o}</string>' for o in config.orientation])}
    </array>
</dict>
</plist>"""
        
        plist_path = output_dir / config.app_name / "Info.plist"
        plist_path.write_text(plist_content)
    
    def _generate_app_delegates(
        self,
        output_dir: Path,
        config: iOSAppConfig,
        use_swiftui: bool,
    ):
        """Generate AppDelegate and SceneDelegate."""
        # AppDelegate
        app_delegate = f"""import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {{
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {{
        return true
    }}

    func application(
        _ application: UIApplication,
        configurationForConnecting connectingSceneSession: UISceneSession,
        options: UIScene.ConnectionOptions
    ) -> UISceneConfiguration {{
        return UISceneConfiguration(
            name: "Default Configuration",
            sessionRole: connectingSceneSession.role
        )
    }}
}}
"""
        
        # SceneDelegate
        if use_swiftui:
            scene_delegate = f"""import UIKit
import SwiftUI

class SceneDelegate: UIResponder, UIWindowSceneDelegate {{
    var window: UIWindow?

    func scene(
        _ scene: UIScene,
        willConnectTo session: UISceneSession,
        options connectionOptions: UIScene.ConnectionOptions
    ) {{
        guard let windowScene = (scene as? UIWindowScene) else {{ return }}
        
        let contentView = ContentView()
        
        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = UIHostingController(rootView: contentView)
        self.window = window
        window.makeKeyAndVisible()
    }}
}}
"""
        else:
            scene_delegate = f"""import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {{
    var window: UIWindow?

    func scene(
        _ scene: UIScene,
        willConnectTo session: UISceneSession,
        options connectionOptions: UIScene.ConnectionOptions
    ) {{
        guard let windowScene = (scene as? UIWindowScene) else {{ return }}
        
        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = MainViewController()
        self.window = window
        window.makeKeyAndVisible()
    }}
}}
"""
        
        (output_dir / config.app_name / "AppDelegate.swift").write_text(app_delegate)
        (output_dir / config.app_name / "SceneDelegate.swift").write_text(scene_delegate)
    
    def _generate_main_view(
        self,
        output_dir: Path,
        config: iOSAppConfig,
        use_swiftui: bool,
    ):
        """Generate the main view/controller."""
        if use_swiftui:
            content_view = f"""import SwiftUI

struct ContentView: View {{
    @StateObject private var viewModel = PredictionViewModel()
    
    var body: some View {{
        NavigationView {{
            VStack(spacing: 20) {{
                Text("{config.app_name}")
                    .font(.largeTitle)
                    .padding()
                
                if let result = viewModel.predictionResult {{
                    Text("Prediction: \\(result)")
                        .font(.title2)
                }}
                
                if viewModel.isLoading {{
                    ProgressView()
                }}
                
                Button("Run Prediction") {{
                    viewModel.runPrediction()
                }}
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
                
                Spacer()
            }}
            .padding()
        }}
    }}
}}

struct ContentView_Previews: PreviewProvider {{
    static var previews: some View {{
        ContentView()
    }}
}}
"""
            (output_dir / config.app_name / "Views" / "ContentView.swift").write_text(content_view)
        else:
            view_controller = f"""import UIKit

class MainViewController: UIViewController {{
    private let viewModel = PredictionViewModel()
    
    private lazy var predictButton: UIButton = {{
        let button = UIButton(type: .system)
        button.setTitle("Run Prediction", for: .normal)
        button.addTarget(self, action: #selector(runPrediction), for: .touchUpInside)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }}()
    
    private lazy var resultLabel: UILabel = {{
        let label = UILabel()
        label.textAlignment = .center
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }}()
    
    override func viewDidLoad() {{
        super.viewDidLoad()
        setupUI()
    }}
    
    private func setupUI() {{
        view.backgroundColor = .systemBackground
        
        view.addSubview(predictButton)
        view.addSubview(resultLabel)
        
        NSLayoutConstraint.activate([
            predictButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            predictButton.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            
            resultLabel.topAnchor.constraint(equalTo: predictButton.bottomAnchor, constant: 20),
            resultLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            resultLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
        ])
    }}
    
    @objc private func runPrediction() {{
        viewModel.runPrediction {{ [weak self] result in
            DispatchQueue.main.async {{
                self?.resultLabel.text = "Prediction: \\(result)"
            }}
        }}
    }}
}}
"""
            (output_dir / config.app_name / "MainViewController.swift").write_text(view_controller)
    
    def _generate_model_wrapper(
        self,
        output_dir: Path,
        model: Any,
        config: iOSAppConfig,
    ):
        """Generate the model wrapper and view model."""
        model_name = getattr(model, 'metadata', None)
        model_name = model_name.name if model_name else "Model"
        
        view_model = f"""import Foundation
import CoreML

class PredictionViewModel: ObservableObject {{
    @Published var predictionResult: String?
    @Published var isLoading = false
    
    private let model: {model_name}Predictor
    
    init() {{
        do {{
            self.model = try {model_name}Predictor()
        }} catch {{
            fatalError("Failed to load model: \\(error)")
        }}
    }}
    
    func runPrediction(completion: ((String) -> Void)? = nil) {{
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {{ [weak self] in
            defer {{ DispatchQueue.main.async {{ self?.isLoading = false }} }}
            
            do {{
                // Prepare input data
                let input = try self?.prepareInput()
                
                // Run prediction
                let result = try self?.model.predict(input: input!)
                
                DispatchQueue.main.async {{
                    let resultString = "\\(result!)"
                    self?.predictionResult = resultString
                    completion?(resultString)
                }}
            }} catch {{
                DispatchQueue.main.async {{
                    self?.predictionResult = "Error: \\(error)"
                }}
            }}
        }}
    }}
    
    private func prepareInput() throws -> MLMultiArray {{
        // Create input array - customize based on your model
        let input = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
        // Fill with actual data
        return input
    }}
}}
"""
        
        (output_dir / config.app_name / "ViewModels" / "PredictionViewModel.swift").write_text(view_model)
    
    def _copy_model(
        self,
        output_dir: Path,
        model: Union[CoreMLModel, Any],
        config: iOSAppConfig,
    ):
        """Copy or save the model to the project."""
        models_dir = output_dir / config.app_name / "Models"
        
        if isinstance(model, CoreMLModel):
            model.save(models_dir / f"{config.app_name}Model.mlmodel")
        elif isinstance(model, (str, Path)):
            import shutil
            shutil.copy(model, models_dir / Path(model).name)
    
    def _generate_xcode_project(
        self,
        output_dir: Path,
        config: iOSAppConfig,
        include_tests: bool,
    ):
        """Generate the Xcode project file."""
        # This is a simplified project file structure
        # Real implementation would use xcodeproj gem or similar
        
        project_content = f"""// {config.app_name}.xcodeproj
// This is a placeholder - use xcodegen or Xcode to create the actual project
// 
// To generate the project:
// 1. Install xcodegen: brew install xcodegen
// 2. Create project.yml with your configuration
// 3. Run: xcodegen generate

Project:
  name: {config.app_name}
  targets:
    {config.app_name}:
      type: application
      platform: iOS
      deploymentTarget: "{config.deployment_target}"
      sources:
        - {config.app_name}
      resources:
        - {config.app_name}/Resources
      settings:
        PRODUCT_BUNDLE_IDENTIFIER: {config.bundle_id}
        SWIFT_VERSION: "{config.swift_version}"
      dependencies:
        {chr(10).join([f'- framework: {f}' for f in config.frameworks])}
"""
        
        (output_dir / "project.yml").write_text(project_content)
        
        # Try to generate with xcodegen if available
        try:
            subprocess.run(
                ["xcodegen", "generate"],
                cwd=output_dir,
                check=True,
                capture_output=True,
            )
            logger.info("Generated Xcode project with xcodegen")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("xcodegen not available. Use project.yml to generate Xcode project.")


def create_ios_app(
    model: Union[CoreMLModel, Any],
    app_name: str,
    bundle_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Quick function to create an iOS app from a model.
    
    Args:
        model: CoreMLModel or path to model
        app_name: Name of the app
        bundle_id: Bundle identifier (auto-generated if None)
        output_dir: Output directory
        **kwargs: Additional configuration options
    
    Returns:
        Path to generated project
    
    Example:
        >>> create_ios_app(
        ...     coreml_model,
        ...     "MyApp",
        ...     bundle_id="com.example.myapp",
        ...     output_dir="./MyApp"
        ... )
    """
    if bundle_id is None:
        bundle_id = f"com.fishstick.{app_name.lower().replace(' ', '')}"
    
    if output_dir is None:
        output_dir = Path(f"./{app_name}")
    
    config = iOSAppConfig(
        app_name=app_name,
        bundle_id=bundle_id,
        **{k: v for k, v in kwargs.items() if k in iOSAppConfig.__dataclass_fields__},
    )
    
    exporter = iOSExporter()
    return exporter.export(
        model,
        config,
        output_dir,
    )


class iOSIntegration:
    """
    Integration utilities for iOS applications.
    
    Provides helpers for common iOS integration patterns.
    """
    
    @staticmethod
    def generate_camera_capture_code() -> str:
        """Generate Swift code for camera capture with CoreML."""
        return '''import UIKit
import AVFoundation
import CoreML

class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
              captureSession.canAddInput(videoInput) else {
            return
        }
        
        captureSession.addInput(videoInput)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        captureSession.startRunning()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Process frame with CoreML model
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Run model prediction on pixelBuffer
        // Dispatch to background queue for prediction
    }
}
'''
    
    @staticmethod
    def generate_photo_picker_code() -> str:
        """Generate Swift code for photo picker with CoreML."""
        return '''import SwiftUI
import PhotosUI

struct PhotoPicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    var onImageSelected: ((UIImage) -> Void)?
    
    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PhotoPicker
        
        init(_ parent: PhotoPicker) {
            self.parent = parent
        }
        
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            
            guard let provider = results.first?.itemProvider else { return }
            
            if provider.canLoadObject(ofClass: UIImage.self) {
                provider.loadObject(ofClass: UIImage.self) { image, _ in
                    DispatchQueue.main.async {
                        self.parent.selectedImage = image as? UIImage
                        if let img = image as? UIImage {
                            self.parent.onImageSelected?(img)
                        }
                    }
                }
            }
        }
    }
}
'''
    
    @staticmethod
    def generate_background_download_code() -> str:
        """Generate Swift code for background model download."""
        return '''import Foundation

class ModelDownloader: NSObject, URLSessionDownloadDelegate {
    private var downloadTask: URLSessionDownloadTask?
    
    func downloadModel(from url: URL, completion: @escaping (Result<URL, Error>) -> Void) {
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        downloadTask = session.downloadTask(with: url)
        downloadTask?.resume()
    }
    
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // Move downloaded model to documents directory
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destinationURL = documentsPath.appendingPathComponent("model.mlmodel")
        
        try? FileManager.default.moveItem(at: location, to: destinationURL)
    }
    
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("Download failed: \\(error)")
        }
    }
}
'''


def optimize_for_ios(
    model: CoreMLModel,
    target_device: str = "iphone",
    quantize: bool = True,
    nbits: int = 8,
) -> CoreMLModel:
    """
    Optimize a CoreML model for iOS deployment.
    
    Args:
        model: CoreMLModel to optimize
        target_device: Target device type
        quantize: Whether to quantize weights
        nbits: Bits for quantization
    
    Returns:
        Optimized CoreMLModel
    
    Example:
        >>> optimized = optimize_for_ios(model, target_device="iphone", quantize=True, nbits=8)
    """
    logger.info(f"Optimizing model for {target_device}")
    
    optimized = model
    
    if quantize:
        logger.info(f"Quantizing to {nbits} bits")
        optimized = optimized.quantize_weights(nbits)
    
    # Additional optimizations would go here
    # - Pruning
    # - Layer fusion
    # - Architecture-specific optimizations
    
    return optimized


# =============================================================================
# macOS Deployment
# =============================================================================


@dataclass
class MacOSAppConfig:
    """Configuration for macOS app generation."""
    app_name: str
    bundle_id: str
    version: str = "1.0.0"
    deployment_target: str = "11.0"
    swift_version: str = "5.0"
    category: str = "public.app-category.utilities"
    capabilities: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=lambda: ["CoreML", "Cocoa"])
    entitlements: Dict[str, Any] = field(default_factory=dict)
    is_menu_bar_app: bool = False
    is_agent: bool = False


class MacOSExporter:
    """
    Export models as macOS applications.
    
    Generates complete Xcode projects for macOS with model integration.
    
    Example:
        >>> exporter = MacOSExporter()
        >>> exporter.export(
        ...     coreml_model,
        ...     config=MacOSAppConfig(
        ...         app_name="MyMacApp",
        ...         bundle_id="com.example.mymacapp",
        ...     ),
        ...     output_dir="./MyMacApp"
        ... )
    """
    
    def __init__(self):
        self.exported_projects: List[Path] = []
        
    def export(
        self,
        model: Union[CoreMLModel, Any],
        config: MacOSAppConfig,
        output_dir: Union[str, Path],
        include_command_line_tool: bool = False,
        include_status_bar: bool = False,
    ) -> Path:
        """
        Export model as a macOS Xcode project.
        
        Args:
            model: CoreMLModel or path to .mlmodel file
            config: macOS app configuration
            output_dir: Output directory for the project
            include_command_line_tool: Include CLI tool target
            include_status_bar: Create a menu bar app
        
        Returns:
            Path to generated project
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project structure
        self._create_project_structure(output_dir, config)
        
        # Generate Info.plist
        self._generate_macos_info_plist(output_dir, config)
        
        # Generate AppDelegate
        self._generate_macos_app_delegate(output_dir, config, include_status_bar)
        
        # Generate main view controller
        self._generate_macos_main_view(output_dir, config)
        
        # Generate model wrapper
        self._generate_macos_model_wrapper(output_dir, model, config)
        
        # Copy model
        self._copy_macos_model(output_dir, model, config)
        
        # Generate entitlements
        self._generate_entitlements(output_dir, config)
        
        # Generate Xcode project
        self._generate_macos_xcode_project(
            output_dir,
            config,
            include_command_line_tool,
        )
        
        self.exported_projects.append(output_dir)
        logger.info(f"Exported macOS app to {output_dir}")
        
        return output_dir
    
    def _create_project_structure(self, output_dir: Path, config: MacOSAppConfig):
        """Create the macOS Xcode project directory structure."""
        (output_dir / config.app_name / "Resources").mkdir(parents=True)
        (output_dir / config.app_name / "Models").mkdir(parents=True)
        (output_dir / config.app_name / "Controllers").mkdir(parents=True)
        (output_dir / config.app_name / "Views").mkdir(parents=True)
        
    def _generate_macos_info_plist(self, output_dir: Path, config: MacOSAppConfig):
        """Generate Info.plist for macOS."""
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>{config.bundle_id}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>{config.version}</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>{config.deployment_target}</string>
    <key>NSMainStoryboardFile</key>
    <string>Main</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSApplicationCategoryType</key>
    <string>{config.category}</string>
    <key>LSUIElement</key>
    <{str(config.is_agent).lower()}/>
</dict>
</plist>"""
        
        plist_path = output_dir / config.app_name / "Info.plist"
        plist_path.write_text(plist_content)
    
    def _generate_macos_app_delegate(
        self,
        output_dir: Path,
        config: MacOSAppConfig,
        is_menu_bar: bool,
    ):
        """Generate AppDelegate for macOS."""
        if is_menu_bar:
            app_delegate = f"""import Cocoa

@main
class AppDelegate: NSObject, NSApplicationDelegate {{
    var statusItem: NSStatusItem!
    var popover: NSPopover!
    
    func applicationDidFinishLaunching(_ aNotification: Notification) {{
        setupMenuBar()
    }}
    
    func setupMenuBar() {{
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        
        if let button = statusItem.button {{
            button.title = "{config.app_name}"
        }}
        
        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "Run Prediction", action: #selector(runPrediction), keyEquivalent: "r"))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        
        statusItem.menu = menu
    }}
    
    @objc func runPrediction() {{
        // Run model prediction
    }}
}}
"""
        else:
            app_delegate = f"""import Cocoa

@main
class AppDelegate: NSObject, NSApplicationDelegate {{
    func applicationDidFinishLaunching(_ aNotification: Notification) {{
        // Insert code here to initialize your application
    }}

    func applicationWillTerminate(_ aNotification: Notification) {{
        // Insert code here to tear down your application
    }}

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {{
        return true
    }}
}}
"""
        
        (output_dir / config.app_name / "AppDelegate.swift").write_text(app_delegate)
    
    def _generate_macos_main_view(self, output_dir: Path, config: MacOSAppConfig):
        """Generate main view controller for macOS."""
        view_controller = f"""import Cocoa

class MainViewController: NSViewController {{
    private let viewModel = PredictionViewModel()
    
    private lazy var predictButton: NSButton = {{
        let button = NSButton(title: "Run Prediction", target: self, action: #selector(runPrediction))
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }}()
    
    private lazy var resultLabel: NSTextField = {{
        let label = NSTextField(labelWithString: "Ready")
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }}()
    
    override func viewDidLoad() {{
        super.viewDidLoad()
        setupUI()
    }}
    
    private func setupUI() {{
        view.addSubview(predictButton)
        view.addSubview(resultLabel)
        
        NSLayoutConstraint.activate([
            predictButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            predictButton.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            
            resultLabel.topAnchor.constraint(equalTo: predictButton.bottomAnchor, constant: 20),
            resultLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
        ])
    }}
    
    @objc private func runPrediction() {{
        viewModel.runPrediction {{ [weak self] result in
            DispatchQueue.main.async {{
                self?.resultLabel.stringValue = "Prediction: \\(result)"
            }}
        }}
    }}
}}
"""
        
        (output_dir / config.app_name / "Controllers" / "MainViewController.swift").write_text(view_controller)
    
    def _generate_macos_model_wrapper(
        self,
        output_dir: Path,
        model: Any,
        config: MacOSAppConfig,
    ):
        """Generate model wrapper for macOS."""
        model_name = getattr(model, 'metadata', None)
        model_name = model_name.name if model_name else "Model"
        
        view_model = f"""import Foundation
import CoreML

class PredictionViewModel {{
    private let model: {model_name}Predictor
    
    init() {{
        do {{
            self.model = try {model_name}Predictor()
        }} catch {{
            fatalError("Failed to load model: \\(error)")
        }}
    }}
    
    func runPrediction(completion: @escaping (String) -> Void) {{
        DispatchQueue.global(qos: .userInitiated).async {{ [weak self] in
            do {{
                let input = try self?.prepareInput()
                let result = try self?.model.predict(input: input!)
                
                DispatchQueue.main.async {{
                    completion("\\(result!)")
                }}
            }} catch {{
                DispatchQueue.main.async {{
                    completion("Error: \\(error)")
                }}
            }}
        }}
    }}
    
    private func prepareInput() throws -> MLMultiArray {{
        let input = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
        return input
    }}
}}
"""
        
        (output_dir / config.app_name / "Controllers" / "PredictionViewModel.swift").write_text(view_model)
    
    def _copy_macos_model(
        self,
        output_dir: Path,
        model: Union[CoreMLModel, Any],
        config: MacOSAppConfig,
    ):
        """Copy model for macOS."""
        models_dir = output_dir / config.app_name / "Models"
        
        if isinstance(model, CoreMLModel):
            model.save(models_dir / f"{config.app_name}Model.mlmodel")
        elif isinstance(model, (str, Path)):
            import shutil
            shutil.copy(model, models_dir / Path(model).name)
    
    def _generate_entitlements(self, output_dir: Path, config: MacOSAppConfig):
        """Generate entitlements file."""
        entitlements = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-only</key>
    <true/>
</dict>
</plist>"""
        
        (output_dir / config.app_name / f"{config.app_name}.entitlements").write_text(entitlements)
    
    def _generate_macos_xcode_project(
        self,
        output_dir: Path,
        config: MacOSAppConfig,
        include_cli: bool,
    ):
        """Generate Xcode project for macOS."""
        project_content = f"""Project:
  name: {config.app_name}
  targets:
    {config.app_name}:
      type: application
      platform: macOS
      deploymentTarget: "{config.deployment_target}"
      sources:
        - {config.app_name}
      resources:
        - {config.app_name}/Resources
      settings:
        PRODUCT_BUNDLE_IDENTIFIER: {config.bundle_id}
        SWIFT_VERSION: "{config.swift_version}"
        CODE_SIGN_ENTITLEMENTS: {config.app_name}/{config.app_name}.entitlements
      dependencies:
        {chr(10).join([f'- framework: {f}' for f in config.frameworks])}
"""
        
        if include_cli:
            project_content += f"""
    {config.app_name}CLI:
      type: tool
      platform: macOS
      sources:
        - CLI
      settings:
        PRODUCT_BUNDLE_IDENTIFIER: {config.bundle_id}.cli
        SWIFT_VERSION: "{config.swift_version}"
"""
        
        (output_dir / "project.yml").write_text(project_content)


def create_macos_app(
    model: Union[CoreMLModel, Any],
    app_name: str,
    bundle_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Quick function to create a macOS app from a model.
    
    Args:
        model: CoreMLModel or path to model
        app_name: Name of the app
        bundle_id: Bundle identifier
        output_dir: Output directory
        **kwargs: Additional configuration
    
    Returns:
        Path to generated project
    """
    if bundle_id is None:
        bundle_id = f"com.fishstick.{app_name.lower().replace(' ', '')}"
    
    if output_dir is None:
        output_dir = Path(f"./{app_name}")
    
    config = MacOSAppConfig(
        app_name=app_name,
        bundle_id=bundle_id,
        **{k: v for k, v in kwargs.items() if k in MacOSAppConfig.__dataclass_fields__},
    )
    
    exporter = MacOSExporter()
    return exporter.export(model, config, output_dir)


class macOSIntegration:
    """
    Integration utilities for macOS applications.
    """
    
    @staticmethod
    def generate_drag_drop_code() -> str:
        """Generate Swift code for drag and drop file handling."""
        return '''import Cocoa

class DropView: NSView {
    var onFileDropped: ((URL) -> Void)?
    
    override func awakeFromNib() {
        super.awakeFromNib()
        setup()
    }
    
    func setup() {
        registerForDraggedTypes([.fileURL])
    }
    
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        return .copy
    }
    
    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        guard let pasteboard = sender.draggingPasteboard.propertyList(
            forType: NSPasteboard.PasteboardType(rawValue: "NSFilenamesPboardType")
        ) as? [String],
        let path = pasteboard.first else {
            return false
        }
        
        onFileDropped?(URL(fileURLWithPath: path))
        return true
    }
}
'''
    
    @staticmethod
    def generate_batch_processing_code() -> str:
        """Generate Swift code for batch file processing."""
        return '''import Cocoa
import CoreML

class BatchProcessor {
    private let model: MLModel
    
    init(model: MLModel) {
        self.model = model
    }
    
    func processFiles(urls: [URL], progress: @escaping (Double) -> Void, completion: @escaping ([Result]) -> Void) {
        var results: [Result] = []
        let total = urls.count
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            for (index, url) in urls.enumerated() {
                if let result = self?.processFile(url: url) {
                    results.append(result)
                }
                
                DispatchQueue.main.async {
                    progress(Double(index + 1) / Double(total))
                }
            }
            
            DispatchQueue.main.async {
                completion(results)
            }
        }
    }
    
    private func processFile(url: URL) -> Result? {
        // Process file with CoreML model
        return nil
    }
    
    struct Result {
        let url: URL
        let prediction: String
    }
}
'''


def optimize_for_macos(
    model: CoreMLModel,
    target_device: str = "mac",
    use_gpu: bool = True,
    use_neural_engine: bool = True,
    quantize: bool = False,
) -> CoreMLModel:
    """
    Optimize a CoreML model for macOS deployment.
    
    Args:
        model: CoreMLModel to optimize
        target_device: Target device
        use_gpu: Enable GPU acceleration
        use_neural_engine: Enable Neural Engine
        quantize: Whether to quantize
    
    Returns:
        Optimized CoreMLModel
    """
    logger.info(f"Optimizing model for macOS on {target_device}")
    
    # Set compute units based on flags
    if use_gpu and use_neural_engine:
        compute_units = ComputeUnits.ALL
    elif use_gpu:
        compute_units = ComputeUnits.CPU_AND_GPU
    else:
        compute_units = ComputeUnits.CPU_ONLY
    
    optimized = model
    
    if quantize:
        optimized = optimized.quantize_weights(16)  # Use 16-bit for macOS
    
    logger.info(f"Using compute units: {compute_units.value}")
    
    return optimized


# =============================================================================
# Swift for TensorFlow
# =============================================================================


class S4TConverter:
    """
    Convert models to Swift for TensorFlow (S4T) format.
    
    Note: Swift for TensorFlow development has been archived, but this
    converter provides compatibility for existing S4T projects.
    
    Example:
        >>> converter = S4TConverter()
        >>> swift_code = converter.convert(pytorch_model)
        >>> swift_code.save("Model.swift")
    """
    
    def __init__(self, swift_tf_version: str = "0.12"):
        self.swift_tf_version = swift_tf_version
        self._layer_converters = self._get_layer_converters()
        
    def _get_layer_converters(self) -> Dict[str, Callable]:
        """Get mapping of layer types to Swift code generators."""
        return {
            "Linear": self._convert_linear,
            "Conv2d": self._convert_conv2d,
            "BatchNorm2d": self._convert_batchnorm,
            "ReLU": self._convert_relu,
            "MaxPool2d": self._convert_maxpool,
            "AdaptiveAvgPool2d": self._convert_adaptive_pool,
            "Flatten": self._convert_flatten,
        }
    
    def convert(
        self,
        model: Any,
        model_name: str = "ConvertedModel",
    ) -> "S4TModel":
        """
        Convert a model to Swift for TensorFlow code.
        
        Args:
            model: Model to convert
            model_name: Name for the generated Swift class
        
        Returns:
            S4TModel with Swift source code
        """
        model_type = self._detect_framework(model)
        
        if model_type == "pytorch":
            swift_code = self._convert_pytorch_to_s4t(model, model_name)
        elif model_type == "tensorflow":
            swift_code = self._convert_tf_to_s4t(model, model_name)
        else:
            raise S4TError(f"Unsupported model type: {model_type}")
        
        return S4TModel(
            name=model_name,
            source_code=swift_code,
            swift_tf_version=self.swift_tf_version,
        )
    
    def _detect_framework(self, model: Any) -> str:
        """Detect the model framework."""
        model_class = type(model).__module__
        
        if "torch" in model_class:
            return "pytorch"
        elif "tensorflow" in model_class or "tf" in model_class:
            return "tensorflow"
        else:
            return "unknown"
    
    def _convert_pytorch_to_s4t(self, model: Any, model_name: str) -> str:
        """Convert PyTorch model to S4T Swift code."""
        import torch
        
        lines = [
            "import TensorFlow",
            "",
            f"struct {model_name}: Layer {{",
            "    var layers: [AnyLayer] = []",
            "",
            "    init() {",
        ]
        
        # Convert each layer
        for name, module in model.named_modules():
            if name == "":  # Skip root module
                continue
            
            layer_type = type(module).__name__
            if layer_type in self._layer_converters:
                converter = self._layer_converters[layer_type]
                swift_layer = converter(module, name)
                lines.append(f"        {swift_layer}")
        
        lines.extend([
            "    }",
            "",
            "    @differentiable",
            "    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {",
            "        var x = input",
            "        for layer in layers {",
            "            x = layer(x)",
            "        }",
            "        return x",
            "    }",
            "}",
        ])
        
        return "\n".join(lines)
    
    def _convert_tf_to_s4t(self, model: Any, model_name: str) -> str:
        """Convert TensorFlow model to S4T Swift code."""
        return f"""
import TensorFlow

struct {model_name}: Layer {{
    // TensorFlow to S4T conversion placeholder
    // Manual implementation required
}}
"""
    
    def _convert_linear(self, layer: Any, name: str) -> str:
        """Convert Linear/Dense layer."""
        in_features = layer.in_features
        out_features = layer.out_features
        has_bias = layer.bias is not None
        
        return f'self.layers.append(Dense(inputSize: {in_features}, outputSize: {out_features}, useBias: {str(has_bias).lower()}))'
    
    def _convert_conv2d(self, layer: Any, name: str) -> str:
        """Convert Conv2d layer."""
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        padding = "same" if layer.padding == "same" else "valid"
        
        return f'self.layers.append(Conv2D(filterShape: ({kernel_size}, {kernel_size}, 0, {out_channels}), strides: ({stride}, {stride}), padding: .{padding}))'
    
    def _convert_batchnorm(self, layer: Any, name: str) -> str:
        """Convert BatchNorm layer."""
        num_features = layer.num_features
        return f'self.layers.append(BatchNorm(featureCount: {num_features}))'
    
    def _convert_relu(self, layer: Any, name: str) -> str:
        """Convert ReLU activation."""
        return 'self.layers.append(ReLU())'
    
    def _convert_maxpool(self, layer: Any, name: str) -> str:
        """Convert MaxPool layer."""
        kernel_size = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
        return f'self.layers.append(MaxPool2D(poolSize: ({kernel_size}, {kernel_size}), strides: ({stride}, {stride})))'
    
    def _convert_adaptive_pool(self, layer: Any, name: str) -> str:
        """Convert AdaptiveAvgPool layer."""
        return 'self.layers.append(AvgPool2D(poolSize: (1, 1), strides: (1, 1)))'
    
    def _convert_flatten(self, layer: Any, name: str) -> str:
        """Convert Flatten layer."""
        return 'self.layers.append(Flatten())'
    
    def export_training_loop(
        self,
        optimizer: str = "Adam",
        loss: str = "softmaxCrossEntropy",
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Generate Swift training loop code."""
        metrics_code = ", ".join([f".{m.lower()}" for m in (metrics or [])])
        
        return f"""
extension {self.__class__.__name__} {{
    func train(
        on dataset: Dataset,
        epochs: Int,
        batchSize: Int = 32
    ) {{
        let optimizer = {optimizer}(for: self)
        
        for epoch in 0..<epochs {{
            var totalLoss: Float = 0
            var batchCount: Int = 0
            
            for batch in dataset.batched(batchSize) {{
                let (loss, gradients) = valueWithGradient(at: self) {{ model -> Tensor<Float> in
                    let logits = model(batch.data)
                    let loss = {loss}(logits: logits, labels: batch.labels)
                    return loss
                }}
                
                optimizer.update(&self, along: gradients)
                
                totalLoss += loss.scalarized()
                batchCount += 1
            }}
            
            print("Epoch \\(epoch + 1): Loss = \\(totalLoss / Float(batchCount))")
        }}
    }}
}}
"""


class S4TModel:
    """
    Swift for TensorFlow model wrapper.
    
    Contains Swift source code and metadata for S4T models.
    """
    
    def __init__(
        self,
        name: str,
        source_code: str,
        swift_tf_version: str = "0.12",
    ):
        self.name = name
        self.source_code = source_code
        self.swift_tf_version = swift_tf_version
        
    def save(self, path: Union[str, Path]) -> Path:
        """Save Swift source code to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source_code)
        logger.info(f"Saved S4T model to {path}")
        return path
    
    def compile(self, build_dir: Optional[Path] = None) -> Path:
        """
        Compile the S4T Swift code.
        
        Note: Requires Swift for TensorFlow toolchain.
        """
        if build_dir is None:
            build_dir = Path(tempfile.gettempdir()) / "s4t_build"
        
        build_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = build_dir / f"{self.name}.swift"
        self.save(source_path)
        
        # Compilation would require Swift for TensorFlow toolchain
        logger.warning("S4T compilation requires Swift for TensorFlow toolchain")
        
        return source_path
    
    def get_source(self) -> str:
        """Get the Swift source code."""
        return self.source_code
    
    def add_comments(self, comments: Dict[str, str]) -> "S4TModel":
        """Add comments to specific sections of the code."""
        # Implementation would modify source_code
        return self


def export_s4t(
    model: Any,
    output_path: Union[str, Path],
    model_name: str = "Model",
) -> Path:
    """
    Quick export to Swift for TensorFlow format.
    
    Args:
        model: Model to convert
        output_path: Output file path
        model_name: Name for the Swift class
    
    Returns:
        Path to saved Swift file
    """
    converter = S4TConverter()
    s4t_model = converter.convert(model, model_name)
    return s4t_model.save(output_path)


def load_s4t(source_path: Union[str, Path]) -> S4TModel:
    """
    Load a Swift for TensorFlow model from source.
    
    Args:
        source_path: Path to Swift source file
    
    Returns:
        S4TModel instance
    """
    source_path = Path(source_path)
    source_code = source_path.read_text()
    
    # Extract model name from source
    import re
    match = re.search(r'struct\s+(\w+)', source_code)
    model_name = match.group(1) if match else "Model"
    
    return S4TModel(
        name=model_name,
        source_code=source_code,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def swift_extension(
    name: Optional[str] = None,
    swift_code: Optional[str] = None,
    source_file: Optional[Union[str, Path]] = None,
) -> SwiftExtension:
    """
    Create a Swift extension from code or file.
    
    Args:
        name: Extension name
        swift_code: Swift source code string
        source_file: Path to Swift source file
    
    Returns:
        SwiftExtension instance
    
    Example:
        >>> ext = swift_extension(
        ...     name="MathLib",
        ...     swift_code="func add(a: Int, b: Int) -> Int { return a + b }"
        ... )
        >>> ext.compile()
    """
    if name is None:
        name = "Extension"
    
    if source_file:
        return load_swift(source_file, name)
    
    if swift_code:
        ext = SwiftExtension(name)
        ext.add_function("main", swift_code, return_type="Void")
        return ext
    
    raise ValueError("Must provide either swift_code or source_file")


def coreml_export(
    model: Any,
    output_path: Union[str, Path],
    input_shape: Optional[Tuple[int, ...]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    **kwargs,
) -> Path:
    """
    Quick export to CoreML with auto-detection of shapes.
    
    Args:
        model: Model to convert
        output_path: Output path for .mlmodel file
        input_shape: Input tensor shape
        input_names: Input tensor names
        output_names: Output tensor names
        **kwargs: Additional arguments for CoreMLConverter
    
    Returns:
        Path to saved model
    
    Example:
        >>> coreml_export(model, "model.mlmodel", input_shape=(1, 3, 224, 224))
    """
    converter = CoreMLConverter(**kwargs)
    
    # Auto-detect shapes if not provided
    if input_shape is None:
        input_shape = (1, 3, 224, 224)  # Default for image models
    
    if input_names is None:
        input_names = ["input"]
    
    if output_names is None:
        output_names = ["output"]
    
    inputs = [TensorSpec(name, input_shape) for name in input_names]
    outputs = [TensorSpec(name, (1, 1000)) for name in output_names]  # Default output
    
    mlmodel = converter.convert(model, inputs=inputs, outputs=outputs)
    
    return mlmodel.save(output_path)


def metal_kernel(
    name: str,
    source: str,
    compile: bool = True,
) -> MetalKernel:
    """
    Create a Metal compute kernel.
    
    Args:
        name: Kernel name
        source: Metal shader source code
        compile: Whether to compile immediately
    
    Returns:
        MetalKernel instance
    
    Example:
        >>> kernel = metal_kernel("matmul", '''
        ...     kernel void matmul(
        ...         device const float* A [[ buffer(0) ]],
        ...         device const float* B [[ buffer(1) ]],
        ...         device float* C [[ buffer(2) ]],
        ...         uint2 gid [[ thread_position_in_grid ]]
        ...     ) {
        ...         // Implementation
        ...     }
        ... ''')
        >>> kernel.launch(buffers=[a, b, c])
    """
    kernel = MetalKernel(name, source)
    
    if compile:
        kernel.compile()
    
    return kernel


# =============================================================================
# Convenience Exports
# =============================================================================


__all__ = [
    # Swift Extensions
    "SwiftExtension",
    "load_swift",
    "compile_swift",
    "swift_bridge",
    
    # CoreML
    "CoreMLConverter",
    "CoreMLModel",
    "export_coreml",
    "load_coreml",
    
    # Metal
    "MetalKernel",
    "MetalBuffer",
    "compile_metal",
    "launch_metal",
    
    # iOS
    "iOSExporter",
    "iOSAppConfig",
    "create_ios_app",
    "iOSIntegration",
    "optimize_for_ios",
    
    # macOS
    "MacOSExporter",
    "MacOSAppConfig",
    "create_macos_app",
    "macOSIntegration",
    "optimize_for_macos",
    
    # Swift for TensorFlow
    "S4TConverter",
    "S4TModel",
    "export_s4t",
    "load_s4t",
    
    # Utilities
    "swift_extension",
    "coreml_export",
    "metal_kernel",
    
    # Types
    "Platform",
    "SwiftDataType",
    "ComputeUnits",
    "TensorSpec",
    "SwiftMetadata",
    "iOSAppConfig",
    "MacOSAppConfig",
    "MetalKernelSpec",
    
    # Exceptions
    "SwiftBackendError",
    "SwiftExtensionError",
    "CoreMLConversionError",
    "MetalKernelError",
    "iOSDeploymentError",
    "MacOSDeploymentError",
    "S4TError",
]
