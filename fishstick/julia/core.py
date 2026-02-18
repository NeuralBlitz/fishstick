"""
Fishstick Julia Backend Module

Comprehensive Julia integration for Python applications, providing:
- Julia extension management and compilation
- PyCall bidirectional communication
- Flux.jl machine learning integration
- Zygote automatic differentiation
- Performance optimization and JIT compilation
- Scientific computing capabilities
- GPU acceleration support

Author: Fishstick Team
Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, Generic
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import subprocess
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions and Types
# =============================================================================


class JuliaError(Exception):
    """Base exception for Julia-related errors."""

    pass


class JuliaNotFoundError(JuliaError):
    """Raised when Julia is not found or not properly installed."""

    pass


class PyCallError(JuliaError):
    """Raised when PyCall operations fail."""

    pass


class FluxError(JuliaError):
    """Raised when Flux.jl operations fail."""

    pass


class ZygoteError(JuliaError):
    """Raised when Zygote differentiation fails."""

    pass


class JuliaState(Enum):
    """Julia runtime state."""

    UNINITIALIZED = auto()
    LOADING = auto()
    READY = auto()
    ERROR = auto()
    SHUTDOWN = auto()


T = TypeVar("T")
ArrayLike = Union[List, Tuple, "numpy.ndarray"]


# =============================================================================
# Julia Extensions
# =============================================================================


@dataclass
class JuliaExtension:
    """
    Julia Extension Management

    Handles loading, compiling, and managing Julia extensions for Python.
    """

    name: str
    path: Optional[Path] = None
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    compiled: bool = False
    _julia_module: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.path is None:
            self.path = Path.home() / ".fishstick" / "julia_extensions" / self.name
        self.path = Path(self.path)

    def load(self, julia_instance: Optional[Any] = None) -> Any:
        """Load the Julia extension into the runtime."""
        if not self.path.exists():
            raise JuliaError(f"Extension path not found: {self.path}")

        jl = julia_instance or get_julia_runtime()

        # Add extension path to Julia load path
        jl.eval(f'push!(LOAD_PATH, "{self.path.parent}")')

        # Load the module
        try:
            self._julia_module = jl.eval(f"using {self.name}; {self.name}")
            logger.info(f"Loaded Julia extension: {self.name}")
            return self._julia_module
        except Exception as e:
            raise JuliaError(f"Failed to load extension {self.name}: {e}")

    def compile(self, force: bool = False) -> Path:
        """
        Compile the extension to a system image for faster loading.

        Args:
            force: Force recompilation even if already compiled

        Returns:
            Path to the compiled system image
        """
        if self.compiled and not force:
            return self._get_sysimage_path()

        sysimg_path = self._get_sysimage_path()
        sysimg_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate precompilation script
        precompile_script = self._generate_precompile_script()

        # Compile using PackageCompiler
        jl = get_julia_runtime()
        try:
            jl.eval("using PackageCompiler")
            jl.eval(f'''
            create_sysimage(
                :{self.name},
                sysimage_path="{sysimg_path}",
                precompile_execution_file="{precompile_script}"
            )
            ''')
            self.compiled = True
            logger.info(f"Compiled Julia extension: {self.name}")
            return sysimg_path
        except Exception as e:
            raise JuliaError(f"Failed to compile extension {self.name}: {e}")

    def _get_sysimage_path(self) -> Path:
        """Get the path for the compiled system image."""
        return self.path / "compiled" / f"{self.name}_sysimage.so"

    def _generate_precompile_script(self) -> Path:
        """Generate a precompilation script for PackageCompiler."""
        script_path = self.path / "precompile.jl"
        script_content = f"""
        using {self.name}
        
        # Add precompilation workload here
        # This helps PackageCompiler identify what to compile
        
        """
        script_path.write_text(script_content)
        return script_path

    def reload(self) -> Any:
        """Reload the extension (useful during development)."""
        jl = get_julia_runtime()
        try:
            jl.eval(f"Base.reload({self.name})")
            return self.load()
        except Exception as e:
            raise JuliaError(f"Failed to reload extension {self.name}: {e}")


def load_julia(
    julia_path: Optional[str] = None,
    sysimage: Optional[str] = None,
    threads: int = 4,
    **kwargs,
) -> Any:
    """
    Load and initialize the Julia runtime.

    Args:
        julia_path: Path to Julia executable (auto-detected if None)
        sysimage: Path to custom system image for faster startup
        threads: Number of Julia threads
        **kwargs: Additional arguments passed to Julia initialization

    Returns:
        Julia runtime object (jl)
    """
    global _JULIA_RUNTIME, _JULIA_STATE

    if _JULIA_STATE == JuliaState.READY and _JULIA_RUNTIME is not None:
        return _JULIA_RUNTIME

    _JULIA_STATE = JuliaState.LOADING

    try:
        from julia import Julia

        # Configure Julia
        julia_kwargs = {
            "runtime": julia_path,
            "sysimage": sysimage,
            "threads": threads,
            **kwargs,
        }

        # Filter out None values
        julia_kwargs = {k: v for k, v in julia_kwargs.items() if v is not None}

        # Initialize Julia
        jl = Julia(**julia_kwargs)

        # Set environment variables for optimal performance
        jl.eval('ENV["JULIA_NUM_THREADS"] = {}'.format(threads))
        jl.eval('ENV["OPENBLAS_NUM_THREADS"] = {}'.format(threads))

        # Install essential packages if needed
        _ensure_packages(jl, ["Pkg", "LinearAlgebra", "Random", "Statistics"])

        _JULIA_RUNTIME = jl
        _JULIA_STATE = JuliaState.READY

        logger.info("Julia runtime initialized successfully")
        return jl

    except ImportError:
        raise JuliaNotFoundError(
            "julia package not found. Install with: pip install julia"
        )
    except Exception as e:
        _JULIA_STATE = JuliaState.ERROR
        raise JuliaError(f"Failed to initialize Julia: {e}")


def compile_julia(
    extensions: List[Union[str, JuliaExtension]],
    output_path: Optional[str] = None,
    precompile_scripts: Optional[List[str]] = None,
) -> str:
    """
    Compile Julia code and extensions into a system image.

    Args:
        extensions: List of extension names or JuliaExtension objects
        output_path: Where to save the system image
        precompile_scripts: Scripts to run during precompilation

    Returns:
        Path to the compiled system image
    """
    jl = get_julia_runtime()

    # Convert string names to JuliaExtension objects
    ext_objects = []
    for ext in extensions:
        if isinstance(ext, str):
            ext_objects.append(JuliaExtension(name=ext))
        else:
            ext_objects.append(ext)

    # Determine output path
    if output_path is None:
        output_path = str(
            Path.home() / ".fishstick" / "julia_sysimages" / "fishstick_custom.so"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure PackageCompiler is available
    _ensure_packages(jl, ["PackageCompiler"])

    # Generate precompilation workload
    workload = []
    for ext in ext_objects:
        workload.append(f"using {ext.name}")

    if precompile_scripts:
        for script in precompile_scripts:
            workload.append(f'include("{script}")')

    workload_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jl", delete=False)
    workload_file.write("\n".join(workload))
    workload_file.close()

    try:
        # Create sysimage
        ext_symbols = ", ".join(f":{ext.name}" for ext in ext_objects)
        jl.eval("using PackageCompiler")
        jl.eval(f'''
        create_sysimage(
            [{ext_symbols}],
            sysimage_path="{output_path}",
            precompile_execution_file="{workload_file.name}"
        )
        ''')

        logger.info(f"Compiled system image saved to: {output_path}")
        return output_path

    finally:
        os.unlink(workload_file.name)


def julia_bridge(
    python_objects: Optional[Dict[str, Any]] = None,
    julia_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a bidirectional bridge between Python and Julia.

    Args:
        python_objects: Dictionary of Python objects to expose to Julia
        julia_modules: List of Julia modules to expose to Python

    Returns:
        Dictionary containing bridge objects
    """
    jl = get_julia_runtime()

    bridge = {
        "julia_runtime": jl,
        "python_to_julia": {},
        "julia_to_python": {},
    }

    # Export Python objects to Julia
    if python_objects:
        from julia import Main

        for name, obj in python_objects.items():
            setattr(Main, name, obj)
            bridge["python_to_julia"][name] = obj
            logger.debug(f"Exported Python object to Julia: {name}")

    # Import Julia modules to Python
    if julia_modules:
        for module_name in julia_modules:
            try:
                jl.eval(f"using {module_name}")
                module = jl.eval(module_name)
                bridge["julia_to_python"][module_name] = module
                logger.debug(f"Imported Julia module to Python: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to import Julia module {module_name}: {e}")

    return bridge


# =============================================================================
# PyCall Integration
# =============================================================================


@dataclass
class PyCallIntegration:
    """
    PyCall Bidirectional Integration

    Manages bidirectional communication between Python and Julia using PyCall.
    """

    jl: Any = field(default=None, repr=False)
    _initialized: bool = False
    _python_objects: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.jl is None:
            self.jl = get_julia_runtime()
        self._initialize_pycall()

    def _initialize_pycall(self):
        """Initialize PyCall in Julia."""
        if self._initialized:
            return

        try:
            self.jl.eval("using PyCall")

            # Configure PyCall to use current Python
            python_exe = sys.executable
            self.jl.eval(f'ENV["PYTHON"] = "{python_exe}"')
            self.jl.eval("PyCall.python")

            self._initialized = True
            logger.info("PyCall initialized")
        except Exception as e:
            raise PyCallError(f"Failed to initialize PyCall: {e}")

    def export_to_julia(self, name: str, obj: Any) -> None:
        """
        Export a Python object to Julia.

        Args:
            name: Name to use in Julia
            obj: Python object to export
        """
        from julia import Main

        setattr(Main, name, obj)
        self._python_objects[name] = obj
        logger.debug(f"Exported {name} to Julia")

    def import_from_julia(self, name: str) -> Any:
        """
        Import a Julia object to Python.

        Args:
            name: Name of the Julia object (can include module path)

        Returns:
            The Julia object
        """
        try:
            return self.jl.eval(name)
        except Exception as e:
            raise PyCallError(f"Failed to import {name} from Julia: {e}")

    def call_julia_function(self, function_name: str, *args, **kwargs) -> Any:
        """
        Call a Julia function from Python.

        Args:
            function_name: Name of the Julia function
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        try:
            func = self.jl.eval(function_name)
            return func(*args, **kwargs)
        except Exception as e:
            raise PyCallError(f"Failed to call Julia function {function_name}: {e}")

    def evaluate(self, code: str) -> Any:
        """
        Evaluate Julia code from Python.

        Args:
            code: Julia code to evaluate

        Returns:
            Result of the evaluation
        """
        try:
            return self.jl.eval(code)
        except Exception as e:
            raise PyCallError(f"Julia evaluation error: {e}")

    def create_callback(self, python_func: Callable) -> Any:
        """
        Create a Julia-callable wrapper for a Python function.

        Args:
            python_func: Python function to wrap

        Returns:
            Julia-callable function object
        """
        from julia import Main

        callback_name = f"_pycall_callback_{id(python_func)}"
        setattr(Main, callback_name, python_func)
        return self.jl.eval(f"({callback_name})")


def export_to_julia(
    obj: Any, name: Optional[str] = None, julia_instance: Optional[Any] = None
) -> str:
    """
    Export a Python object to Julia.

    Args:
        obj: Python object to export
        name: Name to use in Julia (auto-generated if None)
        julia_instance: Julia runtime instance

    Returns:
        Name used in Julia
    """
    if name is None:
        name = f"_pyobj_{id(obj)}"

    jl = julia_instance or get_julia_runtime()
    from julia import Main

    setattr(Main, name, obj)

    logger.debug(f"Exported Python object to Julia as: {name}")
    return name


def call_julia_from_python(function: Union[str, Callable], *args, **kwargs) -> Any:
    """
    Call a Julia function from Python.

    Args:
        function: Julia function name or callable
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    jl = get_julia_runtime()

    if isinstance(function, str):
        func = jl.eval(function)
    else:
        func = function

    return func(*args, **kwargs)


class JuliaPythonBridge:
    """
    High-level bridge for Python-Julia interoperability.

    Provides a seamless interface for bidirectional data and function exchange.
    """

    def __init__(self):
        self.jl = get_julia_runtime()
        self.pycall = PyCallIntegration(self.jl)
        self._exported: Dict[str, Any] = {}
        self._imported: Dict[str, Any] = {}

    def export(self, name: str, obj: Any) -> "JuliaPythonBridge":
        """Export a Python object to Julia (chainable)."""
        self.pycall.export_to_julia(name, obj)
        self._exported[name] = obj
        return self

    def import_module(self, module_name: str) -> Any:
        """Import a Julia module."""
        self.jl.eval(f"using {module_name}")
        module = self.jl.eval(module_name)
        self._imported[module_name] = module
        return module

    def import_function(self, function_path: str) -> Callable:
        """Import a Julia function as a Python callable."""
        func = self.jl.eval(function_path)
        return func

    def execute(self, code: str) -> Any:
        """Execute Julia code."""
        return self.pycall.evaluate(code)

    def convert_array(self, array: ArrayLike, to_julia: bool = True) -> Any:
        """
        Convert arrays between Python and Julia.

        Args:
            array: Array to convert
            to_julia: If True, convert Python to Julia; otherwise Julia to Python

        Returns:
            Converted array
        """
        import numpy as np

        if to_julia:
            np_array = np.asarray(array)
            return self.jl.eval("PyArray")(np_array)
        else:
            # Julia arrays are automatically converted to numpy
            return np.asarray(array)

    def create_proxy(self, python_object: Any) -> Any:
        """
        Create a Julia proxy object for a Python object.

        This allows Julia code to access Python object attributes and methods.
        """
        name = export_to_julia(python_object, julia_instance=self.jl)
        return self.jl.eval(f"PyObject({name})")


# =============================================================================
# Flux Integration
# =============================================================================


@dataclass
class FluxConverter:
    """
    Flux.jl Model Converter

    Converts between Python ML models and Flux.jl models.
    """

    jl: Any = field(default=None, repr=False)
    _flux_loaded: bool = False

    def __post_init__(self):
        if self.jl is None:
            self.jl = get_julia_runtime()
        self._ensure_flux()

    def _ensure_flux(self):
        """Ensure Flux.jl is loaded."""
        if self._flux_loaded:
            return

        _ensure_packages(self.jl, ["Flux", "NNlib", "ChainRulesCore"])
        self.jl.eval("using Flux")
        self._flux_loaded = True

    def convert_pytorch_to_flux(
        self, pytorch_model: Any, input_shape: Optional[Tuple[int, ...]] = None
    ) -> Any:
        """
        Convert a PyTorch model to Flux.

        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Expected input shape for validation

        Returns:
            Flux.jl model
        """
        import torch

        # Export PyTorch model structure
        model_dict = pytorch_model.state_dict()

        # Create Flux model equivalent
        flux_model_code = self._generate_flux_model(pytorch_model)

        # Load weights into Flux model
        self.jl.eval(flux_model_code)
        flux_model = self.jl.eval("model")

        # Transfer weights
        for name, param in model_dict.items():
            julia_param = self.jl.eval(f"Flux.params(model).{name}")
            julia_param[:] = param.numpy()

        return flux_model

    def _generate_flux_model(self, pytorch_model: Any) -> str:
        """Generate Flux model code from PyTorch model structure."""
        # This is a simplified version - full implementation would need
        # to handle all layer types
        layers = []

        for name, module in pytorch_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layers.append(f"Dense({module.in_features}, {module.out_features})")
            elif isinstance(module, torch.nn.ReLU):
                layers.append("relu")
            elif isinstance(module, torch.nn.Conv2d):
                layers.append(
                    f"Conv(({module.kernel_size[0]}, {module.kernel_size[1]}), "
                    f"{module.in_channels} => {module.out_channels})"
                )

        flux_code = f"""
        model = Chain(
            {", ".join(layers)}
        )
        """
        return flux_code

    def convert_flux_to_pytorch(
        self, flux_model: Any, pytorch_class: Optional[type] = None
    ) -> Any:
        """
        Convert a Flux model to PyTorch.

        Args:
            flux_model: Flux.jl model
            pytorch_class: Target PyTorch model class

        Returns:
            PyTorch model
        """
        import torch
        import torch.nn as nn

        # Extract Flux model parameters
        flux_params = self.jl.eval("Flux.params(model)")

        if pytorch_class is None:
            # Create equivalent PyTorch model
            pytorch_model = self._create_pytorch_equivalent(flux_model)
        else:
            pytorch_model = pytorch_class()

        # Transfer parameters
        with torch.no_grad():
            for py_param, flux_param in zip(pytorch_model.parameters(), flux_params):
                py_param.copy_(torch.tensor(flux_param))

        return pytorch_model

    def _create_pytorch_equivalent(self, flux_model: Any) -> Any:
        """Create a PyTorch model equivalent to a Flux model."""
        import torch.nn as nn

        # Parse Flux model structure and create equivalent PyTorch model
        # This is a simplified implementation
        return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def export_to_flux(
    model: Any, filename: Optional[str] = None, format: str = "bson"
) -> str:
    """
    Export a model to Flux.jl format.

    Args:
        model: Model to export (Python or Flux model)
        filename: Output filename
        format: Export format ("bson", "jld2", "json")

    Returns:
        Path to exported file
    """
    jl = get_julia_runtime()
    _ensure_packages(jl, ["BSON", "JLD2"])

    if filename is None:
        filename = f"model.{format}"

    if format == "bson":
        jl.eval(f"using BSON")
        export_code = f'''
        using BSON
        BSON.@save "{filename}" model
        '''
    elif format == "jld2":
        jl.eval(f"using JLD2")
        export_code = f'''
        using JLD2
        jldsave("{filename}"; model)
        '''
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Export current model
    from julia import Main

    setattr(Main, "model", model)
    jl.eval(export_code)

    logger.info(f"Exported model to Flux format: {filename}")
    return filename


def load_flux_model(filename: str, format: str = "bson") -> Any:
    """
    Load a Flux.jl model.

    Args:
        filename: Path to model file
        format: File format ("bson", "jld2", "json")

    Returns:
        Loaded Flux model
    """
    jl = get_julia_runtime()

    if format == "bson":
        _ensure_packages(jl, ["BSON"])
        jl.eval(f"using BSON")
        model = jl.eval(f'BSON.load("{filename}")[:model]')
    elif format == "jld2":
        _ensure_packages(jl, ["JLD2"])
        jl.eval(f"using JLD2")
        model = jl.eval(f'JLD2.load("{filename}", "model")')
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Loaded Flux model from: {filename}")
    return model


@dataclass
class FluxModel:
    """
    Flux.jl Model Wrapper

    Provides a Pythonic interface to Flux.jl models.
    """

    julia_model: Any
    jl: Any = field(default=None, repr=False)
    _converter: Optional[FluxConverter] = field(default=None, repr=False)

    def __post_init__(self):
        if self.jl is None:
            self.jl = get_julia_runtime()
        if self._converter is None:
            self._converter = FluxConverter(self.jl)

    def predict(self, x: ArrayLike) -> Any:
        """Run inference on input data."""
        import numpy as np

        x_jl = self.jl.eval("PyArray")(np.asarray(x))
        result = self.julia_model(x_jl)
        return np.asarray(result)

    def train(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            x_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size

        Returns:
            Training history
        """
        import numpy as np

        _ensure_packages(self.jl, ["Flux", "MLUtils"])
        self.jl.eval("using Flux, MLUtils")

        x_data = self.jl.eval("PyArray")(np.asarray(x_train))
        y_data = self.jl.eval("PyArray")(np.asarray(y_train))

        # Create data loader
        loader = self.jl.eval(f"DataLoader((x_data, y_data), batchsize={batch_size})")

        # Define loss and optimizer
        self.jl.eval(f"opt = Adam({learning_rate})")

        # Training loop
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            epoch_loss = []

            # Julia training code
            train_code = f"""
            Flux.train!(
                (m, x, y) -> Flux.Losses.logitcrossentropy(m(x), y),
                Flux.params(model),
                loader,
                opt
            )
            """

            from julia import Main

            setattr(Main, "model", self.julia_model)
            setattr(Main, "loader", loader)
            self.jl.eval(train_code)

            # Compute epoch metrics
            loss = self.jl.eval("Flux.Losses.logitcrossentropy(model(x_data), y_data)")
            history["loss"].append(float(loss))

        return history

    def save(self, filename: str, format: str = "bson") -> None:
        """Save the model to disk."""
        from julia import Main

        setattr(Main, "model", self.julia_model)
        export_to_flux(self.julia_model, filename, format)

    @classmethod
    def load(cls, filename: str, format: str = "bson") -> "FluxModel":
        """Load a model from disk."""
        model = load_flux_model(filename, format)
        return cls(julia_model=model)


# =============================================================================
# Differentiation (Zygote)
# =============================================================================


@dataclass
class ZygoteIntegration:
    """
    Zygote Automatic Differentiation Integration

    Provides automatic differentiation capabilities using Zygote.jl.
    """

    jl: Any = field(default=None, repr=False)
    _zygote_loaded: bool = False

    def __post_init__(self):
        if self.jl is None:
            self.jl = get_julia_runtime()
        self._ensure_zygote()

    def _ensure_zygote(self):
        """Ensure Zygote is loaded."""
        if self._zygote_loaded:
            return

        _ensure_packages(self.jl, ["Zygote", "ChainRulesCore"])
        self.jl.eval("using Zygote")
        self._zygote_loaded = True

    def automatic_differentiation(
        self,
        func: Union[str, Callable],
        inputs: Union[ArrayLike, List[ArrayLike]],
        return_gradients: bool = True,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Compute automatic differentiation of a function.

        Args:
            func: Function to differentiate (Julia code string or callable)
            inputs: Input values
            return_gradients: Whether to return gradients

        Returns:
            Tuple of (function_value, gradients)
        """
        import numpy as np

        # Prepare inputs
        if isinstance(inputs, (list, tuple)):
            jl_inputs = [self.jl.eval("PyArray")(np.asarray(x)) for x in inputs]
        else:
            jl_inputs = [self.jl.eval("PyArray")(np.asarray(inputs))]

        # Set up function in Julia
        if isinstance(func, str):
            func_code = f"""
            function _ad_func(x)
                {func}
            end
            """
            self.jl.eval(func_code)
            julia_func = self.jl.eval("_ad_func")
        else:
            # Export Python function to Julia
            from julia import Main

            func_name = f"_py_func_{id(func)}"
            setattr(Main, func_name, func)
            julia_func = self.jl.eval(f"({func_name})")

        # Compute using Zygote
        if return_gradients:
            result = self.jl.eval(f"Zygote.withgradient({julia_func}, {jl_inputs[0]})")
            value = result.val
            grads = result.grad
            return value, grads
        else:
            value = julia_func(*jl_inputs)
            return value, None

    def gradient_computation(
        self,
        loss_function: Union[str, Callable],
        params: List[ArrayLike],
        inputs: ArrayLike,
        targets: Optional[ArrayLike] = None,
    ) -> List[Any]:
        """
        Compute gradients of a loss function with respect to parameters.

        Args:
            loss_function: Loss function
            params: Model parameters
            inputs: Input data
            targets: Target labels

        Returns:
            List of gradients for each parameter
        """
        import numpy as np

        # Convert to Julia arrays
        jl_params = [self.jl.eval("PyArray")(np.asarray(p)) for p in params]
        jl_inputs = self.jl.eval("PyArray")(np.asarray(inputs))

        if targets is not None:
            jl_targets = self.jl.eval("PyArray")(np.asarray(targets))
        else:
            jl_targets = None

        # Set up loss function
        if isinstance(loss_function, str):
            self.jl.eval(f"_loss_func = (params, x, y) -> begin {loss_function} end")
            loss_func = self.jl.eval("_loss_func")
        else:
            from julia import Main

            func_name = f"_loss_{id(loss_function)}"
            setattr(Main, func_name, loss_function)
            loss_func = self.jl.eval(f"({func_name})")

        # Compute gradient
        grad_code = """
        Zygote.gradient(ps -> loss_func(ps, x, y), params)
        """

        from julia import Main

        setattr(Main, "params", jl_params)
        setattr(Main, "x", jl_inputs)
        setattr(Main, "y", jl_targets)
        setattr(Main, "loss_func", loss_func)

        grads = self.jl.eval(grad_code)
        return grads

    def hessian_computation(self, func: Union[str, Callable], inputs: ArrayLike) -> Any:
        """
        Compute the Hessian matrix of a function.

        Args:
            func: Function to compute Hessian of
            inputs: Input values

        Returns:
            Hessian matrix
        """
        import numpy as np

        jl_inputs = self.jl.eval("PyArray")(np.asarray(inputs))

        # Set up function
        if isinstance(func, str):
            self.jl.eval(f"_hess_func = x -> begin {func} end")
            hess_func = self.jl.eval("_hess_func")
        else:
            from julia import Main

            func_name = f"_hess_{id(func)}"
            setattr(Main, func_name, func)
            hess_func = self.jl.eval(f"({func_name})")

        # Compute Hessian using ForwardDiff (more stable for Hessians)
        _ensure_packages(self.jl, ["ForwardDiff"])
        self.jl.eval("using ForwardDiff")

        hessian = self.jl.eval(f"ForwardDiff.hessian({hess_func}, {jl_inputs})")
        return np.asarray(hessian)

    def jacobian(self, func: Union[str, Callable], inputs: ArrayLike) -> Any:
        """
        Compute the Jacobian matrix of a vector-valued function.

        Args:
            func: Vector-valued function
            inputs: Input values

        Returns:
            Jacobian matrix
        """
        import numpy as np

        jl_inputs = self.jl.eval("PyArray")(np.asarray(inputs))

        if isinstance(func, str):
            self.jl.eval(f"_jac_func = x -> begin {func} end")
            jac_func = self.jl.eval("_jac_func")
        else:
            from julia import Main

            func_name = f"_jac_{id(func)}"
            setattr(Main, func_name, func)
            jac_func = self.jl.eval(f"({func_name})")

        jacobian = self.jl.eval(f"Zygote.jacobian({jac_func}, {jl_inputs})")
        return np.asarray(jacobian)

    def pullback(
        self,
        func: Union[str, Callable],
        inputs: ArrayLike,
        seed: Optional[ArrayLike] = None,
    ) -> Tuple[Any, Callable]:
        """
        Compute the pullback (vector-Jacobian product).

        Args:
            func: Function to compute pullback for
            inputs: Input values
            seed: Seed vector for pullback

        Returns:
            Tuple of (function_value, pullback_function)
        """
        import numpy as np

        jl_inputs = self.jl.eval("PyArray")(np.asarray(inputs))

        if isinstance(func, str):
            self.jl.eval(f"_pb_func = x -> begin {func} end")
            pb_func = self.jl.eval("_pb_func")
        else:
            from julia import Main

            func_name = f"_pb_{id(func)}"
            setattr(Main, func_name, func)
            pb_func = self.jl.eval(f"({func_name})")

        y, back = self.jl.eval(f"Zygote.pullback({pb_func}, {jl_inputs})")

        return y, back


def automatic_differentiation(
    func: Union[str, Callable], inputs: ArrayLike, julia_instance: Optional[Any] = None
) -> Tuple[Any, Any]:
    """
    Compute function value and gradient using automatic differentiation.

    Args:
        func: Function to differentiate
        inputs: Input values
        julia_instance: Julia runtime instance

    Returns:
        Tuple of (value, gradient)
    """
    zygote = ZygoteIntegration(julia_instance)
    return zygote.automatic_differentiation(func, inputs, return_gradients=True)


def gradient_computation(
    loss_func: Union[str, Callable],
    params: List[ArrayLike],
    inputs: ArrayLike,
    targets: Optional[ArrayLike] = None,
    julia_instance: Optional[Any] = None,
) -> List[Any]:
    """
    Compute gradients of loss with respect to parameters.

    Args:
        loss_func: Loss function
        params: Model parameters
        inputs: Input data
        targets: Target labels
        julia_instance: Julia runtime instance

    Returns:
        List of gradients
    """
    zygote = ZygoteIntegration(julia_instance)
    return zygote.gradient_computation(loss_func, params, inputs, targets)


def hessian_computation(
    func: Union[str, Callable], inputs: ArrayLike, julia_instance: Optional[Any] = None
) -> Any:
    """
    Compute Hessian matrix.

    Args:
        func: Function to compute Hessian of
        inputs: Input values
        julia_instance: Julia runtime instance

    Returns:
        Hessian matrix
    """
    zygote = ZygoteIntegration(julia_instance)
    return zygote.hessian_computation(func, inputs)


# =============================================================================
# Performance Optimization
# =============================================================================


@dataclass
class JuliaOptimizer:
    """
    Julia Performance Optimizer

    Provides tools for optimizing Julia code performance.
    """

    jl: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.jl is None:
            self.jl = get_julia_runtime()

    def jit_compilation(
        self, func: Union[str, Callable], argument_types: Optional[List[type]] = None
    ) -> Callable:
        """
        Apply JIT compilation to a function.

        In Julia, all functions are JIT compiled by default. This method
        provides additional optimization hints.

        Args:
            func: Function to optimize
            argument_types: Expected argument types for specialization

        Returns:
            Optimized function
        """
        if isinstance(func, str):
            # Compile Julia code with @fastmath and @inbounds
            optimized_code = f"""
            @inline function _jit_func(args...)
                @fastmath @inbounds begin
                    {func}
                end
            end
            """
            self.jl.eval(optimized_code)
            return self.jl.eval("_jit_func")
        else:
            # Python function - wrap in Julia
            from julia import Main

            func_name = f"_jit_py_{id(func)}"
            setattr(Main, func_name, func)

            wrapped_code = f"""
            function _jit_wrapped(args...)
                @fastmath @inbounds begin
                    return {func_name}(args...)
                end
            end
            """
            self.jl.eval(wrapped_code)
            return self.jl.eval("_jit_wrapped")

    def optimize_loop(
        self, loop_code: str, vectorize: bool = True, parallel: bool = False
    ) -> str:
        """
        Optimize loop code with various strategies.

        Args:
            loop_code: Julia loop code to optimize
            vectorize: Use vectorized operations
            parallel: Use parallel execution

        Returns:
            Optimized code
        """
        optimizations = []

        if vectorize:
            optimizations.append("using LoopVectorization")
            loop_code = loop_code.replace("for ", "@turbo for ")

        if parallel:
            optimizations.append("using Base.Threads")
            loop_code = loop_code.replace("for ", "@threads for ")

        optimized = "\n".join(optimizations) + "\n" + loop_code
        return optimized

    def profile_code(self, code: str, runs: int = 1000) -> Dict[str, Any]:
        """
        Profile Julia code performance.

        Args:
            code: Julia code to profile
            runs: Number of benchmark runs

        Returns:
            Profiling results
        """
        _ensure_packages(self.jl, ["BenchmarkTools"])
        self.jl.eval("using BenchmarkTools")

        # Create benchmark
        self.jl.eval(f"_bench_func() = begin {code} end")
        result = self.jl.eval(f"@benchmark _bench_func() samples={runs}")

        return {
            "minimum_time": float(result.time.min),
            "median_time": float(result.time.median),
            "mean_time": float(result.time.mean),
            "maximum_time": float(result.time.max),
            "memory_allocations": int(result.memory.allocs),
            "memory_bytes": int(result.memory.bytes),
        }

    def allocate_gpu_arrays(
        self, shape: Tuple[int, ...], dtype: str = "Float32", device: int = 0
    ) -> Any:
        """
        Allocate GPU arrays for CUDA computations.

        Args:
            shape: Array shape
            dtype: Data type
            device: GPU device index

        Returns:
            GPU array
        """
        _ensure_packages(self.jl, ["CUDA"])
        self.jl.eval("using CUDA")

        shape_str = ", ".join(str(s) for s in shape)
        array = self.jl.eval(f"CUDA.zeros({dtype}, ({shape_str}))")
        return array


def jit_compilation(
    func: Union[str, Callable], julia_instance: Optional[Any] = None
) -> Callable:
    """
    Apply JIT compilation to optimize a function.

    Args:
        func: Function to compile
        julia_instance: Julia runtime instance

    Returns:
        Compiled function
    """
    optimizer = JuliaOptimizer(julia_instance)
    return optimizer.jit_compilation(func)


def parallel_julia(
    func: Callable,
    data_chunks: List[ArrayLike],
    n_workers: Optional[int] = None,
    julia_instance: Optional[Any] = None,
) -> List[Any]:
    """
    Execute function in parallel using Julia's threading.

    Args:
        func: Function to execute
        data_chunks: Data divided into chunks
        n_workers: Number of parallel workers
        julia_instance: Julia runtime instance

    Returns:
        List of results
    """
    jl = julia_instance or get_julia_runtime()

    if n_workers is None:
        n_workers = jl.eval("Threads.nthreads()")

    # Export function to Julia
    from julia import Main

    func_name = f"_par_func_{id(func)}"
    setattr(Main, func_name, func)

    # Create parallel execution code
    jl.eval(f"chunks = {data_chunks}")
    jl.eval(f"results = Vector{{Any}}(undef, length(chunks))")

    parallel_code = f"""
    @threads for i in 1:length(chunks)
        results[i] = {func_name}(chunks[i])
    end
    results
    """

    results = jl.eval(parallel_code)
    return list(results)


def gpu_acceleration(
    arrays: List[ArrayLike], operation: str, julia_instance: Optional[Any] = None
) -> Any:
    """
    Execute operations on GPU using CUDA.jl.

    Args:
        arrays: Input arrays to transfer to GPU
        operation: Julia code for the GPU operation
        julia_instance: Julia runtime instance

    Returns:
        Result (transferred back to CPU)
    """
    import numpy as np

    jl = julia_instance or get_julia_runtime()
    _ensure_packages(jl, ["CUDA"])
    jl.eval("using CUDA")

    # Transfer arrays to GPU
    gpu_arrays = []
    for i, arr in enumerate(arrays):
        arr_name = f"_gpu_arr_{i}"
        setattr(jl.Main, arr_name, jl.eval("CuArray")(np.asarray(arr)))
        gpu_arrays.append(arr_name)

    # Execute operation
    gpu_code = operation.format(
        **{f"arr{i}": name for i, name in enumerate(gpu_arrays)}
    )
    result = jl.eval(gpu_code)

    # Transfer result back to CPU
    return np.asarray(result)


# =============================================================================
# Scientific Computing
# =============================================================================


class DifferentialEquations:
    """
    DifferentialEquations.jl Integration

    Provides ODE, PDE, and SDE solving capabilities.
    """

    def __init__(self, julia_instance: Optional[Any] = None):
        self.jl = julia_instance or get_julia_runtime()
        self._ensure_deps()

    def _ensure_deps(self):
        """Ensure DifferentialEquations package is loaded."""
        _ensure_packages(self.jl, ["DifferentialEquations", "OrdinaryDiffEq"])
        self.jl.eval("using DifferentialEquations")

    def solve_ode(
        self,
        equation: Union[str, Callable],
        initial_condition: ArrayLike,
        tspan: Tuple[float, float],
        solver: str = "Tsit5()",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Solve an ordinary differential equation.

        Args:
            equation: ODE definition (dy/dt = f(y, t, p))
            initial_condition: Initial state
            tspan: Time span (t_start, t_end)
            solver: Solver algorithm
            **kwargs: Additional solver options

        Returns:
            Solution dictionary
        """
        import numpy as np

        u0 = self.jl.eval("PyArray")(np.asarray(initial_condition))
        t0, tf = tspan

        # Set up ODE function
        if isinstance(equation, str):
            ode_code = f"""
            function ode_func!(du, u, p, t)
                {equation}
            end
            """
            self.jl.eval(ode_code)
            ode_func = self.jl.eval("ode_func!")
        else:
            from julia import Main

            func_name = f"_ode_{id(equation)}"
            setattr(Main, func_name, equation)
            ode_func = self.jl.eval(f"({func_name})")

        # Create and solve problem
        problem_code = f"""
        prob = ODEProblem(ode_func!, u0, ({t0}, {tf}))
        sol = solve(prob, {solver}; {self._kwargs_to_julia(**kwargs)})
        sol
        """

        solution = self.jl.eval(problem_code)

        return {
            "t": np.asarray(solution.t),
            "u": np.asarray(solution.u),
            "success": solution.retcode == "Success",
            "stats": {
                "nf": solution.stats.nf,
                "naccept": solution.stats.naccept,
                "nreject": solution.stats.nreject,
            },
        }

    def solve_pde(
        self,
        equation: str,
        boundary_conditions: Dict[str, Any],
        domain: Dict[str, Tuple[float, float]],
        initial_condition: ArrayLike,
        tspan: Tuple[float, float],
        discretization: str = "FiniteDifference",
    ) -> Dict[str, Any]:
        """
        Solve a partial differential equation.

        Args:
            equation: PDE definition
            boundary_conditions: Boundary condition specifications
            domain: Spatial domain
            initial_condition: Initial condition
            tspan: Time span
            discretization: Discretization method

        Returns:
            Solution dictionary
        """
        _ensure_packages(self.jl, ["MethodOfLines", "DomainSets", "ModelingToolkit"])
        self.jl.eval("using MethodOfLines, DomainSets, ModelingToolkit")

        # Build PDE problem
        pde_code = f"""
        @parameters t {" ".join(domain.keys())}
        @variables u(..)
        Dt = Differential(t)
        Dxx = Differential(x)^2
        
        eq = {equation}
        
        bcs = {boundary_conditions}
        domains = {domain}
        
        @named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        
        discretization = MOLFiniteDifference([x => 0.1], t)
        prob = discretize(pde_system, discretization)
        sol = solve(prob, Tsit5())
        sol
        """

        solution = self.jl.eval(pde_code)

        import numpy as np

        return {
            "t": np.asarray(solution.t),
            "u": np.asarray(solution.u),
        }

    def _kwargs_to_julia(self, **kwargs) -> str:
        """Convert Python kwargs to Julia keyword arguments."""
        julia_kwargs = []
        for key, value in kwargs.items():
            if isinstance(value, str):
                julia_kwargs.append(f"{key}={value}")
            elif isinstance(value, bool):
                julia_kwargs.append(f"{key}={'true' if value else 'false'}")
            else:
                julia_kwargs.append(f"{key}={value}")
        return ", ".join(julia_kwargs)


class Optimization:
    """
    Optimization.jl Integration

    Provides optimization capabilities for various problem types.
    """

    def __init__(self, julia_instance: Optional[Any] = None):
        self.jl = julia_instance or get_julia_runtime()
        self._ensure_deps()

    def _ensure_deps(self):
        """Ensure Optimization package is loaded."""
        _ensure_packages(
            self.jl, ["Optimization", "OptimizationOptimJL", "OptimizationNLopt"]
        )
        self.jl.eval("using Optimization")

    def minimize(
        self,
        objective: Union[str, Callable],
        initial_guess: ArrayLike,
        method: str = "LBFGS()",
        bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
        constraints: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Minimize an objective function.

        Args:
            objective: Objective function
            initial_guess: Starting point
            method: Optimization algorithm
            bounds: (lower_bounds, upper_bounds)
            constraints: List of constraint strings
            **kwargs: Additional optimizer options

        Returns:
            Optimization result
        """
        import numpy as np

        x0 = self.jl.eval("PyArray")(np.asarray(initial_guess))

        # Set up objective
        if isinstance(objective, str):
            obj_code = f"""
            obj_func(u, p) = begin {objective} end
            """
            self.jl.eval(obj_code)
            obj_func = self.jl.eval("obj_func")
        else:
            from julia import Main

            func_name = f"_obj_{id(objective)}"
            setattr(Main, func_name, objective)
            obj_func = self.jl.eval(f"({func_name})")

        # Create optimization problem
        opt_code = f"""
        opt_f = OptimizationFunction(obj_func, Optimization.AutoZygote())
        prob = OptimizationProblem(opt_f, x0)
        sol = solve(prob, {method})
        sol
        """

        solution = self.jl.eval(opt_code)

        return {
            "minimum": float(solution.minimum),
            "minimizer": np.asarray(solution.minimizer),
            "iterations": solution.iterations,
            "success": solution.retcode == "Success",
        }

    def least_squares(
        self,
        residual_func: Union[str, Callable],
        initial_guess: ArrayLike,
        data: ArrayLike,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Solve a nonlinear least squares problem.

        Args:
            residual_func: Residual function
            initial_guess: Starting parameters
            data: Data to fit
            **kwargs: Additional options

        Returns:
            Fitting result
        """
        # Use LeastSquaresOptim or similar
        _ensure_packages(self.jl, ["LeastSquaresOptim"])
        self.jl.eval("using LeastSquaresOptim")

        import numpy as np

        x0 = self.jl.eval("PyArray")(np.asarray(initial_guess))
        y_data = self.jl.eval("PyArray")(np.asarray(data))

        ls_code = f"""
        function residuals!(out, x)
            out .= {residual_func}(x) .- y_data
            out
        end
        
        prob = LeastSquaresProblem(
            x = x0,
            f! = residuals!,
            output_length = length(y_data)
        )
        
        res = optimize!(prob, Dogleg())
        res
        """

        result = self.jl.eval(ls_code)

        return {
            "minimizer": np.asarray(result.minimizer),
            "ssr": float(result.ssr),
            "iterations": result.iterations,
        }


class LinearAlgebra:
    """
    Julia Linear Algebra Integration

    High-performance linear algebra operations.
    """

    def __init__(self, julia_instance: Optional[Any] = None):
        self.jl = julia_instance or get_julia_runtime()
        self.jl.eval("using LinearAlgebra")

    def svd(self, matrix: ArrayLike, full_matrices: bool = False) -> Dict[str, Any]:
        """
        Compute singular value decomposition.

        Args:
            matrix: Input matrix
            full_matrices: Return full U and Vh matrices

        Returns:
            Dictionary with U, S, Vh
        """
        import numpy as np

        A = self.jl.eval("PyArray")(np.asarray(matrix))

        svd_result = self.jl.eval(f"svd(A, full={str(full_matrices).lower()})")

        return {
            "U": np.asarray(svd_result.U),
            "S": np.asarray(svd_result.S),
            "Vh": np.asarray(svd_result.Vt),
        }

    def eigen(self, matrix: ArrayLike) -> Dict[str, Any]:
        """
        Compute eigendecomposition.

        Args:
            matrix: Input matrix

        Returns:
            Dictionary with eigenvalues and eigenvectors
        """
        import numpy as np

        A = self.jl.eval("PyArray")(np.asarray(matrix))
        eigen_result = self.jl.eval("eigen(A)")

        return {
            "eigenvalues": np.asarray(eigen_result.values),
            "eigenvectors": np.asarray(eigen_result.vectors),
        }

    def solve(self, A: ArrayLike, b: ArrayLike, method: str = "backslash") -> Any:
        """
        Solve linear system Ax = b.

        Args:
            A: Coefficient matrix
            b: Right-hand side
            method: Solution method

        Returns:
            Solution vector x
        """
        import numpy as np

        A_jl = self.jl.eval("PyArray")(np.asarray(A))
        b_jl = self.jl.eval("PyArray")(np.asarray(b))

        if method == "backslash":
            x = self.jl.eval("A_jl \\ b_jl")
        elif method == "lu":
            x = self.jl.eval("lu(A_jl) \\ b_jl")
        elif method == "qr":
            x = self.jl.eval("qr(A_jl) \\ b_jl")
        else:
            raise ValueError(f"Unknown method: {method}")

        return np.asarray(x)

    def matrix_power(self, A: ArrayLike, n: int) -> Any:
        """Compute matrix power A^n."""
        import numpy as np

        A_jl = self.jl.eval("PyArray")(np.asarray(A))
        result = self.jl.eval(f"A_jl^{n}")
        return np.asarray(result)

    def condition_number(self, A: ArrayLike, p: int = 2) -> float:
        """Compute condition number."""
        A_jl = self.jl.eval("PyArray")(np.asarray(A))
        return float(self.jl.eval(f"cond(A_jl, {p})"))


class Statistics:
    """
    Julia Statistics Integration

    Statistical computing and analysis.
    """

    def __init__(self, julia_instance: Optional[Any] = None):
        self.jl = julia_instance or get_julia_runtime()
        self.jl.eval("using Statistics")

    def descriptive_stats(self, data: ArrayLike) -> Dict[str, float]:
        """
        Compute descriptive statistics.

        Args:
            data: Input data

        Returns:
            Dictionary of statistics
        """
        import numpy as np

        x = self.jl.eval("PyArray")(np.asarray(data))

        return {
            "mean": float(self.jl.eval("mean(x)")),
            "median": float(self.jl.eval("median(x)")),
            "std": float(self.jl.eval("std(x)")),
            "var": float(self.jl.eval("var(x)")),
            "min": float(self.jl.eval("minimum(x)")),
            "max": float(self.jl.eval("maximum(x)")),
            "q25": float(self.jl.eval("quantile(x, 0.25)")),
            "q75": float(self.jl.eval("quantile(x, 0.75)")),
        }

    def correlation(self, x: ArrayLike, y: ArrayLike) -> float:
        """Compute Pearson correlation coefficient."""
        import numpy as np

        x_jl = self.jl.eval("PyArray")(np.asarray(x))
        y_jl = self.jl.eval("PyArray")(np.asarray(y))

        return float(self.jl.eval("cor(x_jl, y_jl)"))

    def covariance_matrix(self, data: ArrayLike) -> Any:
        """Compute covariance matrix."""
        import numpy as np

        X = self.jl.eval("PyArray")(np.asarray(data))
        cov = self.jl.eval("cov(X)")
        return np.asarray(cov)

    def hypothesis_test(
        self, data: ArrayLike, test_type: str = "t-test", **kwargs
    ) -> Dict[str, Any]:
        """
        Perform statistical hypothesis tests.

        Args:
            data: Input data
            test_type: Type of test
            **kwargs: Test-specific parameters

        Returns:
            Test results
        """
        _ensure_packages(self.jl, ["HypothesisTests"])
        self.jl.eval("using HypothesisTests")

        import numpy as np

        x = self.jl.eval("PyArray")(np.asarray(data))

        if test_type == "t-test":
            mu0 = kwargs.get("mu0", 0)
            test_result = self.jl.eval(f"OneSampleTTest(x, {mu0})")
        elif test_type == "wilcoxon":
            test_result = self.jl.eval("SignedRankTest(x)")
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        return {
            "statistic": float(test_result.statistic),
            "p_value": float(test_result.pvalue),
            "confidence_interval": np.asarray(test_result.ci),
        }


# =============================================================================
# Utilities
# =============================================================================


def julia_extension(
    name: str,
    path: Optional[Union[str, Path]] = None,
    version: str = "1.0.0",
    dependencies: Optional[List[str]] = None,
) -> JuliaExtension:
    """
    Create and configure a Julia extension.

    Args:
        name: Extension name
        path: Path to extension files
        version: Extension version
        dependencies: Required Julia packages

    Returns:
        Configured JuliaExtension
    """
    return JuliaExtension(
        name=name,
        path=Path(path) if path else None,
        version=version,
        dependencies=dependencies or [],
    )


def pycall_integration(julia_instance: Optional[Any] = None) -> PyCallIntegration:
    """
    Create a PyCall integration instance.

    Args:
        julia_instance: Julia runtime instance

    Returns:
        PyCallIntegration instance
    """
    return PyCallIntegration(jl=julia_instance)


def flux_export(
    model: Any, filename: Optional[str] = None, format: str = "bson"
) -> str:
    """
    Export a model to Flux.jl format.

    Convenience function for model export.

    Args:
        model: Model to export
        filename: Output filename
        format: Export format

    Returns:
        Path to exported file
    """
    return export_to_flux(model, filename, format)


# =============================================================================
# Global State Management
# =============================================================================


_JULIA_RUNTIME: Optional[Any] = None
_JULIA_STATE: JuliaState = JuliaState.UNINITIALIZED


def get_julia_runtime() -> Any:
    """Get the initialized Julia runtime, or initialize it."""
    global _JULIA_RUNTIME
    if _JULIA_RUNTIME is None:
        _JULIA_RUNTIME = load_julia()
    return _JULIA_RUNTIME


def shutdown_julia():
    """Shutdown the Julia runtime."""
    global _JULIA_RUNTIME, _JULIA_STATE
    if _JULIA_RUNTIME is not None:
        try:
            _JULIA_RUNTIME.eval("exit()")
        except:
            pass
        _JULIA_RUNTIME = None
        _JULIA_STATE = JuliaState.SHUTDOWN
        logger.info("Julia runtime shutdown")


def _ensure_packages(jl: Any, packages: List[str]):
    """Ensure Julia packages are installed and loaded."""
    for pkg in packages:
        try:
            jl.eval(f"using {pkg}")
        except:
            logger.info(f"Installing Julia package: {pkg}")
            jl.eval(f'import Pkg; Pkg.add("{pkg}")')
            jl.eval(f"using {pkg}")


def julia_version() -> str:
    """Get the Julia version."""
    jl = get_julia_runtime()
    return jl.eval("VERSION")


def check_julia_installation() -> bool:
    """Check if Julia is properly installed."""
    try:
        subprocess.run(["julia", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def julia_context(**kwargs):
    """
    Context manager for Julia runtime.

    Ensures proper cleanup when done.

    Example:
        with julia_context(threads=8) as jl:
            result = jl.eval("1 + 1")
    """
    jl = load_julia(**kwargs)
    try:
        yield jl
    finally:
        pass  # Julia cleanup handled atexit


@contextmanager
def flux_context():
    """Context manager for Flux.jl operations."""
    jl = get_julia_runtime()
    converter = FluxConverter(jl)
    try:
        yield converter
    finally:
        pass


@contextmanager
def zygote_context():
    """Context manager for Zygote AD operations."""
    jl = get_julia_runtime()
    zygote = ZygoteIntegration(jl)
    try:
        yield zygote
    finally:
        pass


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core
    "JuliaError",
    "JuliaNotFoundError",
    "PyCallError",
    "FluxError",
    "ZygoteError",
    "JuliaState",
    # Julia Extensions
    "JuliaExtension",
    "load_julia",
    "compile_julia",
    "julia_bridge",
    # PyCall
    "PyCallIntegration",
    "export_to_julia",
    "call_julia_from_python",
    "JuliaPythonBridge",
    # Flux
    "FluxConverter",
    "export_to_flux",
    "load_flux_model",
    "FluxModel",
    # Zygote
    "ZygoteIntegration",
    "automatic_differentiation",
    "gradient_computation",
    "hessian_computation",
    # Performance
    "JuliaOptimizer",
    "jit_compilation",
    "parallel_julia",
    "gpu_acceleration",
    # Scientific Computing
    "DifferentialEquations",
    "Optimization",
    "LinearAlgebra",
    "Statistics",
    # Utilities
    "julia_extension",
    "pycall_integration",
    "flux_export",
    # State Management
    "get_julia_runtime",
    "shutdown_julia",
    "julia_version",
    "check_julia_installation",
    # Context Managers
    "julia_context",
    "flux_context",
    "zygote_context",
]

# Register cleanup at exit
import atexit

atexit.register(shutdown_julia)
