"""
fishstick C++ Backend Core Module
================================
Comprehensive C++ extensions, CUDA kernels, custom operations, bindings,
and optimization utilities for high-performance deep learning.

This module provides:
- C++ Extensions: JIT compilation and loading of C++ extensions
- CUDA Kernels: CUDA kernel management and execution
- Custom Operations: Custom PyTorch operations with autograd support
- Bindings: Pybind11 integration for Python-C++ interop
- Optimization: TorchScript, ONNX, TensorRT, OpenVINO conversion
- Memory: CUDA memory pool management
- Parallel: OpenMP and thread pool utilities
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, Sequence,
    TypeVar, Generic, Set, Type
)
from pathlib import Path
import os
import sys
import tempfile
import subprocess
import hashlib
import json
import warnings
from contextlib import contextmanager
from functools import wraps
import threading
import concurrent.futures
import ctypes
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.cpp_extension import (
    load as _cpp_load,
    load_inline as _cpp_load_inline,
    CUDAExtension,
    CppExtension,
    BuildExtension
)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar('T')
DeviceType = Union[str, torch.device]
ShapeType = Union[Tuple[int, ...], List[int]]
KernelFunc = Callable[..., Any]


# =============================================================================
# C++ Extensions
# =============================================================================

class CppExtensionBuilder:
    """Builder for C++ extensions with automatic compilation and caching."""
    
    _cache_dir: Path = Path.home() / ".cache" / "fishstick" / "cpp_extensions"
    _loaded_extensions: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    def __init__(
        self,
        name: str,
        sources: List[Union[str, Path]],
        extra_cflags: Optional[List[str]] = None,
        extra_cuda_cflags: Optional[List[str]] = None,
        extra_ldflags: Optional[List[str]] = None,
        extra_include_paths: Optional[List[str]] = None,
        extra_library_paths: Optional[List[str]] = None,
        libraries: Optional[List[str]] = None,
        build_directory: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        with_cuda: bool = False,
        is_python_module: bool = True,
        is_standalone: bool = False,
    ):
        self.name = name
        self.sources = [Path(s) for s in sources]
        self.extra_cflags = extra_cflags or ["-O3", "-fopenmp"]
        self.extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
        self.extra_ldflags = extra_ldflags or ["-lgomp"]
        self.extra_include_paths = extra_include_paths or []
        self.extra_library_paths = extra_library_paths or []
        self.libraries = libraries or []
        self.build_directory = Path(build_directory) if build_directory else self._cache_dir / name
        self.verbose = verbose
        self.with_cuda = with_cuda
        self.is_python_module = is_python_module
        self.is_standalone = is_standalone
        
        self.build_directory.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self) -> str:
        """Compute hash of sources and flags for caching."""
        hasher = hashlib.sha256()
        
        for source in self.sources:
            if source.exists():
                hasher.update(source.read_bytes())
        
        config = {
            "extra_cflags": self.extra_cflags,
            "extra_cuda_cflags": self.extra_cuda_cflags,
            "extra_ldflags": self.extra_ldflags,
            "libraries": self.libraries,
            "with_cuda": self.with_cuda,
        }
        hasher.update(json.dumps(config, sort_keys=True).encode())
        
        return hasher.hexdigest()[:16]
    
    def build(self) -> Any:
        """Build and return the extension module."""
        cache_key = f"{self.name}_{self._compute_hash()}"
        
        with self._lock:
            if cache_key in self._loaded_extensions:
                return self._loaded_extensions[cache_key]
            
            try:
                if self.with_cuda and not torch.cuda.is_available():
                    warnings.warn("CUDA not available, building CPU-only extension")
                    self.with_cuda = False
                
                extension = _cpp_load(
                    name=cache_key,
                    sources=[str(s) for s in self.sources],
                    extra_cflags=self.extra_cflags,
                    extra_cuda_cflags=self.extra_cuda_cflags if self.with_cuda else None,
                    extra_ldflags=self.extra_ldflags,
                    extra_include_paths=self.extra_include_paths,
                    extra_library_paths=self.extra_library_paths,
                    build_directory=str(self.build_directory),
                    verbose=self.verbose,
                    with_cuda=self.with_cuda,
                    is_python_module=self.is_python_module,
                    is_standalone=self.is_standalone,
                )
                
                self._loaded_extensions[cache_key] = extension
                return extension
                
            except Exception as e:
                raise RuntimeError(f"Failed to build C++ extension '{self.name}': {e}")


class InlineCppExtensionBuilder:
    """Builder for inline C++ extensions from source strings."""
    
    _cache_dir: Path = Path.home() / ".cache" / "fishstick" / "inline_cpp"
    _loaded_extensions: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    def __init__(
        self,
        name: str,
        cpp_sources: Union[str, List[str]],
        cuda_sources: Optional[Union[str, List[str]]] = None,
        functions: Optional[List[str]] = None,
        extra_cflags: Optional[List[str]] = None,
        extra_cuda_cflags: Optional[List[str]] = None,
        extra_ldflags: Optional[List[str]] = None,
        extra_include_paths: Optional[List[str]] = None,
        build_directory: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        with_cuda: bool = False,
    ):
        self.name = name
        self.cpp_sources = [cpp_sources] if isinstance(cpp_sources, str) else cpp_sources
        self.cuda_sources = ([cuda_sources] if isinstance(cuda_sources, str) else cuda_sources) if cuda_sources else None
        self.functions = functions or []
        self.extra_cflags = extra_cflags or ["-O3", "-fopenmp"]
        self.extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
        self.extra_ldflags = extra_ldflags or ["-lgomp"]
        self.extra_include_paths = extra_include_paths or []
        self.build_directory = Path(build_directory) if build_directory else self._cache_dir / name
        self.verbose = verbose
        self.with_cuda = with_cuda and torch.cuda.is_available()
        
        self.build_directory.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self) -> str:
        """Compute hash of source code."""
        hasher = hashlib.sha256()
        
        for source in self.cpp_sources:
            hasher.update(source.encode())
        
        if self.cuda_sources:
            for source in self.cuda_sources:
                hasher.update(source.encode())
        
        return hasher.hexdigest()[:16]
    
    def build(self) -> Any:
        """Build and return the inline extension module."""
        cache_key = f"{self.name}_{self._compute_hash()}"
        
        with self._lock:
            if cache_key in self._loaded_extensions:
                return self._loaded_extensions[cache_key]
            
            try:
                cpp_source = "\n\n".join(self.cpp_sources)
                cuda_source = "\n\n".join(self.cuda_sources) if self.cuda_sources else None
                
                extension = _cpp_load_inline(
                    name=cache_key,
                    cpp_sources=cpp_source,
                    cuda_sources=cuda_source,
                    functions=self.functions,
                    extra_cflags=self.extra_cflags,
                    extra_cuda_cflags=self.extra_cuda_cflags if self.with_cuda else None,
                    extra_ldflags=self.extra_ldflags,
                    extra_include_paths=self.extra_include_paths,
                    build_directory=str(self.build_directory),
                    verbose=self.verbose,
                    with_cuda=self.with_cuda,
                )
                
                self._loaded_extensions[cache_key] = extension
                return extension
                
            except Exception as e:
                raise RuntimeError(f"Failed to build inline C++ extension '{self.name}': {e}")


class CppExtension:
    """
    C++ Extension wrapper for building and loading C++ extensions.
    
    Example:
        >>> extension = CppExtension(
        ...     name="my_ops",
        ...     sources=["ops.cpp"],
        ...     extra_cflags=["-O3"]
        ... )
        >>> module = extension.build()
        >>> result = module.my_function(torch.randn(10))
    """
    
    def __init__(
        self,
        name: str,
        sources: List[Union[str, Path]],
        extra_cflags: Optional[List[str]] = None,
        extra_cuda_cflags: Optional[List[str]] = None,
        extra_ldflags: Optional[List[str]] = None,
        extra_include_paths: Optional[List[str]] = None,
        extra_library_paths: Optional[List[str]] = None,
        libraries: Optional[List[str]] = None,
        build_directory: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        with_cuda: bool = False,
    ):
        self.name = name
        self.sources = sources
        self.extra_cflags = extra_cflags
        self.extra_cuda_cflags = extra_cuda_cflags
        self.extra_ldflags = extra_ldflags
        self.extra_include_paths = extra_include_paths
        self.extra_library_paths = extra_library_paths
        self.libraries = libraries
        self.build_directory = build_directory
        self.verbose = verbose
        self.with_cuda = with_cuda
        
        self._builder = CppExtensionBuilder(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            extra_library_paths=extra_library_paths,
            libraries=libraries,
            build_directory=build_directory,
            verbose=verbose,
            with_cuda=with_cuda,
        )
        self._module: Optional[Any] = None
    
    def build(self) -> Any:
        """Build and return the extension module."""
        if self._module is None:
            self._module = self._builder.build()
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the built module."""
        if self._module is None:
            self._module = self.build()
        return getattr(self._module, name)


def load_cpp(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    **kwargs: Any
) -> Any:
    """
    Load a C++ extension from source files.
    
    Args:
        name: Extension name
        sources: List of C++ source files
        extra_cflags: Extra compiler flags
        extra_ldflags: Extra linker flags
        **kwargs: Additional arguments for CppExtension
    
    Returns:
        Loaded extension module
    
    Example:
        >>> module = load_cpp("my_ops", ["ops.cpp", "utils.cpp"])
        >>> result = module.forward(x)
    """
    extension = CppExtension(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        **kwargs
    )
    return extension.build()


def compile_cpp(
    sources: List[Union[str, Path]],
    output: Optional[Union[str, Path]] = None,
    extra_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    shared: bool = True,
) -> Path:
    """
    Compile C++ source files to a shared library or executable.
    
    Args:
        sources: Source files to compile
        output: Output path (default: auto-generated)
        extra_cflags: Extra compiler flags
        extra_ldflags: Extra linker flags
        shared: Whether to build shared library
    
    Returns:
        Path to compiled output
    """
    sources = [Path(s) for s in sources]
    
    if output is None:
        output = sources[0].with_suffix(".so" if shared else "")
    else:
        output = Path(output)
    
    cflags = extra_cflags or ["-O3", "-std=c++17", "-fPIC", "-fopenmp"]
    ldflags = extra_ldflags or ["-lgomp"]
    
    if shared:
        cflags.append("-shared")
    
    cmd = ["g++"] + cflags + [str(s) for s in sources] + ldflags + ["-o", str(output)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compilation failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("g++ compiler not found. Please install build-essential")


def jit_compile(
    cpp_code: str,
    name: str = "jit_module",
    functions: Optional[List[str]] = None,
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    cuda_code: Optional[str] = None,
    with_cuda: bool = False,
) -> Any:
    """
    JIT compile C++ code from a string.
    
    Args:
        cpp_code: C++ source code
        name: Module name
        functions: Function names to expose
        extra_cflags: Extra compiler flags
        extra_cuda_cflags: Extra CUDA compiler flags
        cuda_code: Optional CUDA source code
        with_cuda: Whether to compile with CUDA
    
    Returns:
        Compiled module
    
    Example:
        >>> code = """
        ...     #include <torch/extension.h>
        ...     torch::Tensor my_op(torch::Tensor x) { return x * 2; }
        ... """
        >>> module = jit_compile(code, functions=["my_op"])
        >>> result = module.my_op(torch.randn(10))
    """
    builder = InlineCppExtensionBuilder(
        name=name,
        cpp_sources=cpp_code,
        cuda_sources=cuda_code,
        functions=functions,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=with_cuda,
    )
    return builder.build()


# =============================================================================
# CUDA Kernels
# =============================================================================

@dataclass
class CudaKernelConfig:
    """Configuration for CUDA kernel execution."""
    block_size: Tuple[int, int, int] = (256, 1, 1)
    grid_size: Optional[Tuple[int, int, int]] = None
    shared_memory: int = 0
    stream: Optional[torch.cuda.Stream] = None
    non_blocking: bool = False
    
    def __post_init__(self):
        if self.stream is None and torch.cuda.is_available():
            self.stream = torch.cuda.current_stream()


class CudaKernel:
    """
    CUDA Kernel wrapper for launching GPU kernels.
    
    Example:
        >>> kernel = CudaKernel("matmul", config=CudaKernelConfig(block_size=(16, 16, 1)))
        >>> kernel.launch(input_tensor, output_tensor)
    """
    
    def __init__(
        self,
        name: str,
        kernel_func: Optional[Callable] = None,
        config: Optional[CudaKernelConfig] = None,
    ):
        self.name = name
        self.kernel_func = kernel_func
        self.config = config or CudaKernelConfig()
        self._compiled_kernel: Optional[Any] = None
    
    def compile(self, source: str) -> "CudaKernel":
        """Compile CUDA source code for this kernel."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        # JIT compile CUDA code
        module = jit_compile(
            cpp_code="",  # C++ wrapper code will be generated
            cuda_code=source,
            name=f"cuda_{self.name}",
            functions=[self.name],
            with_cuda=True,
        )
        
        self.kernel_func = getattr(module, self.name)
        return self
    
    def launch(self, *args: Any, **kwargs: Any) -> Any:
        """Launch the CUDA kernel with given arguments."""
        if self.kernel_func is None:
            raise RuntimeError(f"Kernel '{self.name}' not compiled")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        return self.kernel_func(*args, **kwargs)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Convenience method to launch kernel."""
        return self.launch(*args, **kwargs)
    
    def benchmark(
        self,
        *args: Any,
        num_warmup: int = 10,
        num_iters: int = 100,
    ) -> Dict[str, float]:
        """Benchmark kernel execution time."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(num_warmup):
            self.launch(*args)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iters):
            self.launch(*args)
        end.record()
        
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        
        return {
            "mean_ms": elapsed_ms / num_iters,
            "total_ms": elapsed_ms,
            "num_iters": num_iters,
        }


class CudaKernelCache:
    """Cache for compiled CUDA kernels."""
    
    _kernels: Dict[str, CudaKernel] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get(cls, name: str) -> Optional[CudaKernel]:
        return cls._kernels.get(name)
    
    @classmethod
    def set(cls, name: str, kernel: CudaKernel) -> None:
        with cls._lock:
            cls._kernels[name] = kernel
    
    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._kernels.clear()


def load_cuda(
    source_path: Union[str, Path],
    kernel_names: List[str],
    **kwargs: Any
) -> Dict[str, CudaKernel]:
    """
    Load CUDA kernels from a source file.
    
    Args:
        source_path: Path to CUDA source file
        kernel_names: Names of kernels to load
        **kwargs: Additional compilation arguments
    
    Returns:
        Dictionary of kernel name to CudaKernel
    
    Example:
        >>> kernels = load_cuda("kernels.cu", ["matmul", "conv2d"])
        >>> result = kernels["matmul"].launch(a, b, c)
    """
    source_path = Path(source_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"CUDA source not found: {source_path}")
    
    source = source_path.read_text()
    
    module = jit_compile(
        cpp_code="",
        cuda_code=source,
        name=f"cuda_{source_path.stem}",
        functions=kernel_names,
        with_cuda=True,
        **kwargs
    )
    
    kernels = {}
    for name in kernel_names:
        kernel = CudaKernel(
            name=name,
            kernel_func=getattr(module, name),
        )
        kernels[name] = kernel
        CudaKernelCache.set(name, kernel)
    
    return kernels


def compile_cuda(
    source: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    arch: Optional[str] = None,
    extra_flags: Optional[List[str]] = None,
) -> Path:
    """
    Compile CUDA source to a library.
    
    Args:
        source: CUDA source file or code
        output: Output path
        arch: CUDA architecture (e.g., "sm_70")
        extra_flags: Extra compiler flags
    
    Returns:
        Path to compiled library
    """
    if arch is None and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch = f"sm_{capability[0]}{capability[1]}"
    
    flags = extra_flags or ["-O3", "--use_fast_math", "-Xcompiler", "-fPIC"]
    
    if isinstance(source, (str, Path)) and Path(source).exists():
        source_path = Path(source)
        source_code = None
    else:
        source_path = None
        source_code = str(source)
    
    if output is None:
        if source_path:
            output = source_path.with_suffix(".so")
        else:
            output = Path("cuda_module.so")
    else:
        output = Path(output)
    
    cmd = ["nvcc", "-shared"]
    
    if arch:
        cmd.extend(["-arch", arch])
    
    cmd.extend(flags)
    
    if source_path:
        cmd.append(str(source_path))
    else:
        # Use stdin for inline code
        cmd.extend(["-x", "cu", "-"])
    
    cmd.extend(["-o", str(output)])
    
    try:
        if source_code:
            subprocess.run(cmd, input=source_code, check=True, capture_output=True, text=True)
        else:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"CUDA compilation failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("nvcc not found. Please install CUDA toolkit")


def launch_kernel(
    kernel: Union[CudaKernel, Callable],
    *args: Any,
    block_size: Tuple[int, ...] = (256,),
    grid_size: Optional[Tuple[int, ...]] = None,
    shared_memory: int = 0,
    stream: Optional[torch.cuda.Stream] = None,
) -> Any:
    """
    Launch a CUDA kernel with specified configuration.
    
    Args:
        kernel: CudaKernel instance or callable
        *args: Kernel arguments
        block_size: Thread block dimensions
        grid_size: Grid dimensions
        shared_memory: Shared memory size in bytes
        stream: CUDA stream
    
    Returns:
        Kernel result
    
    Example:
        >>> result = launch_kernel(my_kernel, input_tensor, output_tensor, 
        ...                        block_size=(256, 1, 1), grid_size=(10, 1, 1))
    """
    if isinstance(kernel, CudaKernel):
        return kernel.launch(*args)
    else:
        return kernel(*args)


# =============================================================================
# Custom Operations
# =============================================================================

@dataclass
class OpMetadata:
    """Metadata for custom operations."""
    name: str
    num_inputs: int
    num_outputs: int
    has_cuda_kernel: bool = False
    has_backward: bool = False
    deterministic: bool = True
    nondeterministic_seeded: bool = False


class CustomOp(torch.autograd.Function):
    """
    Base class for custom operations with autograd support.
    
    Example:
        >>> class MyCustomOp(CustomOp):
        ...     @staticmethod
        ...     def forward(ctx, x, weight):
        ...         ctx.save_for_backward(x, weight)
        ...         return x @ weight.t()
        ...     
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         x, weight = ctx.saved_tensors
        ...         return grad_output @ weight, grad_output.t() @ x
        >>> 
        >>> my_op = MyCustomOp.apply
        >>> result = my_op(input, weight)
    """
    
    metadata: OpMetadata = OpMetadata(name="custom_op", num_inputs=1, num_outputs=1)
    
    @classmethod
    def register(cls, name: str, forward_fn: Callable, backward_fn: Optional[Callable] = None):
        """Register a custom operation."""
        OpRegistry.register(name, cls)
        return cls


class OpRegistry:
    """Registry for custom operations."""
    
    _ops: Dict[str, Type[CustomOp]] = {}
    _metadata: Dict[str, OpMetadata] = {}
    _lock = threading.Lock()
    
    @classmethod
    def register(
        cls,
        name: str,
        op_class: Type[CustomOp],
        metadata: Optional[OpMetadata] = None,
    ) -> None:
        """Register a custom operation."""
        with cls._lock:
            cls._ops[name] = op_class
            if metadata:
                cls._metadata[name] = metadata
            elif hasattr(op_class, 'metadata'):
                cls._metadata[name] = op_class.metadata
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[CustomOp]]:
        """Get a registered operation by name."""
        return cls._ops.get(name)
    
    @classmethod
    def get_metadata(cls, name: str) -> Optional[OpMetadata]:
        """Get metadata for a registered operation."""
        return cls._metadata.get(name)
    
    @classmethod
    def list_ops(cls) -> List[str]:
        """List all registered operation names."""
        return list(cls._ops.keys())
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister an operation."""
        with cls._lock:
            cls._ops.pop(name, None)
            cls._metadata.pop(name, None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered operations."""
        with cls._lock:
            cls._ops.clear()
            cls._metadata.clear()


def register_op(
    name: str,
    forward_fn: Optional[Callable] = None,
    backward_fn: Optional[Callable] = None,
    has_cuda_kernel: bool = False,
) -> Callable:
    """
    Decorator to register a custom operation.
    
    Args:
        name: Operation name
        forward_fn: Forward function
        backward_fn: Backward function
        has_cuda_kernel: Whether CUDA kernel is available
    
    Returns:
        Registered operation class or decorator
    
    Example:
        >>> @register_op("custom_matmul")
        ... class CustomMatmul(CustomOp):
        ...     @staticmethod
        ...     def forward(ctx, a, b):
        ...         return torch.matmul(a, b)
        ...     
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         return grad_output, grad_output
    """
    def decorator(op_class: Type[CustomOp]) -> Type[CustomOp]:
        metadata = OpMetadata(
            name=name,
            num_inputs=getattr(op_class, 'num_inputs', 1),
            num_outputs=getattr(op_class, 'num_outputs', 1),
            has_cuda_kernel=has_cuda_kernel,
            has_backward=backward_fn is not None or hasattr(op_class, 'backward'),
        )
        OpRegistry.register(name, op_class, metadata)
        return op_class
    
    if forward_fn is None:
        return decorator
    else:
        # Create a custom op class from functions
        class GeneratedOp(CustomOp):
            @staticmethod
            def forward(ctx, *args, **kwargs):
                return forward_fn(*args, **kwargs)
            
            @staticmethod
            def backward(ctx, *grad_outputs):
                if backward_fn:
                    return backward_fn(*grad_outputs)
                return grad_outputs
        
        return decorator(GeneratedOp)


def load_op(name: str) -> Optional[Callable]:
    """
    Load a registered custom operation.
    
    Args:
        name: Operation name
    
    Returns:
        Operation apply function or None
    
    Example:
        >>> op = load_op("custom_matmul")
        >>> result = op(a, b)
    """
    op_class = OpRegistry.get(name)
    if op_class is None:
        return None
    return op_class.apply


# =============================================================================
# Bindings (Pybind11)
# =============================================================================

class Pybind11Module:
    """
    Pybind11 module builder for creating Python bindings.
    
    Example:
        >>> module = Pybind11Module("my_module")
        >>> module.export_function("add", add_impl, ["float", "float"])
        >>> module.export_class("MyClass", MyClass, [
        ...     ("__init__", ["int"]),
        ...     ("compute", ["float"], "float"),
        ... ])
        >>> py_module = module.build()
    """
    
    def __init__(self, name: str):
        self.name = name
        self._functions: List[Dict[str, Any]] = []
        self._classes: List[Dict[str, Any]] = []
        self._module: Optional[Any] = None
    
    def export_function(
        self,
        name: str,
        func: Callable,
        arg_types: Optional[List[str]] = None,
        return_type: Optional[str] = None,
        docstring: str = "",
    ) -> "Pybind11Module":
        """Export a function to the module."""
        self._functions.append({
            "name": name,
            "func": func,
            "arg_types": arg_types,
            "return_type": return_type,
            "docstring": docstring,
        })
        return self
    
    def export_class(
        self,
        name: str,
        cls: Type,
        methods: Optional[List[Tuple[str, List[str], Optional[str]]]] = None,
        docstring: str = "",
    ) -> "Pybind11Module":
        """Export a class to the module."""
        self._classes.append({
            "name": name,
            "class": cls,
            "methods": methods or [],
            "docstring": docstring,
        })
        return self
    
    def build(self) -> Any:
        """Build and return the Python module."""
        if self._module is not None:
            return self._module
        
        # Generate C++ binding code
        cpp_code = self._generate_bindings()
        
        # Compile bindings
        self._module = jit_compile(
            cpp_code=cpp_code,
            name=self.name,
            functions=[f["name"] for f in self._functions],
        )
        
        return self._module
    
    def _generate_bindings(self) -> str:
        """Generate C++ binding code."""
        lines = [
            "#include <torch/extension.h>",
            "#include <pybind11/pybind11.h>",
            "",
            f"namespace py = pybind11;",
            "",
        ]
        
        # Add function declarations
        for func_info in self._functions:
            lines.append(f"// {func_info['name']}")
        
        lines.extend([
            "",
            f"PYBIND11_MODULE({self.name}, m) {{",
        ])
        
        # Export functions
        for func_info in self._functions:
            lines.append(f'    m.def("{func_info["name"]}", &{func_info["name"]}, "{func_info["docstring"]}");')
        
        # Export classes
        for cls_info in self._classes:
            lines.append(f'    py::class_<{cls_info["name"]}>(m, "{cls_info["name"]}")')
            for method in cls_info["methods"]:
                method_name = method[0]
                lines.append(f'        .def("{method_name}", &{cls_info["name"]}::{method_name})')
            lines.append("        ;")
        
        lines.append("}")
        
        return "\n".join(lines)


def create_bindings(
    name: str,
    functions: Optional[Dict[str, Callable]] = None,
    classes: Optional[Dict[str, Type]] = None,
) -> Pybind11Module:
    """
    Create Pybind11 bindings module.
    
    Args:
        name: Module name
        functions: Dictionary of function names to callables
        classes: Dictionary of class names to types
    
    Returns:
        Pybind11Module instance
    
    Example:
        >>> module = create_bindings("my_module", {
        ...     "add": lambda a, b: a + b,
        ...     "multiply": lambda a, b: a * b,
        ... })
        >>> py_mod = module.build()
    """
    module = Pybind11Module(name)
    
    if functions:
        for func_name, func in functions.items():
            module.export_function(func_name, func)
    
    if classes:
        for cls_name, cls in classes.items():
            module.export_class(cls_name, cls)
    
    return module


def export_function(
    module: Pybind11Module,
    name: str,
    func: Callable,
    **kwargs: Any
) -> Pybind11Module:
    """
    Export a function to a Pybind11 module.
    
    Args:
        module: Pybind11Module instance
        name: Function name
        func: Function to export
        **kwargs: Additional export options
    
    Returns:
        Updated module
    """
    return module.export_function(name, func, **kwargs)


def export_class(
    module: Pybind11Module,
    name: str,
    cls: Type,
    **kwargs: Any
) -> Pybind11Module:
    """
    Export a class to a Pybind11 module.
    
    Args:
        module: Pybind11Module instance
        name: Class name
        cls: Class to export
        **kwargs: Additional export options
    
    Returns:
        Updated module
    """
    return module.export_class(name, cls, **kwargs)


# =============================================================================
# Optimization
# =============================================================================

class TorchScriptCompile:
    """
    TorchScript compilation for model optimization.
    
    Example:
        >>> compiler = TorchScriptCompile(optimize=True)
        >>> scripted = compiler.compile(model, example_input)
        >>> compiler.save(scripted, "model.pt")
    """
    
    def __init__(
        self,
        optimize: bool = True,
        freeze: bool = False,
        set_profile: bool = False,
        check_trace: bool = True,
    ):
        self.optimize = optimize
        self.freeze = freeze
        self.set_profile = set_profile
        self.check_trace = check_trace
    
    def compile(
        self,
        model: nn.Module,
        example_inputs: Any,
        method: str = "trace",
    ) -> torch.jit.ScriptModule:
        """
        Compile model to TorchScript.
        
        Args:
            model: Model to compile
            example_inputs: Example inputs for tracing
            method: "trace" or "script"
        
        Returns:
            Compiled ScriptModule
        """
        model.eval()
        
        if isinstance(example_inputs, Tensor):
            example_inputs = (example_inputs,)
        elif not isinstance(example_inputs, (tuple, list)):
            example_inputs = (example_inputs,)
        
        if method == "trace":
            scripted = torch.jit.trace(
                model,
                example_inputs,
                check_trace=self.check_trace,
            )
        elif method == "script":
            scripted = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if self.optimize:
            scripted = torch.jit.optimize_for_inference(scripted)
        
        if self.freeze:
            scripted = torch.jit.freeze(scripted)
        
        return scripted
    
    def save(self, module: torch.jit.ScriptModule, path: Union[str, Path]) -> None:
        """Save compiled module to disk."""
        module.save(str(path))
    
    def load(self, path: Union[str, Path]) -> torch.jit.ScriptModule:
        """Load compiled module from disk."""
        return torch.jit.load(str(path))
    
    def benchmark(
        self,
        module: torch.jit.ScriptModule,
        example_inputs: Any,
        num_iters: int = 100,
    ) -> Dict[str, float]:
        """Benchmark compiled module."""
        if isinstance(example_inputs, Tensor):
            example_inputs = (example_inputs,)
        
        # Warmup
        for _ in range(10):
            module(*example_inputs)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(num_iters):
                module(*example_inputs)
            end.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            import time
            start = time.time()
            for _ in range(num_iters):
                module(*example_inputs)
            elapsed_ms = (time.time() - start) * 1000
        
        return {
            "mean_ms": elapsed_ms / num_iters,
            "total_ms": elapsed_ms,
            "num_iters": num_iters,
        }


class ONNXExport:
    """
    Export models to ONNX format.
    
    Example:
        >>> exporter = ONNXExport(input_names=["input"], output_names=["output"])
        >>> exporter.export(model, example_input, "model.onnx")
        >>> session = exporter.create_session("model.onnx")
    """
    
    def __init__(
        self,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14,
    ):
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]
        self.dynamic_axes = dynamic_axes or {}
        self.opset_version = opset_version
    
    def export(
        self,
        model: nn.Module,
        example_inputs: Any,
        output_path: Union[str, Path],
        **kwargs: Any
    ) -> None:
        """
        Export model to ONNX.
        
        Args:
            model: Model to export
            example_inputs: Example inputs
            output_path: Output file path
            **kwargs: Additional export arguments
        """
        model.eval()
        
        if isinstance(example_inputs, Tensor):
            example_inputs = (example_inputs,)
        elif not isinstance(example_inputs, (tuple, list)):
            example_inputs = (example_inputs,)
        
        torch.onnx.export(
            model,
            example_inputs,
            str(output_path),
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            opset_version=self.opset_version,
            **kwargs
        )
    
    def export_with_validation(
        self,
        model: nn.Module,
        example_inputs: Any,
        output_path: Union[str, Path],
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> bool:
        """
        Export model and validate output matches PyTorch.
        
        Returns:
            True if validation passes
        """
        import onnxruntime as ort
        import numpy as np
        
        self.export(model, example_inputs, output_path)
        
        # Run PyTorch
        model.eval()
        with torch.no_grad():
            if isinstance(example_inputs, (tuple, list)):
                pytorch_output = model(*example_inputs)
            else:
                pytorch_output = model(example_inputs)
        
        # Run ONNX Runtime
        session = ort.InferenceSession(str(output_path))
        
        if isinstance(example_inputs, (tuple, list)):
            onnx_inputs = {
                name: inp.cpu().numpy() if isinstance(inp, Tensor) else inp
                for name, inp in zip(self.input_names, example_inputs)
            }
        else:
            onnx_inputs = {
                self.input_names[0]: example_inputs.cpu().numpy() if isinstance(example_inputs, Tensor) else example_inputs
            }
        
        onnx_output = session.run(None, onnx_inputs)
        
        # Compare
        if isinstance(pytorch_output, (tuple, list)):
            for po, oo in zip(pytorch_output, onnx_output):
                if not np.allclose(po.cpu().numpy(), oo, rtol=rtol, atol=atol):
                    return False
        else:
            if not np.allclose(pytorch_output.cpu().numpy(), onnx_output[0], rtol=rtol, atol=atol):
                return False
        
        return True
    
    def create_session(self, model_path: Union[str, Path], **kwargs: Any) -> Any:
        """Create ONNX Runtime session from exported model."""
        try:
            import onnxruntime as ort
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            return ort.InferenceSession(str(model_path), sess_options, **kwargs)
        except ImportError:
            raise ImportError("onnxruntime not installed")


class TensorRTConvert:
    """
    Convert models to TensorRT for optimized inference.
    
    Example:
        >>> converter = TensorRTConvert(fp16=True)
        >>> engine = converter.convert(model, example_input)
        >>> output = converter.inference(engine, input_tensor)
    """
    
    def __init__(
        self,
        max_workspace_size: int = 1 << 30,
        fp16: bool = False,
        int8: bool = False,
        strict: bool = False,
        max_batch_size: int = 1,
    ):
        self.max_workspace_size = max_workspace_size
        self.fp16 = fp16
        self.int8 = int8
        self.strict = strict
        self.max_batch_size = max_batch_size
        self._trt_logger: Optional[Any] = None
        self._builder: Optional[Any] = None
    
    def _get_trt(self) -> Any:
        """Import and return TensorRT module."""
        try:
            import tensorrt as trt
            return trt
        except ImportError:
            raise ImportError("TensorRT not installed. Install with: pip install tensorrt")
    
    def convert(
        self,
        model: nn.Module,
        example_inputs: Any,
        min_shape: Optional[Tuple[int, ...]] = None,
        opt_shape: Optional[Tuple[int, ...]] = None,
        max_shape: Optional[Tuple[int, ...]] = None,
    ) -> Any:
        """
        Convert model to TensorRT engine.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            min_shape: Minimum input shape
            opt_shape: Optimal input shape
            max_shape: Maximum input shape
        
        Returns:
            TensorRT engine
        """
        trt = self._get_trt()
        
        # Create builder
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._builder = trt.Builder(self._trt_logger)
        
        # Create network
        network = self._builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # Create builder config
        config = self._builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        
        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.int8:
            config.set_flag(trt.BuilderFlag.INT8)
        if self.strict:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        
        # Parse ONNX model
        parser = trt.OnnxParser(network, self._trt_logger)
        
        # Export model to ONNX
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        
        try:
            model.eval()
            torch.onnx.export(
                model,
                example_inputs,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=14,
            )
            
            with open(onnx_path, "rb") as f:
                parser.parse(f.read())
            
            # Build engine
            if min_shape and opt_shape and max_shape:
                profile = self._builder.create_optimization_profile()
                profile.set_shape("input", min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
            
            engine = self._builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            return engine
            
        finally:
            import os
            os.unlink(onnx_path)
    
    def serialize(self, engine: Any, path: Union[str, Path]) -> None:
        """Serialize TensorRT engine to file."""
        with open(path, "wb") as f:
            f.write(engine.serialize())
    
    def deserialize(self, path: Union[str, Path]) -> Any:
        """Deserialize TensorRT engine from file."""
        trt = self._get_trt()
        runtime = trt.Runtime(self._trt_logger or trt.Logger(trt.Logger.WARNING))
        
        with open(path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())
    
    def inference(self, engine: Any, input_tensor: Tensor) -> Tensor:
        """Run inference with TensorRT engine."""
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Create context
        context = engine.create_execution_context()
        
        # Allocate device memory
        d_input = cuda.mem_alloc(input_tensor.numel() * input_tensor.element_size())
        d_output = cuda.mem_alloc(input_tensor.numel() * input_tensor.element_size())
        
        # Copy input to device
        cuda.memcpy_htod(d_input, input_tensor.cpu().numpy())
        
        # Execute
        context.execute_v2([int(d_input), int(d_output)])
        
        # Copy output to host
        output = torch.empty_like(input_tensor)
        cuda.memcpy_dtoh(output.cpu().numpy(), d_output)
        
        return output


class OpenVINOConvert:
    """
    Convert models to OpenVINO format for optimized inference on Intel hardware.
    
    Example:
        >>> converter = OpenVINOConvert()
        >>> model_xml, model_bin = converter.convert(model, example_input)
        >>> compiled = converter.compile_model(model_xml)
        >>> output = converter.inference(compiled, input_tensor)
    """
    
    def __init__(
        self,
        precision: str = "FP32",
        mean_values: Optional[List[float]] = None,
        scale_values: Optional[List[float]] = None,
    ):
        self.precision = precision
        self.mean_values = mean_values
        self.scale_values = scale_values
    
    def _get_openvino(self) -> Any:
        """Import and return OpenVINO module."""
        try:
            import openvino.runtime as ov
            return ov
        except ImportError:
            raise ImportError("OpenVINO not installed. Install with: pip install openvino")
    
    def convert(
        self,
        model: nn.Module,
        example_inputs: Any,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """
        Convert model to OpenVINO format.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            output_dir: Output directory
        
        Returns:
            Tuple of (model_xml_path, model_bin_path)
        """
        try:
            from openvino.tools import mo
        except ImportError:
            raise ImportError("OpenVINO Model Optimizer not installed")
        
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to ONNX first
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        
        try:
            model.eval()
            torch.onnx.export(
                model,
                example_inputs,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=14,
            )
            
            # Convert ONNX to OpenVINO
            model_name = "model"
            output_path = output_dir / model_name
            
            mo.convert_model(
                onnx_path,
                output_dir=str(output_dir),
                model_name=model_name,
                data_type=self.precision,
                mean_values=self.mean_values,
                scale_values=self.scale_values,
            )
            
            return output_path.with_suffix(".xml"), output_path.with_suffix(".bin")
            
        finally:
            import os
            os.unlink(onnx_path)
    
    def compile_model(
        self,
        model_path: Union[str, Path],
        device: str = "AUTO",
    ) -> Any:
        """
        Compile OpenVINO model for inference.
        
        Args:
            model_path: Path to model XML file
            device: Device to use (CPU, GPU, AUTO, etc.)
        
        Returns:
            Compiled model
        """
        ov = self._get_openvino()
        core = ov.Core()
        model = core.read_model(str(model_path))
        return core.compile_model(model, device)
    
    def inference(self, compiled_model: Any, input_tensor: Tensor) -> Tensor:
        """Run inference with compiled OpenVINO model."""
        input_data = input_tensor.cpu().numpy()
        output = compiled_model([input_data])
        return torch.from_numpy(list(output.values())[0])


# =============================================================================
# Memory Management
# =============================================================================

@dataclass
class CudaMemoryPoolConfig:
    """Configuration for CUDA memory pool."""
    device_id: int = 0
    initial_size: int = 1024 * 1024 * 1024  # 1GB
    growth_factor: float = 2.0
    max_size: int = 8 * 1024 * 1024 * 1024  # 8GB
    alignment: int = 512


class CudaMemoryPool:
    """
    CUDA memory pool for efficient memory allocation.
    
    Example:
        >>> pool = CudaMemoryPool(device_id=0)
        >>> tensor = pool.allocate("key1", (1000, 1000), torch.float32)
        >>> pool.release("key1", tensor)
        >>> stats = pool.get_stats()
    """
    
    def __init__(self, config: Optional[CudaMemoryPoolConfig] = None, device_id: int = 0):
        self.config = config or CudaMemoryPoolConfig(device_id=device_id)
        self.device_id = self.config.device_id
        self._pools: Dict[str, List[Tensor]] = {}
        self._allocated: Dict[str, int] = {}
        self._peak_allocated: int = 0
        self._lock = threading.Lock()
    
    def allocate(
        self,
        key: str,
        shape: ShapeType,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """
        Allocate tensor from pool or create new.
        
        Args:
            key: Pool key
            shape: Tensor shape
            dtype: Data type
        
        Returns:
            Allocated tensor
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        with self._lock:
            shape = tuple(shape)
            
            # Try to find matching tensor in pool
            if key in self._pools:
                for i, tensor in enumerate(self._pools[key]):
                    if tensor.shape == shape and tensor.dtype == dtype:
                        # Reuse tensor
                        self._pools[key].pop(i)
                        tensor.zero_()
                        self._allocated[key] = self._allocated.get(key, 0) + 1
                        return tensor
            
            # Allocate new tensor
            tensor = torch.empty(
                shape,
                dtype=dtype,
                device=f"cuda:{self.device_id}",
            )
            
            self._allocated[key] = self._allocated.get(key, 0) + 1
            
            # Update peak
            current = torch.cuda.memory_allocated(self.device_id)
            if current > self._peak_allocated:
                self._peak_allocated = current
            
            return tensor
    
    def release(self, key: str, tensor: Tensor) -> None:
        """
        Release tensor back to pool.
        
        Args:
            key: Pool key
            tensor: Tensor to release
        """
        with self._lock:
            if key not in self._pools:
                self._pools[key] = []
            
            self._pools[key].append(tensor)
            self._allocated[key] = max(0, self._allocated.get(key, 0) - 1)
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear pool or specific key.
        
        Args:
            key: Specific key to clear, or None for all
        """
        with self._lock:
            if key is not None:
                if key in self._pools:
                    self._pools[key].clear()
            else:
                for k in self._pools:
                    self._pools[k].clear()
                self._allocated.clear()
                torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            pool_sizes = {k: len(v) for k, v in self._pools.items()}
            
            stats = {
                "device_id": self.device_id,
                "pooled_tensors": sum(pool_sizes.values()),
                "pool_sizes": pool_sizes,
                "allocated": dict(self._allocated),
                "peak_allocated_bytes": self._peak_allocated,
            }
            
            if torch.cuda.is_available():
                stats["cuda_allocated"] = torch.cuda.memory_allocated(self.device_id)
                stats["cuda_reserved"] = torch.cuda.memory_reserved(self.device_id)
                stats["cuda_max_allocated"] = torch.cuda.max_memory_allocated(self.device_id)
            
            return stats


def allocate_cuda(
    shape: ShapeType,
    dtype: torch.dtype = torch.float32,
    device_id: int = 0,
    pool: Optional[CudaMemoryPool] = None,
    pool_key: str = "default",
) -> Tensor:
    """
    Allocate CUDA tensor with optional pooling.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        device_id: CUDA device
        pool: Optional memory pool
        pool_key: Pool key
    
    Returns:
        Allocated tensor
    
    Example:
        >>> tensor = allocate_cuda((1000, 1000), torch.float32)
        >>> # With pooling
        >>> pool = CudaMemoryPool()
        >>> tensor = allocate_cuda((1000, 1000), pool=pool, pool_key="activations")
    """
    if pool is not None:
        return pool.allocate(pool_key, shape, dtype)
    
    return torch.empty(shape, dtype=dtype, device=f"cuda:{device_id}")


def free_cuda(
    tensor: Tensor,
    pool: Optional[CudaMemoryPool] = None,
    pool_key: str = "default",
) -> None:
    """
    Free CUDA tensor, optionally returning to pool.
    
    Args:
        tensor: Tensor to free
        pool: Optional memory pool
        pool_key: Pool key
    
    Example:
        >>> free_cuda(tensor)
        >>> # With pooling
        >>> free_cuda(tensor, pool=pool, pool_key="activations")
    """
    if pool is not None:
        pool.release(pool_key, tensor)
    else:
        del tensor


def memory_stats(device_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get CUDA memory statistics.
    
    Args:
        device_id: Device ID, or None for current
    
    Returns:
        Memory statistics dictionary
    
    Example:
        >>> stats = memory_stats()
        >>> print(f"Allocated: {stats['allocated_gb']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    stats = {
        "device_id": device_id,
        "device_name": torch.cuda.get_device_name(device_id),
        "allocated_bytes": torch.cuda.memory_allocated(device_id),
        "reserved_bytes": torch.cuda.memory_reserved(device_id),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device_id),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device_id),
        "allocated_gb": torch.cuda.memory_allocated(device_id) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device_id) / (1024**3),
    }
    
    props = torch.cuda.get_device_properties(device_id)
    stats["total_memory_gb"] = props.total_memory / (1024**3)
    stats["compute_capability"] = f"{props.major}.{props.minor}"
    
    return stats


# =============================================================================
# Parallel Processing
# =============================================================================

class OpenMPParallel:
    """
    OpenMP parallel execution utilities.
    
    Example:
        >>> parallel = OpenMPParallel(num_threads=8)
        >>> results = parallel.map(func, data)
    """
    
    def __init__(self, num_threads: Optional[int] = None):
        self.num_threads = num_threads or os.cpu_count() or 1
        self._extension: Optional[Any] = None
    
    def _get_extension(self) -> Any:
        """Get or create OpenMP extension."""
        if self._extension is not None:
            return self._extension
        
        # Compile OpenMP utilities
        cpp_code = """
        #include <torch/extension.h>
        #include <omp.h>
        
        int get_num_threads() {
            return omp_get_max_threads();
        }
        
        void set_num_threads(int n) {
            omp_set_num_threads(n);
        }
        
        double parallel_for(int n, const std::function<void(int)>& fn) {
            double start = omp_get_wtime();
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                fn(i);
            }
            return omp_get_wtime() - start;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("get_num_threads", &get_num_threads, "Get number of OpenMP threads");
            m.def("set_num_threads", &set_num_threads, "Set number of OpenMP threads");
        }
        """
        
        self._extension = jit_compile(
            cpp_code=cpp_code,
            name="openmp_utils",
            extra_cflags=["-O3", "-fopenmp"],
            extra_ldflags=["-lgomp"],
        )
        
        return self._extension
    
    def set_num_threads(self, n: int) -> None:
        """Set number of OpenMP threads."""
        ext = self._get_extension()
        ext.set_num_threads(n)
        self.num_threads = n
    
    def get_num_threads(self) -> int:
        """Get number of OpenMP threads."""
        ext = self._get_extension()
        return ext.get_num_threads()
    
    def map(
        self,
        func: Callable[[Any], T],
        iterable: Sequence[Any],
        chunksize: int = 1,
    ) -> List[T]:
        """Map function over iterable in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            return list(executor.map(func, iterable, chunksize=chunksize))
    
    def reduce(
        self,
        func: Callable[[T, T], T],
        iterable: Sequence[T],
        initializer: Optional[T] = None,
    ) -> T:
        """Parallel reduction."""
        from functools import reduce as functools_reduce
        
        # Simple parallel reduce using divide and conquer
        def parallel_reduce_chunk(chunk: Sequence[T]) -> T:
            return functools_reduce(func, chunk, initializer)
        
        chunk_size = max(1, len(iterable) // self.num_threads)
        chunks = [iterable[i:i+chunk_size] for i in range(0, len(iterable), chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            chunk_results = list(executor.map(parallel_reduce_chunk, chunks))
        
        return functools_reduce(func, chunk_results, initializer)


class ThreadPool:
    """
    Thread pool for parallel execution.
    
    Example:
        >>> pool = ThreadPool(num_workers=8)
        >>> results = pool.map(func, data)
        >>> pool.shutdown()
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or os.cpu_count() or 1
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.Lock()
    
    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create executor."""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.num_workers
                    )
        return self._executor
    
    def map(
        self,
        func: Callable[[Any], T],
        iterable: Sequence[Any],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> List[T]:
        """Map function over iterable."""
        executor = self._get_executor()
        return list(executor.map(func, iterable, timeout=timeout, chunksize=chunksize))
    
    def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """Submit function for execution."""
        executor = self._get_executor()
        return executor.submit(func, *args, **kwargs)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread pool."""
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=wait)
                self._executor = None
    
    def __enter__(self) -> "ThreadPool":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()


def parallel_for(
    func: Callable[[int], T],
    n: int,
    num_workers: Optional[int] = None,
    backend: str = "thread",
) -> List[T]:
    """
    Parallel for loop.
    
    Args:
        func: Function to call for each index
        n: Number of iterations
        num_workers: Number of workers
        backend: "thread" or "process"
    
    Returns:
        List of results
    
    Example:
        >>> results = parallel_for(lambda i: i**2, 100, num_workers=8)
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    if backend == "thread":
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(func, range(n)))
    elif backend == "process":
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(func, range(n)))
    else:
        raise ValueError(f"Unknown backend: {backend}")


def parallel_reduce(
    func: Callable[[T, T], T],
    iterable: Sequence[T],
    initializer: Optional[T] = None,
    num_workers: Optional[int] = None,
) -> T:
    """
    Parallel reduction.
    
    Args:
        func: Reduction function
        iterable: Iterable to reduce
        initializer: Initial value
        num_workers: Number of workers
    
    Returns:
        Reduced value
    
    Example:
        >>> result = parallel_reduce(lambda a, b: a + b, range(1000), initializer=0)
    """
    from functools import reduce as functools_reduce
    
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    chunk_size = max(1, len(iterable) // num_workers)
    chunks = [iterable[i:i+chunk_size] for i in range(0, len(iterable), chunk_size)]
    
    def reduce_chunk(chunk: Sequence[T]) -> T:
        return functools_reduce(func, chunk, initializer)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(reduce_chunk, chunks))
    
    return functools_reduce(func, chunk_results, initializer)


# =============================================================================
# Utility Functions
# =============================================================================

def cpp_extension(
    name: str,
    sources: List[Union[str, Path]],
    with_cuda: bool = False,
    **kwargs: Any
) -> Any:
    """
    Convenience function to create and build a C++ extension.
    
    Args:
        name: Extension name
        sources: Source files
        with_cuda: Whether to compile with CUDA
        **kwargs: Additional arguments
    
    Returns:
        Built extension module
    
    Example:
        >>> module = cpp_extension("my_ops", ["ops.cpp"], with_cuda=True)
        >>> result = module.forward(x)
    """
    return load_cpp(name, sources, with_cuda=with_cuda, **kwargs)


def cuda_jit(
    cuda_code: str,
    kernel_names: List[str],
    **kwargs: Any
) -> Dict[str, CudaKernel]:
    """
    JIT compile CUDA code and return kernels.
    
    Args:
        cuda_code: CUDA source code
        kernel_names: Names of kernels to expose
        **kwargs: Additional compilation arguments
    
    Returns:
        Dictionary of kernel name to CudaKernel
    
    Example:
        >>> code = '''
        ... __global__ void add(float* a, float* b, float* c, int n) {
        ...     int i = blockIdx.x * blockDim.x + threadIdx.x;
        ...     if (i < n) c[i] = a[i] + b[i];
        ... }
        ... '''
        >>> kernels = cuda_jit(code, ["add"])
        >>> kernels["add"].launch(a, b, c, n)
    """
    module = jit_compile(
        cpp_code="",
        cuda_code=cuda_code,
        name="cuda_jit_module",
        functions=kernel_names,
        with_cuda=True,
        **kwargs
    )
    
    kernels = {}
    for name in kernel_names:
        kernel = CudaKernel(
            name=name,
            kernel_func=getattr(module, name),
        )
        kernels[name] = kernel
        CudaKernelCache.set(name, kernel)
    
    return kernels


def custom_op(
    name: str,
    forward: Callable,
    backward: Optional[Callable] = None,
    setup_context: Optional[Callable] = None,
    **kwargs: Any
) -> Callable:
    """
    Decorator to create a custom operation.
    
    Args:
        name: Operation name
        forward: Forward function
        backward: Optional backward function
        setup_context: Optional context setup function
        **kwargs: Additional metadata
    
    Returns:
        Decorated function
    
    Example:
        >>> @custom_op("my_matmul", backward=my_backward)
        ... def my_matmul_forward(ctx, a, b):
        ...     return torch.matmul(a, b)
    """
    class Op(CustomOp):
        @staticmethod
        def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            if setup_context:
                setup_context(ctx, *args, **kwargs)
            return forward(*args, **kwargs)
        
        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            if backward:
                return backward(ctx, *grad_outputs)
            return grad_outputs
    
    Op.metadata = OpMetadata(
        name=name,
        num_inputs=kwargs.get("num_inputs", 1),
        num_outputs=kwargs.get("num_outputs", 1),
        has_backward=backward is not None,
    )
    
    OpRegistry.register(name, Op, Op.metadata)
    
    return Op.apply


# =============================================================================
# Additional Utilities
# =============================================================================

class CompilationCache:
    """Cache for compiled extensions and kernels."""
    
    _cache: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        with cls._lock:
            cls._cache[key] = value
    
    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._cache.clear()
    
    @classmethod
    def keys(cls) -> List[str]:
        return list(cls._cache.keys())


@contextmanager
def cuda_stream(device: Optional[int] = None):
    """Context manager for CUDA stream."""
    if not torch.cuda.is_available():
        yield None
        return
    
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        yield stream
    torch.cuda.synchronize()


@contextmanager
def record_cuda_graph(
    model: nn.Module,
    example_inputs: Any,
    warmup: int = 3,
):
    """Context manager for CUDA graph recording."""
    if not torch.cuda.is_available():
        yield model
        return
    
    model.eval()
    graph = torch.cuda.CUDAGraph()
    
    # Warmup
    for _ in range(warmup):
        if isinstance(example_inputs, (list, tuple)):
            _ = model(*example_inputs)
        else:
            _ = model(example_inputs)
    
    torch.cuda.synchronize()
    
    # Record
    with torch.cuda.graph(graph):
        if isinstance(example_inputs, (list, tuple)):
            static_output = model(*example_inputs)
        else:
            static_output = model(example_inputs)
    
    yield graph
    
    graph.reset()


def benchmark_cuda_kernel(
    kernel: Callable,
    *args: Any,
    num_warmup: int = 10,
    num_iters: int = 100,
    synchronize: bool = True,
) -> Dict[str, float]:
    """
    Benchmark a CUDA kernel.
    
    Args:
        kernel: Kernel callable
        *args: Kernel arguments
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations
        synchronize: Whether to synchronize
    
    Returns:
        Benchmark statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # Warmup
    for _ in range(num_warmup):
        kernel(*args)
    
    if synchronize:
        torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        kernel(*args)
    end.record()
    
    if synchronize:
        torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    
    return {
        "mean_ms": elapsed_ms / num_iters,
        "total_ms": elapsed_ms,
        "num_iters": num_iters,
        "std_ms": 0.0,  # Would require multiple runs for std
    }


def get_device_info(device: Optional[int] = None) -> Dict[str, Any]:
    """Get comprehensive device information."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    if device is None:
        device = torch.cuda.current_device()
    
    props = torch.cuda.get_device_properties(device)
    
    return {
        "cuda_available": True,
        "device_id": device,
        "name": torch.cuda.get_device_name(device),
        "major": props.major,
        "minor": props.minor,
        "total_memory_gb": props.total_memory / (1024**3),
        "multi_processor_count": props.multi_processor_count,
        "warp_size": props.warp_size,
        "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
        "memory_clock_rate_mhz": props.memory_clock_rate / 1000,
        "memory_bus_width_bits": props.memory_bus_width,
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # C++ Extensions
    "CppExtension",
    "CppExtensionBuilder",
    "InlineCppExtensionBuilder",
    "load_cpp",
    "compile_cpp",
    "jit_compile",
    
    # CUDA Kernels
    "CudaKernel",
    "CudaKernelConfig",
    "CudaKernelCache",
    "load_cuda",
    "compile_cuda",
    "launch_kernel",
    
    # Custom Operations
    "CustomOp",
    "OpMetadata",
    "OpRegistry",
    "register_op",
    "load_op",
    
    # Bindings
    "Pybind11Module",
    "create_bindings",
    "export_function",
    "export_class",
    
    # Optimization
    "TorchScriptCompile",
    "ONNXExport",
    "TensorRTConvert",
    "OpenVINOConvert",
    
    # Memory
    "CudaMemoryPool",
    "CudaMemoryPoolConfig",
    "allocate_cuda",
    "free_cuda",
    "memory_stats",
    
    # Parallel
    "OpenMPParallel",
    "ThreadPool",
    "parallel_for",
    "parallel_reduce",
    
    # Utilities
    "cpp_extension",
    "cuda_jit",
    "custom_op",
    
    # Additional utilities
    "CompilationCache",
    "cuda_stream",
    "record_cuda_graph",
    "benchmark_cuda_kernel",
    "get_device_info",
    
    # Types
    "DeviceType",
    "ShapeType",
    "KernelFunc",
]
