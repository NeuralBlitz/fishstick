"""
fishstick C++ Backend Module
===========================

High-performance C++ and CUDA backend for fishstick with:
- C++ Extensions compilation and loading
- CUDA kernel management
- Custom PyTorch operations
- Pybind11 bindings
- Model optimization (TorchScript, ONNX, TensorRT, OpenVINO)
- Memory pooling
- Parallel processing
"""

from .core import (
    # C++ Extensions
    CppExtension,
    CppExtensionBuilder,
    InlineCppExtensionBuilder,
    load_cpp,
    compile_cpp,
    jit_compile,
    # CUDA Kernels
    CudaKernel,
    CudaKernelConfig,
    CudaKernelCache,
    load_cuda,
    compile_cuda,
    launch_kernel,
    # Custom Operations
    CustomOp,
    OpMetadata,
    OpRegistry,
    register_op,
    load_op,
    # Bindings
    Pybind11Module,
    create_bindings,
    export_function,
    export_class,
    # Optimization
    TorchScriptCompile,
    ONNXExport,
    TensorRTConvert,
    OpenVINOConvert,
    # Memory
    CudaMemoryPool,
    CudaMemoryPoolConfig,
    allocate_cuda,
    free_cuda,
    memory_stats,
    # Parallel
    OpenMPParallel,
    ThreadPool,
    parallel_for,
    parallel_reduce,
    # Utilities
    cpp_extension,
    cuda_jit,
    custom_op,
    CompilationCache,
    cuda_stream,
    record_cuda_graph,
    benchmark_cuda_kernel,
    get_device_info,
    # Types
    DeviceType,
    ShapeType,
    KernelFunc,
)

__version__ = "0.1.0"

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
