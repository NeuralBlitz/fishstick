"""
fishstick Swift Backend Module
==============================

Apple platform deployment tools for fishstick models.

This module provides comprehensive Swift backend integration:

1. Swift Extensions (swift_extension, load_swift, compile_swift, swift_bridge)
   - Python-Swift interoperability layer
   - Dynamic library compilation
   - Function bridging

2. CoreML (CoreMLConverter, CoreMLModel, export_coreml, load_coreml)
   - Convert PyTorch, TensorFlow, ONNX models to CoreML
   - Model quantization and optimization
   - Swift code generation

3. Metal (MetalKernel, MetalBuffer, compile_metal, launch_metal)
   - GPU compute shaders
   - High-performance kernels
   - Buffer management

4. iOS Deployment (iOSExporter, iOSAppConfig, create_ios_app, iOSIntegration)
   - Complete Xcode project generation
   - SwiftUI/UIKit app templates
   - Camera and photo integration

5. macOS Deployment (MacOSExporter, MacOSAppConfig, create_macos_app, macOSIntegration)
   - macOS app generation
   - Menu bar apps
   - Batch processing tools

6. Swift for TensorFlow (S4TConverter, S4TModel, export_s4t, load_s4t)
   - S4T model conversion
   - Swift code generation
   - Training loop templates

7. Utilities (swift_extension, coreml_export, metal_kernel)
   - Quick conversion functions
   - Helper utilities

Quick Start:
    >>> from fishstick.swift.core import CoreMLConverter, export_coreml
    >>>
    >>> # Convert your model
    >>> converter = CoreMLConverter()
    >>> mlmodel = converter.convert(model, inputs=[...], outputs=[...])
    >>> mlmodel.save("MyModel.mlmodel")
    >>>
    >>> # Or use quick export
    >>> export_coreml(model, "MyModel.mlmodel", input_shape=(1, 3, 224, 224))

Requirements:
    - macOS with Xcode installed (for CoreML and Metal)
    - coremltools: pip install coremltools
    - Swift toolchain (for Swift Extensions)
"""

from .core import (
    # Swift Extensions
    SwiftExtension,
    load_swift,
    compile_swift,
    swift_bridge,
    # CoreML
    CoreMLConverter,
    CoreMLModel,
    export_coreml,
    load_coreml,
    # Metal
    MetalKernel,
    MetalBuffer,
    compile_metal,
    launch_metal,
    # iOS
    iOSExporter,
    iOSAppConfig,
    create_ios_app,
    iOSIntegration,
    optimize_for_ios,
    # macOS
    MacOSExporter,
    MacOSAppConfig,
    create_macos_app,
    macOSIntegration,
    optimize_for_macos,
    # Swift for TensorFlow
    S4TConverter,
    S4TModel,
    export_s4t,
    load_s4t,
    # Utilities
    swift_extension,
    coreml_export,
    metal_kernel,
    # Types
    Platform,
    SwiftDataType,
    ComputeUnits,
    TensorSpec,
    SwiftMetadata,
    MetalKernelSpec,
    # Exceptions
    SwiftBackendError,
    SwiftExtensionError,
    CoreMLConversionError,
    MetalKernelError,
    iOSDeploymentError,
    MacOSDeploymentError,
    S4TError,
)

__version__ = "0.1.0"
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
