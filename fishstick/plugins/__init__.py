"""
Fishstick Plugin System

A comprehensive plugin system for extending fishstick functionality.
"""

from fishstick.plugins.core import (
    # Exceptions
    PluginError,
    PluginNotFoundError,
    PluginLoadError,
    PluginDependencyError,
    HookError,
    ExtensionError,
    ConfigurationError,
    # Enums
    PluginState,
    PluginPriority,
    HookPriority,
    # Data Classes
    PluginMetadata,
    PluginInfo,
    HookInfo,
    ExtensionInfo,
    DependencyNode,
    # Plugin Interface
    PluginInterface,
    BasePlugin,
    ModelPlugin,
    DataPlugin,
    TrainerPlugin,
    # Plugin Manager
    PluginManager,
    Manager,
    # Discovery
    PluginDiscovery,
    Discovery,
    # Hooks
    HookSystem,
    Hooks,
    HookRegistry,
    Registry,
    # Extensions
    ExtensionPoint,
    Point,
    ExtensionRegistry,
    # Configuration
    PluginConfig,
    Config,
    # Dependencies
    DependencyGraph,
    Graph,
    DependencyResolver,
    Resolve,
    # Utilities
    PluginLoader,
    Load,
    create_plugin,
    Create,
    plugin_decorator,
    Decorator,
)

__version__ = "1.0.0"
__all__ = [
    # Exceptions
    "PluginError",
    "PluginNotFoundError",
    "PluginLoadError",
    "PluginDependencyError",
    "HookError",
    "ExtensionError",
    "ConfigurationError",
    # Enums
    "PluginState",
    "PluginPriority",
    "HookPriority",
    # Data Classes
    "PluginMetadata",
    "PluginInfo",
    "HookInfo",
    "ExtensionInfo",
    "DependencyNode",
    # Plugin Interface
    "PluginInterface",
    "BasePlugin",
    "ModelPlugin",
    "DataPlugin",
    "TrainerPlugin",
    # Plugin Manager
    "PluginManager",
    "Manager",
    # Discovery
    "PluginDiscovery",
    "Discovery",
    # Hooks
    "HookSystem",
    "Hooks",
    "HookRegistry",
    "Registry",
    # Extensions
    "ExtensionPoint",
    "Point",
    "ExtensionRegistry",
    # Configuration
    "PluginConfig",
    "Config",
    # Dependencies
    "DependencyGraph",
    "Graph",
    "DependencyResolver",
    "Resolve",
    # Utilities
    "PluginLoader",
    "Load",
    "create_plugin",
    "Create",
    "plugin_decorator",
    "Decorator",
]
