"""
Fishstick Plugin System Core Module

A comprehensive plugin system for extending fishstick functionality.
Provides plugin management, discovery, hooks, extensions, and dependency resolution.
"""

from __future__ import annotations

import abc
import ast
import functools
import importlib
import importlib.util
import inspect
import json
import logging
import os
import pkgutil
import sys
import threading
import types
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)


# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class PluginError(Exception):
    """Base exception for plugin-related errors."""

    pass


class PluginNotFoundError(PluginError):
    """Raised when a plugin is not found."""

    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""

    pass


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies cannot be resolved."""

    pass


class HookError(PluginError):
    """Raised when a hook operation fails."""

    pass


class ExtensionError(PluginError):
    """Raised when an extension operation fails."""

    pass


class ConfigurationError(PluginError):
    """Raised when plugin configuration is invalid."""

    pass


# ============================================================================
# Enums and Constants
# ============================================================================


class PluginState(Enum):
    """Plugin lifecycle states."""

    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    ENABLED = auto()
    DISABLED = auto()
    ERROR = auto()


class PluginPriority(Enum):
    """Plugin priority levels for loading order."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class HookPriority(Enum):
    """Hook execution priority."""

    FIRST = -100
    EARLY = -50
    NORMAL = 0
    LATE = 50
    LAST = 100


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PluginMetadata:
    """Plugin metadata information."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    email: str = ""
    url: str = ""
    license: str = "MIT"
    min_platform_version: str = "1.0.0"
    max_platform_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    priority: PluginPriority = PluginPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None
    config_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "email": self.email,
            "url": self.url,
            "license": self.license,
            "min_platform_version": self.min_platform_version,
            "max_platform_version": self.max_platform_version,
            "tags": self.tags,
            "category": self.category,
            "priority": self.priority.name,
            "dependencies": self.dependencies,
            "optional_dependencies": self.optional_dependencies,
            "conflicts": self.conflicts,
            "provides": self.provides,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginMetadata:
        """Create metadata from dictionary."""
        data = data.copy()
        if "priority" in data and isinstance(data["priority"], str):
            data["priority"] = PluginPriority[data["priority"].upper()]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""

    metadata: PluginMetadata
    module: Optional[types.ModuleType] = None
    instance: Optional[BasePlugin] = None
    state: PluginState = PluginState.UNLOADED
    path: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    load_time: Optional[float] = None

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def is_active(self) -> bool:
        return self.state == PluginState.ENABLED


@dataclass
class HookInfo:
    """Information about a registered hook."""

    name: str
    callback: Callable[..., Any]
    priority: int
    plugin_name: Optional[str] = None
    once: bool = False
    async_callback: bool = False


@dataclass
class ExtensionInfo:
    """Information about an extension."""

    name: str
    extension_point: str
    provider: str
    factory: Callable[..., Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Plugin Interface
# ============================================================================


class PluginInterface(abc.ABC):
    """
    Abstract base class defining the plugin interface.
    All plugins must implement this interface.
    """

    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abc.abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """
        Initialize the plugin.

        Args:
            context: Plugin context with configuration and services

        Returns:
            True if initialization succeeded
        """
        pass

    @abc.abstractmethod
    def shutdown(self) -> bool:
        """
        Shutdown the plugin.

        Returns:
            True if shutdown succeeded
        """
        pass

    def enable(self) -> bool:
        """Enable the plugin. Override if needed."""
        return True

    def disable(self) -> bool:
        """Disable the plugin. Override if needed."""
        return True

    def get_config_schema(self) -> Optional[Dict[str, Any]]:
        """Return configuration schema for this plugin."""
        return None

    def on_config_changed(self, config: Dict[str, Any]) -> None:
        """Called when plugin configuration changes."""
        pass


class BasePlugin(PluginInterface):
    """
    Base class for all plugins.
    Provides common functionality and can be extended for specific plugin types.
    """

    _metadata: ClassVar[Optional[PluginMetadata]] = None

    def __init__(self):
        self._context: Optional[Dict[str, Any]] = None
        self._enabled = False
        self._initialized = False

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        if self._metadata is None:
            raise PluginError(f"Plugin {self.__class__.__name__} must define metadata")
        return self._metadata

    @classmethod
    def set_metadata(cls, metadata: PluginMetadata) -> None:
        """Set plugin metadata (typically called by plugin decorator)."""
        cls._metadata = metadata

    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        try:
            self._context = context
            self._do_initialize()
            self._initialized = True
            logger.info(f"Plugin {self.metadata.name} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize plugin {self.metadata.name}: {e}")
            return False

    def _do_initialize(self) -> None:
        """Override this method for custom initialization."""
        pass

    def shutdown(self) -> bool:
        """Shutdown the plugin."""
        try:
            if self._enabled:
                self.disable()
            self._do_shutdown()
            self._initialized = False
            logger.info(f"Plugin {self.metadata.name} shutdown")
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown plugin {self.metadata.name}: {e}")
            return False

    def _do_shutdown(self) -> None:
        """Override this method for custom shutdown."""
        pass

    def enable(self) -> bool:
        """Enable the plugin."""
        if not self._initialized:
            logger.warning(
                f"Cannot enable plugin {self.metadata.name}: not initialized"
            )
            return False

        try:
            self._do_enable()
            self._enabled = True
            logger.info(f"Plugin {self.metadata.name} enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable plugin {self.metadata.name}: {e}")
            return False

    def _do_enable(self) -> None:
        """Override this method for custom enable logic."""
        pass

    def disable(self) -> bool:
        """Disable the plugin."""
        try:
            self._do_disable()
            self._enabled = False
            logger.info(f"Plugin {self.metadata.name} disabled")
            return True
        except Exception as e:
            logger.error(f"Failed to disable plugin {self.metadata.name}: {e}")
            return False

    def _do_disable(self) -> None:
        """Override this method for custom disable logic."""
        pass

    def get_service(self, name: str) -> Any:
        """Get a service from the plugin context."""
        if self._context and "services" in self._context:
            return self._context["services"].get(name)
        return None

    def register_hook(
        self, hook_name: str, callback: Callable[..., Any], priority: int = 0
    ) -> None:
        """Register a hook (requires hook system in context)."""
        if self._context and "hook_system" in self._context:
            hook_system: HookSystem = self._context["hook_system"]
            hook_system.add_hook(hook_name, callback, priority, self.metadata.name)


class ModelPlugin(BasePlugin):
    """
    Base class for model-related plugins.
    Provides hooks for model lifecycle events.
    """

    def on_model_load(self, model: Any) -> Any:
        """Called when a model is loaded."""
        return model

    def on_model_save(self, model: Any, path: str) -> bool:
        """Called when a model is saved."""
        return True

    def on_model_train_start(self, model: Any, config: Dict[str, Any]) -> None:
        """Called when model training starts."""
        pass

    def on_model_train_end(self, model: Any, metrics: Dict[str, float]) -> None:
        """Called when model training ends."""
        pass

    def on_model_predict(self, model: Any, inputs: Any) -> Any:
        """Called during model prediction."""
        return inputs

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about models provided by this plugin."""
        return {}


class DataPlugin(BasePlugin):
    """
    Base class for data-related plugins.
    Provides hooks for data processing events.
    """

    def on_data_load(self, data: Any, source: str) -> Any:
        """Called when data is loaded."""
        return data

    def on_data_preprocess(self, data: Any) -> Any:
        """Called during data preprocessing."""
        return data

    def on_data_transform(self, data: Any, transform_name: str) -> Any:
        """Called during data transformation."""
        return data

    def on_data_save(self, data: Any, path: str) -> bool:
        """Called when data is saved."""
        return True

    def get_data_loaders(self) -> Dict[str, Callable[..., Any]]:
        """Return data loaders provided by this plugin."""
        return {}

    def get_transforms(self) -> Dict[str, Callable[..., Any]]:
        """Return data transforms provided by this plugin."""
        return {}


class TrainerPlugin(BasePlugin):
    """
    Base class for trainer-related plugins.
    Provides hooks for training lifecycle events.
    """

    def on_training_start(self, trainer: Any, config: Dict[str, Any]) -> None:
        """Called when training starts."""
        pass

    def on_training_end(self, trainer: Any, results: Dict[str, Any]) -> None:
        """Called when training ends."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> Any:
        """Called at the start of each batch."""
        return batch

    def on_batch_end(
        self, trainer: Any, batch: Any, batch_idx: int, outputs: Any, loss: float
    ) -> None:
        """Called at the end of each batch."""
        pass

    def on_backward(self, trainer: Any, loss: Any) -> Any:
        """Called during backward pass."""
        return loss

    def on_optimizer_step(self, trainer: Any, optimizer: Any) -> None:
        """Called during optimizer step."""
        pass

    def get_callbacks(self) -> Dict[str, Callable[..., Any]]:
        """Return training callbacks provided by this plugin."""
        return {}


# ============================================================================
# Plugin Discovery
# ============================================================================


class PluginDiscovery:
    """
    Discovers plugins from various sources.
    Supports file system scanning, module path scanning, and entry points.
    """

    def __init__(self):
        self._discovered: Dict[str, PluginMetadata] = {}
        self._discovery_paths: List[Path] = []

    def add_discovery_path(self, path: Union[str, Path]) -> None:
        """Add a path to search for plugins."""
        self._discovery_paths.append(Path(path))

    def scan_for_plugins(
        self, paths: Optional[List[Union[str, Path]]] = None
    ) -> Dict[str, PluginMetadata]:
        """
        Scan for plugins in the specified paths.

        Args:
            paths: Paths to scan. If None, uses registered discovery paths.

        Returns:
            Dictionary mapping plugin names to metadata
        """
        search_paths = [Path(p) for p in paths] if paths else self._discovery_paths
        discovered = {}

        for path in search_paths:
            if not path.exists():
                logger.warning(f"Discovery path does not exist: {path}")
                continue

            # Scan Python files
            for py_file in path.rglob("*.py"):
                try:
                    metadata = self._extract_metadata_from_file(py_file)
                    if metadata:
                        discovered[metadata.name] = metadata
                        logger.debug(
                            f"Discovered plugin: {metadata.name} from {py_file}"
                        )
                except Exception as e:
                    logger.debug(f"Failed to extract metadata from {py_file}: {e}")

            # Scan plugin.json files
            for json_file in path.rglob("plugin.json"):
                try:
                    metadata = self._load_metadata_from_json(json_file)
                    if metadata:
                        discovered[metadata.name] = metadata
                        logger.debug(
                            f"Discovered plugin: {metadata.name} from {json_file}"
                        )
                except Exception as e:
                    logger.debug(f"Failed to load metadata from {json_file}: {e}")

        self._discovered.update(discovered)
        return discovered

    def discover_plugins(
        self, package_prefix: str = "fishstick.plugins"
    ) -> Dict[str, PluginMetadata]:
        """
        Discover plugins in installed packages.

        Args:
            package_prefix: Package prefix to search under

        Returns:
            Dictionary mapping plugin names to metadata
        """
        discovered = {}

        try:
            # Find all modules under the package prefix
            for importer, modname, ispkg in pkgutil.iter_modules():
                if modname.startswith(package_prefix.replace(".", "_")) or (
                    ispkg and modname.startswith(package_prefix.split(".")[-1])
                ):
                    try:
                        module = importlib.import_module(modname)
                        metadata = self._extract_metadata_from_module(module)
                        if metadata:
                            discovered[metadata.name] = metadata
                    except Exception as e:
                        logger.debug(f"Failed to import module {modname}: {e}")
        except Exception as e:
            logger.warning(f"Error during plugin discovery: {e}")

        self._discovered.update(discovered)
        return discovered

    def register_plugin(
        self, metadata: PluginMetadata, module_path: Optional[Path] = None
    ) -> None:
        """
        Manually register a plugin.

        Args:
            metadata: Plugin metadata
            module_path: Optional path to plugin module
        """
        self._discovered[metadata.name] = metadata
        if module_path:
            logger.debug(f"Registered plugin {metadata.name} from {module_path}")
        else:
            logger.debug(f"Registered plugin {metadata.name}")

    def get_discovered(self) -> Dict[str, PluginMetadata]:
        """Get all discovered plugins."""
        return self._discovered.copy()

    def _extract_metadata_from_file(self, file_path: Path) -> Optional[PluginMetadata]:
        """Extract plugin metadata from a Python file."""
        try:
            with open(file_path, "r") as f:
                tree = ast.parse(f.read())

            # Look for plugin decorator or metadata definition
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check for plugin decorator
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if isinstance(
                                decorator.func, ast.Name
                            ) and decorator.func.id in ("plugin", "create_plugin"):
                                # Extract metadata from decorator arguments
                                return self._extract_metadata_from_decorator(
                                    decorator, file_path
                                )
                        elif isinstance(decorator, ast.Name):
                            if decorator.id in ("plugin", "create_plugin"):
                                # Look for metadata in class attributes
                                return self._extract_metadata_from_class(
                                    node, file_path
                                )

                elif isinstance(node, ast.Assign):
                    # Look for __plugin_metadata__ assignment
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "__plugin_metadata__"
                        ):
                            if isinstance(node.value, ast.Dict):
                                metadata_dict = ast.literal_eval(node.value)
                                return PluginMetadata.from_dict(metadata_dict)

            return None
        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")
            return None

    def _extract_metadata_from_decorator(
        self, decorator: ast.Call, file_path: Path
    ) -> Optional[PluginMetadata]:
        """Extract metadata from plugin decorator."""
        try:
            kwargs = {}
            for keyword in decorator.keywords:
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)

            if "name" not in kwargs:
                kwargs["name"] = file_path.stem

            return PluginMetadata(**kwargs)
        except Exception as e:
            logger.debug(f"Error extracting metadata from decorator: {e}")
            return None

    def _extract_metadata_from_class(
        self, class_def: ast.ClassDef, file_path: Path
    ) -> Optional[PluginMetadata]:
        """Extract metadata from plugin class definition."""
        try:
            metadata_dict = {"name": class_def.name}

            for item in class_def.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            if target.id == "__version__":
                                metadata_dict["version"] = ast.literal_eval(item.value)
                            elif target.id == "__description__":
                                metadata_dict["description"] = ast.literal_eval(
                                    item.value
                                )
                            elif target.id == "__author__":
                                metadata_dict["author"] = ast.literal_eval(item.value)

            return PluginMetadata(**metadata_dict)
        except Exception as e:
            logger.debug(f"Error extracting metadata from class: {e}")
            return None

    def _extract_metadata_from_module(
        self, module: types.ModuleType
    ) -> Optional[PluginMetadata]:
        """Extract metadata from an imported module."""
        if hasattr(module, "__plugin_metadata__"):
            metadata_dict = module.__plugin_metadata__
            if isinstance(metadata_dict, dict):
                return PluginMetadata.from_dict(metadata_dict)

        # Look for plugin classes
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BasePlugin):
                if hasattr(obj, "_metadata") and obj._metadata is not None:
                    return obj._metadata

        return None

    def _load_metadata_from_json(self, json_path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from a JSON file."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return PluginMetadata.from_dict(data)
        except Exception as e:
            logger.debug(f"Error loading metadata from {json_path}: {e}")
            return None


# ============================================================================
# Hook System
# ============================================================================


class HookSystem:
    """
    Manages hooks for plugin extensibility.
    Allows plugins to register callbacks for specific events.
    """

    def __init__(self):
        self._hooks: Dict[str, List[HookInfo]] = defaultdict(list)
        self._lock = threading.RLock()

    def add_hook(
        self,
        name: str,
        callback: Callable[..., Any],
        priority: int = 0,
        plugin_name: Optional[str] = None,
        once: bool = False,
    ) -> None:
        """
        Add a hook callback.

        Args:
            name: Hook name/event
            callback: Function to call when hook is executed
            priority: Execution priority (lower = earlier)
            plugin_name: Name of the plugin registering the hook
            once: If True, remove after first execution
        """
        with self._lock:
            hook_info = HookInfo(
                name=name,
                callback=callback,
                priority=priority,
                plugin_name=plugin_name,
                once=once,
            )

            # Insert in priority order
            hooks = self._hooks[name]
            idx = 0
            for i, existing in enumerate(hooks):
                if existing.priority > priority:
                    idx = i
                    break
                idx = i + 1

            hooks.insert(idx, hook_info)
            logger.debug(f"Added hook '{name}' with priority {priority}")

    def remove_hook(
        self,
        name: str,
        callback: Optional[Callable[..., Any]] = None,
        plugin_name: Optional[str] = None,
    ) -> bool:
        """
        Remove a hook callback.

        Args:
            name: Hook name
            callback: Specific callback to remove (if None, removes all for plugin)
            plugin_name: Remove all hooks for this plugin

        Returns:
            True if any hooks were removed
        """
        with self._lock:
            if name not in self._hooks:
                return False

            original_count = len(self._hooks[name])

            if callback is not None:
                self._hooks[name] = [
                    h for h in self._hooks[name] if h.callback != callback
                ]
            elif plugin_name is not None:
                self._hooks[name] = [
                    h for h in self._hooks[name] if h.plugin_name != plugin_name
                ]
            else:
                self._hooks[name] = []

            removed = original_count - len(self._hooks[name])
            if removed > 0:
                logger.debug(f"Removed {removed} hook(s) from '{name}'")

            return removed > 0

    def execute_hooks(
        self,
        name: str,
        *args,
        stop_on_false: bool = False,
        aggregate_results: bool = False,
        **kwargs,
    ) -> Any:
        """
        Execute all callbacks for a hook.

        Args:
            name: Hook name
            *args: Positional arguments to pass to callbacks
            stop_on_false: Stop execution if a callback returns False
            aggregate_results: Aggregate all results into a list
            **kwargs: Keyword arguments to pass to callbacks

        Returns:
            Result from last callback, or aggregated results, or None
        """
        with self._lock:
            hooks = self._hooks.get(name, [])

        if not hooks:
            return [] if aggregate_results else None

        results = []
        hooks_to_remove = []

        for hook in hooks:
            try:
                result = hook.callback(*args, **kwargs)

                if aggregate_results:
                    results.append(result)

                if hook.once:
                    hooks_to_remove.append(hook)

                if stop_on_false and result is False:
                    break

            except Exception as e:
                logger.error(f"Hook '{name}' callback failed: {e}")
                if hook.plugin_name:
                    logger.error(f"  (from plugin: {hook.plugin_name})")

        # Remove once hooks
        if hooks_to_remove:
            with self._lock:
                for hook in hooks_to_remove:
                    if hook in self._hooks[name]:
                        self._hooks[name].remove(hook)

        return results if aggregate_results else (results[-1] if results else None)

    def has_hooks(self, name: str) -> bool:
        """Check if a hook has any registered callbacks."""
        return name in self._hooks and len(self._hooks[name]) > 0

    def get_hook_names(self) -> List[str]:
        """Get all registered hook names."""
        return list(self._hooks.keys())

    def get_hook_count(self, name: Optional[str] = None) -> int:
        """Get number of registered hooks."""
        if name:
            return len(self._hooks.get(name, []))
        return sum(len(hooks) for hooks in self._hooks.values())

    def clear_hooks(self, name: Optional[str] = None) -> None:
        """Clear all hooks (or hooks for a specific name)."""
        with self._lock:
            if name:
                self._hooks[name] = []
            else:
                self._hooks.clear()


class HookRegistry:
    """
    Registry for hook names and documentation.
    Helps maintain a catalog of available hooks.
    """

    def __init__(self):
        self._hooks: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str = "",
        signature: Optional[str] = None,
        return_type: Optional[str] = None,
    ) -> None:
        """
        Register a hook for documentation purposes.

        Args:
            name: Hook name
            description: Description of when/why this hook is called
            signature: Expected callback signature
            return_type: Expected return type
        """
        self._hooks[name] = {
            "name": name,
            "description": description,
            "signature": signature or "(*args, **kwargs)",
            "return_type": return_type or "Any",
        }

    def get_hook_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered hook."""
        return self._hooks.get(name)

    def list_hooks(self) -> List[str]:
        """List all registered hook names."""
        return list(self._hooks.keys())

    def get_all_hooks(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered hook information."""
        return self._hooks.copy()


# ============================================================================
# Extension System
# ============================================================================


T = TypeVar("T")


class ExtensionPoint(Generic[T]):
    """
    Defines an extension point where plugins can contribute functionality.
    """

    def __init__(
        self,
        name: str,
        interface: Type[T],
        description: str = "",
        singleton: bool = False,
    ):
        """
        Initialize an extension point.

        Args:
            name: Unique name for this extension point
            interface: Interface/Protocol that extensions must implement
            description: Description of what this extension point provides
            singleton: If True, only one extension can be registered
        """
        self.name = name
        self.interface = interface
        self.description = description
        self.singleton = singleton
        self._extensions: Dict[str, ExtensionInfo] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        provider: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register an extension.

        Args:
            name: Extension name
            factory: Factory function that creates the extension
            provider: Name of the plugin providing this extension
            metadata: Additional metadata
        """
        if self.singleton and self._extensions:
            raise ExtensionError(
                f"Extension point '{self.name}' is singleton and already has an extension"
            )

        self._extensions[name] = ExtensionInfo(
            name=name,
            extension_point=self.name,
            provider=provider,
            factory=factory,
            metadata=metadata or {},
        )
        logger.debug(f"Registered extension '{name}' at point '{self.name}'")

    def unregister(self, name: str) -> bool:
        """Unregister an extension."""
        if name in self._extensions:
            del self._extensions[name]
            logger.debug(f"Unregistered extension '{name}' from '{self.name}'")
            return True
        return False

    def get_extension(self, name: str) -> Optional[T]:
        """Get a specific extension instance."""
        if name not in self._extensions:
            return None

        ext_info = self._extensions[name]
        return ext_info.factory()

    def get_extensions(self) -> Dict[str, T]:
        """Get all extension instances."""
        return {name: ext.factory() for name, ext in self._extensions.items()}

    def get_extension_names(self) -> List[str]:
        """Get names of all registered extensions."""
        return list(self._extensions.keys())

    def has_extension(self, name: str) -> bool:
        """Check if an extension is registered."""
        return name in self._extensions

    def clear(self) -> None:
        """Clear all extensions."""
        self._extensions.clear()


class ExtensionRegistry:
    """
    Central registry for all extension points.
    """

    def __init__(self):
        self._points: Dict[str, ExtensionPoint] = {}

    def register_extension_point(self, point: ExtensionPoint) -> None:
        """Register an extension point."""
        self._points[point.name] = point
        logger.debug(f"Registered extension point '{point.name}'")

    def create_extension_point(
        self,
        name: str,
        interface: Type[T],
        description: str = "",
        singleton: bool = False,
    ) -> ExtensionPoint[T]:
        """
        Create and register a new extension point.

        Args:
            name: Extension point name
            interface: Interface type
            description: Description
            singleton: Whether only one extension is allowed

        Returns:
            The created ExtensionPoint
        """
        point = ExtensionPoint(name, interface, description, singleton)
        self.register_extension_point(point)
        return point

    def get_extension_point(self, name: str) -> Optional[ExtensionPoint]:
        """Get an extension point by name."""
        return self._points.get(name)

    def register_extension(
        self,
        point_name: str,
        ext_name: str,
        factory: Callable[..., Any],
        provider: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register an extension at a specific extension point.

        Args:
            point_name: Name of the extension point
            ext_name: Name of the extension
            factory: Factory function
            provider: Provider plugin name
            metadata: Additional metadata

        Returns:
            True if registration succeeded
        """
        point = self._points.get(point_name)
        if point is None:
            logger.error(f"Extension point '{point_name}' not found")
            return False

        try:
            point.register(ext_name, factory, provider, metadata)
            return True
        except ExtensionError as e:
            logger.error(f"Failed to register extension: {e}")
            return False

    def get_extensions(self, point_name: str) -> Dict[str, Any]:
        """Get all extensions for an extension point."""
        point = self._points.get(point_name)
        if point:
            return point.get_extensions()
        return {}

    def list_extension_points(self) -> List[str]:
        """List all extension point names."""
        return list(self._points.keys())

    def clear(self) -> None:
        """Clear all extension points and extensions."""
        self._points.clear()


# ============================================================================
# Configuration
# ============================================================================


class PluginConfig:
    """
    Manages plugin configuration.
    Supports JSON-based configuration with schema validation.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file/directory
        """
        self._config_path = Path(config_path) if config_path else None
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._defaults: Dict[str, Dict[str, Any]] = {}

    def configure_plugin(
        self,
        plugin_name: str,
        config: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Configure a plugin.

        Args:
            plugin_name: Name of the plugin
            config: Configuration dictionary
            schema: Optional JSON schema for validation

        Returns:
            True if configuration was valid and applied
        """
        if schema:
            if not self.validate_config(config, schema):
                return False
            self._schemas[plugin_name] = schema

        # Merge with defaults if they exist
        if plugin_name in self._defaults:
            merged = self._defaults[plugin_name].copy()
            merged.update(config)
            config = merged

        self._configs[plugin_name] = config
        logger.debug(f"Configured plugin '{plugin_name}'")
        return True

    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get configuration for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Configuration dictionary (may be empty)
        """
        return self._configs.get(plugin_name, {}).copy()

    def set_default_config(self, plugin_name: str, defaults: Dict[str, Any]) -> None:
        """Set default configuration for a plugin."""
        self._defaults[plugin_name] = defaults.copy()

        # Apply defaults if no config exists
        if plugin_name not in self._configs:
            self._configs[plugin_name] = defaults.copy()

    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a schema.

        Args:
            config: Configuration to validate
            schema: JSON schema

        Returns:
            True if valid
        """
        try:
            # Simple schema validation
            # In production, use jsonschema library
            required = schema.get("required", [])
            properties = schema.get("properties", {})

            # Check required fields
            for field in required:
                if field not in config:
                    logger.error(f"Missing required config field: {field}")
                    return False

            # Validate field types
            for key, value in config.items():
                if key in properties:
                    prop = properties[key]
                    prop_type = prop.get("type")

                    if prop_type == "string" and not isinstance(value, str):
                        logger.error(f"Config field '{key}' must be a string")
                        return False
                    elif prop_type == "integer" and not isinstance(value, int):
                        logger.error(f"Config field '{key}' must be an integer")
                        return False
                    elif prop_type == "number" and not isinstance(value, (int, float)):
                        logger.error(f"Config field '{key}' must be a number")
                        return False
                    elif prop_type == "boolean" and not isinstance(value, bool):
                        logger.error(f"Config field '{key}' must be a boolean")
                        return False
                    elif prop_type == "array" and not isinstance(value, list):
                        logger.error(f"Config field '{key}' must be an array")
                        return False
                    elif prop_type == "object" and not isinstance(value, dict):
                        logger.error(f"Config field '{key}' must be an object")
                        return False

            return True
        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    def load_from_file(self, file_path: Union[str, Path]) -> bool:
        """
        Load configuration from a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            True if loaded successfully
        """
        try:
            path = Path(file_path)
            with open(path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for plugin_name, config in data.items():
                    self.configure_plugin(plugin_name, config)

            logger.info(f"Loaded configuration from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def save_to_file(self, file_path: Union[str, Path]) -> bool:
        """
        Save configuration to a JSON file.

        Args:
            file_path: Path to save to

        Returns:
            True if saved successfully
        """
        try:
            path = Path(file_path)
            with open(path, "w") as f:
                json.dump(self._configs, f, indent=2)

            logger.info(f"Saved configuration to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all plugin configurations."""
        return self._configs.copy()


# ============================================================================
# Dependencies
# ============================================================================


@dataclass
class DependencyNode:
    """Node in the dependency graph."""

    name: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    resolved: bool = False


class DependencyGraph:
    """
    Manages plugin dependencies as a directed graph.
    """

    def __init__(self):
        self._nodes: Dict[str, DependencyNode] = {}

    def add_node(self, name: str, dependencies: Optional[Set[str]] = None) -> None:
        """Add a node to the graph."""
        if name not in self._nodes:
            self._nodes[name] = DependencyNode(name)

        if dependencies:
            self._nodes[name].dependencies.update(dependencies)

            # Add reverse edges
            for dep in dependencies:
                if dep not in self._nodes:
                    self._nodes[dep] = DependencyNode(dep)
                self._nodes[dep].dependents.add(name)

    def remove_node(self, name: str) -> None:
        """Remove a node from the graph."""
        if name not in self._nodes:
            return

        node = self._nodes[name]

        # Remove reverse edges
        for dep in node.dependencies:
            if dep in self._nodes:
                self._nodes[dep].dependents.discard(name)

        # Remove from dependents' dependencies
        for dependent in node.dependents:
            if dependent in self._nodes:
                self._nodes[dependent].dependencies.discard(name)

        del self._nodes[name]

    def topological_sort(self) -> List[str]:
        """
        Perform topological sort on the graph.

        Returns:
            List of node names in dependency order

        Raises:
            PluginDependencyError: If a cycle is detected
        """
        visited: Set[str] = set()
        temp_mark: Set[str] = set()
        result: List[str] = []

        def visit(node_name: str) -> None:
            if node_name in temp_mark:
                raise PluginDependencyError(
                    f"Circular dependency detected involving '{node_name}'"
                )

            if node_name in visited:
                return

            temp_mark.add(node_name)

            if node_name in self._nodes:
                for dep in self._nodes[node_name].dependencies:
                    visit(dep)

            temp_mark.remove(node_name)
            visited.add(node_name)
            result.append(node_name)

        for name in self._nodes:
            if name not in visited:
                visit(name)

        return result

    def get_dependencies(self, name: str, recursive: bool = False) -> Set[str]:
        """Get dependencies of a node."""
        if name not in self._nodes:
            return set()

        deps = self._nodes[name].dependencies.copy()

        if recursive:
            all_deps = set()
            to_process = list(deps)

            while to_process:
                dep = to_process.pop()
                if dep not in all_deps:
                    all_deps.add(dep)
                    if dep in self._nodes:
                        for sub_dep in self._nodes[dep].dependencies:
                            if sub_dep not in all_deps:
                                to_process.append(sub_dep)

            deps = all_deps

        return deps

    def get_dependents(self, name: str) -> Set[str]:
        """Get nodes that depend on a given node."""
        if name not in self._nodes:
            return set()

        return self._nodes[name].dependents.copy()

    def has_cycles(self) -> bool:
        """Check if the graph has cycles."""
        try:
            self.topological_sort()
            return False
        except PluginDependencyError:
            return True


class DependencyResolver:
    """
    Resolves plugin dependencies.
    """

    def __init__(self, plugin_manager: PluginManager):
        self._manager = plugin_manager
        self._graph = DependencyGraph()

    def check_dependencies(
        self, plugin_name: str, metadata: Optional[PluginMetadata] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if all dependencies for a plugin are satisfied.

        Args:
            plugin_name: Name of the plugin
            metadata: Plugin metadata (if None, will be looked up)

        Returns:
            Tuple of (success, missing_dependencies)
        """
        if metadata is None:
            plugin_info = self._manager.get_plugin_info(plugin_name)
            if plugin_info is None:
                return False, [plugin_name]
            metadata = plugin_info.metadata

        missing = []

        for dep in metadata.dependencies:
            if not self._manager.is_plugin_loaded(dep):
                missing.append(dep)

        # Check conflicts
        for conflict in metadata.conflicts:
            if self._manager.is_plugin_loaded(conflict):
                logger.error(f"Plugin '{plugin_name}' conflicts with '{conflict}'")
                return False, [f"conflict:{conflict}"]

        return len(missing) == 0, missing

    def resolve_load_order(self, plugin_names: List[str]) -> List[str]:
        """
        Determine the correct load order for plugins based on dependencies.

        Args:
            plugin_names: List of plugin names to order

        Returns:
            List of plugin names in load order
        """
        # Build dependency graph
        self._graph = DependencyGraph()

        metadata_map: Dict[str, PluginMetadata] = {}

        for name in plugin_names:
            plugin_info = self._manager.get_plugin_info(name)
            if plugin_info:
                metadata_map[name] = plugin_info.metadata
                deps = set(plugin_info.metadata.dependencies)
                self._graph.add_node(name, deps)

        # Get topological order
        try:
            ordered = self._graph.topological_sort()
            # Filter to only requested plugins
            ordered = [name for name in ordered if name in plugin_names]

            # Sort by priority within same dependency level
            def priority_key(name: str) -> int:
                if name in metadata_map:
                    return metadata_map[name].priority.value
                return PluginPriority.NORMAL.value

            # Group by dependencies
            result = []
            loaded: Set[str] = set()

            while len(result) < len(plugin_names):
                progress = False

                for name in ordered:
                    if name in loaded:
                        continue

                    plugin_info = self._manager.get_plugin_info(name)
                    if plugin_info is None:
                        continue

                    deps = set(plugin_info.metadata.dependencies)
                    if deps.issubset(loaded):
                        result.append(name)
                        loaded.add(name)
                        progress = True

                if not progress:
                    # Deadlock - circular dependency
                    remaining = [n for n in plugin_names if n not in loaded]
                    raise PluginDependencyError(
                        f"Cannot resolve dependencies for: {remaining}"
                    )

            return result

        except PluginDependencyError:
            raise

    def install_dependencies(
        self, plugin_name: str, metadata: Optional[PluginMetadata] = None
    ) -> bool:
        """
        Install missing dependencies for a plugin.

        Args:
            plugin_name: Name of the plugin
            metadata: Plugin metadata

        Returns:
            True if all dependencies were installed
        """
        # This would typically install Python packages
        # For now, just log what's needed
        if metadata is None:
            plugin_info = self._manager.get_plugin_info(plugin_name)
            if plugin_info is None:
                return False
            metadata = plugin_info.metadata

        success, missing = self.check_dependencies(plugin_name, metadata)

        if not success and missing:
            logger.info(f"Plugin '{plugin_name}' requires: {missing}")
            logger.info("Please install missing dependencies manually")
            return False

        return True


# ============================================================================
# Plugin Manager
# ============================================================================


class PluginManager:
    """
    Central manager for the plugin system.
    Handles loading, unloading, and managing plugins.
    """

    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._discovery = PluginDiscovery()
        self._hooks = HookSystem()
        self._hook_registry = HookRegistry()
        self._extensions = ExtensionRegistry()
        self._config = PluginConfig()
        self._resolver = DependencyResolver(self)
        self._lock = threading.RLock()
        self._context: Dict[str, Any] = {
            "hook_system": self._hooks,
            "extension_registry": self._extensions,
            "plugin_manager": self,
        }

    def load_plugin(
        self,
        plugin_path: Union[str, Path, types.ModuleType],
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Load a plugin from a path or module.

        Args:
            plugin_path: Path to plugin file/directory, or module
            config: Optional configuration for the plugin

        Returns:
            True if plugin was loaded successfully
        """
        with self._lock:
            try:
                if isinstance(plugin_path, types.ModuleType):
                    return self._load_from_module(plugin_path, config)

                path = Path(plugin_path)

                if path.is_file() and path.suffix == ".py":
                    return self._load_from_file(path, config)
                elif path.is_dir():
                    return self._load_from_directory(path, config)
                else:
                    # Try as module name
                    return self._load_from_module_name(str(plugin_path), config)

            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_path}: {e}")
                return False

    def _load_from_file(
        self, file_path: Path, config: Optional[Dict[str, Any]]
    ) -> bool:
        """Load plugin from a Python file."""
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return self._load_from_module(module, config, file_path)

    def _load_from_directory(
        self, dir_path: Path, config: Optional[Dict[str, Any]]
    ) -> bool:
        """Load plugin from a directory."""
        # Look for __init__.py or plugin.py
        init_file = dir_path / "__init__.py"
        plugin_file = dir_path / "plugin.py"

        if init_file.exists():
            return self._load_from_file(init_file, config)
        elif plugin_file.exists():
            return self._load_from_file(plugin_file, config)
        else:
            raise PluginLoadError(
                f"No plugin found in {dir_path} (need __init__.py or plugin.py)"
            )

    def _load_from_module(
        self,
        module: types.ModuleType,
        config: Optional[Dict[str, Any]],
        path: Optional[Path] = None,
    ) -> bool:
        """Load plugin from an imported module."""
        # Find plugin classes in the module
        plugin_class = None
        metadata = None

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BasePlugin):
                if obj is not BasePlugin:
                    plugin_class = obj
                    if hasattr(obj, "_metadata") and obj._metadata is not None:
                        metadata = obj._metadata
                    break

        if plugin_class is None:
            # Try to get metadata from module
            metadata = self._discovery._extract_metadata_from_module(module)
            if metadata is None:
                raise PluginLoadError(f"No plugin class found in {module.__name__}")

        # If no metadata from class, try module
        if metadata is None:
            metadata = PluginMetadata(name=module.__name__)

        # Check if already loaded
        if metadata.name in self._plugins:
            logger.warning(f"Plugin '{metadata.name}' is already loaded")
            return False

        # Check dependencies
        if metadata.dependencies:
            success, missing = self._resolver.check_dependencies(
                metadata.name, metadata
            )
            if not success:
                raise PluginDependencyError(
                    f"Missing dependencies for '{metadata.name}': {missing}"
                )

        # Create plugin instance
        instance = None
        if plugin_class is not None:
            instance = plugin_class()

        # Create plugin info
        plugin_info = PluginInfo(
            metadata=metadata,
            module=module,
            instance=instance,
            state=PluginState.LOADING,
            path=path,
            config=config or {},
        )

        self._plugins[metadata.name] = plugin_info

        # Initialize the plugin
        if instance is not None:
            plugin_context = self._context.copy()
            plugin_context["config"] = config or {}

            if not instance.initialize(plugin_context):
                plugin_info.state = PluginState.ERROR
                plugin_info.error_message = "Initialization failed"
                return False

            plugin_info.state = PluginState.LOADED

            # Enable the plugin
            if instance.enable():
                plugin_info.state = PluginState.ENABLED
            else:
                plugin_info.state = PluginState.DISABLED
        else:
            plugin_info.state = PluginState.LOADED

        logger.info(f"Loaded plugin: {metadata.name}")

        # Execute load hooks
        self._hooks.execute_hooks(
            "plugin.loaded", plugin_name=metadata.name, plugin_info=plugin_info
        )

        return True

    def _load_from_module_name(
        self, module_name: str, config: Optional[Dict[str, Any]]
    ) -> bool:
        """Load plugin from a module name."""
        try:
            module = importlib.import_module(module_name)
            return self._load_from_module(module, config)
        except ImportError as e:
            raise PluginLoadError(f"Cannot import module {module_name}: {e}")

    def unload_plugin(self, plugin_name: str, force: bool = False) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of the plugin to unload
            force: Force unload even if other plugins depend on it

        Returns:
            True if plugin was unloaded
        """
        with self._lock:
            if plugin_name not in self._plugins:
                logger.warning(f"Plugin '{plugin_name}' is not loaded")
                return False

            plugin_info = self._plugins[plugin_name]

            # Check for dependents
            if not force:
                dependents = self._resolver._graph.get_dependents(plugin_name)
                loaded_dependents = [d for d in dependents if d in self._plugins]
                if loaded_dependents:
                    logger.error(
                        f"Cannot unload '{plugin_name}' - required by: {loaded_dependents}"
                    )
                    return False

            # Shutdown the plugin
            if plugin_info.instance:
                plugin_info.instance.shutdown()

            # Remove hooks registered by this plugin
            for hook_name in self._hooks.get_hook_names():
                self._hooks.remove_hook(hook_name, plugin_name=plugin_name)

            # Remove from registry
            del self._plugins[plugin_name]

            # Update dependency graph
            self._resolver._graph.remove_node(plugin_name)

            logger.info(f"Unloaded plugin: {plugin_name}")

            # Execute unload hooks
            self._hooks.execute_hooks("plugin.unloaded", plugin_name=plugin_name)

            return True

    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.

        Args:
            plugin_name: Name of the plugin to reload

        Returns:
            True if plugin was reloaded
        """
        with self._lock:
            if plugin_name not in self._plugins:
                logger.warning(f"Plugin '{plugin_name}' is not loaded")
                return False

            plugin_info = self._plugins[plugin_name]
            path = plugin_info.path
            config = plugin_info.config

            # Unload
            if not self.unload_plugin(plugin_name):
                return False

            # Reload module if possible
            if path and path.exists():
                return self.load_plugin(path, config)
            elif plugin_info.module:
                # Try to reload the module
                importlib.reload(plugin_info.module)
                return self._load_from_module(plugin_info.module, config)

            return False

    def list_plugins(self, state: Optional[PluginState] = None) -> List[PluginInfo]:
        """
        List loaded plugins.

        Args:
            state: Filter by state (if None, returns all)

        Returns:
            List of plugin information
        """
        plugins = list(self._plugins.values())

        if state:
            plugins = [p for p in plugins if p.state == state]

        return plugins

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a loaded plugin."""
        return self._plugins.get(plugin_name)

    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is loaded."""
        return plugin_name in self._plugins

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        if plugin_name not in self._plugins:
            return False
        return self._plugins[plugin_name].state == PluginState.ENABLED

    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a loaded plugin."""
        if plugin_name not in self._plugins:
            return False

        plugin_info = self._plugins[plugin_name]

        if plugin_info.state == PluginState.ENABLED:
            return True

        if plugin_info.instance:
            if plugin_info.instance.enable():
                plugin_info.state = PluginState.ENABLED
                return True

        return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a loaded plugin."""
        if plugin_name not in self._plugins:
            return False

        plugin_info = self._plugins[plugin_name]

        if plugin_info.state == PluginState.DISABLED:
            return True

        if plugin_info.instance:
            if plugin_info.instance.disable():
                plugin_info.state = PluginState.DISABLED
                return True

        return False

    def discover_and_load(
        self, paths: Optional[List[Union[str, Path]]] = None, auto_load: bool = True
    ) -> List[str]:
        """
        Discover and optionally load plugins from paths.

        Args:
            paths: Paths to search for plugins
            auto_load: If True, load discovered plugins

        Returns:
            List of discovered plugin names
        """
        discovered = self._discovery.scan_for_plugins(paths)

        loaded = []
        if auto_load:
            for name, metadata in discovered.items():
                # Skip if already loaded
                if name in self._plugins:
                    continue

                # Try to find and load the plugin
                # This would need the actual path, which we'd need to store
                logger.info(f"Discovered plugin '{name}' - use load_plugin() to load")

        return list(discovered.keys())

    def get_context(self) -> Dict[str, Any]:
        """Get the plugin context."""
        return self._context.copy()

    def shutdown_all(self) -> None:
        """Shutdown all loaded plugins."""
        with self._lock:
            # Unload in reverse dependency order
            order = self._resolver.resolve_load_order(list(self._plugins.keys()))

            for plugin_name in reversed(order):
                self.unload_plugin(plugin_name, force=True)

    @property
    def discovery(self) -> PluginDiscovery:
        """Get the plugin discovery instance."""
        return self._discovery

    @property
    def hooks(self) -> HookSystem:
        """Get the hook system instance."""
        return self._hooks

    @property
    def hook_registry(self) -> HookRegistry:
        """Get the hook registry instance."""
        return self._hook_registry

    @property
    def extensions(self) -> ExtensionRegistry:
        """Get the extension registry instance."""
        return self._extensions

    @property
    def config(self) -> PluginConfig:
        """Get the plugin config instance."""
        return self._config

    @property
    def resolver(self) -> DependencyResolver:
        """Get the dependency resolver instance."""
        return self._resolver


# ============================================================================
# Utilities
# ============================================================================


class PluginLoader:
    """
    Utility class for loading plugins from various sources.
    """

    @staticmethod
    def load_from_file(path: Union[str, Path]) -> Optional[types.ModuleType]:
        """Load a module from a file path."""
        try:
            file_path = Path(path)
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.error(f"Failed to load module from {path}: {e}")
            return None

    @staticmethod
    def load_from_directory(path: Union[str, Path]) -> Optional[types.ModuleType]:
        """Load a module from a directory."""
        try:
            dir_path = Path(path)

            # Try __init__.py first
            init_file = dir_path / "__init__.py"
            if init_file.exists():
                return PluginLoader.load_from_file(init_file)

            # Try plugin.py
            plugin_file = dir_path / "plugin.py"
            if plugin_file.exists():
                return PluginLoader.load_from_file(plugin_file)

            return None
        except Exception as e:
            logger.error(f"Failed to load module from {path}: {e}")
            return None

    @staticmethod
    def load_from_module_name(name: str) -> Optional[types.ModuleType]:
        """Load a module by name."""
        try:
            return importlib.import_module(name)
        except ImportError as e:
            logger.error(f"Failed to import module {name}: {e}")
            return None


def create_plugin(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    plugin_class: Optional[Type[BasePlugin]] = None,
    **kwargs,
) -> Union[Type[BasePlugin], Callable[[Type[BasePlugin]], Type[BasePlugin]]]:
    """
    Create a plugin class or decorator with metadata.

    Can be used as:
        @create_plugin(name="my_plugin", version="1.0.0")
        class MyPlugin(BasePlugin):
            pass

    Or:
        MyPlugin = create_plugin(name="my_plugin", version="1.0.0", plugin_class=MyPluginBase)

    Args:
        name: Plugin name
        version: Plugin version
        description: Plugin description
        author: Plugin author
        plugin_class: Optional base class (if not using as decorator)
        **kwargs: Additional metadata fields

    Returns:
        Decorator function or plugin class
    """
    metadata = PluginMetadata(
        name=name, version=version, description=description, author=author, **kwargs
    )

    def decorator(cls: Type[BasePlugin]) -> Type[BasePlugin]:
        """Decorator to apply metadata to plugin class."""
        if not issubclass(cls, BasePlugin):
            raise PluginError(f"Plugin class must inherit from BasePlugin: {cls}")

        cls.set_metadata(metadata)

        # Store metadata on class for easy access
        cls.__plugin_metadata__ = metadata.to_dict()

        return cls

    if plugin_class is not None:
        return decorator(plugin_class)

    return decorator


def plugin_decorator(
    name: Optional[str] = None,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    dependencies: Optional[List[str]] = None,
    **kwargs,
) -> Callable[[Type[BasePlugin]], Type[BasePlugin]]:
    """
    Decorator for marking a class as a plugin.

    Usage:
        @plugin_decorator(name="my_plugin", version="1.0.0")
        class MyPlugin(BasePlugin):
            pass

    Args:
        name: Plugin name (defaults to class name)
        version: Plugin version
        description: Plugin description
        author: Plugin author
        dependencies: List of plugin dependencies
        **kwargs: Additional metadata fields

    Returns:
        Decorator function
    """

    def decorator(cls: Type[BasePlugin]) -> Type[BasePlugin]:
        plugin_name = name or cls.__name__

        metadata = PluginMetadata(
            name=plugin_name,
            version=version,
            description=description,
            author=author,
            dependencies=dependencies or [],
            **kwargs,
        )

        if not issubclass(cls, BasePlugin):
            raise PluginError(f"Plugin class must inherit from BasePlugin: {cls}")

        cls.set_metadata(metadata)
        cls.__plugin_metadata__ = metadata.to_dict()

        return cls

    return decorator


# Convenience aliases for exports
Manager = PluginManager
Interface = PluginInterface
Base = BasePlugin
Model = ModelPlugin
Data = DataPlugin
Trainer = TrainerPlugin
Discovery = PluginDiscovery
Hooks = HookSystem
Registry = HookRegistry
Point = ExtensionPoint
Config = PluginConfig
Resolve = DependencyResolver
Graph = DependencyGraph
Load = PluginLoader


# Export tuple for type hints
Tuple = tuple


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
    # Type exports
    "T",
    "Tuple",
]
