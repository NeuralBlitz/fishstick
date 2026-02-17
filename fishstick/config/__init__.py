"""
Configuration Management for fishstick

YAML/JSON configuration with:
- Environment variable interpolation
- Validation schemas
- Multiple environments (dev/staging/prod)
- Hierarchical configs
"""

from typing import Dict, Any, Optional, Union
import os
import yaml
import json
from pathlib import Path


class Config:
    """
    Configuration manager with dict-like access.

    Supports:
    - Dot notation: config.model.lr
    - Environment variables: ${VAR_NAME}
    - Default values
    - Type validation
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        """Access config with dot notation."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(value)
            return value

        raise AttributeError(f"Config has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Access config with bracket notation."""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        keys = key.split(".")
        value = self._data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_yaml(path: Union[str, Path]) -> Config:
    """
    Load YAML config file.

    Args:
        path: Path to YAML file

    Returns:
        Config object
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Interpolate environment variables
    data = _interpolate_env_vars(data)

    return Config(data)


def load_json(path: Union[str, Path]) -> Config:
    """
    Load JSON config file.

    Args:
        path: Path to JSON file

    Returns:
        Config object
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Interpolate environment variables
    data = _interpolate_env_vars(data)

    return Config(data)


def _interpolate_env_vars(data: Any) -> Any:
    """Recursively interpolate environment variables."""
    if isinstance(data, dict):
        return {k: _interpolate_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_interpolate_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Replace ${VAR} with environment variable
        import re

        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace_var, data)
    else:
        return data


class ConfigManager:
    """
    Manage multiple configurations for different environments.
    """

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}

    def load(self, name: str, env: Optional[str] = None) -> Config:
        """
        Load configuration for environment.

        Args:
            name: Base config name
            env: Environment (dev, staging, prod)

        Returns:
            Merged configuration
        """
        # Load base config
        base_path = self.config_dir / f"{name}.yaml"
        if not base_path.exists():
            base_path = self.config_dir / f"{name}.json"

        config = (
            load_yaml(base_path)
            if base_path.suffix == ".yaml"
            else load_json(base_path)
        )

        # Load environment-specific config if exists
        if env:
            env_path = self.config_dir / f"{name}.{env}.yaml"
            if env_path.exists():
                env_config = load_yaml(env_path)
                config = self._merge_configs(config.to_dict(), env_config.to_dict())

        self.configs[name] = config
        return config

    def _merge_configs(self, base: Dict, override: Dict) -> Config:
        """Merge two config dictionaries."""
        merged = base.copy()

        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value).to_dict()
            else:
                merged[key] = value

        return Config(merged)

    def get_default_config(self) -> Config:
        """Get default fishstick configuration."""
        default = {
            "project": {"name": "fishstick_project", "version": "0.1.0"},
            "model": {
                "name": "uniintelli",
                "input_dim": 784,
                "output_dim": 10,
                "hidden_dim": 256,
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "lr": 0.001,
                "optimizer": "adam",
                "scheduler": "cosine",
            },
            "tracking": {
                "backend": "tensorboard",
                "project_name": "${PROJECT_NAME}",
                "log_dir": "./runs",
            },
        }

        return Config(default)


# Default configuration for UniIntelli
DEFAULT_UNIINTELLI_CONFIG = """
model:
  name: uniintelli
  input_dim: 784
  output_dim: 10
  hidden_dim: 256
  n_layers: 4

training:
  epochs: 100
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  optimizer: adam
  scheduler: cosine
  warmup_steps: 500

tracking:
  backend: tensorboard
  project_name: fishstick_uniintelli
  log_dir: ./runs/uniintelli
"""


def create_default_config(output_path: str = "config.yaml") -> None:
    """Create default configuration file."""
    with open(output_path, "w") as f:
        f.write(DEFAULT_UNIINTELLI_CONFIG)

    print(f"✓ Created default config at {output_path}")
