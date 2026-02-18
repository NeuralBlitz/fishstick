"""
Experiment Configuration

Comprehensive experiment configuration management.

Classes:
- ExperimentConfig: Main experiment configuration
- DataConfig: Data configuration
- ModelConfig: Model configuration
- TrainingConfig: Training configuration
- Hyperparameters: Hyperparameter management
- ConfigValidator: Configuration validation
- ConfigLoader: Load configurations from files
- ConfigMerger: Merge multiple configurations
- SweepConfig: Hyperparameter sweep configuration
"""

from typing import Optional, Dict, List, Any, Union, TypeVar, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import yaml
import itertools
from copy import deepcopy
from datetime import datetime


T = TypeVar("T")


@dataclass
class DataConfig:
    """Configuration for data loading."""

    dataset_name: str = ""
    data_dir: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    transforms: Optional[Dict[str, Any]] = None
    augmentation: Optional[Dict[str, Any]] = None
    num_classes: int = 10
    image_size: int = 224
    channels: int = 3


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_name: str = ""
    model_class: str = ""
    pretrained: bool = False
    pretrained_path: Optional[str] = None
    num_classes: int = 10
    input_channels: int = 3
    input_size: int = 224
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    activation: str = "relu"
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    residual: bool = False
    attention: bool = False
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    momentum: float = 0.9
    nesterov: bool = False
    warmup_epochs: int = 0
    warmup_lr: float = 1e-5
    gradient_clip: Optional[float] = None
    mixed_precision: bool = False
    accumulation_steps: int = 1
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    checkpoint_frequency: int = 1
    log_frequency: int = 10
    eval_frequency: int = 1
    seed: int = 42
    device: str = "cuda"
    accelerator: str = "auto"
    precision: int = 32
    num_nodes: int = 1
    num_gpus: int = 1
    distributed_backend: str = "nccl"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_name: str = "experiment"
    project_name: str = "default_project"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(
        default_factory=lambda: {
            "log_dir": "./logs",
            "use_tensorboard": True,
            "use_wandb": False,
            "wandb_project": None,
            "log_gradients": False,
            "log_weights": False,
        }
    )
    reproducibility: Dict[str, Any] = field(
        default_factory=lambda: {
            "seed": 42,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
    )
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration as dictionary
        """
        config_dict = asdict(self)
        return config_dict

    def to_json(self, path: Optional[Path] = None) -> str:
        """Convert config to JSON.

        Args:
            path: Optional path to save JSON

        Returns:
            JSON string
        """
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, default=str)

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(json_str)

        return json_str

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Convert config to YAML.

        Args:
            path: Optional path to save YAML

        Returns:
            YAML string
        """
        config_dict = self.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(yaml_str)

        return yaml_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ExperimentConfig instance
        """
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            experiment_name=config_dict.get("experiment_name", "experiment"),
            project_name=config_dict.get("project_name", "default_project"),
            description=config_dict.get("description", ""),
            tags=config_dict.get("tags", []),
            data=data_config,
            model=model_config,
            training=training_config,
            hyperparameters=config_dict.get("hyperparameters", {}),
            logging=config_dict.get("logging", {}),
            reproducibility=config_dict.get("reproducibility", {}),
            notes=config_dict.get("notes", ""),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load config from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            ExperimentConfig instance
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            ExperimentConfig instance
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load config from file (auto-detect format).

        Args:
            path: Path to config file

        Returns:
            ExperimentConfig instance
        """
        path = Path(path)
        if path.suffix == ".json":
            return cls.from_json(path)
        elif path.suffix in [".yaml", ".yml"]:
            return cls.from_yaml(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    def copy(self) -> "ExperimentConfig":
        """Create a deep copy of the config.

        Returns:
            Copy of config
        """
        return deepcopy(self)

    def update(self, updates: Dict[str, Any]) -> "ExperimentConfig":
        """Update config with new values.

        Args:
            updates: Dictionary of updates

        Returns:
            Updated config
        """
        config = self.copy()
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif "." in key:
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if hasattr(self, key):
            return getattr(self, key)

        if "." in key:
            parts = key.split(".")
            obj = self
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return default
            return obj

        return default


class ConfigValidator:
    """Validate experiment configurations."""

    @staticmethod
    def validate(config: ExperimentConfig) -> List[str]:
        """Validate configuration and return list of errors.

        Args:
            config: Configuration to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not config.experiment_name:
            errors.append("Experiment name is required")

        if config.training.epochs <= 0:
            errors.append("Epochs must be positive")

        if config.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")

        if config.data.batch_size <= 0:
            errors.append("Batch size must be positive")

        if (
            config.data.train_split + config.data.val_split + config.data.test_split
            != 1.0
        ):
            errors.append("Data splits must sum to 1.0")

        if config.data.num_workers < 0:
            errors.append("Number of workers must be non-negative")

        if config.model.dropout < 0 or config.model.dropout > 1:
            errors.append("Dropout must be between 0 and 1")

        if (
            config.training.gradient_clip is not None
            and config.training.gradient_clip <= 0
        ):
            errors.append("Gradient clip must be positive if specified")

        return errors

    @staticmethod
    def validate_or_raise(config: ExperimentConfig) -> None:
        """Validate config and raise ValueError if invalid.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        errors = ConfigValidator.validate(config)
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")


class ConfigMerger:
    """Merge multiple configurations."""

    @staticmethod
    def merge(
        base_config: ExperimentConfig,
        override_config: ExperimentConfig,
        strategy: str = "override",
    ) -> ExperimentConfig:
        """Merge two configurations.

        Args:
            base_config: Base configuration
            override_config: Override configuration
            strategy: Merge strategy ('override', 'extend', 'replace')

        Returns:
            Merged configuration
        """
        base_dict = base_config.to_dict()
        override_dict = override_config.to_dict()

        merged = ConfigMerger._merge_dict(base_dict, override_dict, strategy)

        return ExperimentConfig.from_dict(merged)

    @staticmethod
    def _merge_dict(
        base: Dict[str, Any],
        override: Dict[str, Any],
        strategy: str,
    ) -> Dict[str, Any]:
        """Recursively merge dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary
            strategy: Merge strategy

        Returns:
            Merged dictionary
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key not in result:
                result[key] = deepcopy(value)
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = ConfigMerger._merge_dict(result[key], value, strategy)
            elif strategy == "replace":
                result[key] = deepcopy(value)
            elif (
                strategy == "extend"
                and isinstance(value, list)
                and isinstance(result[key], list)
            ):
                result[key] = result[key] + value
            else:
                result[key] = deepcopy(value)

        return result


class SweepConfig:
    """Hyperparameter sweep configuration."""

    def __init__(
        self,
        parameters: Dict[str, List[Any]],
        method: str = "grid",
        metric_name: str = "val_loss",
        metric_mode: str = "min",
        goal: str = "minimize",
    ):
        """Initialize sweep configuration.

        Args:
            parameters: Dictionary of parameter names to value lists
            method: Sweep method ('grid', 'random', 'bayesian')
            metric_name: Metric to optimize
            metric_mode: 'min' or 'max'
            goal: Optimization goal
        """
        self.parameters = parameters
        self.method = method
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.goal = goal
        self._parameter_combinations: Optional[List[Dict[str, Any]]] = None

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations.

        Returns:
            List of parameter dictionaries
        """
        if self._parameter_combinations is not None:
            return self._parameter_combinations

        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        combinations = list(itertools.product(*param_values))

        self._parameter_combinations = [
            dict(zip(param_names, combo)) for combo in combinations
        ]

        return self._parameter_combinations

    def sample_random(self, n_samples: int) -> List[Dict[str, Any]]:
        """Sample random parameter combinations.

        Args:
            n_samples: Number of samples

        Returns:
            List of sampled parameter dictionaries
        """
        import random

        combinations = self.generate_combinations()
        return random.sample(combinations, min(n_samples, len(combinations)))

    def to_config(self, params: Dict[str, Any]) -> ExperimentConfig:
        """Create config from sweep parameters.

        Args:
            params: Parameter dictionary

        Returns:
            ExperimentConfig with applied parameters
        """
        config = ExperimentConfig()
        config.hyperparameters = params
        return config

    def __len__(self) -> int:
        """Get number of parameter combinations."""
        return len(self.generate_combinations())

    def __iter__(self):
        """Iterate over parameter combinations."""
        for params in self.generate_combinations():
            yield self.to_config(params)


class Hyperparameters:
    """Hyperparameter management utilities."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize hyperparameters.

        Args:
            params: Parameter dictionary
        """
        self.params = params or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get hyperparameter value.

        Args:
            key: Parameter key
            default: Default value

        Returns:
            Parameter value
        """
        return self.params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set hyperparameter value.

        Args:
            key: Parameter key
            value: Parameter value
        """
        self.params[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update hyperparameters.

        Args:
            updates: Dictionary of updates
        """
        self.params.update(updates)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Parameter dictionary
        """
        return self.params.copy()

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Hyperparameters":
        """Create from experiment config.

        Args:
            config: Experiment configuration

        Returns:
            Hyperparameters instance
        """
        params = {
            "learning_rate": config.training.learning_rate,
            "batch_size": config.data.batch_size,
            "epochs": config.training.epochs,
            "weight_decay": config.training.weight_decay,
            "dropout": config.model.dropout,
            **config.hyperparameters,
        }
        return cls(params)


class ConfigLoader:
    """Load configurations from various sources."""

    @staticmethod
    def load(path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from file.

        Args:
            path: Path to configuration file

        Returns:
            ExperimentConfig instance
        """
        return ExperimentConfig.from_file(path)

    @staticmethod
    def load_multiple(paths: List[Union[str, Path]]) -> List[ExperimentConfig]:
        """Load multiple configurations.

        Args:
            paths: List of configuration file paths

        Returns:
            List of ExperimentConfig instances
        """
        return [ConfigLoader.load(path) for path in paths]

    @staticmethod
    def load_directory(
        directory: Union[str, Path], pattern: str = "*.yaml"
    ) -> List[ExperimentConfig]:
        """Load all configs from directory.

        Args:
            directory: Directory containing configs
            pattern: File pattern to match

        Returns:
            List of ExperimentConfig instances
        """
        directory = Path(directory)
        config_files = list(directory.glob(pattern))
        return [ConfigLoader.load(f) for f in config_files]

    @staticmethod
    def create_default() -> ExperimentConfig:
        """Create default configuration.

        Returns:
            Default ExperimentConfig
        """
        return ExperimentConfig(
            experiment_name="default_experiment",
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
        )
