"""
🥫 fishstick Sauce - The Secret Recipe

Pre-configured, ready-to-train models for all 6 frameworks.
Just like fishsticks need sauce, your AI needs these models!

Includes different sizes (small, base, large) and task-specific variants.
"""

from typing import Optional, Dict, Any, List
import torch
from torch import nn


class SauceBottle:
    """
    🥫 The Sauce Bottle - Your secret ingredient collection

    Pre-configured fishstick models ready for training.
    Just pick your flavor and go!
    """

    # Model configurations for different sizes
    CONFIGS = {
        # UniIntelli configurations
        "uniintelli_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 2,
            "description": "UniIntelli-Small: 500K params, fast training",
        },
        "uniintelli_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UniIntelli-Base: 1.8M params, balanced performance",
        },
        "uniintelli_large": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 512,
            "n_layers": 6,
            "description": "UniIntelli-Large: 7M params, best accuracy",
        },
        # HSCA configurations
        "hsca_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "HSCA-Small: 2M params, energy-conserving",
        },
        "hsca_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "HSCA-Base: 6.5M params, symplectic dynamics",
        },
        # UIA configurations
        "uia_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "UIA-Small: 800K params, categorical-Hamiltonian",
        },
        "uia_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UIA-Base: 1.7M params, CHNP+RG",
        },
        # SCIF configurations
        "scif_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "SCIF-Small: 1.5M params, fiber bundles",
        },
        "scif_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "SCIF-Base: 3.8M params, symplectic-categorical",
        },
        # UIF configurations
        "uif_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 64,
            "n_layers": 2,
            "description": "UIF-Small: 150K params, 4-layer stack",
        },
        "uif_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 4,
            "description": "UIF-Base: 367K params, Category→Geo→Dyn→Verify",
        },
        # UIS configurations
        "uis_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "UIS-Small: 400K params, quantum-inspired",
        },
        "uis_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UIS-Base: 861K params, complete synthesis",
        },
    }

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all available models with descriptions."""
        return {name: config["description"] for name, config in cls.CONFIGS.items()}

    @classmethod
    def create_model(cls, model_name: str, **overrides) -> nn.Module:
        """
        Create a model from the zoo.

        Args:
            model_name: Name of the model (e.g., 'uniintelli_base')
            **overrides: Override any config parameters

        Returns:
            Instantiated model ready for training
        """
        if model_name not in cls.CONFIGS:
            raise ValueError(
                f"Model {model_name} not found. Available: {list(cls.CONFIGS.keys())}"
            )

        config = cls.CONFIGS[model_name].copy()
        config.pop("description")  # Remove non-model parameter
        config.update(overrides)

        # Determine which framework to use
        if "uniintelli" in model_name:
            from fishstick.frameworks.uniintelli import create_uniintelli

            return create_uniintelli(**config)
        elif "hsca" in model_name:
            from fishstick.frameworks.hsca import create_hsca

            return create_hsca(**config)
        elif "uia" in model_name:
            from fishstick.frameworks.uia import create_uia

            return create_uia(**config)
        elif "scif" in model_name:
            from fishstick.frameworks.scif import create_scif

            return create_scif(**config)
        elif "uif" in model_name:
            from fishstick.frameworks.uif import create_uif

            return create_uif(**config)
        elif "uis" in model_name:
            from fishstick.frameworks.uis import create_uis

            return create_uis(**config)
        else:
            raise ValueError(f"Unknown model type: {model_name}")


class TaskModels:
    """
    Task-specific model presets.
    """

    @staticmethod
    def image_classifier(
        num_classes: int = 10, framework: str = "uniintelli", size: str = "base"
    ) -> nn.Module:
        """
        Create image classification model.

        Args:
            num_classes: Number of output classes
            framework: Which framework to use
            size: 'small', 'base', or 'large'

        Returns:
            Model configured for image classification
        """
        model_name = f"{framework}_{size}"
        model = SauceBottle.create_model(model_name, output_dim=num_classes)
        return model

    @staticmethod
    def text_classifier(
        num_classes: int = 2,
        vocab_size: int = 10000,
        framework: str = "hsca",
        size: str = "base",
    ) -> nn.Module:
        """
        Create text classification model.

        Args:
            num_classes: Number of classes
            vocab_size: Size of vocabulary
            framework: Which framework to use
            size: Model size

        Returns:
            Model configured for text classification
        """
        model_name = f"{framework}_{size}"
        model = SauceBottle.create_model(
            model_name, input_dim=vocab_size, output_dim=num_classes
        )
        return model

    @staticmethod
    def regression_model(
        output_dim: int = 1,
        input_dim: int = 100,
        framework: str = "uia",
        size: str = "base",
    ) -> nn.Module:
        """
        Create regression model.

        Args:
            output_dim: Number of output values
            input_dim: Input feature dimension
            framework: Which framework to use
            size: Model size

        Returns:
            Model configured for regression
        """
        model_name = f"{framework}_{size}"
        model = SauceBottle.create_model(
            model_name, input_dim=input_dim, output_dim=output_dim
        )
        return model

    @staticmethod
    def time_series_forecaster(
        forecast_horizon: int = 24,
        input_dim: int = 10,
        framework: str = "uis",
        size: str = "base",
    ) -> nn.Module:
        """
        Create time series forecasting model.

        Args:
            forecast_horizon: How many steps to predict
            input_dim: Number of input features
            framework: Which framework to use
            size: Model size

        Returns:
            Model configured for time series
        """
        model_name = f"{framework}_{size}"
        model = SauceBottle.create_model(
            model_name, input_dim=input_dim, output_dim=forecast_horizon
        )
        return model


class TransferLearning:
    """
    Transfer learning utilities for fishstick models.
    """

    @staticmethod
    def load_pretrained(
        model_name: str, checkpoint_path: Optional[str] = None
    ) -> nn.Module:
        """
        Load a pretrained model.

        Args:
            model_name: Model architecture name
            checkpoint_path: Path to checkpoint (optional)

        Returns:
            Model with loaded weights
        """
        model = SauceBottle.create_model(model_name)

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Loaded pretrained weights from {checkpoint_path}")

        return model

    @staticmethod
    def freeze_layers(model: nn.Module, num_layers: int = 2) -> nn.Module:
        """
        Freeze first N layers for transfer learning.

        Args:
            model: Model to modify
            num_layers: Number of layers to freeze

        Returns:
            Model with frozen layers
        """
        layers_frozen = 0
        for name, param in model.named_parameters():
            if layers_frozen < num_layers:
                param.requires_grad = False
                layers_frozen += 1
            else:
                break

        print(f"✓ Froze {layers_frozen} layers")
        return model

    @staticmethod
    def replace_head(model: nn.Module, new_output_dim: int) -> nn.Module:
        """
        Replace the final layer for transfer learning.

        Args:
            model: Model to modify
            new_output_dim: New output dimension

        Returns:
            Model with new head
        """
        # Find and replace final layer
        # This is framework-specific
        if hasattr(model, "decoder"):
            # UniIntelli, HSCA, etc.
            old_linear = model.decoder
            model.decoder = nn.Linear(old_linear.in_features, new_output_dim)
        elif hasattr(model, "head"):
            # Vision Transformer style
            old_linear = model.head
            model.head = nn.Linear(old_linear.in_features, new_output_dim)

        print(f"✓ Replaced head with output dim {new_output_dim}")
        return model


class ModelBuilder:
    """
    Fluent API for building custom models.
    """

    def __init__(self):
        self.config = {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
        }
        self.framework = "uniintelli"

    def with_framework(self, framework: str) -> "ModelBuilder":
        """Set framework."""
        self.framework = framework
        return self

    def with_input_dim(self, dim: int) -> "ModelBuilder":
        """Set input dimension."""
        self.config["input_dim"] = dim
        return self

    def with_output_dim(self, dim: int) -> "ModelBuilder":
        """Set output dimension."""
        self.config["output_dim"] = dim
        return self

    def with_hidden_dim(self, dim: int) -> "ModelBuilder":
        """Set hidden dimension."""
        self.config["hidden_dim"] = dim
        return self

    def with_layers(self, n_layers: int) -> "ModelBuilder":
        """Set number of layers."""
        self.config["n_layers"] = n_layers
        return self

    def build(self) -> nn.Module:
        """Build the model."""
        return SauceBottle.create_model(f"{self.framework}_base", **self.config)


# Convenience functions
def list_available_models() -> Dict[str, str]:
    """List all available models."""
    return SauceBottle.list_models()


def create_model(model_name: str, **kwargs) -> nn.Module:
    """Create a model from the zoo."""
    return SauceBottle.create_model(model_name, **kwargs)


def build_model() -> ModelBuilder:
    """Get a model builder for fluent API."""
    return ModelBuilder()


# Example usage documentation
"""
Usage Examples:

# 🥫 fishstick Sauce - How to use

# 1. Simple model creation
from fishstick.sauce import create_model

model = create_model('uniintelli_base')
# or
model = create_model('hsca_large', output_dim=100)

# 2. Task-specific models
from fishstick.sauce import TaskModels

# Image classification
model = TaskModels.image_classifier(num_classes=1000, framework='uniintelli')

# Text classification
model = TaskModels.text_classifier(num_classes=2, vocab_size=10000)

# Regression
model = TaskModels.regression_model(output_dim=1, input_dim=50)

# 3. List all available sauce bottles
from fishstick.sauce import list_available_models

models = list_available_models()
for name, desc in models.items():
    print(f"🥫 {name}: {desc}")

# 4. Fluent builder API
from fishstick.sauce import build_model

model = (build_model()
    .with_framework('hsca')
    .with_input_dim(100)
    .with_output_dim(10)
    .with_hidden_dim(512)
    .with_layers(6)
    .build())

# 5. Transfer learning
from fishstick.sauce import TransferLearning

# Load pretrained
model = TransferLearning.load_pretrained('uniintelli_base', 'checkpoint.pt')

# Freeze layers
model = TransferLearning.freeze_layers(model, num_layers=3)

# Replace head for new task
model = TransferLearning.replace_head(model, new_output_dim=100)
"""
