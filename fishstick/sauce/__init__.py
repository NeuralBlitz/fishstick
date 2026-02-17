"""
🥫 fishstick Sauce - The Secret Recipe

Pre-configured, ready-to-train models for all 6 frameworks.
Just like fishsticks need sauce, your AI needs these models!

Includes different sizes (small, base, large) and task-specific variants.
"""

from typing import Optional, Dict, Any, List
import os
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
        "crls_g_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "CRLS-G-Small: 600K params, categorical renormalization",
        },
        "crls_g_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "CRLS-G-Base: 1.5M params, categorical RG flows",
        },
        "toposformer_h_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "ToposFormer-H-Small: 700K params, sheaf integration",
        },
        "toposformer_h_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "ToposFormer-H-Base: 1.8M params, topos-theoretic",
        },
        "uif_i_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 64,
            "n_layers": 2,
            "description": "UIF-I-Small: 200K params, renormalized attention",
        },
        "uif_i_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 4,
            "description": "UIF-I-Base: 500K params, RAM architecture",
        },
        "uis_j_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "UIS-J-Small: 550K params, node-at-attention",
        },
        "uis_j_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UIS-J-Base: 1.4M params, NAA mechanism",
        },
        "uia_k_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "UIA-K-Small: 750K params, sheaf-LSTM",
        },
        "uia_k_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UIA-K-Base: 1.9M params, fiber bundle attention",
        },
        "crls_l_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "CRLS-L-Small: 800K params, mathematical physics",
        },
        "crls_l_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "CRLS-L-Base: 2.0M params, MIP framework",
        },
        "uia_m_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "UIA-M-Small: 650K params, neural flow",
        },
        "uia_m_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UIA-M-Base: 1.6M params, symplectic dynamics",
        },
        "uis_n_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "UIS-N-Small: 700K params, cross-synthetic",
        },
        "uis_n_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UIS-N-Base: 1.8M params, CS-NAA mechanism",
        },
        "uia_o_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "UIA-O-Small: 600K params, sheaf-theoretic",
        },
        "uia_o_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UIA-O-Base: 1.5M params, STNN architecture",
        },
        "uif_p_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 64,
            "n_layers": 2,
            "description": "UIF-P-Small: 250K params, RG-informed",
        },
        "uif_p_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 4,
            "description": "UIF-P-Base: 620K params, hierarchical networks",
        },
        "uinet_q_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "UINet-Q-Small: 550K params, categorical quantum",
        },
        "uinet_q_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "UINet-Q-Base: 1.4M params, CQNA architecture",
        },
        "uif_r_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 64,
            "n_layers": 2,
            "description": "UIF-R-Small: 300K params, Fisher natural gradient",
        },
        "uif_r_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 4,
            "description": "UIF-R-Base: 750K params, comprehensive blueprint",
        },
        "usif_s_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "USIF-S-Small: 500K params, quantum categorical",
        },
        "usif_s_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "USIF-S-Base: 1.2M params, QCNN architecture",
        },
        "uif_t_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 64,
            "n_layers": 2,
            "description": "UIF-T-Small: 280K params, Hamiltonian-RG",
        },
        "uif_t_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 4,
            "description": "UIF-T-Base: 700K params, flow optimizer",
        },
        "usif_u_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "USIF-U-Small: 450K params, thermodynamic bounds",
        },
        "usif_u_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "USIF-U-Base: 1.1M params, info bounds framework",
        },
        "uif_v_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 64,
            "n_layers": 2,
            "description": "UIF-V-Small: 220K params, info-theoretic",
        },
        "uif_v_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 4,
            "description": "UIF-V-Base: 550K params, dynamics framework",
        },
        "mca_w_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "MCA-W-Small: 650K params, meta-cognitive",
        },
        "mca_w_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "MCA-W-Base: 1.6M params, architecture framework",
        },
        "ttsik_x_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "TTSIK-X-Small: 720K params, topos-theoretic symplectic",
        },
        "ttsik_x_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "TTSIK-X-Base: 1.8M params, intelligence kernel",
        },
        "ctna_y_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "CTNA-Y-Small: 600K params, categorical-thermodynamic",
        },
        "ctna_y_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "CTNA-Y-Base: 1.5M params, neural architecture",
        },
        "scif_z_small": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 128,
            "n_layers": 3,
            "description": "SCIF-Z-Small: 700K params, symplectic-categorical",
        },
        "scif_z_base": {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 256,
            "n_layers": 4,
            "description": "SCIF-Z-Base: 1.7M params, intelligence framework",
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

        # Handle n_layers for models that don't support it
        n_layers = config.pop("n_layers", None)
        config.update(overrides)

        # Determine which framework to use
        # Check more specific patterns first (G-Z frameworks) before general ones (A-F)
        if model_name.startswith("crls_g"):
            from fishstick.frameworks.crls import create_crls

            return create_crls(**config)
        elif model_name.startswith("toposformer_h"):
            from fishstick.frameworks.toposformer import create_toposformer

            return create_toposformer(**config)
        elif model_name.startswith("uif_i"):
            from fishstick.frameworks.uif_i import create_uif_i

            return create_uif_i(**config)
        elif model_name.startswith("uis_j"):
            from fishstick.frameworks.uis_j import create_uis_j

            return create_uis_j(**config)
        elif model_name.startswith("uia_k"):
            from fishstick.frameworks.uia_k import create_uia_k

            return create_uia_k(**config)
        elif model_name.startswith("crls_l"):
            from fishstick.frameworks.crls_l import create_crls_l

            return create_crls_l(**config)
        elif model_name.startswith("uia_m"):
            from fishstick.frameworks.uia_m import create_uia_m

            return create_uia_m(**config)
        elif model_name.startswith("uis_n"):
            from fishstick.frameworks.uis_n import create_uis_n

            return create_uis_n(**config)
        elif model_name.startswith("uia_o"):
            from fishstick.frameworks.uia_o import create_uia_o

            return create_uia_o(**config)
        elif model_name.startswith("uif_p"):
            from fishstick.frameworks.uif_p import create_uif_p

            return create_uif_p(**config)
        elif model_name.startswith("uinet_q"):
            from fishstick.frameworks.uinet_q import create_uinet_q

            return create_uinet_q(**config)
        elif model_name.startswith("uif_r"):
            from fishstick.frameworks.uif_r import create_uif_r

            return create_uif_r(**config)
        elif model_name.startswith("usif_s"):
            from fishstick.frameworks.usif_s import create_usif_s

            return create_usif_s(**config)
        elif model_name.startswith("uif_t"):
            from fishstick.frameworks.uif_t import create_uif_t

            return create_uif_t(**config)
        elif model_name.startswith("usif_u"):
            from fishstick.frameworks.usif_u import create_usif_u

            return create_usif_u(**config)
        elif model_name.startswith("uif_v"):
            from fishstick.frameworks.uif_v import create_uif_v

            return create_uif_v(**config)
        elif model_name.startswith("mca_w"):
            from fishstick.frameworks.mca_w import create_mca_w

            return create_mca_w(**config)
        elif model_name.startswith("ttsik_x"):
            from fishstick.frameworks.ttsik_x import create_ttsik_x

            return create_ttsik_x(**config)
        elif model_name.startswith("ctna_y"):
            from fishstick.frameworks.ctna_y import create_ctna_y

            return create_ctna_y(**config)
        elif model_name.startswith("scif_z"):
            from fishstick.frameworks.scif_z import create_scif_z

            return create_scif_z(**config)
        # Now check general A-F frameworks
        elif "uniintelli" in model_name:
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
