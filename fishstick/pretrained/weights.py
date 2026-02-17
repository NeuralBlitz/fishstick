"""
Pretrained Weights

Utilities for handling pretrained model weights.
"""

from typing import Dict, Optional, Any
import torch
from pathlib import Path


class Weights:
    """Container for pretrained model weights."""

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.state_dict = state_dict
        self.metadata = metadata or {}

    def save(self, path: str) -> None:
        """Save weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "Weights":
        """Load weights from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        return cls(
            state_dict=checkpoint["state_dict"],
            metadata=checkpoint.get("metadata", {}),
        )

    def apply_to_model(self, model: torch.nn.Module, strict: bool = True) -> None:
        """Apply weights to a model."""
        model.load_state_dict(self.state_dict, strict=strict)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    @staticmethod
    def from_url(url: str, map_location: str = "cpu") -> "Weights":
        """Download and load weights from URL."""
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=map_location)
        return cls(state_dict=state_dict)


class WeightInitializer:
    """Initialize model weights with pretrained weights."""

    @staticmethod
    def kaiming_normal(model: torch.nn.Module, nonlinearity: str = "relu") -> None:
        """Kaiming normal initialization."""
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    @staticmethod
    def xavier_uniform(model: torch.nn.Module) -> None:
        """Xavier uniform initialization."""
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    @staticmethod
    def orthogonal(model: torch.nn.Module) -> None:
        """Orthogonal initialization."""
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    @staticmethod
    def nalu(model: torch.nn.Module) -> None:
        """NALU-inspired initialization."""
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class WeightAdapter:
    """Adapt pretrained weights for different model architectures."""

    @staticmethod
    def adapt_keys(state_dict: Dict, key_mapping: Dict[str, str]) -> Dict:
        """Adapt state dict keys according to mapping."""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key_mapping.get(key, key)
            new_state_dict[new_key] = value
        return new_state_dict

    @staticmethod
    def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
        """Remove prefix from state dict keys."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict

    @staticmethod
    def add_prefix(state_dict: Dict, prefix: str) -> Dict:
        """Add prefix to state dict keys."""
        return {f"{prefix}{key}": value for key, value in state_dict.items()}

    @staticmethod
    def filter_keys(state_dict: Dict, keep_keys: list) -> Dict:
        """Keep only specified keys."""
        return {k: v for k, v in state_dict.items() if k in keep_keys}


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict:
    """Load a checkpoint and optionally restore optimizer state."""
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    result = {"model": model}

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        result["optimizer"] = optimizer

    if "epoch" in checkpoint:
        result["epoch"] = checkpoint["epoch"]

    if "metrics" in checkpoint:
        result["metrics"] = checkpoint["metrics"]

    return result


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
) -> None:
    """Save a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)
