"""
Training Visualization Tools
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TrainingVisualizer:
    """Visualize training metrics and progress."""

    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

    def update(self, metrics: Dict[str, float]) -> None:
        """Update history with new metrics."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def plot_loss(self, save_path: Optional[str] = None) -> None:
        """Plot training and validation loss."""
        plt.figure(figsize=(10, 6))

        if self.history["train_loss"]:
            plt.plot(self.history["train_loss"], label="Train Loss", linewidth=2)

        if self.history["val_loss"]:
            plt.plot(self.history["val_loss"], label="Val Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        else:
            plt.savefig(self.save_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_accuracy(self, save_path: Optional[str] = None) -> None:
        """Plot training and validation accuracy."""
        plt.figure(figsize=(10, 6))

        if self.history["train_acc"]:
            plt.plot(self.history["train_acc"], label="Train Accuracy", linewidth=2)

        if self.history["val_acc"]:
            plt.plot(self.history["val_acc"], label="Val Accuracy", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        else:
            plt.savefig(
                self.save_dir / "accuracy_curve.png", dpi=150, bbox_inches="tight"
            )
        plt.close()

    def plot_lr(self, save_path: Optional[str] = None) -> None:
        """Plot learning rate schedule."""
        plt.figure(figsize=(10, 6))

        if self.history["lr"]:
            plt.plot(self.history["lr"], linewidth=2, color="orange")

        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        else:
            plt.savefig(self.save_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_all(self) -> None:
        """Plot all available metrics."""
        if self.history["train_loss"] or self.history["val_loss"]:
            self.plot_loss()

        if self.history["train_acc"] or self.history["val_acc"]:
            self.plot_accuracy()

        if self.history["lr"]:
            self.plot_lr()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))

        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        else:
            plt.savefig(
                self.save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight"
            )
        plt.close()


class MetricsDashboard:
    """Interactive training dashboard."""

    def __init__(self):
        self.metrics = {}

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = {"steps": [], "values": []}

        self.metrics[name]["steps"].append(step)
        self.metrics[name]["values"].append(value)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        for name, data in self.metrics.items():
            if data["values"]:
                summary[name] = {
                    "min": min(data["values"]),
                    "max": max(data["values"]),
                    "mean": sum(data["values"]) / len(data["values"]),
                    "latest": data["values"][-1],
                }
        return summary
