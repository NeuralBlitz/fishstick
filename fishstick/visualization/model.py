"""
Model Visualization Tools
"""

from typing import Optional, Dict, List
import torch
from torch import nn
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


class ModelVisualizer:
    """Visualize model architecture and structure."""

    def __init__(self, model: nn.Module):
        self.model = model

    def summary(self, input_shape: tuple) -> Dict:
        """Generate model summary similar to Keras."""
        summary_dict = {
            "layers": [],
            "total_params": 0,
            "trainable_params": 0,
            "non_trainable_params": 0,
        }

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                num_params = sum(p.numel() for p in module.parameters())
                trainable = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )

                summary_dict["layers"].append(
                    {
                        "name": name,
                        "type": module.__class__.__name__,
                        "params": num_params,
                        "trainable": trainable,
                    }
                )

                summary_dict["total_params"] += num_params
                summary_dict["trainable_params"] += trainable
                summary_dict["non_trainable_params"] += num_params - trainable

        return summary_dict

    def print_summary(self, input_shape: tuple) -> None:
        """Print formatted model summary."""
        summary = self.summary(input_shape)

        print("=" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
        print("=" * 80)

        for layer in summary["layers"]:
            print(f"{layer['name']:<30} {'?':<25} {layer['params']:<15,}")

        print("=" * 80)
        print(f"Total params: {summary['total_params']:,}")
        print(f"Trainable params: {summary['trainable_params']:,}")
        print(f"Non-trainable params: {summary['non_trainable_params']:,}")
        print("=" * 80)

    def visualize_architecture(self, save_path: str = "model_arch.png") -> None:
        """Visualize model architecture as a graph."""
        G = nx.DiGraph()

        # Add nodes for each layer
        for name, module in self.model.named_modules():
            if name:  # Skip root
                G.add_node(name, type=module.__class__.__name__)

        # Add edges based on forward pass (simplified)
        prev_node = None
        for name, module in self.model.named_modules():
            if name:
                if prev_node:
                    G.add_edge(prev_node, name)
                prev_node = name

        # Draw
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=2)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=2000,
            font_size=8,
            arrows=True,
        )

        plt.title("Model Architecture")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Architecture visualization saved to {save_path}")

    def plot_weight_distribution(self, save_path: str = "weight_dist.png") -> None:
        """Plot distribution of weights."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and idx < 6:
                axes[idx].hist(param.data.cpu().numpy().flatten(), bins=50)
                axes[idx].set_title(f"{name}")
                axes[idx].set_xlabel("Value")
                axes[idx].set_ylabel("Frequency")
                idx += 1

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Weight distribution saved to {save_path}")

    def plot_gradient_flow(self, save_path: str = "gradient_flow.png") -> None:
        """Plot gradient flow through layers."""
        ave_grads = []
        layers = []

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())

        plt.figure(figsize=(12, 8))
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="r")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("Average Gradient")
        plt.title("Gradient Flow")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gradient flow saved to {save_path}")

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def model_size_mb(self) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
