"""
Interactive Training Visualization Dashboard for fishstick

A comprehensive dashboard for real-time visualization of training metrics,
layer activations, attention maps, and predictions.

Supports both matplotlib (static) and plotly (interactive) backends.
Can be used in Jupyter notebooks, standalone scripts, or as a web server.
"""

from typing import Optional, Dict, List, Callable, Any, Union, Tuple
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod
import threading
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class RealTimePlot:
    """
    Real-time plotting utility with non-blocking updates.

    Supports both matplotlib and plotly backends, works in Jupyter notebooks
    and standalone scripts.

    Example:
        >>> plot = RealTimePlot(backend='plotly')
        >>> plot.add_line('loss', x=[1, 2, 3], y=[0.5, 0.3, 0.2])
        >>> plot.update()  # Non-blocking update
        >>> plot.save('training.html')
    """

    def __init__(
        self,
        backend: str = "matplotlib",
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Training Visualization",
    ):
        """
        Initialize real-time plot.

        Args:
            backend: 'matplotlib' or 'plotly'
            figsize: Figure size (width, height)
            title: Plot title
        """
        self.backend = backend
        self.figsize = figsize
        self.title = title
        self.data = defaultdict(lambda: {"x": [], "y": [], "type": "line", "style": {}})
        self._is_jupyter = self._check_jupyter()
        self._fig = None

        if backend == "matplotlib":
            self._init_matplotlib()
        elif backend == "plotly":
            self._init_plotly()
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'matplotlib' or 'plotly'"
            )

    def _check_jupyter(self) -> bool:
        """Check if running in Jupyter notebook."""
        try:
            from IPython import get_ipython

            if get_ipython() is None:
                return False
            return "IPKernelApp" in get_ipython().config
        except:
            return False

    def _init_matplotlib(self):
        """Initialize matplotlib backend."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self.plt = plt
        self._fig, self._ax = plt.subplots(figsize=self.figsize)
        self._ax.set_title(self.title)
        self._lines = {}
        self._initialized = False
        plt.ion()  # Interactive mode

    def _init_plotly(self):
        """Initialize plotly backend."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            self.go = go
            self._fig = make_subplots(rows=1, cols=1)
            self._fig.update_layout(title=self.title)

            if self._is_jupyter:
                from IPython.display import display

                self._display_handle = display(self._fig, display_id=True)
        except ImportError:
            raise ImportError(
                "plotly is required for plotly backend. Install with: pip install plotly"
            )

    def add_line(
        self, name: str, x: Optional[List] = None, y: Optional[List] = None, **style
    ):
        """
        Add or update a line series.

        Args:
            name: Series name
            x: X coordinates
            y: Y coordinates
            **style: Line style (color, linewidth, etc.)
        """
        if x is not None:
            self.data[name]["x"] = list(x)
        if y is not None:
            self.data[name]["y"] = list(y)
        self.data[name]["type"] = "line"
        self.data[name]["style"] = style

    def add_scatter(
        self, name: str, x: Optional[List] = None, y: Optional[List] = None, **style
    ):
        """Add or update a scatter series."""
        if x is not None:
            self.data[name]["x"] = list(x)
        if y is not None:
            self.data[name]["y"] = list(y)
        self.data[name]["type"] = "scatter"
        self.data[name]["style"] = style

    def append(self, name: str, x: float, y: float):
        """Append a point to a series."""
        self.data[name]["x"].append(x)
        self.data[name]["y"].append(y)

    def update(self):
        """Update the plot (non-blocking)."""
        if self.backend == "matplotlib":
            self._update_matplotlib()
        else:
            self._update_plotly()

    def _update_matplotlib(self):
        """Update matplotlib plot."""
        if not self._initialized:
            self._ax.clear()
            self._ax.set_title(self.title)

        for name, series in self.data.items():
            if len(series["x"]) > 0 and len(series["y"]) > 0:
                if series["type"] == "line":
                    self._ax.plot(
                        series["x"], series["y"], label=name, **series["style"]
                    )
                else:
                    self._ax.scatter(
                        series["x"], series["y"], label=name, **series["style"]
                    )

        if len(self.data) > 0:
            self._ax.legend()
            self._ax.grid(True, alpha=0.3)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self._initialized = True

    def _update_plotly(self):
        """Update plotly plot."""
        self._fig.data = []

        for name, series in self.data.items():
            if len(series["x"]) > 0 and len(series["y"]) > 0:
                if series["type"] == "line":
                    self._fig.add_trace(
                        self.go.Scatter(
                            x=series["x"],
                            y=series["y"],
                            mode="lines",
                            name=name,
                            **series["style"],
                        )
                    )
                else:
                    self._fig.add_trace(
                        self.go.Scatter(
                            x=series["x"],
                            y=series["y"],
                            mode="markers",
                            name=name,
                            **series["style"],
                        )
                    )

        if self._is_jupyter and hasattr(self, "_display_handle"):
            self._display_handle.update(self._fig)

    def clear(self):
        """Clear all data."""
        self.data.clear()

    def save(self, filepath: str):
        """Save plot to file."""
        if self.backend == "matplotlib":
            self._fig.savefig(filepath, dpi=150, bbox_inches="tight")
        else:
            self._fig.write_html(filepath)

    def show(self):
        """Show the plot (blocking for matplotlib)."""
        if self.backend == "matplotlib":
            self.plt.show()
        else:
            self._fig.show()


class LayerVisualizer:
    """
    Visualize layer activations, weights, gradients, and feature maps.

    Hooks into model layers to capture intermediate outputs during forward/backward passes.

    Example:
        >>> model = MyModel()
        >>> visualizer = LayerVisualizer(model)
        >>> visualizer.register_hooks()
        >>> output = model(input_data)
        >>> visualizer.plot_activations('conv1')  # Visualize conv1 activations
        >>> visualizer.plot_weight_distribution()  # Plot weight histograms
    """

    def __init__(self, model: nn.Module, save_dir: str = "visualizations/layers"):
        """
        Initialize layer visualizer.

        Args:
            model: PyTorch model to visualize
            save_dir: Directory to save visualizations
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.activations = {}
        self.gradients = {}
        self.hooks = []
        self._hook_handles = []

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """
        Register forward and backward hooks on specified layers.

        Args:
            layer_names: List of layer names to hook (None = all layers)
        """
        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                if len(list(module.children())) == 0:  # Leaf modules only
                    handle = module.register_forward_hook(self._make_forward_hook(name))
                    self._hook_handles.append(handle)

                    handle = module.register_backward_hook(
                        self._make_backward_hook(name)
                    )
                    self._hook_handles.append(handle)

    def _make_forward_hook(self, name: str):
        """Create forward hook for layer."""

        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook

    def _make_backward_hook(self, name: str):
        """Create backward hook for layer."""

        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients[name] = grad_output[0].detach()

        return hook

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def plot_activations(
        self,
        layer_name: str,
        max_channels: int = 16,
        save_path: Optional[str] = None,
        interactive: bool = False,
    ):
        """
        Plot activation heatmaps for a layer.

        Args:
            layer_name: Name of the layer
            max_channels: Maximum number of channels to plot
            save_path: Path to save plot (None = auto-generate)
            interactive: Use plotly for interactive plot
        """
        if layer_name not in self.activations:
            print(f"No activations captured for {layer_name}")
            return

        activation = self.activations[layer_name]

        # Handle different tensor shapes
        if activation.ndim == 2:  # Linear layer (batch, features)
            self._plot_1d_activations(activation, layer_name, save_path, interactive)
        elif (
            activation.ndim == 3
        ):  # (batch, seq_len, features) or (batch, channels, seq_len)
            self._plot_2d_activations(
                activation, layer_name, max_channels, save_path, interactive
            )
        elif activation.ndim == 4:  # Conv layer (batch, channels, height, width)
            self._plot_conv_activations(
                activation, layer_name, max_channels, save_path, interactive
            )
        else:
            print(f"Unsupported activation shape: {activation.shape}")

    def _plot_1d_activations(
        self,
        activation: torch.Tensor,
        layer_name: str,
        save_path: Optional[str],
        interactive: bool,
    ):
        """Plot 1D activations (e.g., from linear layers)."""
        if interactive:
            import plotly.express as px

            act_np = activation.cpu().numpy()
            fig = px.imshow(act_np, aspect="auto", title=f"Activations: {layer_name}")
            if save_path:
                fig.write_html(self.save_dir / save_path)
            else:
                fig.show()
        else:
            import matplotlib.pyplot as plt

            act_np = activation.cpu().numpy()

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(act_np, aspect="auto", cmap="viridis")
            ax.set_title(f"Activations: {layer_name}")
            ax.set_xlabel("Features")
            ax.set_ylabel("Batch")
            plt.colorbar(im, ax=ax)

            if save_path is None:
                save_path = f"{layer_name}_activations.png"
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
            plt.close()

    def _plot_2d_activations(
        self,
        activation: torch.Tensor,
        layer_name: str,
        max_channels: int,
        save_path: Optional[str],
        interactive: bool,
    ):
        """Plot 2D activations."""
        import matplotlib.pyplot as plt

        act_np = activation[0].cpu().numpy()  # Take first sample

        if act_np.ndim == 2:
            # (seq_len, features)
            n_rows = min(max_channels, act_np.shape[0])
        else:
            n_rows = min(max_channels, act_np.shape[0])

        n_cols = min(8, (n_rows + 7) // 8)
        n_rows = (n_rows + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        for i in range(min(n_rows * n_cols, act_np.shape[0])):
            row, col = i // n_cols, i % n_cols
            if act_np.ndim == 2:
                axes[row, col].plot(act_np[i])
                axes[row, col].set_title(f"Seq {i}")
            else:
                im = axes[row, col].imshow(act_np[i], cmap="viridis")
                axes[row, col].set_title(f"Ch {i}")
                axes[row, col].axis("off")

        # Hide unused subplots
        for i in range(min(n_rows * n_cols, act_np.shape[0]), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis("off")

        plt.suptitle(f"Activations: {layer_name}")
        plt.tight_layout()

        if save_path is None:
            save_path = f"{layer_name}_activations.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_conv_activations(
        self,
        activation: torch.Tensor,
        layer_name: str,
        max_channels: int,
        save_path: Optional[str],
        interactive: bool,
    ):
        """Plot convolutional layer feature maps."""
        import matplotlib.pyplot as plt

        act_np = activation[0].cpu().numpy()  # Take first sample
        n_channels = min(max_channels, act_np.shape[0])

        n_cols = min(8, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.array(axes).reshape(n_rows, n_cols)

        for i in range(n_channels):
            row, col = i // n_cols, i % n_cols
            im = axes[row, col].imshow(act_np[i], cmap="viridis")
            axes[row, col].set_title(f"Ch {i}")
            axes[row, col].axis("off")

        for i in range(n_channels, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis("off")

        plt.suptitle(f"Feature Maps: {layer_name}")
        plt.tight_layout()

        if save_path is None:
            save_path = f"{layer_name}_feature_maps.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_weight_distribution(
        self,
        max_layers: int = 12,
        save_path: Optional[str] = None,
        interactive: bool = False,
    ):
        """
        Plot weight distribution histograms for each layer.

        Args:
            max_layers: Maximum number of layers to plot
            save_path: Path to save plot
            interactive: Use plotly for interactive plot
        """
        if interactive:
            self._plot_weight_distribution_plotly(max_layers, save_path)
        else:
            self._plot_weight_distribution_mpl(max_layers, save_path)

    def _plot_weight_distribution_mpl(self, max_layers: int, save_path: Optional[str]):
        """Plot weight distribution with matplotlib."""
        import matplotlib.pyplot as plt

        n_rows = (max_layers + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3 * n_rows))
        axes = axes.flatten()

        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and "weight" in name and idx < max_layers:
                weights = param.data.cpu().numpy().flatten()
                axes[idx].hist(weights, bins=50, alpha=0.7, edgecolor="black")
                axes[idx].set_title(f"{name[:30]}..." if len(name) > 30 else name)
                axes[idx].set_xlabel("Weight Value")
                axes[idx].set_ylabel("Frequency")
                axes[idx].grid(True, alpha=0.3)
                idx += 1

        for i in range(idx, len(axes)):
            axes[i].axis("off")

        plt.suptitle("Weight Distributions by Layer")
        plt.tight_layout()

        if save_path is None:
            save_path = "weight_distributions.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_weight_distribution_plotly(
        self, max_layers: int, save_path: Optional[str]
    ):
        """Plot weight distribution with plotly."""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        n_rows = (max_layers + 2) // 3
        fig = make_subplots(rows=n_rows, cols=3, subplot_titles=[])

        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and "weight" in name and idx < max_layers:
                weights = param.data.cpu().numpy().flatten()
                row, col = idx // 3 + 1, idx % 3 + 1
                fig.add_trace(
                    go.Histogram(x=weights, name=name[:30], nbinsx=50), row=row, col=col
                )
                idx += 1

        fig.update_layout(
            title="Weight Distributions by Layer", showlegend=False, height=300 * n_rows
        )

        if save_path is None:
            save_path = "weight_distributions.html"
        fig.write_html(self.save_dir / save_path)

    def plot_gradient_flow(
        self, save_path: Optional[str] = None, interactive: bool = False
    ):
        """
        Plot gradient flow through layers.

        Args:
            save_path: Path to save plot
            interactive: Use plotly for interactive plot
        """
        avg_grads = []
        layer_names = []

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                layer_names.append(name)
                avg_grads.append(param.grad.abs().mean().item())

        if not avg_grads:
            print("No gradients available. Run backward pass first.")
            return

        if interactive:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(avg_grads))),
                    y=avg_grads,
                    mode="lines+markers",
                    name="Gradient Flow",
                )
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title="Gradient Flow Through Layers",
                xaxis_title="Layer",
                yaxis_title="Average Gradient",
                xaxis_ticktext=layer_names,
                xaxis_tickvals=list(range(len(layer_names))),
                xaxis_tickangle=45,
                height=600,
            )

            if save_path is None:
                save_path = "gradient_flow.html"
            fig.write_html(self.save_dir / save_path)
        else:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(14, 6))
            plt.plot(avg_grads, alpha=0.6, color="b", linewidth=2, marker="o")
            plt.hlines(0, 0, len(avg_grads), linewidth=1, color="r", linestyle="--")
            plt.xticks(
                range(len(layer_names)), layer_names, rotation="vertical", fontsize=8
            )
            plt.xlim(xmin=0, xmax=len(avg_grads) - 1)
            plt.xlabel("Layers")
            plt.ylabel("Average Gradient")
            plt.title("Gradient Flow")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path is None:
                save_path = "gradient_flow.png"
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
            plt.close()

    def plot_feature_map_grid(
        self, layer_name: str, input_data: torch.Tensor, save_path: Optional[str] = None
    ):
        """
        Visualize feature maps as a grid overlay on input.

        Args:
            layer_name: Layer to visualize
            input_data: Input image (CHW or HWC format)
            save_path: Path to save plot
        """
        if layer_name not in self.activations:
            print(f"No activations for {layer_name}. Run forward pass first.")
            return

        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        activation = self.activations[layer_name][0]  # First sample

        # Handle input
        if input_data.ndim == 3:
            if input_data.shape[0] in [1, 3]:
                input_np = input_data.cpu().permute(1, 2, 0).numpy()
            else:
                input_np = input_data.cpu().numpy()
        else:
            input_np = input_data.cpu().numpy()

        # Normalize input
        input_np = (input_np - input_np.min()) / (
            input_np.max() - input_np.min() + 1e-8
        )

        # Create grid of feature maps
        n_channels = min(16, activation.shape[0])
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

        for i in range(n_channels):
            ax = plt.subplot(n_rows, n_cols, i + 1)

            # Upsample activation to match input size
            from torch.nn.functional import interpolate

            act = activation[i].unsqueeze(0).unsqueeze(0)
            act_upsampled = (
                interpolate(
                    act, size=input_np.shape[:2], mode="bilinear", align_corners=False
                )[0, 0]
                .cpu()
                .numpy()
            )

            # Normalize activation
            act_upsampled = (act_upsampled - act_upsampled.min()) / (
                act_upsampled.max() - act_upsampled.min() + 1e-8
            )

            # Overlay
            ax.imshow(input_np)
            ax.imshow(act_upsampled, cmap="jet", alpha=0.5)
            ax.set_title(f"Channel {i}")
            ax.axis("off")

        plt.suptitle(f"Feature Map Overlays: {layer_name}")
        plt.tight_layout()

        if save_path is None:
            save_path = f"{layer_name}_overlays.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()


class AttentionVisualizer:
    """
    Visualize attention mechanisms in transformer models.

    Supports attention heatmaps, rollout visualization, and multi-head comparisons.

    Example:
        >>> model = TransformerModel()
        >>> attn_vis = AttentionVisualizer(model)
        >>> output = model(input_ids)
        >>> attn_vis.plot_attention_heatmap(layer_idx=0, head_idx=0)
        >>> attn_vis.plot_attention_rollout()
    """

    def __init__(self, model: nn.Module, save_dir: str = "visualizations/attention"):
        """
        Initialize attention visualizer.

        Args:
            model: Transformer model
            save_dir: Directory to save visualizations
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.attention_weights = {}
        self.tokens = None
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture attention weights."""

        def make_hook(name):
            def hook(module, input, output):
                # Try to extract attention weights from different output formats
                if isinstance(output, tuple):
                    attn_weights = output[1] if len(output) > 1 else None
                else:
                    attn_weights = getattr(output, "attentions", None)

                if attn_weights is not None:
                    self.attention_weights[name] = attn_weights.detach()

            return hook

        # Hook into attention modules
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                handle = module.register_forward_hook(make_hook(name))

    def set_tokens(self, tokens: List[str]):
        """Set token labels for visualization."""
        self.tokens = tokens

    def plot_attention_heatmap(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None,
        interactive: bool = False,
    ):
        """
        Plot attention weights as a heatmap.

        Args:
            attention_weights: Attention tensor (layers, heads, seq_len, seq_len)
            layer_idx: Layer index to visualize
            head_idx: Head index (None = average over heads)
            save_path: Path to save plot
            interactive: Use plotly for interactive plot
        """
        if attention_weights is None:
            # Try to get from hooks
            if not self.attention_weights:
                print("No attention weights captured. Run forward pass first.")
                return
            # Use first available
            attention_weights = list(self.attention_weights.values())[0]

        if attention_weights.ndim == 4:  # (batch, heads, seq, seq)
            attn = attention_weights[0]  # First batch
        else:
            attn = attention_weights

        if layer_idx < attn.shape[0]:
            attn = attn[layer_idx]

        if head_idx is not None:
            attn = attn[head_idx]
        else:
            attn = attn.mean(dim=0)  # Average over heads

        attn_np = attn.cpu().numpy()

        if interactive:
            self._plot_attention_plotly(attn_np, save_path)
        else:
            self._plot_attention_mpl(attn_np, save_path)

    def _plot_attention_mpl(self, attn_np: np.ndarray, save_path: Optional[str]):
        """Plot attention heatmap with matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(attn_np, cmap="viridis", aspect="auto")

        if self.tokens and len(self.tokens) == attn_np.shape[0]:
            ax.set_xticks(range(len(self.tokens)))
            ax.set_yticks(range(len(self.tokens)))
            ax.set_xticklabels(self.tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(self.tokens, fontsize=8)

        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title("Attention Heatmap")

        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()

        if save_path is None:
            save_path = "attention_heatmap.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_attention_plotly(self, attn_np: np.ndarray, save_path: Optional[str]):
        """Plot interactive attention heatmap with plotly."""
        import plotly.express as px

        if self.tokens and len(self.tokens) == attn_np.shape[0]:
            fig = px.imshow(
                attn_np,
                x=self.tokens,
                y=self.tokens,
                color_continuous_scale="Viridis",
                title="Attention Heatmap",
            )
        else:
            fig = px.imshow(
                attn_np, color_continuous_scale="Viridis", title="Attention Heatmap"
            )

        fig.update_layout(
            xaxis_title="Key Position", yaxis_title="Query Position", height=700
        )

        if save_path is None:
            save_path = "attention_heatmap.html"
        fig.write_html(self.save_dir / save_path)

    def plot_attention_rollout(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        discard_ratio: float = 0.9,
        save_path: Optional[str] = None,
        interactive: bool = False,
    ):
        """
        Compute and plot attention rollout.

        Attention rollout combines attention across layers to show the full
        attention path from input to output.

        Args:
            attention_weights: Attention tensor (layers, heads, seq_len, seq_len)
            discard_ratio: Ratio of low attention weights to discard
            save_path: Path to save plot
            interactive: Use plotly for interactive plot
        """
        if attention_weights is None:
            if not self.attention_weights:
                print("No attention weights captured. Run forward pass first.")
                return
            attention_weights = list(self.attention_weights.values())[0]

        if attention_weights.ndim == 4:
            attention_weights = attention_weights[0]  # First batch

        # Average over heads and apply residual connections
        attn_avg = attention_weights.mean(dim=1)  # (layers, seq, seq)
        n_layers, seq_len, _ = attn_avg.shape

        # Add residual connections
        residual_att = torch.eye(seq_len, device=attn_avg.device).unsqueeze(0)
        attn_avg = attn_avg + residual_att
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)

        # Compute rollout (matrix multiplication across layers)
        rollout = attn_avg[0]
        for i in range(1, n_layers):
            rollout = torch.matmul(attn_avg[i], rollout)

        # Discard low attention weights
        if discard_ratio > 0:
            flat = rollout.view(-1)
            threshold = torch.quantile(flat, discard_ratio)
            rollout = rollout * (rollout > threshold)

        rollout_np = rollout.cpu().numpy()

        if interactive:
            self._plot_attention_plotly(rollout_np, save_path)
        else:
            self._plot_attention_mpl(rollout_np, save_path)

    def plot_multi_head_comparison(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int = 0,
        max_heads: int = 8,
        save_path: Optional[str] = None,
    ):
        """
        Plot attention patterns for multiple heads in a grid.

        Args:
            attention_weights: Attention tensor (batch, layers, heads, seq, seq)
            layer_idx: Layer index
            max_heads: Maximum number of heads to display
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt

        if attention_weights.ndim == 5:
            attn = attention_weights[0, layer_idx]  # (heads, seq, seq)
        else:
            attn = attention_weights[layer_idx]

        n_heads = min(max_heads, attn.shape[0])
        n_cols = 4
        n_rows = (n_heads + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_heads):
            row, col = i // n_cols, i % n_cols
            im = axes[row, col].imshow(
                attn[i].cpu().numpy(), cmap="viridis", aspect="auto"
            )
            axes[row, col].set_title(f"Head {i}")
            axes[row, col].set_xlabel("Key")
            axes[row, col].set_ylabel("Query")

        for i in range(n_heads, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis("off")

        plt.suptitle(f"Multi-Head Attention (Layer {layer_idx})")
        plt.tight_layout()

        if save_path is None:
            save_path = f"multi_head_layer{layer_idx}.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def visualize_bert_attention(
        self, input_ids: torch.Tensor, tokenizer: Any, save_path: Optional[str] = None
    ):
        """
        Specialized visualization for BERT-like models.

        Args:
            input_ids: Input token IDs
            tokenizer: HuggingFace tokenizer
            save_path: Path to save plot
        """
        # Forward pass to get attention
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            attentions = outputs.attentions  # Tuple of tensors

        # Stack into single tensor
        attn_tensor = torch.stack(
            [a[0] for a in attentions]
        )  # (layers, heads, seq, seq)

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        self.set_tokens(tokens)

        # Plot all layers
        import matplotlib.pyplot as plt

        n_layers = attn_tensor.shape[0]
        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        axes = axes.flatten()

        for layer_idx in range(n_layers):
            # Average over heads
            avg_attn = attn_tensor[layer_idx].mean(dim=0).cpu().numpy()

            im = axes[layer_idx].imshow(avg_attn, cmap="viridis", aspect="auto")
            axes[layer_idx].set_title(f"Layer {layer_idx}")

            if layer_idx >= n_layers - n_cols:  # Only label bottom row
                axes[layer_idx].set_xticks(range(len(tokens)))
                axes[layer_idx].set_xticklabels(
                    tokens, rotation=90, ha="right", fontsize=6
                )
            else:
                axes[layer_idx].set_xticks([])

            if layer_idx % n_cols == 0:  # Only label left column
                axes[layer_idx].set_yticks(range(len(tokens)))
                axes[layer_idx].set_yticklabels(tokens, fontsize=6)
            else:
                axes[layer_idx].set_yticks([])

        for i in range(n_layers, len(axes)):
            axes[i].axis("off")

        plt.suptitle("BERT Attention Across All Layers")
        plt.tight_layout()

        if save_path is None:
            save_path = "bert_attention_all_layers.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()


class PredictionVisualizer:
    """
    Visualize model predictions including confusion matrices,
    ROC curves, PR curves, and misclassification analysis.

    Example:
        >>> pred_vis = PredictionVisualizer(class_names=['cat', 'dog', 'bird'])
        >>> pred_vis.update(preds, targets)
        >>> pred_vis.plot_confusion_matrix()
        >>> pred_vis.plot_roc_curve()
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        save_dir: str = "visualizations/predictions",
    ):
        """
        Initialize prediction visualizer.

        Args:
            class_names: List of class names
            num_classes: Number of classes (inferred if not provided)
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names
        self.num_classes = num_classes

        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.misclassified_data = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
    ):
        """
        Update with new predictions.

        Args:
            predictions: Predicted class indices
            targets: Ground truth class indices
            probabilities: Prediction probabilities (for ROC/PR curves)
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())

    def plot_confusion_matrix(
        self,
        normalize: bool = False,
        save_path: Optional[str] = None,
        interactive: bool = False,
    ):
        """
        Plot confusion matrix.

        Args:
            normalize: Normalize by row (true class)
            save_path: Path to save plot
            interactive: Use plotly for interactive plot
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(self.targets, self.predictions)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

        # Infer num_classes if not provided
        if self.num_classes is None:
            self.num_classes = cm.shape[0]

        # Create class names if not provided
        if self.class_names is None:
            self.class_names = [f"Class {i}" for i in range(self.num_classes)]

        if interactive:
            self._plot_confusion_matrix_plotly(cm, normalize, save_path)
        else:
            self._plot_confusion_matrix_mpl(cm, normalize, save_path)

    def _plot_confusion_matrix_mpl(
        self, cm: np.ndarray, normalize: bool, save_path: Optional[str]
    ):
        """Plot confusion matrix with matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()

        if save_path is None:
            save_path = f"confusion_matrix{'_norm' if normalize else ''}.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_confusion_matrix_plotly(
        self, cm: np.ndarray, normalize: bool, save_path: Optional[str]
    ):
        """Plot interactive confusion matrix with plotly."""
        import plotly.express as px

        fig = px.imshow(
            cm,
            x=self.class_names,
            y=self.class_names,
            color_continuous_scale="Blues",
            aspect="auto",
            title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
        )

        fig.update_layout(
            xaxis_title="Predicted Label", yaxis_title="True Label", height=700
        )

        if save_path is None:
            save_path = f"confusion_matrix{'_norm' if normalize else ''}.html"
        fig.write_html(self.save_dir / save_path)

    def plot_roc_curve(
        self, save_path: Optional[str] = None, interactive: bool = False
    ):
        """
        Plot ROC curves for each class.

        Args:
            save_path: Path to save plot
            interactive: Use plotly for interactive plot
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        if not self.probabilities:
            print("No probabilities available. Pass probabilities to update().")
            return

        probs = np.array(self.probabilities)
        targets = np.array(self.targets)

        # Binarize labels
        n_classes = probs.shape[1]
        targets_bin = label_binarize(targets, classes=range(n_classes))

        if interactive:
            import plotly.graph_objects as go

            fig = go.Figure()

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)

                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        name=f"{class_name} (AUC = {roc_auc:.2f})",
                        mode="lines",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    name="Random",
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                )
            )

            fig.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=600,
                hovermode="closest",
            )

            if save_path is None:
                save_path = "roc_curves.html"
            fig.write_html(self.save_dir / save_path)
        else:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)

                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path is None:
                save_path = "roc_curves.png"
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
            plt.close()

    def plot_pr_curve(self, save_path: Optional[str] = None, interactive: bool = False):
        """
        Plot Precision-Recall curves for each class.

        Args:
            save_path: Path to save plot
            interactive: Use plotly for interactive plot
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize

        if not self.probabilities:
            print("No probabilities available. Pass probabilities to update().")
            return

        probs = np.array(self.probabilities)
        targets = np.array(self.targets)

        n_classes = probs.shape[1]
        targets_bin = label_binarize(targets, classes=range(n_classes))

        if interactive:
            import plotly.graph_objects as go

            fig = go.Figure()

            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(
                    targets_bin[:, i], probs[:, i]
                )
                avg_precision = average_precision_score(targets_bin[:, i], probs[:, i])

                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        name=f"{class_name} (AP = {avg_precision:.2f})",
                        mode="lines",
                    )
                )

            fig.update_layout(
                title="Precision-Recall Curves",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=600,
                hovermode="closest",
            )

            if save_path is None:
                save_path = "pr_curves.html"
            fig.write_html(self.save_dir / save_path)
        else:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))

            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(
                    targets_bin[:, i], probs[:, i]
                )
                avg_precision = average_precision_score(targets_bin[:, i], probs[:, i])

                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                plt.plot(
                    recall, precision, label=f"{class_name} (AP = {avg_precision:.2f})"
                )

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curves")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path is None:
                save_path = "pr_curves.png"
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
            plt.close()

    def analyze_misclassifications(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        device: str = "cuda",
        max_samples: int = 16,
        save_path: Optional[str] = None,
    ):
        """
        Analyze and visualize misclassified samples with image overlays.

        Args:
            data_loader: Data loader
            model: Model for predictions
            device: Device to use
            max_samples: Maximum number of samples to show
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt

        model.eval()
        misclassified = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                # Find misclassified
                mask = preds != labels
                if mask.any():
                    misclassified.extend(
                        [
                            {
                                "image": images[i].cpu(),
                                "true": labels[i].item(),
                                "pred": preds[i].item(),
                                "conf": outputs[i].softmax(dim=0).max().item(),
                            }
                            for i in range(len(mask))
                            if mask[i]
                        ]
                    )

                if len(misclassified) >= max_samples:
                    break

        misclassified = misclassified[:max_samples]
        n_samples = len(misclassified)

        if n_samples == 0:
            print("No misclassifications found!")
            return

        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, sample in enumerate(misclassified):
            img = sample["image"]

            # Handle different channel configurations
            if img.ndim == 3:
                if img.shape[0] in [1, 3]:
                    img = img.permute(1, 2, 0).numpy()
                else:
                    img = img.numpy()

            # Normalize for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            if img.shape[2] == 1:
                img = img.squeeze()
                axes[idx].imshow(img, cmap="gray")
            else:
                axes[idx].imshow(img)

            true_label = (
                self.class_names[sample["true"]]
                if self.class_names
                else f"Class {sample['true']}"
            )
            pred_label = (
                self.class_names[sample["pred"]]
                if self.class_names
                else f"Class {sample['pred']}"
            )

            axes[idx].set_title(
                f"True: {true_label}\nPred: {pred_label}\nConf: {sample['conf']:.2%}",
                fontsize=8,
            )
            axes[idx].axis("off")

        for idx in range(n_samples, len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Misclassification Analysis")
        plt.tight_layout()

        if save_path is None:
            save_path = "misclassifications.png"
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Found {len(misclassified)} misclassifications")


class TrainingDashboard:
    """
    Main training dashboard combining all visualizers with real-time updates.

    Integrates with the fishstick Trainer class via callbacks.

    Example:
        >>> dashboard = TrainingDashboard(real_time=True, backend='plotly')
        >>> trainer = Trainer(model, optimizer, loss_fn, callbacks=[dashboard])
        >>> trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        save_dir: str = "visualizations/dashboard",
        real_time: bool = True,
        backend: str = "matplotlib",
        update_interval: int = 1,
        metrics_to_plot: Optional[List[str]] = None,
        interactive: bool = False,
    ):
        """
        Initialize training dashboard.

        Args:
            save_dir: Directory to save visualizations
            real_time: Enable real-time plotting
            backend: 'matplotlib' or 'plotly'
            update_interval: Update plot every N epochs
            metrics_to_plot: List of metrics to plot (None = all)
            interactive: Use interactive plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.real_time = real_time
        self.backend = backend
        self.update_interval = update_interval
        self.metrics_to_plot = metrics_to_plot
        self.interactive = interactive

        self.history = defaultdict(list)
        self.plots = {}
        self._lock = threading.Lock()

        if real_time:
            self._init_plots()

    def _init_plots(self):
        """Initialize real-time plots."""
        self.plots["loss"] = RealTimePlot(
            backend=self.backend, figsize=(12, 6), title="Loss"
        )
        self.plots["accuracy"] = RealTimePlot(
            backend=self.backend, figsize=(12, 6), title="Accuracy"
        )
        self.plots["learning_rate"] = RealTimePlot(
            backend=self.backend, figsize=(10, 5), title="Learning Rate"
        )

    def update(self, epoch: int, logs: Dict[str, float]):
        """
        Update dashboard with new metrics.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metric values
        """
        with self._lock:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.history[key].append(value)

            # Update real-time plots
            if self.real_time and epoch % self.update_interval == 0:
                self._update_plots(epoch)

    def _update_plots(self, epoch: int):
        """Update all real-time plots."""
        # Loss plot
        if "loss" in self.history:
            self.plots["loss"].add_line(
                "train_loss",
                list(range(len(self.history["loss"]))),
                self.history["loss"],
                color="blue",
            )

        if "val_loss" in self.history:
            self.plots["loss"].add_line(
                "val_loss",
                list(range(len(self.history["val_loss"]))),
                self.history["val_loss"],
                color="orange",
            )

        self.plots["loss"].update()

        # Accuracy plot
        if "accuracy" in self.history:
            self.plots["accuracy"].add_line(
                "train_accuracy",
                list(range(len(self.history["accuracy"]))),
                self.history["accuracy"],
                color="green",
            )

        if "val_accuracy" in self.history:
            self.plots["accuracy"].add_line(
                "val_accuracy",
                list(range(len(self.history["val_accuracy"]))),
                self.history["val_accuracy"],
                color="red",
            )

        self.plots["accuracy"].update()

        # Learning rate plot
        if "lr" in self.history:
            self.plots["learning_rate"].add_line(
                "learning_rate",
                list(range(len(self.history["lr"]))),
                self.history["lr"],
                color="purple",
            )
            self.plots["learning_rate"].update()

    def plot_all_metrics(self, save_path: Optional[str] = None):
        """
        Plot all metrics in a comprehensive dashboard.

        Args:
            save_path: Path to save combined plot
        """
        import matplotlib.pyplot as plt

        # Determine grid size
        n_plots = len(self.history)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (metric, values) in enumerate(self.history.items()):
            if len(values) > 0:
                axes[idx].plot(values, linewidth=2)
                axes[idx].set_title(metric.replace("_", " ").title())
                axes[idx].set_xlabel("Epoch")
                axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(self.history), len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Training Metrics Dashboard")
        plt.tight_layout()

        if save_path is None:
            save_path = self.save_dir / "all_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_history(self, filepath: Optional[str] = None):
        """
        Save training history to JSON.

        Args:
            filepath: Path to save JSON file
        """
        if filepath is None:
            filepath = self.save_dir / "training_history.json"

        with open(filepath, "w") as f:
            json.dump(dict(self.history), f, indent=2)

    def load_history(self, filepath: str):
        """
        Load training history from JSON.

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, "r") as f:
            data = json.load(f)
            self.history = defaultdict(list, data)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.

        Returns:
            Dictionary with min, max, mean, and last values for each metric
        """
        summary = {}
        for metric, values in self.history.items():
            if values:
                summary[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "last": values[-1],
                    "best_epoch": values.index(min(values))
                    if "loss" in metric
                    else values.index(max(values)),
                }
        return summary

    def print_summary(self):
        """Print formatted summary of training."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        for metric, stats in summary.items():
            print(f"\n{metric.upper()}:")
            print(
                f"  Best:     {stats['min']:.4f}"
                if "loss" in metric
                else f"  Best:     {stats['max']:.4f}"
            )
            print(f"  Final:    {stats['last']:.4f}")
            print(f"  Mean:     {stats['mean']:.4f}")
            print(f"  Epoch:    {stats['best_epoch'] + 1}")

        print("=" * 60)


class DashboardCallback:
    """
    Callback to integrate TrainingDashboard with fishstick Trainer.

    Example:
        >>> from fishstick.training.advanced import Trainer, Callback
        >>>
        >>> dashboard = TrainingDashboard(real_time=True)
        >>> callback = DashboardCallback(dashboard)
        >>>
        >>> trainer = Trainer(model, optimizer, loss_fn, callbacks=[callback])
        >>> trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(self, dashboard: TrainingDashboard, save_on_epoch: bool = True):
        """
        Initialize dashboard callback.

        Args:
            dashboard: TrainingDashboard instance
            save_on_epoch: Save plots after each epoch
        """
        self.dashboard = dashboard
        self.save_on_epoch = save_on_epoch

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict):
        """Called at end of each epoch."""
        self.dashboard.update(epoch, logs)

        if self.save_on_epoch:
            self.dashboard.plot_all_metrics()
            self.dashboard.save_history()

    def on_train_end(self, trainer: "Trainer"):
        """Called at end of training."""
        self.dashboard.plot_all_metrics()
        self.dashboard.save_history()
        self.dashboard.print_summary()


class DashboardServer:
    """
    Web-based dashboard server using Flask with WebSocket support.

    Provides real-time training visualization through a web browser.

    Example:
        >>> server = DashboardServer(port=5000)
        >>> server.start()
        >>>
        >>> # During training
        >>> server.update_metrics(epoch=5, loss=0.3, accuracy=0.85)
        >>>
        >>> server.stop()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        save_dir: str = "visualizations/server",
    ):
        """
        Initialize dashboard server.

        Args:
            host: Host address
            port: Port number
            save_dir: Directory to save static files
        """
        self.host = host
        self.port = port
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.app = None
        self.socketio = None
        self.server_thread = None
        self._running = False

        self._init_server()

    def _init_server(self):
        """Initialize Flask server."""
        try:
            from flask import Flask, render_template_string, jsonify
            from flask_socketio import SocketIO, emit
        except ImportError:
            raise ImportError(
                "Flask and Flask-SocketIO required. Install with: pip install flask flask-socketio"
            )

        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "fishstick-dashboard-secret"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""
        from flask import render_template_string

        @self.app.route("/")
        def index():
            return render_template_string(self._get_html_template())

        @self.app.route("/api/metrics")
        def get_metrics():
            from flask import jsonify

            return jsonify(self.metrics_history)

        @self.socketio.on("connect")
        def handle_connect():
            print(f"Client connected from {self.socketio.server.eio.sockets}")

    def _get_html_template(self) -> str:
        """Get HTML template for dashboard."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Fishstick Training Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: #2196F3;
            color: white;
            padding: 20px;
            margin: -20px -20px 20px -20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.85em;
        }
        .status.active {
            background: #4CAF50;
            color: white;
        }
        .status.inactive {
            background: #f44336;
            color: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐟 Fishstick Training Dashboard</h1>
        <span id="status" class="status inactive">Disconnected</span>
    </div>
    
    <div class="grid">
        <div class="card">
            <div class="metric-label">Current Epoch</div>
            <div class="metric-value" id="epoch">-</div>
        </div>
        <div class="card">
            <div class="metric-label">Loss</div>
            <div class="metric-value" id="loss">-</div>
        </div>
        <div class="card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value" id="accuracy">-</div>
        </div>
        <div class="card">
            <div class="metric-label">Learning Rate</div>
            <div class="metric-value" id="lr">-</div>
        </div>
    </div>
    
    <div class="grid" style="margin-top: 20px;">
        <div class="card">
            <h3>Loss Over Time</h3>
            <div id="loss-plot" style="height: 400px;"></div>
        </div>
        <div class="card">
            <h3>Accuracy Over Time</h3>
            <div id="accuracy-plot" style="height: 400px;"></div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let lossData = {x: [], y: [], mode: 'lines', name: 'Loss'};
        let valLossData = {x: [], y: [], mode: 'lines', name: 'Val Loss'};
        let accData = {x: [], y: [], mode: 'lines', name: 'Accuracy'};
        let valAccData = {x: [], y: [], mode: 'lines', name: 'Val Accuracy'};
        
        socket.on('connect', function() {
            document.getElementById('status').textContent = 'Connected';
            document.getElementById('status').className = 'status active';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').className = 'status inactive';
        });
        
        socket.on('metrics', function(data) {
            document.getElementById('epoch').textContent = data.epoch || '-';
            document.getElementById('loss').textContent = data.loss ? data.loss.toFixed(4) : '-';
            document.getElementById('accuracy').textContent = data.accuracy ? (data.accuracy * 100).toFixed(1) + '%' : '-';
            document.getElementById('lr').textContent = data.lr ? data.lr.toExponential(2) : '-';
            
            // Update plots
            if (data.epoch !== undefined) {
                if (data.loss !== undefined) {
                    lossData.x.push(data.epoch);
                    lossData.y.push(data.loss);
                }
                if (data.val_loss !== undefined) {
                    valLossData.x.push(data.epoch);
                    valLossData.y.push(data.val_loss);
                }
                if (data.accuracy !== undefined) {
                    accData.x.push(data.epoch);
                    accData.y.push(data.accuracy);
                }
                if (data.val_accuracy !== undefined) {
                    valAccData.x.push(data.epoch);
                    valAccData.y.push(data.val_accuracy);
                }
                
                Plotly.newPlot('loss-plot', [lossData, valLossData], {
                    margin: {t: 10},
                    xaxis: {title: 'Epoch'},
                    yaxis: {title: 'Loss'}
                });
                
                Plotly.newPlot('accuracy-plot', [accData, valAccData], {
                    margin: {t: 10},
                    xaxis: {title: 'Epoch'},
                    yaxis: {title: 'Accuracy', tickformat: ',.0%'}
                });
            }
        });
    </script>
</body>
</html>
        """

    def start(self, blocking: bool = False):
        """
        Start the dashboard server.

        Args:
            blocking: If True, block until server stops
        """
        if self._running:
            print("Server already running")
            return

        self._running = True
        self.metrics_history = []

        if blocking:
            print(f"Starting dashboard server at http://{self.host}:{self.port}")
            self.socketio.run(self.app, host=self.host, port=self.port)
        else:
            self.server_thread = threading.Thread(
                target=self.socketio.run,
                kwargs={"app": self.app, "host": self.host, "port": self.port},
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            print(f"Dashboard server started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the dashboard server."""
        self._running = False
        # Note: Flask-SocketIO doesn't have a clean stop method
        print("Server stopping...")

    def update_metrics(self, **metrics):
        """
        Update metrics and broadcast to connected clients.

        Args:
            **metrics: Metric name-value pairs (e.g., epoch=5, loss=0.3)
        """
        if not self._running:
            return

        self.metrics_history.append(metrics)
        self.socketio.emit("metrics", metrics)

    def update_plots(self, plot_data: Dict[str, Any]):
        """
        Update plot data and broadcast to clients.

        Args:
            plot_data: Dictionary with plot updates
        """
        if not self._running:
            return

        self.socketio.emit("plots", plot_data)


# Convenience functions for quick visualization
def quick_plot_loss(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    save_path: str = "loss.png",
):
    """Quickly plot training and validation loss."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss", linewidth=2)
    if val_loss:
        plt.plot(val_loss, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def quick_plot_metrics(metrics: Dict[str, List[float]], save_path: str = "metrics.png"):
    """Quickly plot multiple metrics."""
    import matplotlib.pyplot as plt

    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, values) in enumerate(metrics.items()):
        axes[idx].plot(values, linewidth=2)
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Epoch")
        axes[idx].grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_interactive_dashboard(
    history: Dict[str, List[float]], save_path: str = "dashboard.html"
):
    """Create an interactive HTML dashboard with plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly required for interactive dashboard")
        return

    n_plots = len(history)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(history.keys()))

    for idx, (name, values) in enumerate(history.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        fig.add_trace(
            go.Scatter(x=list(range(len(values))), y=values, mode="lines", name=name),
            row=row,
            col=col,
        )

    fig.update_layout(title="Training Dashboard", height=300 * n_rows, showlegend=False)

    fig.write_html(save_path)
    print(f"Interactive dashboard saved to {save_path}")


# Example usage and demonstrations
def example_basic_usage():
    """Example of basic dashboard usage."""
    print("=" * 60)
    print("Example: Basic Training Dashboard")
    print("=" * 60)

    # Simulate training
    dashboard = TrainingDashboard(real_time=False, save_dir="examples/dashboard_basic")

    for epoch in range(10):
        loss = 1.0 / (epoch + 1)
        val_loss = 1.2 / (epoch + 1)
        accuracy = min(0.95, 0.5 + epoch * 0.05)
        val_accuracy = min(0.93, 0.48 + epoch * 0.045)

        logs = {
            "loss": loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "lr": 0.001 * (0.9**epoch),
        }

        dashboard.update(epoch, logs)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.4f}")

    dashboard.plot_all_metrics()
    dashboard.save_history()
    dashboard.print_summary()


def example_layer_visualization():
    """Example of layer visualization."""
    print("\n" + "=" * 60)
    print("Example: Layer Visualization")
    print("=" * 60)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.fc1 = nn.Linear(16 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, 10)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleModel()
    visualizer = LayerVisualizer(model, save_dir="examples/layers")

    # Register hooks and run forward pass
    visualizer.register_hooks()

    input_data = torch.randn(4, 1, 28, 28)
    output = model(input_data)

    # Compute gradients
    loss = output.sum()
    loss.backward()

    # Visualize
    visualizer.plot_activations("conv1")
    visualizer.plot_activations("conv2")
    visualizer.plot_weight_distribution()
    visualizer.plot_gradient_flow()

    visualizer.remove_hooks()

    print("Layer visualizations saved to examples/layers/")


def example_attention_visualization():
    """Example of attention visualization."""
    print("\n" + "=" * 60)
    print("Example: Attention Visualization")
    print("=" * 60)

    try:
        from transformers import BertModel, BertTokenizer

        # Load pretrained BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        model.eval()

        visualizer = AttentionVisualizer(model, save_dir="examples/attention")

        # Process text
        text = "The quick brown fox jumps over the lazy dog"
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Get attention weights
        attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)

        # Visualize
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        visualizer.set_tokens(tokens)

        visualizer.plot_attention_heatmap(attentions, layer_idx=0, head_idx=0)
        visualizer.plot_attention_heatmap(attentions, layer_idx=-1, head_idx=None)
        visualizer.plot_multi_head_comparison(attentions, layer_idx=0)

        print("Attention visualizations saved to examples/attention/")

    except ImportError:
        print("transformers library required for this example")


def example_prediction_visualization():
    """Example of prediction visualization."""
    print("\n" + "=" * 60)
    print("Example: Prediction Visualization")
    print("=" * 60)

    # Simulate predictions
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5

    targets = np.random.randint(0, n_classes, n_samples)
    predictions = targets.copy()

    # Add some noise
    noise_idx = np.random.choice(n_samples, size=100, replace=False)
    predictions[noise_idx] = np.random.randint(0, n_classes, 100)

    # Generate probabilities
    probabilities = np.random.rand(n_samples, n_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Update visualizer
    visualizer = PredictionVisualizer(
        class_names=["Airplane", "Car", "Bird", "Cat", "Dog"],
        save_dir="examples/predictions",
    )

    visualizer.update(
        torch.tensor(predictions), torch.tensor(targets), torch.tensor(probabilities)
    )

    # Plot
    visualizer.plot_confusion_matrix()
    visualizer.plot_confusion_matrix(normalize=True)
    visualizer.plot_roc_curve()
    visualizer.plot_pr_curve()

    print("Prediction visualizations saved to examples/predictions/")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_layer_visualization()
    example_prediction_visualization()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check the 'examples/' directory for visualizations.")
    print("=" * 60)
