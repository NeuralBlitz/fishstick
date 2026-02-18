"""
Results Visualization

Comprehensive visualization utilities for experiment results.

Classes:
- ResultsVisualizer: Main visualization class
- MetricPlotter: Plot metric curves
- HeatmapPlotter: Plot heatmaps
- ConfusionMatrixPlotter: Plot confusion matrices
- TrainingProgressPlotter: Plot training progress
- ComparisonPlotter: Plot comparison charts
- DistributionPlotter: Plot distributions
- ImageGridPlotter: Plot image grids
- InteractivePlotter: Interactive plotly plots
"""

from typing import Optional, Dict, List, Any, Union, Tuple
from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from seaborn import heatmap


class MetricPlotter:
    """Plot metric curves."""

    @staticmethod
    def plot_single(
        steps: List[int],
        values: List[float],
        title: str = "",
        xlabel: str = "Step",
        ylabel: str = "Value",
        color: str = "blue",
        save_path: Optional[Path] = None,
        show_std: bool = False,
        std_values: Optional[List[float]] = None,
    ) -> Figure:
        """Plot single metric curve.

        Args:
            steps: List of steps
            values: List of values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Line color
            save_path: Path to save figure
            show_std: Show standard deviation band
            std_values: Standard deviation values

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(steps, values, color=color, linewidth=2, label=ylabel)

        if show_std and std_values:
            ax.fill_between(
                steps,
                np.array(values) - np.array(std_values),
                np.array(values) + np.array(std_values),
                alpha=0.2,
                color=color,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig

    @staticmethod
    def plot_multiple(
        metrics: Dict[str, Tuple[List[int], List[float]]],
        title: str = "",
        xlabel: str = "Step",
        ylabel: str = "Value",
        colors: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Plot multiple metrics.

        Args:
            metrics: Dictionary of metric name to (steps, values)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            colors: List of colors
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = colors or list(mcolors.TABLEAU_COLORS.values())[: len(metrics)]

        for idx, (name, (steps, values)) in enumerate(metrics.items()):
            ax.plot(
                steps, values, linewidth=2, label=name, color=colors[idx % len(colors)]
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig


class HeatmapPlotter:
    """Plot heatmaps."""

    @staticmethod
    def plot_matrix(
        data: np.ndarray,
        title: str = "",
        xlabels: Optional[List[str]] = None,
        ylabels: Optional[List[str]] = None,
        cmap: str = "viridis",
        save_path: Optional[Path] = None,
        annot: bool = True,
        fmt: str = ".2f",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Figure:
        """Plot heatmap matrix.

        Args:
            data: 2D data array
            title: Plot title
            xlabels: X-axis labels
            ylabels: Y-axis labels
            colormap: Colormap name
            save_path: Path to save figure
            annot: Show values in cells
            fmt: Format string for annotations
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

        if xlabels:
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=45, ha="right")

        if ylabels:
            ax.set_yticks(np.arange(len(ylabels)))
            ax.set_yticklabels(ylabels)

        if annot:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(
                        j,
                        i,
                        format(data[i, j], fmt),
                        ha="center",
                        va="center",
                        color="white" if data[i, j] > (vmin + vmax) / 2 else "black",
                    )

        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig


class ConfusionMatrixPlotter:
    """Plot confusion matrices."""

    @staticmethod
    def plot(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        normalize: bool = False,
        save_path: Optional[Path] = None,
        cmap: str = "Blues",
    ) -> Figure:
        """Plot confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            normalize: Whether to normalize
            save_path: Path to save figure
            cmap: Colormap name

        Returns:
            Matplotlib figure
        """
        if normalize:
            cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        if class_names:
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
                xlabel="Predicted label",
                ylabel="True label",
            )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], ".2f"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig


class TrainingProgressPlotter:
    """Plot training progress."""

    @staticmethod
    def plot_loss_curves(
        train_losses: List[float],
        val_losses: List[float],
        steps_per_epoch: int,
        title: str = "Training Progress",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Plot training and validation loss curves.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            steps_per_epoch: Number of steps per epoch
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        train_steps = [i * steps_per_epoch for i in range(len(train_losses))]
        val_steps = [i * steps_per_epoch for i in range(len(val_losses))]

        ax1.plot(train_steps, train_losses, label="Train Loss", color="blue")
        ax1.plot(val_steps, val_losses, label="Val Loss", color="red")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            range(len(train_losses)), train_losses, label="Train Loss", color="blue"
        )
        ax2.plot(range(len(val_losses)), val_losses, label="Val Loss", color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss by Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig

    @staticmethod
    def plot_metrics_dashboard(
        metrics: Dict[str, List[float]],
        title: str = "Metrics Dashboard",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Plot metrics dashboard.

        Args:
            metrics: Dictionary of metric name to values
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        colors = list(mcolors.TABLEAU_COLORS.values())

        for idx, (name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            ax.plot(values, color=colors[idx % len(colors)], linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        for idx in range(len(metrics), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig


class ComparisonPlotter:
    """Plot comparison charts."""

    @staticmethod
    def plot_bar_comparison(
        categories: List[str],
        values: Dict[str, List[float]],
        title: str = "Comparison",
        ylabel: str = "Value",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Plot bar chart comparison.

        Args:
            categories: List of category names
            values: Dictionary of group name to values
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        n_categories = len(categories)
        n_groups = len(values)
        bar_width = 0.8 / n_groups

        colors = list(mcolors.TABLEAU_COLORS.values())

        for idx, (group_name, group_values) in enumerate(values.items()):
            x = np.arange(n_categories) + idx * bar_width
            ax.bar(
                x,
                group_values,
                bar_width,
                label=group_name,
                color=colors[idx % len(colors)],
            )

        ax.set_xlabel("Category")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(n_categories) + bar_width * (n_groups - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig

    @staticmethod
    def plot_box_comparison(
        data: Dict[str, List[float]],
        title: str = "Distribution Comparison",
        ylabel: str = "Value",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Plot box plot comparison.

        Args:
            data: Dictionary of group name to values
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        groups = list(data.keys())
        values = list(data.values())

        bp = ax.boxplot(values, labels=groups, patch_artist=True)

        colors = list(mcolors.TABLEAU_COLORS.values())
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig


class DistributionPlotter:
    """Plot distributions."""

    @staticmethod
    def plot_histogram(
        data: np.ndarray,
        bins: int = 30,
        title: str = "Distribution",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        kde: bool = True,
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Plot histogram.

        Args:
            data: Data array
            bins: Number of bins
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            kde: Show KDE curve
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data, bins=bins, alpha=0.7, edgecolor="black")

        if kde:
            try:
                from scipy import stats

                kde_x = np.linspace(data.min(), data.max(), 100)
                kde_y = stats.gaussian_kde(data)(kde_x)
                ax.plot(kde_x, kde_y, "r-", linewidth=2, label="KDE")
                ax.legend()
            except ImportError:
                pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig

    @staticmethod
    def plot_multiple_distributions(
        data: Dict[str, np.ndarray],
        title: str = "Distribution Comparison",
        xlabel: str = "Value",
        save_path: Optional[Path] = None,
    ) -> Figure:
        """Plot multiple distributions.

        Args:
            data: Dictionary of group name to data arrays
            title: Plot title
            xlabel: X-axis label
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = list(mcolors.TABLEAU_COLORS.values())

        for idx, (name, values) in enumerate(data.items()):
            ax.hist(
                values, bins=30, alpha=0.5, label=name, color=colors[idx % len(colors)]
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig


class ImageGridPlotter:
    """Plot image grids."""

    @staticmethod
    def plot_grid(
        images: np.ndarray,
        nrow: int = 4,
        ncol: int = 4,
        title: str = "Image Grid",
        save_path: Optional[Path] = None,
        labels: Optional[List[str]] = None,
    ) -> Figure:
        """Plot image grid.

        Args:
            images: Array of images (N, H, W, C) or (N, C, H, W)
            nrow: Number of rows
            ncol: Number of columns
            title: Plot title
            save_path: Path to save figure
            labels: Optional list of labels

        Returns:
            Matplotlib figure
        """
        n_images = min(len(images), nrow * ncol)
        images = images[:n_images]

        if images.ndim == 4:
            images = np.transpose(images, (0, 2, 3, 1))

        images = np.clip(images, 0, 1)

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        axes = axes.flatten() if nrow * ncol > 1 else [axes]

        for idx in range(nrow * ncol):
            if idx < len(images):
                axes[idx].imshow(images[idx])
                if labels and idx < len(labels):
                    axes[idx].set_title(labels[idx], fontsize=8)
            axes[idx].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig


class ResultsVisualizer:
    """Main results visualization class."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize results visualizer.

        Args:
            output_dir: Directory for saving plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path("plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metric_plotter = MetricPlotter()
        self.heatmap_plotter = HeatmapPlotter()
        self.confusion_matrix_plotter = ConfusionMatrixPlotter()
        self.training_progress_plotter = TrainingProgressPlotter()
        self.comparison_plotter = ComparisonPlotter()
        self.distribution_plotter = DistributionPlotter()
        self.image_grid_plotter = ImageGridPlotter()

    def plot_metric(
        self,
        name: str,
        steps: List[int],
        values: List[float],
        title: Optional[str] = None,
    ) -> Figure:
        """Plot a single metric.

        Args:
            name: Metric name
            steps: List of steps
            values: List of values
            title: Optional title

        Returns:
            Figure
        """
        title = title or f"{name} over Time"
        save_path = self.output_dir / f"{name}.png"
        return self.metric_plotter.plot_single(
            steps, values, title, save_path=save_path
        )

    def plot_metrics(
        self,
        metrics: Dict[str, Tuple[List[int], List[float]]],
        title: str = "Metrics Comparison",
    ) -> Figure:
        """Plot multiple metrics.

        Args:
            metrics: Dictionary of metric name to (steps, values)
            title: Plot title

        Returns:
            Figure
        """
        save_path = self.output_dir / "metrics_comparison.png"
        return self.metric_plotter.plot_multiple(metrics, title, save_path=save_path)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
    ) -> Figure:
        """Plot confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: Class names
            title: Plot title

        Returns:
            Figure
        """
        save_path = self.output_dir / "confusion_matrix.png"
        return self.confusion_matrix_plotter.plot(
            cm, class_names, title, save_path=save_path
        )

    def plot_training_progress(
        self,
        train_losses: List[float],
        val_losses: List[float],
        steps_per_epoch: int,
    ) -> Figure:
        """Plot training progress.

        Args:
            train_losses: Training losses
            val_losses: Validation losses
            steps_per_epoch: Steps per epoch

        Returns:
            Figure
        """
        save_path = self.output_dir / "training_progress.png"
        return self.training_progress_plotter.plot_loss_curves(
            train_losses, val_losses, steps_per_epoch, save_path=save_path
        )

    def plot_metrics_dashboard(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Metrics Dashboard",
    ) -> Figure:
        """Plot metrics dashboard.

        Args:
            metrics: Metrics dictionary
            title: Dashboard title

        Returns:
            Figure
        """
        save_path = self.output_dir / "metrics_dashboard.png"
        return self.training_progress_plotter.plot_metrics_dashboard(
            metrics, title, save_path=save_path
        )

    def plot_bar_comparison(
        self,
        categories: List[str],
        values: Dict[str, List[float]],
        title: str = "Comparison",
    ) -> Figure:
        """Plot bar chart comparison.

        Args:
            categories: Categories
            values: Values per group
            title: Plot title

        Returns:
            Figure
        """
        save_path = self.output_dir / "bar_comparison.png"
        return self.comparison_plotter.plot_bar_comparison(
            categories, values, title, save_path=save_path
        )

    def plot_box_comparison(
        self,
        data: Dict[str, List[float]],
        title: str = "Distribution Comparison",
    ) -> Figure:
        """Plot box plot comparison.

        Args:
            data: Data dictionary
            title: Plot title

        Returns:
            Figure
        """
        save_path = self.output_dir / "box_comparison.png"
        return self.comparison_plotter.plot_box_comparison(
            data, title, save_path=save_path
        )

    def plot_distribution(
        self,
        data: np.ndarray,
        name: str,
        title: Optional[str] = None,
    ) -> Figure:
        """Plot distribution.

        Args:
            data: Data array
            name: Distribution name
            title: Optional title

        Returns:
            Figure
        """
        title = title or f"Distribution of {name}"
        save_path = self.output_dir / f"distribution_{name}.png"
        return self.distribution_plotter.plot_histogram(
            data, title=title, save_path=save_path
        )

    def plot_image_grid(
        self,
        images: np.ndarray,
        name: str,
        nrow: int = 4,
        ncol: int = 4,
        labels: Optional[List[str]] = None,
    ) -> Figure:
        """Plot image grid.

        Args:
            images: Images array
            name: Grid name
            nrow: Number of rows
            ncol: Number of columns
            labels: Optional labels

        Returns:
            Figure
        """
        save_path = self.output_dir / f"grid_{name}.png"
        return self.image_grid_plotter.plot_grid(
            images, nrow, ncol, save_path=save_path, labels=labels
        )
