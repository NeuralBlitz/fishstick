"""
Data Visualization Tools
"""

from typing import Optional, List, Tuple
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class DataVisualizer:
    """Visualize datasets and data distributions."""

    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def plot_images(
        self,
        images: Tensor,
        labels: Optional[List] = None,
        n_rows: int = 4,
        n_cols: int = 4,
        save_path: str = "sample_images.png",
    ) -> None:
        """Plot a grid of images."""
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img = images[idx]

                # Handle different channel configurations
                if img.ndim == 3:
                    if img.shape[0] in [1, 3]:  # CHW format
                        img = img.permute(1, 2, 0)
                    img = img.cpu().numpy()

                    if img.shape[2] == 1:
                        img = img.squeeze()
                        ax.imshow(img, cmap="gray")
                    else:
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        ax.imshow(img)

                if labels and idx < len(labels):
                    ax.set_title(str(labels[idx]))

                ax.axis("off")
            else:
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Images saved to {self.save_dir / save_path}")

    def plot_class_distribution(
        self,
        labels: List[int],
        class_names: Optional[List[str]] = None,
        save_path: str = "class_distribution.png",
    ) -> None:
        """Plot class distribution."""
        unique, counts = np.unique(labels, return_counts=True)

        plt.figure(figsize=(12, 6))

        if class_names:
            tick_labels = [
                class_names[i] if i < len(class_names) else str(i) for i in unique
            ]
        else:
            tick_labels = [str(i) for i in unique]

        plt.bar(range(len(unique)), counts)
        plt.xticks(range(len(unique)), tick_labels, rotation=45, ha="right")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Class distribution saved to {self.save_dir / save_path}")

    def plot_feature_distribution(
        self,
        features: np.ndarray,
        labels: Optional[List[int]] = None,
        save_path: str = "feature_distribution.png",
    ) -> None:
        """Plot feature distributions (for 1D or 2D features)."""
        if features.ndim == 1 or (features.ndim == 2 and features.shape[1] == 1):
            # 1D features - histogram
            plt.figure(figsize=(10, 6))
            plt.hist(features.flatten(), bins=50, alpha=0.7)
            plt.xlabel("Feature Value")
            plt.ylabel("Frequency")
            plt.title("Feature Distribution")
            plt.grid(True, alpha=0.3)

        elif features.ndim == 2 and features.shape[1] == 2:
            # 2D features - scatter plot
            plt.figure(figsize=(10, 8))
            if labels is not None:
                for label in np.unique(labels):
                    mask = np.array(labels) == label
                    plt.scatter(
                        features[mask, 0],
                        features[mask, 1],
                        label=f"Class {label}",
                        alpha=0.6,
                    )
                plt.legend()
            else:
                plt.scatter(features[:, 0], features[:, 1], alpha=0.6)

            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Feature Distribution (2D)")
            plt.grid(True, alpha=0.3)

        else:
            # Higher dimensional - show first few dimensions
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i in range(min(6, features.shape[1] if features.ndim > 1 else 1)):
                if features.ndim > 1:
                    data = features[:, i]
                else:
                    data = features

                axes[i].hist(data, bins=50, alpha=0.7)
                axes[i].set_title(f"Feature {i + 1}")
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()

        plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Feature distribution saved to {self.save_dir / save_path}")

    def plot_tsne(
        self,
        features: np.ndarray,
        labels: List[int],
        save_path: str = "tsne.png",
        n_components: int = 2,
    ) -> None:
        """Plot t-SNE visualization of features."""
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=n_components, random_state=42)
            embedded = tsne.fit_transform(features)

            plt.figure(figsize=(12, 10))

            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                if n_components == 2:
                    plt.scatter(
                        embedded[mask, 0],
                        embedded[mask, 1],
                        c=[colors[i]],
                        label=f"Class {label}",
                        alpha=0.6,
                    )
                else:
                    ax = plt.axes(projection="3d")
                    ax.scatter(
                        embedded[mask, 0],
                        embedded[mask, 1],
                        embedded[mask, 2],
                        c=[colors[i]],
                        label=f"Class {label}",
                        alpha=0.6,
                    )

            plt.legend()
            plt.title("t-SNE Visualization")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"t-SNE visualization saved to {self.save_dir / save_path}")

        except ImportError:
            print("scikit-learn not installed. Skipping t-SNE.")

    def plot_pca(
        self,
        features: np.ndarray,
        labels: List[int],
        save_path: str = "pca.png",
    ) -> None:
        """Plot PCA visualization of features."""
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            embedded = pca.fit_transform(features)

            plt.figure(figsize=(12, 10))

            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(
                    embedded[mask, 0],
                    embedded[mask, 1],
                    c=[colors[i]],
                    label=f"Class {label}",
                    alpha=0.6,
                )

            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            plt.legend()
            plt.title("PCA Visualization")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"PCA visualization saved to {self.save_dir / save_path}")

        except ImportError:
            print("scikit-learn not installed. Skipping PCA.")
