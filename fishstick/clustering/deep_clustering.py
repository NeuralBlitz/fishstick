"""
Deep clustering methods combining neural networks with clustering.

Implements:
- DEC (Deep Embedded Clustering)
- IDEC (Improved Deep Embedded Clustering)
- DCN (Deep Clustering Network)
- ClusterGAN
- VaDE (Variational Deep Embedding)
- JULE (Joint Unsupervised Learning)
"""

from typing import Optional, List, Tuple, Callable
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from .base import ClustererBase, ClusterResult


@dataclass
class DeepClusteringResult(ClusterResult):
    """Result of deep clustering."""

    embeddings: Optional[Tensor] = None
    cluster_centers: Optional[Tensor] = None
    reconstruction_loss: Optional[float] = None


class Encoder(nn.Module):
    """Autoencoder encoder network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.latent = nn.Linear(prev_dim, latent_dim)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.latent(x)


class Decoder(nn.Module):
    """Autoencoder decoder network."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = latent_dim

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.output = nn.Linear(prev_dim, output_dim)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            self.activation = F.relu

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output(x)


class AutoEncoder(nn.Module):
    """Autoencoder for deep clustering pretraining."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class DEC(nn.Module):
    """
    Deep Embedded Clustering (DEC).

    Simultaneously learns cluster assignments and feature representations.

    Args:
        input_dim: Input feature dimension
        n_clusters: Number of clusters
        hidden_dims: List of hidden layer dimensions
        latent_dim: Latent space dimension
        alpha: Softmax temperature parameter
    """

    def __init__(
        self,
        input_dim: int,
        n_clusters: int,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 10,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.cluster_center = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x: Tensor, return_embeddings: bool = False):
        z = self.encoder(x)
        q = self._get_clustering_distribution(z)

        if return_embeddings:
            return q, z
        return q

    def _get_clustering_distribution(self, z: Tensor) -> Tensor:
        """Compute soft cluster assignments."""
        dist = torch.cdist(z, self.cluster_center)
        q = (1.0 + dist**2 / self.alpha) ** (-(self.alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    def predict(self, x: Tensor) -> Tensor:
        """Hard cluster assignment prediction."""
        z = self.encoder(x)
        dist = torch.cdist(z, self.cluster_center)
        return dist.argmin(dim=1)


class DECTrainer:
    """Trainer for DEC model."""

    def __init__(
        self,
        model: DEC,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def pretrain(
        self,
        X: Tensor,
        epochs: int = 100,
        batch_size: int = 256,
    ) -> AutoEncoder:
        """Pretrain autoencoder."""
        input_dim = X.shape[1]
        hidden_dims = [256, 128]
        latent_dim = self.model.cluster_center.shape[1]

        ae = AutoEncoder(input_dim, hidden_dims, latent_dim)
        if X.is_cuda:
            ae = ae.cuda()

        optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                x = batch[0]
                if x.is_cuda:
                    x = x.cuda()

                x_recon, z = ae(x)
                loss = F.mse_loss(x_recon, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(
                    f"Pretrain Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}"
                )

        with torch.no_grad():
            embeddings = ae.encoder(X)
            if X.is_cuda:
                embeddings = embeddings.cuda()

        kmeans = KMeansModel(n_clusters=self.model.n_clusters)
        kmeans.fit(embeddings)
        self.model.cluster_center.data = kmeans.centroids

        return ae

    def fit(
        self,
        X: Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        update_interval: int = 1,
    ):
        """Train DEC model."""
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                x = batch[0]
                if x.is_cuda:
                    x = x.cuda()

                q, z = self.model(x, return_embeddings=True)

                p = self._target_distribution(q)

                dist = torch.cdist(z, self.model.cluster_center)
                loss = F.kl_div(q.log(), p, reduction="batchmean")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}"
                )

    def _target_distribution(self, q: Tensor) -> Tensor:
        """Compute target distribution for clustering."""
        p = q**2 / q.sum(dim=0)
        p = p / p.sum(dim=1, keepdim=True)
        return p


class KMeansModel:
    """Simple K-means for initializing cluster centers."""

    def __init__(self, n_clusters: int, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "KMeansModel":
        """Fit K-means."""
        n_samples = X.shape[0]
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        indices = torch.randperm(n_samples)[: self.n_clusters]
        centroids = X[indices].clone()

        for _ in range(100):
            distances = torch.cdist(X, centroids)
            labels = distances.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)

            if torch.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        self.centroids = centroids
        return self


class IDEC(nn.Module):
    """
    Improved Deep Embedded Clustering (IDEC).

    Combines reconstruction loss with clustering loss.

    Args:
        input_dim: Input feature dimension
        n_clusters: Number of clusters
        hidden_dims: List of hidden dimensions
        latent_dim: Latent dimension
        alpha: Softmax temperature
        gamma: Weight for reconstruction loss
    """

    def __init__(
        self,
        input_dim: int,
        n_clusters: int,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 10,
        alpha: float = 1.0,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)
        self.cluster_center = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x: Tensor, return_embeddings: bool = False):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        q = self._get_clustering_distribution(z)

        if return_embeddings:
            return q, x_recon, z
        return q, x_recon

    def _get_clustering_distribution(self, z: Tensor) -> Tensor:
        dist = torch.cdist(z, self.cluster_center)
        q = (1.0 + dist**2 / self.alpha) ** (-(self.alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)
        return q


class DCN(nn.Module):
    """
    Deep Clustering Network (DCN).

    Uses K-means and autoencoder jointly.

    Args:
        input_dim: Input dimension
        n_clusters: Number of clusters
        hidden_dims: Hidden dimensions
        latent_dim: Latent dimension
    """

    def __init__(
        self,
        input_dim: int,
        n_clusters: int,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 10,
    ):
        super().__init__()
        self.n_clusters = n_clusters

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)
        self.cluster_center = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x: Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class ClusterGAN(nn.Module):
    """
    ClusterGAN - Clustering with GANs.

    Uses GAN framework for clustering.

    Args:
        latent_dim: Latent dimension
        n_clusters: Number of clusters
        input_dim: Input dimension
        hidden_dims: Hidden dimensions for generator/discriminator
    """

    def __init__(
        self,
        latent_dim: int,
        n_clusters: int,
        input_dim: int,
        hidden_dims: List[int] = [128, 256],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.input_dim = input_dim

        self.generator = nn.Sequential(
            nn.Linear(latent_dim + n_clusters, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], input_dim),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], 1),
        )

    def generate(self, z: Tensor, c: Tensor) -> Tensor:
        """Generate samples from latent and cluster."""
        return self.generator(torch.cat([z, c], dim=1))


class VaDE(nn.Module):
    """
    Variational Deep Embedding (VaDE).

    Variational autoencoder for clustering.

    Args:
        input_dim: Input dimension
        n_clusters: Number of clusters
        hidden_dims: Hidden dimensions
        latent_dim: Latent dimension
    """

    def __init__(
        self,
        input_dim: int,
        n_clusters: int,
        hidden_dims: List[int] = [500, 500, 2000],
        latent_dim: int = 10,
    ):
        super().__init__()
        self.n_clusters = n_clusters

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim

        self.decoder = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(prev_dim, input_dim)

        self.cluster_prior = nn.Parameter(torch.ones(n_clusters) / n_clusters)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder(z)
        return self.output_layer(h)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class JULE(nn.Module):
    """
    Joint Unsupervised Learning (JULE).

    Recurrent clustering with CNN features.

    Args:
        input_dim: Input dimension
        n_clusters: Number of clusters
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        input_dim: int,
        n_clusters: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_clusters = n_clusters

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.cluster_layer = nn.Linear(hidden_dim, n_clusters)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        q = F.softmax(self.cluster_layer(h), dim=1)
        return q, h


def create_dec(
    input_dim: int,
    n_clusters: int,
    **kwargs,
) -> DEC:
    """Factory function for DEC."""
    return DEC(input_dim=input_dim, n_clusters=n_clusters, **kwargs)


def create_idec(
    input_dim: int,
    n_clusters: int,
    **kwargs,
) -> IDEC:
    """Factory function for IDEC."""
    return IDEC(input_dim=input_dim, n_clusters=n_clusters, **kwargs)


def create_dcn(
    input_dim: int,
    n_clusters: int,
    **kwargs,
) -> DCN:
    """Factory function for DCN."""
    return DCN(input_dim=input_dim, n_clusters=n_clusters, **kwargs)
