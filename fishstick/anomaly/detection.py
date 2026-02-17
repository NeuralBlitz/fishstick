"""
Comprehensive Anomaly Detection Module

Provides a unified interface for various anomaly detection methods including:
- Reconstruction-based (Autoencoder, VAE, DeepSVDD, AnoGAN, MemAE)
- Distance-based (KNN, LOF, Isolation Forest, OCSVM)
- Density-based (GMM, KDE, Normalizing Flows)
- Self-supervised (Contrastive, Rotation, Jigsaw, Relative Position)
- Attention-based (Transformers, Attention mechanisms)

Includes comprehensive evaluation metrics and visualization tools.
"""

from typing import Optional, Tuple, List, Dict, Union, Callable, Any
from abc import ABC, abstractmethod
import warnings

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis


# =============================================================================
# Base Classes
# =============================================================================


class BaseAnomalyDetector(ABC, nn.Module):
    """Base class for all anomaly detectors."""

    def __init__(self, input_dim: int, device: str = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.is_fitted = False
        self.threshold: Optional[float] = None

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning reconstructions and anomaly scores.

        Returns:
            outputs: Model outputs
            scores: Anomaly scores (higher = more anomalous)
        """
        pass

    @abstractmethod
    def fit(self, x: Union[np.ndarray, Tensor], **kwargs) -> "BaseAnomalyDetector":
        """Fit the detector on normal data."""
        pass

    def predict(
        self, x: Union[np.ndarray, Tensor], threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            x: Input data
            threshold: Decision threshold (uses fitted threshold if None)

        Returns:
            labels: 1 for anomaly, 0 for normal
        """
        scores = self.score(x)
        thresh = threshold if threshold is not None else self.threshold

        if thresh is None:
            raise ValueError("Threshold not set. Call fit() or provide threshold.")

        return (scores > thresh).astype(int)

    def score(self, x: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Compute anomaly scores."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float().to(self.device)
            _, scores = self.forward(x)
            return scores.cpu().numpy()

    def set_threshold(
        self,
        method: str = "percentile",
        value: float = 95.0,
        validation_data: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> "BaseAnomalyDetector":
        """
        Set anomaly detection threshold.

        Args:
            method: 'percentile' or 'std'
            value: Percentile or number of standard deviations
            validation_data: Data to compute threshold from
        """
        if validation_data is None:
            raise ValueError("validation_data required to set threshold")

        scores = self.score(validation_data)

        if method == "percentile":
            self.threshold = np.percentile(scores, value)
        elif method == "std":
            mean, std = scores.mean(), scores.std()
            self.threshold = mean + value * std
        else:
            raise ValueError(f"Unknown method: {method}")

        return self


# =============================================================================
# Reconstruction-Based Methods
# =============================================================================


class AutoencoderAnomalyDetector(BaseAnomalyDetector):
    """
    Standard Autoencoder for anomaly detection.

    Uses reconstruction error as anomaly score.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 16,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(input_dim, device)

        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(device)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to input space."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning reconstruction and anomaly score."""
        z = self.encode(x)
        x_hat = self.decode(z)

        # Reconstruction error as anomaly score
        score = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

        return x_hat, score

    def fit(
        self,
        x: Union[np.ndarray, Tensor],
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> "AutoencoderAnomalyDetector":
        """Fit the autoencoder on normal data."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                x_hat, _ = self.forward(batch_x)
                loss = F.mse_loss(x_hat, batch_x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

        self.is_fitted = True
        return self


class VAEAnomalyDetector(BaseAnomalyDetector):
    """
    Variational Autoencoder for anomaly detection.

    Uses reconstruction error + KL divergence as anomaly score.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        activation: str = "relu",
        device: str = "cpu",
    ):
        super().__init__(input_dim, device)

        self.latent_dim = latent_dim

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                ]
            )
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                ]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(device)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        return activations.get(name, nn.ReLU())

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to input space."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning reconstruction and anomaly score."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        # Reconstruction error + KL divergence as anomaly score
        recon_error = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        # Normalize KL to similar scale as reconstruction
        kl_div = kl_div / self.latent_dim

        score = recon_error + 0.1 * kl_div

        return x_hat, score

    def fit(
        self,
        x: Union[np.ndarray, Tensor],
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        verbose: bool = True,
    ) -> "VAEAnomalyDetector":
        """Fit the VAE on normal data."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                mu, logvar = self.encode(batch_x)
                z = self.reparameterize(mu, logvar)
                x_hat = self.decode(z)

                # ELBO loss
                recon_loss = F.mse_loss(x_hat, batch_x, reduction="sum") / batch_x.size(
                    0
                )
                kl_loss = (
                    -0.5
                    * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    / batch_x.size(0)
                )
                loss = recon_loss + beta * kl_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

        self.is_fitted = True
        return self

    def sample_latent(self, n_samples: int) -> Tensor:
        """Sample from the prior distribution."""
        return torch.randn(n_samples, self.latent_dim, device=self.device)


class DeepSVDD(BaseAnomalyDetector):
    """
    Deep One-Class Classification (Deep SVDD).

    Learns a neural network transformation that maps normal data
    close to a hypersphere center.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 32,
        activation: str = "relu",
        device: str = "cpu",
    ):
        super().__init__(input_dim, device)

        # Build network
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

        # Hypersphere center
        self.center: Optional[Tensor] = None

        self.to(device)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning latent representation and anomaly score."""
        z = self.network(x)

        if self.center is None:
            return z, torch.zeros(x.size(0), device=x.device)

        # Distance from center as anomaly score
        score = torch.sum((z - self.center) ** 2, dim=-1)
        return z, score

    def fit(
        self,
        x: Union[np.ndarray, Tensor],
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        verbose: bool = True,
    ) -> "DeepSVDD":
        """Fit Deep SVDD on normal data."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Compute center as mean of initial forward pass
        self.eval()
        with torch.no_grad():
            x_init = x[: min(1000, len(x))].to(self.device)
            z_init = self.network(x_init)
            self.center = z_init.mean(dim=0)

        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                z = self.network(batch_x)
                loss = torch.sum((z - self.center) ** 2)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

        self.is_fitted = True
        return self


class AnoGAN(nn.Module):
    """
    Anomaly Detection with Generative Adversarial Networks.

    Uses a trained GAN to generate normal data and measures
    reconstruction error in both pixel and feature space.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 100,
        generator_dims: List[int] = [128, 256],
        discriminator_dims: List[int] = [256, 128],
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        # Generator
        generator_layers = []
        prev_dim = latent_dim
        for dim in generator_dims:
            generator_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = dim
        generator_layers.append(nn.Linear(prev_dim, input_dim))
        self.generator = nn.Sequential(*generator_layers)

        # Discriminator
        discriminator_layers = []
        prev_dim = input_dim
        for dim in discriminator_dims:
            discriminator_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            prev_dim = dim
        discriminator_layers.append(nn.Linear(prev_dim, 1))
        self.discriminator = nn.Sequential(*discriminator_layers)

        self.to(device)
        self.is_fitted = False

    def generate(self, z: Tensor) -> Tensor:
        """Generate samples from latent code."""
        return self.generator(z)

    def discriminate(self, x: Tensor) -> Tensor:
        """Discriminate real vs fake."""
        return self.discriminator(x)

    def fit(
        self,
        x: Union[np.ndarray, Tensor],
        epochs: int = 100,
        batch_size: int = 256,
        g_lr: float = 2e-4,
        d_lr: float = 2e-4,
        verbose: bool = True,
    ) -> "AnoGAN":
        """Train the GAN on normal data."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999)
        )
        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999)
        )

        criterion = nn.BCEWithLogitsLoss()

        self.train()
        for epoch in range(epochs):
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                batch_size = batch_x.size(0)

                # Train discriminator
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                d_optimizer.zero_grad()

                real_loss = criterion(self.discriminate(batch_x), real_labels)

                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generate(z)
                fake_loss = criterion(
                    self.discriminate(fake_data.detach()), fake_labels
                )

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                d_optimizer.step()

                # Train generator
                g_optimizer.zero_grad()
                g_loss = criterion(self.discriminate(fake_data), real_labels)
                g_loss.backward()
                g_optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
                )

        self.is_fitted = True
        return self

    def detect(
        self,
        x: Union[np.ndarray, Tensor],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
        lambda_feat: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies by finding optimal latent code.

        Args:
            x: Input data
            n_iterations: Number of optimization steps
            learning_rate: Learning rate for latent code optimization
            lambda_feat: Weight for feature matching loss

        Returns:
            anomaly_scores: Anomaly scores
            reconstructed: Reconstructed samples
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)

        self.eval()

        # Initialize latent codes
        z = torch.randn(
            x.size(0), self.latent_dim, device=self.device, requires_grad=True
        )
        optimizer = torch.optim.Adam([z], lr=learning_rate)

        for _ in range(n_iterations):
            optimizer.zero_grad()

            x_hat = self.generate(z)

            # Residual loss
            residual_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1).sum()

            # Feature matching loss
            feat_real = self.discriminator[:-1](x)
            feat_fake = self.discriminator[:-1](x_hat)
            feat_loss = (
                F.mse_loss(feat_fake, feat_real, reduction="none").mean(dim=-1).sum()
            )

            loss = residual_loss + lambda_feat * feat_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x_hat = self.generate(z)
            scores = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

        return scores.cpu().numpy(), x_hat.cpu().numpy()


class MemAE(nn.Module):
    """
    Memory-Augmented Autoencoder for Anomaly Detection.

    Uses an external memory module to record prototypical normal patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 32,
        mem_dim: int = 50,
        shrink_threshold: float = 1.0 / 50,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mem_dim = mem_dim
        self.shrink_threshold = shrink_threshold
        self.device = device

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Memory
        self.memory = nn.Parameter(torch.randn(mem_dim, latent_dim))
        nn.init.xavier_uniform_(self.memory)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(device)
        self.is_fitted = False

    def query_memory(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Query memory with attention mechanism.

        Returns:
            z_hat: Reconstructed latent code
            w: Attention weights
        """
        # Compute attention scores
        w = torch.matmul(z, self.memory.t())  # [batch, mem_dim]
        w = F.softmax(w, dim=-1)

        # Sparse addressing (shrinkage)
        if self.shrink_threshold > 0:
            w = F.relu(w - self.shrink_threshold)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

        # Read from memory
        z_hat = torch.matmul(w, self.memory)  # [batch, latent_dim]

        return z_hat, w

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            x_hat: Reconstruction
            score: Anomaly score
            w: Attention weights
        """
        z = self.encoder(x)
        z_hat, w = self.query_memory(z)
        x_hat = self.decoder(z_hat)

        score = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

        return x_hat, score, w

    def fit(
        self,
        x: Union[np.ndarray, Tensor],
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        entropy_weight: float = 0.0002,
        verbose: bool = True,
    ) -> "MemAE":
        """Fit MemAE on normal data."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                x_hat, _, w = self.forward(batch_x)

                # Reconstruction loss
                recon_loss = F.mse_loss(x_hat, batch_x)

                # Entropy loss to encourage sparse memory access
                entropy = -torch.sum(w * torch.log(w + 1e-8), dim=-1).mean()

                loss = recon_loss + entropy_weight * entropy
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

        self.is_fitted = True
        return self
