"""
Autoencoder-based Anomaly Detection Module.

This module provides autoencoder-based methods for anomaly detection:
- Variational Autoencoder (VAE)
- Denoising Autoencoder (DAE)
- Contractive Autoencoder (CAE)
- Sparse Autoencoder
- Adversarial Autoencoder
- Attention-based Autoencoder

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class AutoencoderResult:
    """Container for autoencoder-based detection results."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    n_anomalies: int
    reconstructions: Optional[np.ndarray] = None
    latent_mean: Optional[np.ndarray] = None
    latent_logvar: Optional[np.ndarray] = None


class BaseAutoencoderDetector(nn.Module):
    """Base class for autoencoder-based anomaly detectors."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.contamination = contamination
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = self._get_device(device)
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.threshold: Optional[float] = None
        self.is_fitted = False

    def _get_device(self, device: str) -> torch.device:
        """Get torch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _to_tensor(self, X: np.ndarray) -> Tensor:
        """Convert numpy array to tensor."""
        return torch.FloatTensor(X).to(self.device)

    def _create_dataloader(
        self,
        X: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create DataLoader from numpy array."""
        dataset = TensorDataset(torch.FloatTensor(X))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward pass returning reconstruction and optional latent."""
        pass

    @abstractmethod
    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute reconstruction loss."""
        pass

    def fit(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
    ) -> "BaseAutoencoderDetector":
        """Train the autoencoder on normal data."""
        self.to(self.device)
        self.train()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        train_loader = self._create_dataloader(X, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch[0].to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(x)

                if isinstance(outputs, tuple):
                    x_recon = outputs[0]
                else:
                    x_recon = outputs

                loss = self.reconstruction_loss(x, x_recon)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores based on reconstruction error."""
        self.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            outputs = self.forward(X_tensor)

            if isinstance(outputs, tuple):
                x_recon = outputs[0]
            else:
                x_recon = outputs

            errors = torch.mean((X_tensor - x_recon) ** 2, dim=1)
            return errors.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            self.threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return (scores > self.threshold).astype(int)

    def fit_predict(self, X: np.ndarray) -> AutoencoderResult:
        """Fit and predict in one call."""
        self.fit(X)
        scores = self.score(X)
        labels = self.predict(X)
        return AutoencoderResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            n_anomalies=int(np.sum(labels)),
        )


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ReLU

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class VAEAnomalyDetector(BaseAutoencoderDetector):
    """
    Variational Autoencoder for anomaly detection.

    Uses reconstruction probability as anomaly score. Points with
    low reconstruction probability are flagged as anomalies.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list
        Hidden layer dimensions.
    latent_dim : int
        Latent space dimension.
    contamination : float
        Expected proportion of anomalies.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    beta : float
        Beta-VAE regularization weight.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        contamination: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        beta: float = 1.0,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim,
            latent_dim,
            contamination,
            lr,
            batch_size,
            epochs,
            device,
            random_state,
        )
        self.beta = beta
        self.model = VariationalAutoencoder(input_dim, hidden_dims, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        return self.model(x)

    def reconstruction_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """Compute VAE loss (reconstruction + KL divergence)."""
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    def fit(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
    ) -> "VAEAnomalyDetector":
        """Train VAE on normal data."""
        self.to(self.device)
        self.model.train()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_loader = self._create_dataloader(X, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch[0].to(self.device)

                optimizer.zero_grad()
                x_recon, mu, logvar = self.model(x)
                loss = self.reconstruction_loss(x, x_recon, mu, logvar)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores based on reconstruction error + KL."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            x_recon, mu, logvar = self.model(X_tensor)

            recon_error = torch.mean((X_tensor - x_recon) ** 2, dim=1)
            kl_term = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            return (recon_error + self.beta * kl_term).cpu().numpy()


class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder with configurable corruption."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        activation: str = "relu",
        corruption_level: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.corruption_level = corruption_level

        if activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Tanh

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.latent = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent."""
        return self.latent(self.encoder(x))

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def corrupt(self, x: Tensor) -> Tensor:
        """Add Gaussian noise to input."""
        noise = torch.randn_like(x) * self.corruption_level
        return x + noise

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with corruption during training."""
        if self.training:
            x = self.corrupt(x)
        z = self.encode(x)
        return self.decode(z)


class DenoisingAEDetector(BaseAutoencoderDetector):
    """
    Denoising Autoencoder for anomaly detection.

    Learns to reconstruct clean data from corrupted inputs.
    High reconstruction error indicates anomalies.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list
        Hidden layer dimensions.
    latent_dim : int
        Latent space dimension.
    contamination : float
        Expected proportion of anomalies.
    corruption_level : float
        Noise level for corruption during training.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        contamination: float = 0.1,
        corruption_level: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim,
            latent_dim,
            contamination,
            lr,
            batch_size,
            epochs,
            device,
            random_state,
        )
        self.corruption_level = corruption_level
        self.model = DenoisingAutoencoder(
            input_dim, hidden_dims, latent_dim, corruption_level=corruption_level
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(x)

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute MSE loss."""
        return F.mse_loss(x_recon, x, reduction="mean")


class ContractiveAutoencoder(nn.Module):
    """Contractive Autoencoder with Jacobian penalty."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Tanh

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_latent = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        """Encode to latent."""
        h = self.encoder(x)
        return self.fc_latent(h)

    def decode(self, z: Tensor) -> Tensor:
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        z = self.encode(x)
        return self.decode(z)


class ContractiveAEDetector(BaseAutoencoderDetector):
    """
    Contractive Autoencoder for anomaly detection.

    Adds penalty for sensitivity to input changes, learning
    more robust representations.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list
        Hidden layer dimensions.
    latent_dim : int
        Latent space dimension.
    contamination : float
        Expected proportion of anomalies.
    contractive_weight : float
        Weight for Jacobian penalty.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        contamination: float = 0.1,
        contractive_weight: float = 1e-4,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim,
            latent_dim,
            contamination,
            lr,
            batch_size,
            epochs,
            device,
            random_state,
        )
        self.contractive_weight = contractive_weight
        self.model = ContractiveAutoencoder(input_dim, hidden_dims, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(x)

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute reconstruction + contractive loss."""
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        jacobian = []
        x_requires_grad = x.requires_grad_(True)
        z = self.model.encode(x_requires_grad)
        for i in range(z.shape[1]):
            grad = torch.autograd.grad(
                z[:, i].sum(), x_requires_grad, retain_graph=True, create_graph=True
            )[0]
            jacobian.append(grad)

        jacobian = torch.stack(jacobian, dim=2)
        contractive_loss = torch.sum(jacobian**2) / x.shape[0]

        return recon_loss + self.contractive_weight * contractive_loss


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 penalty on latent."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Tanh

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_latent = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        z = self.fc_latent(self.encoder(x))
        return self.decoder(z)

    def get_latent(self, x: Tensor) -> Tensor:
        """Get latent representation."""
        return self.fc_latent(self.encoder(x))


class SparseAEDetector(BaseAutoencoderDetector):
    """
    Sparse Autoencoder for anomaly detection.

    Enforces sparsity in latent representation, learning
    more interpretable features.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list
        Hidden layer dimensions.
    latent_dim : int
        Latent space dimension.
    contamination : float
        Expected proportion of anomalies.
    sparsity_weight : float
        Weight for L1 sparsity penalty.
    sparsity_target : float
        Target sparsity level.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        contamination: float = 0.1,
        sparsity_weight: float = 1e-3,
        sparsity_target: float = 0.05,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim,
            latent_dim,
            contamination,
            lr,
            batch_size,
            epochs,
            device,
            random_state,
        )
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        self.model = SparseAutoencoder(input_dim, hidden_dims, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(x)

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute reconstruction + sparsity loss."""
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        z = self.model.get_latent(x)
        rho_hat = torch.mean(z, dim=0)
        rho = torch.full_like(rho_hat, self.sparsity_target)
        kl_div = rho * (torch.log(rho) - torch.log(rho_hat + 1e-8)) + (1 - rho) * (
            torch.log(1 - rho) - torch.log(1 - rho_hat + 1e-8)
        )
        sparsity_loss = torch.sum(kl_div)

        return recon_loss + self.sparsity_weight * sparsity_loss


class AttentionAutoencoder(nn.Module):
    """Autoencoder with self-attention mechanism."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        n_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
        )

        self.attention = nn.MultiheadAttention(
            hidden_dims[1], n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dims[1])

        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        h = self.encoder(x).unsqueeze(1)
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out).squeeze(1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        recon = self.decoder(z)
        return recon, mu, logvar


class AttentionAEDetector(BaseAutoencoderDetector):
    """
    Attention-based Autoencoder for anomaly detection.

    Uses self-attention to capture feature dependencies.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list
        Hidden layer dimensions.
    latent_dim : int
        Latent space dimension.
    n_heads : int
        Number of attention heads.
    contamination : float
        Expected proportion of anomalies.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 16,
        n_heads: int = 4,
        contamination: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim,
            latent_dim,
            contamination,
            lr,
            batch_size,
            epochs,
            device,
            random_state,
        )
        self.n_heads = n_heads
        self.model = AttentionAutoencoder(input_dim, hidden_dims, latent_dim, n_heads)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        return self.model(x)

    def reconstruction_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """Compute VAE-style loss."""
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        return recon_loss + kl_loss

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        self.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            x_recon, _, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - x_recon) ** 2, dim=1)
            return errors.cpu().numpy()


__all__ = [
    "BaseAutoencoderDetector",
    "AutoencoderResult",
    "VariationalAutoencoder",
    "VAEAnomalyDetector",
    "DenoisingAutoencoder",
    "DenoisingAEDetector",
    "ContractiveAutoencoder",
    "ContractiveAEDetector",
    "SparseAutoencoder",
    "SparseAEDetector",
    "AttentionAutoencoder",
    "AttentionAEDetector",
]
