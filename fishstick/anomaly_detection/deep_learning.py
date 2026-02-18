"""
Deep Learning Anomaly Detection Module.

This module provides deep learning-based anomaly detection methods:
- GAN-based anomaly detection (GANomaly, AnoGAN)
- RNN-based sequence anomaly detection
- Memory-augmented autoencoders
- Attention-based anomaly detection
- Contrastive learning for anomaly detection

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer


@dataclass
class DeepLearningResult:
    """Container for deep learning anomaly detection results."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    n_anomalies: int
    anomaly_indices: np.ndarray
    latent_representations: Optional[np.ndarray] = None
    reconstructions: Optional[np.ndarray] = None


class BaseDeepAnomalyDetector(nn.Module, ABC):
    """Base class for deep learning anomaly detectors."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
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

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward pass."""
        pass

    @abstractmethod
    def compute_loss(self, x: Tensor, **kwargs) -> Tuple[Tensor, Dict]:
        """Compute training loss."""
        pass

    def fit(self, X: np.ndarray) -> "BaseDeepAnomalyDetector":
        """Train the model."""
        self.to(self.device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        X_tensor = self._to_tensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()

                loss_dict = self.compute_loss(x)
                loss = loss_dict.get("total", sum(loss_dict.values()))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        self.eval()
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        self.eval()
        X_tensor = self._to_tensor(X)

        with torch.no_grad():
            scores = self._compute_scores(X_tensor)

        return scores.cpu().numpy()

    @abstractmethod
    def _compute_scores(self, x: Tensor) -> Tensor:
        """Compute anomaly scores from input."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            self.threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return (scores > self.threshold).astype(int)

    def fit_predict(self, X: np.ndarray) -> DeepLearningResult:
        """Fit and predict."""
        self.fit(X)
        scores = self.score(X)
        labels = self.predict(X)
        return DeepLearningResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            n_anomalies=int(np.sum(labels)),
            anomaly_indices=np.where(labels == 1)[0],
        )


class GANomalyDetector(BaseDeepAnomalyDetector):
    """
    GAN-based anomaly detector.

    Uses a generative adversarial network to learn normal data distribution.
    Anomalies are detected based on reconstruction and discriminator scores.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    latent_dim : int
        Latent space dimension.
    hidden_dims : List[int]
        Hidden layer dimensions.
    contamination : float
        Expected proportion of anomalies.
    epochs : int
        Training epochs.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            contamination=contamination,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            random_state=random_state,
        )
        self.hidden_dims = hidden_dims or [64, 128, 256]

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.discriminator = self._build_discriminator()

    def _build_encoder(self) -> nn.Module:
        """Build encoder network."""
        layers = []
        dims = [self.input_dim] + self.hidden_dims[:-1] + [self.latent_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Module:
        """Build decoder network."""
        layers = []
        dims = [self.latent_dim] + self.hidden_dims[:-1][::-1] + [self.input_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)

    def _build_discriminator(self) -> nn.Module:
        """Build discriminator network."""
        layers = []
        dims = [self.input_dim] + self.hidden_dims[::-1] + [1]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        validity = self.discriminator(x_recon)
        return x_recon, z, validity

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute training loss."""
        x_recon, z, validity = self.forward(x)

        recon_loss = F.mse_loss(x_recon, x)

        z_fake = self.encoder(x_recon.detach())
        latent_loss = F.mse_loss(z_fake, z)

        fake_validity = self.discriminator(x_recon)
        adv_loss = F.binary_cross_entropy_with_logits(
            fake_validity, torch.ones_like(fake_validity)
        )

        total = recon_loss + 0.1 * latent_loss + 0.1 * adv_loss

        return {
            "total": total,
            "recon": recon_loss,
            "latent": latent_loss,
            "adv": adv_loss,
        }

    def _compute_scores(self, x: Tensor) -> Tensor:
        """Compute anomaly scores."""
        x_recon, z, _ = self.forward(x)

        recon_error = torch.mean((x - x_recon) ** 2, dim=1)

        z_fake = self.encoder(x_recon)
        latent_error = torch.mean((z - z_fake) ** 2, dim=1)

        scores = recon_error + 0.1 * latent_error
        return scores


class OneClassRNN(BaseDeepAnomalyDetector):
    """
    One-Class RNN for sequence anomaly detection.

    Uses RNN to model normal sequences and detects anomalies
    based on reconstruction error.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dim : int
        RNN hidden dimension.
    num_layers : int
        Number of RNN layers.
    latent_dim : int
        Latent space dimension.
    contamination : float
        Expected proportion of anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        latent_dim: int = 16,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            contamination=contamination,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            random_state=random_state,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        self.decoder_rnn = nn.RNN(
            latent_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        _, h = self.encoder_rnn(x)
        z = self.encoder_fc(h[-1])

        output, _ = self.decoder_rnn(z.unsqueeze(1).repeat(1, x.size(1), 1))
        x_recon = self.decoder_fc(output)

        return x_recon, z

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute training loss."""
        x_recon, _ = self.forward(x)
        loss = F.mse_loss(x_recon, x)
        return {"total": loss, "recon": loss}

    def _compute_scores(self, x: Tensor) -> Tensor:
        """Compute anomaly scores."""
        x_recon, _ = self.forward(x)
        scores = torch.mean((x - x_recon) ** 2, dim=(1, 2))
        return scores


class MemoryAugmentedAutoencoder(BaseDeepAnomalyDetector):
    """
    Memory-augmented autoencoder for anomaly detection.

    Uses a memory module to store normal patterns and detects
    anomalies based on reconstruction error and attention to memory.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    latent_dim : int
        Latent space dimension.
    memory_size : int
        Number of memory slots.
    contamination : float
        Expected proportion of anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        memory_size: int = 50,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            contamination=contamination,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            random_state=random_state,
        )
        self.memory_size = memory_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
        )

        self.memory = nn.Parameter(torch.randn(memory_size, latent_dim))

        self.query_fc = nn.Linear(latent_dim, latent_dim)
        self.key_fc = nn.Linear(latent_dim, latent_dim)
        self.value_fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        z = self.encoder(x)

        query = self.query_fc(z)
        keys = self.key_fc(self.memory)
        values = self.value_fc(self.memory)

        attn_weights = torch.softmax(
            torch.matmul(query, keys.T) / (self.latent_dim**0.5), dim=-1
        )
        memory_output = torch.matmul(attn_weights, values)

        zEnhanced = z + memory_output
        x_recon = self.decoder(zEnhanced)

        return x_recon, z, attn_weights

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute training loss."""
        x_recon, z, attn_weights = self.forward(x)

        recon_loss = F.mse_loss(x_recon, x)

        mem_entropy = -torch.sum(
            attn_weights * torch.log(attn_weights + 1e-10), dim=-1
        ).mean()

        total = recon_loss + 0.01 * mem_entropy

        return {
            "total": total,
            "recon": recon_loss,
            "entropy": mem_entropy,
        }

    def _compute_scores(self, x: Tensor) -> Tensor:
        """Compute anomaly scores."""
        x_recon, z, attn_weights = self.forward(x)

        recon_error = torch.mean((x - x_recon) ** 2, dim=1)

        mem_usage = torch.max(attn_weights, dim=1)[0]

        scores = recon_error + 0.1 * (1 - mem_usage)
        return scores


class AttentionAnomalyDetector(BaseDeepAnomalyDetector):
    """
    Self-attention based anomaly detector.

    Uses multi-head self-attention to capture dependencies
    and detect anomalies based on attention pattern anomalies.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    d_model : int
        Model dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    latent_dim : int
        Latent space dimension.
    contamination : float
        Expected proportion of anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        latent_dim: int = 16,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            contamination=contamination,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            random_state=random_state,
        )
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.latent_fc = nn.Linear(d_model, latent_dim)
        self.recon_fc = nn.Linear(latent_dim, input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x_proj = self.input_proj(x.unsqueeze(1))

        attn_output = self.transformer(x_proj)

        z = self.latent_fc(attn_output.squeeze(1))

        x_recon = self.recon_fc(z)

        return x_recon, z

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute training loss."""
        x_recon, _ = self.forward(x)
        loss = F.mse_loss(x_recon, x)
        return {"total": loss, "recon": loss}

    def _compute_scores(self, x: Tensor) -> Tensor:
        """Compute anomaly scores."""
        x_recon, z = self.forward(x)

        recon_error = torch.mean((x - x_recon) ** 2, dim=1)

        z_std = torch.std(z, dim=1)
        latent_anomaly = -z_std

        scores = recon_error + 0.1 * torch.abs(latent_anomaly)
        return scores


class DeviationNetwork(BaseDeepAnomalyDetector):
    """
    Deviation Network for one-class classification.

    Learns a hypersphere that encompasses normal data.
    Points outside the sphere are anomalies.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dims : List[int]
        Hidden layer dimensions.
    contamination : float
        Expected proportion of anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=input_dim,
            contamination=contamination,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            random_state=random_state,
        )
        self.hidden_dims = hidden_dims or [128, 64]

        layers = []
        dims = [input_dim] + self.hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        self.center: Optional[Tensor] = None
        self.radius = 0.1

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.network(x)

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute training loss."""
        output = self.forward(x)

        if self.center is None:
            self.center = output.mean(dim=0, keepdim=True)

        dist = torch.norm(output - self.center, dim=1)

        loss = torch.clamp(dist - self.radius, min=0).mean()

        return {"total": loss, "deviation": loss}

    def _compute_scores(self, x: Tensor) -> Tensor:
        """Compute anomaly scores (distance from center)."""
        output = self.forward(x)
        scores = torch.norm(output - self.center.to(x.device), dim=1)
        return scores


class OneClassContrastive(BaseDeepAnomalyDetector):
    """
    One-Class Contrastive Learning for anomaly detection.

    Uses contrastive learning to learn normal data embeddings.
    Anomalies have different embedding patterns.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    embedding_dim : int
        Embedding dimension.
    temperature : float
        Temperature for contrastive loss.
    contamination : float
        Expected proportion of anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        temperature: float = 0.1,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=embedding_dim,
            contamination=contamination,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            random_state=random_state,
        )
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        z = self.encoder(x)
        h = self.projector(z)
        return z, h

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute contrastive loss."""
        z, h = self.forward(x)

        h = F.normalize(h, dim=1)
        similarity = torch.matmul(h, h.T) / self.temperature

        labels = torch.arange(len(x), device=x.device)
        loss = F.cross_entropy(similarity, labels)

        return {"total": loss, "contrastive": loss}

    def _compute_scores(self, x: Tensor) -> Tensor:
        """Compute anomaly scores based on embedding density."""
        z, _ = self.forward(x)

        z_normalized = F.normalize(z, dim=1)

        similarity = torch.matmul(z_normalized, z_normalized.T)
        similarity = similarity.fill_diagonal_(float("-inf"))

        max_sim, _ = similarity.max(dim=1)
        scores = -max_sim

        return scores


class DAGMMDetector(BaseDeepAnomalyDetector):
    """
    Deep Autoencoding Gaussian Mixture Model (DAGMM).

    Combines autoencoder with mixture model for density estimation.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    latent_dim : int
        Latent space dimension.
    n_components : int
        Number of GMM components.
    contamination : float
        Expected proportion of anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        n_components: int = 4,
        contamination: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            contamination=contamination,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            random_state=random_state,
        )
        self.n_components = n_components

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim),
        )

        self.estimation = nn.Sequential(
            nn.Linear(latent_dim + 2, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, n_components),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        z = self.encoder(x)
        x_recon = self.decoder(z)

        euclidean = torch.norm(x - x_recon, dim=1, keepdim=True)
        cosine = F.cosine_similarity(x, x_recon, dim=1, keepdim=True)

        z_concat = torch.cat([z, euclidean, cosine], dim=1)
        gamma = self.estimation(z_concat)

        return x_recon, z, gamma

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute DAGMM loss."""
        x_recon, z, gamma = self.forward(x)

        recon_loss = F.mse_loss(x_recon, x)

        z_mean = torch.sum(gamma * z, dim=0)
        z_var = torch.sum(gamma * (z - z_mean) ** 2, dim=0) + 1e-10

        phi = torch.sum(gamma, dim=0) / gamma.size(0)

        log_likelihood = (
            -0.5
            * torch.sum(
                gamma
                * (
                    torch.log(phi.unsqueeze(0))
                    - 0.5 * torch.log(z_var.unsqueeze(0))
                    - 0.5 * ((z - z_mean.unsqueeze(0)) ** 2 / z_var.unsqueeze(0))
                ),
                dim=1,
            ).mean()
        )

        total = recon_loss - log_likelihood

        return {
            "total": total,
            "recon": recon_loss,
            "likelihood": -log_likelihood,
        }

    def _compute_scores(self, x: Tensor) -> Tensor:
        """Compute anomaly scores."""
        x_recon, z, gamma = self.forward(x)

        recon_error = torch.mean((x - x_recon) ** 2, dim=1)

        z_mean = torch.sum(gamma * z, dim=0)
        z_var = torch.sum(gamma * (z - z_mean) ** 2, dim=0) + 1e-10

        phi = torch.sum(gamma, dim=0) / gamma.size(0)
        energy = -0.5 * torch.sum(
            gamma
            * (
                torch.log(phi.unsqueeze(0))
                - 0.5 * torch.log(z_var.unsqueeze(0))
                - 0.5 * ((z - z_mean.unsqueeze(0)) ** 2 / z_var.unsqueeze(0))
            ),
            dim=1,
        )

        scores = recon_error + 0.1 * energy
        return scores
