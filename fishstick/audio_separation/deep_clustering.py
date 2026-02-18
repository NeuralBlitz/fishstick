"""
Deep Clustering for Audio Source Separation

Implementation of deep clustering methods for permutation-invariant
audio source separation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from fishstick.audio_separation.base import SeparationModel, SeparationResult, STFT


class DeepClusteringLoss(nn.Module):
    """Deep Clustering Loss for permutation-invariant training.

    Deep clustering assigns each time-frequency bin to a source through
    learned embeddings, then uses clustering to assign bins to sources.

    Reference:
        Deep Clustering: Discriminative Embeddings for Segmentation
    """

    def __init__(
        self,
        embedding_dim: int = 20,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(
        self,
        embeddings: torch.Tensor,
        source_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep clustering loss.

        Args:
            embeddings: Embeddings of shape (batch, freq, time, embedding_dim)
            source_indicators: Source indicator matrix of shape (batch, freq, time, n_sources)

        Returns:
            Deep clustering loss value
        """
        batch, freq, time, emb_dim = embeddings.shape
        n_sources = source_indicators.shape[-1]

        embeddings = embeddings.reshape(batch, freq * time, emb_dim)
        source_indicators = source_indicators.reshape(batch, freq * time, n_sources)

        V = source_indicators.float()

        Y = V / (V.sum(dim=1, keepdim=True) + 1e-8)

        A = torch.matmul(embeddings, embeddings.transpose(1, 2))

        AY = torch.matmul(A, Y)

        loss = torch.trace(torch.matmul(Y.transpose(1, 2), AY))

        norm_term = torch.sum(A * A, dim=(1, 2)).mean()

        return -loss / (norm_term + 1e-8)

    def compute_permutation_loss(
        self,
        embeddings: torch.Tensor,
        sources: torch.Tensor,
        n_sources: int = 2,
    ) -> torch.Tensor:
        """Compute loss that handles permutations automatically.

        Args:
            embeddings: Embeddings of shape (batch, freq, time, emb_dim)
            sources: Source signals of shape (batch, n_sources, freq, time)
            n_sources: Number of sources

        Returns:
            Permutation-invariant loss
        """
        batch, freq, time, emb_dim = embeddings.shape

        indicators = torch.zeros(batch, freq, time, n_sources, device=embeddings.device)

        for b in range(batch):
            for i in range(n_sources):
                mask = torch.abs(sources[b, i]) > torch.mean(
                    torch.abs(sources[b]), dim=0
                )
                indicators[b, :, :, i] = mask.float()

        return self.forward(embeddings, indicators)


class ClusteringNetwork(nn.Module):
    """Network that produces embeddings for deep clustering.

    Architecture:
    - Encoder: Processes STFT magnitude into embeddings
    - Embedding space: Units are clustered to assign T-F bins to sources
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        embedding_dim: int = 20,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim

        n_freqs = n_fft // 2 + 1

        self.stft = STFT(n_fft, hop_length)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.embedding_layer = nn.Conv2d(128, embedding_dim, kernel_size=1)

    def forward(
        self,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        """Generate embeddings for T-F bins.

        Args:
            mixture: Mixed audio of shape (batch, channels, time)

        Returns:
            Embeddings of shape (batch, freq, time, embedding_dim)
        """
        mix_stft = self.stft(mixture)

        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        mag = torch.abs(mix_stft)
        mag = mag.unsqueeze(1)

        features = self.encoder(mag)

        embeddings = self.embedding_layer(features)

        embeddings = embeddings.permute(0, 2, 3, 1)

        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class EmbeddingExtractor(nn.Module):
    """Extracts embeddings from audio for clustering-based separation."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        embedding_dim: int = 40,
        n_layers: int = 4,
        hidden_channels: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim

        self.stft = STFT(n_fft, hop_length)

        channels = [hidden_channels // (2**i) for i in range(n_layers)]

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                ]
            )
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        self.projection = nn.Conv2d(channels[-1], embedding_dim, kernel_size=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from audio.

        Args:
            audio: Audio tensor of shape (batch, channels, time)

        Returns:
            Normalized embeddings of shape (batch, freq, time, embedding_dim)
        """
        stft = self.stft(audio)

        if stft.dim() == 4:
            stft = stft.squeeze(2)

        mag = torch.abs(stft).unsqueeze(1)

        embeddings = self.encoder(mag)
        embeddings = self.projection(embeddings)

        embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def extract_affinity_matrix(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute affinity matrix from embeddings.

        Args:
            embeddings: Embeddings of shape (batch, freq, time, emb_dim)

        Returns:
            Affinity matrix of shape (batch, T, T) where T = freq * time
        """
        batch, freq, time, emb_dim = embeddings.shape

        embeddings_flat = embeddings.reshape(batch, freq * time, emb_dim)

        affinity = torch.matmul(embeddings_flat, embeddings_flat.transpose(1, 2))

        return affinity


class AgglomerativeClustering:
    """Agglomerative clustering for post-processing embeddings.

    Performs clustering on embeddings to assign T-F bins to sources.
    """

    def __init__(
        self,
        n_sources: int = 2,
        threshold: float = 0.5,
    ):
        self.n_sources = n_sources
        self.threshold = threshold

    def __call__(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Cluster embeddings to assign T-F bins to sources.

        Args:
            embeddings: Embeddings of shape (batch, freq, time, emb_dim)

        Returns:
            Source assignments of shape (batch, freq, time, n_sources)
        """
        batch, freq, time, emb_dim = embeddings.shape

        embeddings_flat = embeddings.reshape(batch, freq * time, emb_dim)

        centers = self._initialize_centers(embeddings_flat)

        assignments = self._assign_clusters(embeddings_flat, centers)

        assignments = assignments.reshape(batch, freq, time, self.n_sources)

        return assignments

    def _initialize_centers(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Initialize cluster centers using k-means++."""
        batch_size, n_points, emb_dim = embeddings.shape

        centers = torch.zeros(
            batch_size, self.n_sources, emb_dim, device=embeddings.device
        )

        indices = torch.randint(0, n_points, (batch_size,), device=embeddings.device)
        centers[:, 0] = embeddings[torch.arange(batch_size), indices]

        for i in range(1, self.n_sources):
            distances = torch.cdist(embeddings, centers[:, :i])
            min_distances = distances.min(dim=2).values

            probs = min_distances**2
            probs = probs / probs.sum(dim=1, keepdim=True)

            indices = torch.multinomial(probs, 1).squeeze(-1)
            centers[:, i] = embeddings[torch.arange(batch_size), indices]

        return centers

    def _assign_clusters(
        self,
        embeddings: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """Assign points to nearest cluster centers."""
        distances = torch.cdist(embeddings, centers)

        assignments = distances.argmin(dim=2)

        one_hot = F.one_hot(assignments, num_classes=self.n_sources).float()

        return one_hot


class DeepClusteringModel(SeparationModel):
    """Complete deep clustering model for source separation."""

    def __init__(
        self,
        n_sources: int = 2,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        embedding_dim: int = 20,
        hidden_dim: int = 256,
    ):
        super().__init__(n_sources, sample_rate, n_fft, hop_length)

        self.clustering_network = ClusteringNetwork(
            n_fft=n_fft,
            hop_length=hop_length,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        self.clusterer = AgglomerativeClustering(n_sources=n_sources)

        self.stft = STFT(n_fft, hop_length)

        self.embedding_dim = embedding_dim

    def forward(
        self,
        mixture: torch.Tensor,
    ) -> SeparationResult:
        """Separate sources using deep clustering.

        Args:
            mixture: Mixed audio

        Returns:
            SeparationResult with separated sources
        """
        embeddings = self.clustering_network(mixture)

        assignments = self.clusterer(embeddings)

        mix_stft = self.stft(mixture)
        if mix_stft.dim() == 4:
            mix_stft = mix_stft.squeeze(2)

        mix_mag = torch.abs(mix_stft)

        separated_sources = []
        for i in range(self.n_sources):
            mask = assignments[:, :, :, i]
            if mask.dim() == 3:
                mask = mask.permute(0, 2, 1)

            source_stft = mix_stft * mask.unsqueeze(-1)
            source_wav = self.stft.inverse(source_stft)
            separated_sources.append(source_wav)

        sources = torch.stack(separated_sources)

        return SeparationResult(
            sources=sources,
            embeddings=embeddings,
            masks=assignments,
        )

    def estimate_sources(
        self,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate sources from mixture."""
        result = self.forward(mixture)
        return result.sources


class ClusterLoss(nn.Module):
    """Combined clustering loss for training deep clustering models."""

    def __init__(
        self,
        embedding_dim: int = 20,
    ):
        super().__init__()
        self.dc_loss = DeepClusteringLoss(embedding_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        source_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clustering loss."""
        return self.dc_loss(embeddings, source_indicators)

    def compute_separation_loss(
        self,
        embeddings: torch.Tensor,
        estimates: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss combining clustering and separation quality.

        Args:
            embeddings: Network embeddings
            estimates: Estimated sources
            references: Reference sources

        Returns:
            Combined loss
        """
        indicators = self._create_indicators(references)

        dc_loss = self.dc_loss(embeddings, indicators)

        sep_loss = F.mse_loss(estimates, references)

        return dc_loss + 0.1 * sep_loss

    def _create_indicators(
        self,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Create source indicator matrix from references."""
        batch, n_sources = references.shape[:2]

        indicators = torch.zeros(batch, n_sources, device=references.device)

        for i in range(n_sources):
            indicators[:, i] = torch.norm(references[:, i], p=2, dim=-1)

        indicators = indicators / (indicators.sum(dim=1, keepdim=True) + 1e-8)

        indicators = indicators.unsqueeze(1).unsqueeze(2)

        return indicators.expand(-1, references.shape[2], references.shape[3], -1)
