"""
Audio Deep Learning Models
"""

from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AudioClassifier(nn.Module):
    """Audio classification model."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_dim * (2**i), kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim * (2**i)),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
            )
            in_ch = hidden_dim * (2**i)

        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv_layers:
            x = conv(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class WaveNetAudio(nn.Module):
    """WaveNet model for audio generation."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        kernel_size: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            hidden_dim if i > 0 else input_dim,
                            hidden_dim,
                            kernel_size,
                            dilation=dilation,
                            padding=(kernel_size - 1) * dilation,
                        ),
                        "gate": nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
                        "skip": nn.Conv1d(hidden_dim, hidden_dim, 1),
                    }
                )
            )

        self.skip_projection = nn.Conv1d(hidden_dim, input_dim, 1)
        self.output_proj = nn.Conv1d(input_dim, input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        skip_connections = []

        for layer in self.layers:
            x = layer["conv"](x)[:, :, -x.size(2) :]

            gates = layer["gate"](x)
            gate_input, gate_transform = gates.chunk(2, dim=1)
            x = torch.tanh(gate_input) * torch.sigmoid(gate_transform)

            skip = layer["skip"](x)
            skip_connections.append(skip)

        x = sum(skip_connections)
        x = F.relu(x)
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_proj(x)

        return x.transpose(1, 2)


class TransformerAudio(nn.Module):
    """Transformer model for audio processing."""

    def __init__(
        self,
        input_dim: int = 80,
        num_classes: int = 10,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class VQVAE(nn.Module):
    """Vector Quantized VAE for audio."""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_embeddings: int = 256,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embedding_dim, 3, padding=1),
        )

        self.quantize = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim, 4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = x.transpose(1, 2)
        z = self.encoder(x)

        quant, indices, commit_loss = self.quantize(z.transpose(1, 2))

        recon = self.decoder(quant.transpose(1, 2))

        return recon.transpose(1, 2), quant, commit_loss


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_flattened = z.view(-1, self.embedding_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_indices).view(z.shape)

        commit_loss = F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()

        return z_q, min_indices, commit_loss


class AudioAutoencoder(nn.Module):
    """Autoencoder for audio compression."""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim, 4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x.transpose(1, 2))
        recon = self.decoder(z)
        return recon.transpose(1, 2)


class SpeechRecognizer(nn.Module):
    """End-to-end speech recognition model."""

    def __init__(
        self,
        input_dim: int = 80,
        num_classes: int = 32,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, 10, padding=5),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 8, stride=2, padding=3),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        return self.fc(x)
