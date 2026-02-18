import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        encoder_layers = []
        channels = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Conv2d(channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        decoder_layers = []
        self.fc_decode = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        channels = hidden_dims[-1]
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels, h_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels = h_dim

        decoder_layers.append(
            nn.ConvTranspose2d(
                channels, in_channels, kernel_size=3, stride=1, padding=1
            )
        )
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 256, 2, 2)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.label_embed = nn.Embedding(num_classes, 32)

        encoder_layers = []
        channels = in_channels + 32
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Conv2d(channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.fc_decode = nn.Linear(latent_dim + 32, hidden_dims[-1] * 4)

        decoder_layers = []
        channels = hidden_dims[-1]
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels, h_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels = h_dim

        decoder_layers.append(
            nn.ConvTranspose2d(
                channels, in_channels, kernel_size=3, stride=1, padding=1
            )
        )
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_embed = self.label_embed(y).unsqueeze(-1).unsqueeze(-1)
        y_embed = y_embed.expand(-1, -1, x.size(2), x.size(3))
        h = torch.cat([x, y_embed], dim=1)

        h = self.encoder(h)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_embed = self.label_embed(y)
        h = torch.cat([z, y_embed], dim=1)
        h = self.fc_decode(h)
        h = h.view(-1, 256, 2, 2)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    def generate(
        self, num_samples: int, class_labels: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, class_labels)


class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_flattened = z.view(-1, self.embedding_dim)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(
            z_q, z.detach()
        )
        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dims: Optional[list] = None,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [128, 256]

        encoder_layers = []
        channels = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Conv2d(channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                ]
            )
            channels = h_dim

        encoder_layers.append(nn.Conv2d(channels, embedding_dim, kernel_size=1))
        self.encoder = nn.Sequential(*encoder_layers)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        decoder_layers = []
        decoder_layers.append(
            nn.ConvTranspose2d(embedding_dim, hidden_dims[-1], kernel_size=1)
        )
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim // 2),
                    nn.ReLU(),
                ]
            )
            channels = h_dim // 2

        decoder_layers.append(
            nn.ConvTranspose2d(
                channels, in_channels, kernel_size=3, stride=1, padding=1
            )
        )
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_q, _, _ = self.vq_layer(z)
        return z_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq_layer(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        indices = torch.randint(
            0, self.num_embeddings, (num_samples, 64), device=device
        )
        z = self.vq_layer.embedding(indices)
        z = z.view(num_samples, self.embedding_dim, 8, 8)
        return self.decode(z)


class BetaVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        beta: float = 4.0,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        encoder_layers = []
        channels = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Conv2d(channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        decoder_layers = []
        channels = hidden_dims[-1]
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels, h_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels = h_dim

        decoder_layers.append(
            nn.ConvTranspose2d(
                channels, in_channels, kernel_size=3, stride=1, padding=1
            )
        )
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 256, 2, 2)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss_function(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss


class FactorVAEEncoder(nn.Module):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: Optional[list] = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        encoder_layers = []
        channels = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Conv2d(channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc = nn.Linear(hidden_dims[-1] * 4, latent_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class FactorVAEDecoder(nn.Module):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: Optional[list] = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.fc_decode = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        decoder_layers = []
        channels = hidden_dims[-1]
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels, h_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels = h_dim

        decoder_layers.append(
            nn.ConvTranspose2d(
                channels, in_channels, kernel_size=3, stride=1, padding=1
            )
        )
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 256, 2, 2)
        return self.decoder(h)


class DisentangledBetaVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        gamma: float = 1.0,
        beta: float = 6.0,
        lr: float = 1e-4,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.beta = beta
        self.lr = lr

        self.encoder = FactorVAEEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = FactorVAEDecoder(in_channels, latent_dim, hidden_dims)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = h[:, : self.latent_dim]
        logvar = h[:, self.latent_dim :]
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def compute_tc_loss(self, z: torch.Tensor) -> torch.Tensor:
        z1 = z[:, : self.latent_dim // 2]
        z2 = z[:, self.latent_dim // 2 :]
        tc = torch.abs(self._compute_mi(z1) - self._compute_mi(z2))
        return tc

    def _compute_mi(self, z: torch.Tensor) -> torch.Tensor:
        return torch.mean(z**2)

    def loss_function(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        tc_loss = self.compute_tc_loss(z)
        loss = recon_loss + self.beta * kl_loss + self.gamma * tc_loss

        return loss, {
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "tc": tc_loss.item(),
            "total": loss.item(),
        }
