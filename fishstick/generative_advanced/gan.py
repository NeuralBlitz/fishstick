import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DCGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_channels: int,
        features_g: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.init_size = 4
        self.fc = nn.Linear(
            latent_dim, features_g * 8 * self.init_size * self.init_size
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(features_g * 8),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,
        features_d: int = 64,
    ):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(features_d * 8 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        return validity


class DCGAN(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 3,
        features_g: int = 64,
        features_d: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels

        self.generator = DCGenerator(latent_dim, img_channels, features_g)
        self.discriminator = DCDiscriminator(img_channels, features_d)

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)

    def d_loss(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> torch.Tensor:
        real_validity = self.discriminator(real_imgs)
        fake_validity = self.discriminator(fake_imgs.detach())
        loss = F.binary_cross_entropy_with_logits(
            real_validity, torch.ones_like(real_validity)
        ) + F.binary_cross_entropy_with_logits(
            fake_validity, torch.zeros_like(fake_validity)
        )
        return loss

    def g_loss(self, fake_imgs: torch.Tensor) -> torch.Tensor:
        fake_validity = self.discriminator(fake_imgs)
        loss = F.binary_cross_entropy_with_logits(
            fake_validity, torch.ones_like(fake_validity)
        )
        return loss


class WGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_channels: int,
        features_g: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.init_size = 4
        self.fc = nn.Linear(
            latent_dim, features_g * 8 * self.init_size * self.init_size
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(features_g * 8),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        return self.conv_blocks(out)


class WGANDiscriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,
        features_d: int = 64,
    ):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(features_d * 8 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def gradient_penalty(
    discriminator: nn.Module,
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolates = interpolates.requires_grad_(True)

    disc_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones_like(disc_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


class WGAN_GP(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 3,
        features_g: int = 64,
        features_d: int = 64,
        gp_lambda: float = 10.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.gp_lambda = gp_lambda

        self.generator = WGANGenerator(latent_dim, img_channels, features_g)
        self.discriminator = WGANDiscriminator(img_channels, features_d)

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)

    def d_loss(
        self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        real_validity = self.discriminator(real_imgs)
        fake_validity = self.discriminator(fake_imgs.detach())
        wasserstein_d = real_validity.mean() - fake_validity.mean()
        gp = gradient_penalty(self.discriminator, real_imgs, fake_imgs, device)
        return -wasserstein_d + self.gp_lambda * gp

    def g_loss(self, fake_imgs: torch.Tensor) -> torch.Tensor:
        fake_validity = self.discriminator(fake_imgs)
        return -fake_validity.mean()


class StyleGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_channels: int,
        n_style: int = 8,
        mapping_layers: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_style = n_style

        layers = []
        in_features = latent_dim
        for _ in range(mapping_layers):
            layers.extend(
                [
                    nn.Linear(in_features, latent_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_features = latent_dim
        self.mapping = nn.Sequential(*layers)

        self.style_layers = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(n_style)]
        )

        self.init_size = 4
        self.fc = nn.Linear(latent_dim, latent_dim * self.init_size * self.init_size)

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        latent_dim, latent_dim // 2, 4, 2, 1, bias=False
                    ),
                    nn.LeakyReLU(0.2),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        latent_dim // 2, latent_dim // 4, 4, 2, 1, bias=False
                    ),
                    nn.LeakyReLU(0.2),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        latent_dim // 4, latent_dim // 8, 4, 2, 1, bias=False
                    ),
                    nn.LeakyReLU(0.2),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        latent_dim // 8, img_channels, 4, 2, 1, bias=False
                    ),
                    nn.Tanh(),
                ),
            ]
        )

    def style_inject(self, w: torch.Tensor, layer_idx: int) -> nn.Module:
        return self.style_layers[layer_idx](w)

    def forward(
        self, z: torch.Tensor, styles: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if styles is None:
            w = self.mapping(z)
            styles = [self.style_layers[i](w) for i in range(self.n_style)]

        out = self.fc(styles[0])
        out = out.view(out.size(0), -1, self.init_size, self.init_size)

        for i, block in enumerate(self.conv_blocks[:-1]):
            out = block(out)
            if i < len(styles) - 1:
                style = styles[i + 1].unsqueeze(-1).unsqueeze(-1)
                out = out * (0.5 + 0.5 * style)

        out = self.conv_blocks[-1](out)
        return out


class StyleGANDiscriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,
        n_layers: int = 4,
    ):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class StyleGAN(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        img_channels: int = 3,
        n_style: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.generator = StyleGANGenerator(latent_dim, img_channels, n_style)
        self.discriminator = StyleGANDiscriminator(img_channels)

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)


class BigGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_channels: int,
        num_classes: int,
        channels: int = 96,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim * 2, channels * 16)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.res_blocks = nn.ModuleList(
            [
                self._res_block(channels * 8, channels * 8),
                self._res_block(channels * 8, channels * 4),
                self._res_block(channels * 4, channels * 2),
                self._res_block(channels * 2, channels),
            ]
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels, img_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def _res_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_embed = self.embed(y)
        h = torch.cat([z, y_embed], dim=-1)
        h = self.fc(h)
        h = h.view(h.size(0), -1, 4, 4)

        for block in self.res_blocks:
            h = block(h)
            h = self.upsample(h)

        return self.to_rgb(h)


class BigGANDiscriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_classes: int,
        channels: int = 96,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.downsample = nn.AvgPool2d(2)

        self.res_blocks = nn.ModuleList(
            [
                self._res_block(channels, channels * 2),
                self._res_block(channels * 2, channels * 4),
                self._res_block(channels * 4, channels * 8),
                self._res_block(channels * 8, channels * 16),
            ]
        )

        self.embed = nn.Embedding(num_classes, channels * 16)
        self.fc = nn.Linear(channels * 16, 1)

    def _res_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)

        for block in self.res_blocks:
            h = block(h)
            h = self.downsample(h)

        h = h.view(h.size(0), -1)
        y_embed = self.embed(y)
        h = h * y_embed

        return self.fc(h)


class BigGAN(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        img_channels: int = 3,
        num_classes: int = 1000,
        channels: int = 96,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.generator = BigGANGenerator(
            latent_dim, img_channels, num_classes, channels
        )
        self.discriminator = BigGANDiscriminator(img_channels, num_classes, channels)

    def generate(
        self, num_samples: int, class_labels: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z, class_labels)


class StyleGAN2Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_channels: int,
        n_style: int = 8,
        mapping_layers: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_style = n_style

        layers = []
        in_features = latent_dim
        for _ in range(mapping_layers):
            layers.extend(
                [
                    nn.Linear(in_features, latent_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_features = latent_dim
        self.mapping = nn.Sequential(*layers)

        self.style_layers = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(n_style)]
        )

        self.init_size = 4
        self.fc = nn.Linear(latent_dim, latent_dim * self.init_size * self.init_size)

        self.conv_blocks = nn.ModuleList(
            [
                self._conv_block(latent_dim, latent_dim // 2),
                self._conv_block(latent_dim // 2, latent_dim // 4),
                self._conv_block(latent_dim // 4, latent_dim // 8),
                self._conv_block(latent_dim // 8, latent_dim // 16),
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(latent_dim // (2**i), img_channels, 1, 1, 0),
                    nn.Tanh(),
                )
                for i in range(n_style)
            ]
        )

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
        )

    def forward(
        self,
        z: torch.Tensor,
        styles: Optional[list] = None,
        noise: Optional[list] = None,
    ) -> torch.Tensor:
        if styles is None:
            w = self.mapping(z)
            styles = [self.style_layers[i](w) for i in range(self.n_style)]

        out = self.fc(styles[0])
        out = out.view(out.size(0), -1, self.init_size, self.init_size)

        for i, block in enumerate(self.conv_blocks):
            out = block(out)
            if i < len(styles) - 1:
                style = styles[i + 1].unsqueeze(-1).unsqueeze(-1)
                out = out * (0.5 + 0.5 * style)
            if i == len(self.conv_blocks) - 1:
                out = self.to_rgb[i](out)

        return out


class StyleGAN2Discriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,
    ):
        super().__init__()

        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_channels, 512, 1, 1, 0),
            nn.LeakyReLU(0.2),
        )

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
                for _ in range(4)
            ]
        )

        self.downsample = nn.AvgPool2d(2)

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)
        h = h.permute(0, 2, 3, 1)

        for block in self.conv_blocks:
            h = block(h)
            h = h.permute(0, 3, 1, 2)
            h = self.downsample(h)
            h = h.permute(0, 2, 3, 1)

        h = h.view(h.size(0), -1)
        return self.fc(h)


class StyleGAN2(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        img_channels: int = 3,
        n_style: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.generator = StyleGAN2Generator(latent_dim, img_channels, n_style)
        self.discriminator = StyleGAN2Discriminator(img_channels)

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)


class StyleGAN3Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_channels: int,
        n_style: int = 8,
        mapping_layers: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_style = n_style

        layers = []
        in_features = latent_dim
        for _ in range(mapping_layers):
            layers.extend(
                [
                    nn.Linear(in_features, latent_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_features = latent_dim
        self.mapping = nn.Sequential(*layers)

        self.const = nn.Parameter(torch.randn(1, latent_dim, 4, 4))

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(latent_dim, latent_dim, 3, 1, 1),
                nn.Conv2d(latent_dim, latent_dim, 3, 1, 1),
                nn.Conv2d(latent_dim, latent_dim // 2, 3, 1, 1),
                nn.Conv2d(latent_dim // 2, latent_dim // 2, 3, 1, 1),
                nn.Conv2d(latent_dim // 2, latent_dim // 4, 3, 1, 1),
                nn.Conv2d(latent_dim // 4, latent_dim // 4, 3, 1, 1),
                nn.Conv2d(latent_dim // 4, latent_dim // 8, 3, 1, 1),
                nn.Conv2d(latent_dim // 8, img_channels, 3, 1, 1),
            ]
        )

        self.noise_strength = nn.Parameter(torch.ones(n_style))

    def forward(self, z: torch.Tensor, styles: Optional[list] = None) -> torch.Tensor:
        if styles is None:
            w = self.mapping(z)
            styles = [w for _ in range(self.n_style)]

        batch_size = z.size(0)
        out = self.const.expand(batch_size, -1, -1, -1)

        for i, conv in enumerate(self.convs):
            if i < len(styles):
                style = styles[i].unsqueeze(-1).unsqueeze(-1)
                out = out * (0.5 + 0.5 * style)

            if i < len(self.convs) - 1:
                out = conv(out)
                out = F.leaky_relu(out, 0.2)

            if i in [1, 3, 5]:
                out = F.interpolate(
                    out, scale_factor=2, mode="bilinear", align_corners=False
                )

        return torch.tanh(out)


class StyleGAN3Discriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,
    ):
        super().__init__()

        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_channels, 512, 1, 1, 0),
            nn.LeakyReLU(0.2),
        )

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
                for _ in range(8)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)

        for conv in self.convs:
            h = conv(h)
            h = F.avg_pool2d(h, 2)

        h = h.view(h.size(0), -1)
        return self.fc(h)


class StyleGAN3(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        img_channels: int = 3,
        n_style: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.generator = StyleGAN3Generator(latent_dim, img_channels, n_style)
        self.discriminator = StyleGAN3Discriminator(img_channels)

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)
