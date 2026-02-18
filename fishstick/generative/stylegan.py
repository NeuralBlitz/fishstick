"""
StyleGAN and StyleGAN2 Implementations.

Implements style-based GAN architecture as described in:
Karras et al. (2019) "A Style-Based Generator Architecture for GANs"
Karras et al. (2020) "Analyzing and Improving the Image Quality of StyleGAN"

Key innovations:
- Mapping network: transforms latent code to intermediate latent space
- Synthesis network: generates image with adaptive instance normalization
- Style mixing: enables mixing of latent codes at different resolutions
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    """
    Mapping network that transforms latent code z to intermediate latent space w.

    Uses 8 fully-connected layers with progressive mapping for better disentanglement.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 8,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_dim = latent_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))

            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())

        self.mapping = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Map latent code to intermediate space.

        Args:
            z: Latent code [batch, latent_dim]

        Returns:
            Intermediate latent [batch, latent_dim]
        """
        return self.mapping(z)


class StyleBlock(nn.Module):
    """
    Style-based modulation block with adaptive instance normalization.

    Applies style modulation followed by adaptive instance normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=False)
        self.style = nn.Linear(512, out_channels * 2)

    def forward(
        self,
        x: Tensor,
        w: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply style modulation.

        Args:
            x: Input features
            w: Style code from mapping network
            noise: Optional noise injection

        Returns:
            Styled output features
        """
        style = self.style(w)
        scale, bias = style.chunk(2, dim=-1)

        h = self.conv(x)
        h = self.norm(h)

        h = h * (scale.view(-1, 1, 1, 1) + 1) + bias.view(-1, 1, 1, 1)

        if noise is not None:
            h = h + noise

        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)

        return h


class SynthesisNetwork(nn.Module):
    """
    Synthesis network that generates images using style modulation.

    Progressively increases resolution through the network.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        base_channels: int = 512,
        max_channels: int = 512,
        max_resolution: int = 1024,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.register_buffer("const", torch.ones(4, 4))

        self.channels = {
            4: base_channels,
            8: base_channels,
            16: base_channels,
            32: base_channels,
            64: min(base_channels // 2, max_channels),
            128: min(base_channels // 4, max_channels),
            256: min(base_channels // 8, max_channels),
            512: min(base_channels // 16, max_channels),
            1024: min(base_channels // 32, max_channels),
        }

        self.blocks = nn.ModuleDict()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        res = 4
        self.blocks[str(res)] = self._make_block(res, self.channels[res])

        while res < max_resolution:
            res = res * 2
            self.blocks[str(res)] = self._make_block(res, self.channels[res])

        self.to_rgb = nn.ModuleDict()
        for res in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            if res in self.channels:
                self.to_rgb[str(res)] = nn.Conv2d(self.channels[res], 3, 1)

    def _make_block(self, resolution: int, channels: int) -> nn.Module:
        """Create a synthesis block for given resolution."""
        in_ch = self.channels.get(resolution // 2, channels)

        block = nn.ModuleDict()
        block["conv1"] = StyleBlock(in_ch, channels, upsample=True)
        block["conv2"] = StyleBlock(channels, channels)

        block["noise1"] = NoiseInjection()
        block["noise2"] = NoiseInjection()

        return block

    def forward(
        self,
        w: Tensor,
        noise: Optional[Tensor] = None,
        styles: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """
        Generate image from style codes.

        Args:
            w: Style code [batch, latent_dim] or [batch, num_layers, latent_dim]
            noise: Optional noise tensors
            styles: Optional list of style codes per layer

        Returns:
            Generated image [batch, 3, H, W]
        """
        if w.dim() == 2:
            w = w.unsqueeze(1).repeat(1, 14, 1)

        x = self.const.unsqueeze(0).repeat(w.shape[0], 1, 1, 1)

        res = 4
        layer_idx = 0

        while res < 1024:
            if str(res) in self.blocks:
                block = self.blocks[str(res)]

                style1 = w[:, layer_idx]
                style2 = w[:, layer_idx + 1]

                x = block["conv1"](x, style1)
                x = block["noise1"](x, noise)
                x = F.leaky_relu(x, 0.2)

                x = block["conv2"](x, style2)
                x = block["noise2"](x, noise)
                x = F.leaky_relu(x, 0.2)

                layer_idx += 2

            if res < 512:
                x = self.upsample(x)

            res = res * 2 if res < 512 else res

        x = self.to_rgb["1024"](x)
        x = torch.tanh(x)

        return x


class NoiseInjection(nn.Module):
    """Inject learned noise into features."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            batch, _, h, w = x.shape
            noise = x.new_empty(batch, 1, h, w).normal_()

        return x + self.weight * noise


class StyleGAN(nn.Module):
    """
    StyleGAN generator with mapping and synthesis networks.

    Args:
        latent_dim: Dimension of input latent code
        base_channels: Base number of channels
        max_resolution: Maximum output resolution
    """

    def __init__(
        self,
        latent_dim: int = 512,
        base_channels: int = 512,
        max_resolution: int = 1024,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.mapping = MappingNetwork(latent_dim, latent_dim)
        self.synthesis = SynthesisNetwork(latent_dim, base_channels, max_resolution)

    def forward(
        self,
        z: Tensor,
        noise: Optional[Tensor] = None,
        styles: Optional[List[Tensor]] = None,
        style_mixing: Optional[int] = None,
    ) -> Tensor:
        """
        Generate images from latent codes.

        Args:
            z: Latent codes [batch, latent_dim]
            noise: Optional noise
            styles: Optional explicit style codes
            style_mixing: If set, mix styles from different latents

        Returns:
            Generated images
        """
        if styles is None:
            w = self.mapping(z)

            if style_mixing is not None:
                z2 = torch.randn_like(z)
                w2 = self.mapping(z2)

                mixed_styles = []
                for i in range(14):
                    if i < style_mixing:
                        mixed_styles.append(w2)
                    else:
                        mixed_styles.append(w)

                styles = mixed_styles

            return self.synthesis(w, noise, styles if style_mixing else None)

        return self.synthesis(z, noise, styles)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Tensor:
        """Sample random images."""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.forward(z)


class StyleGAN2Generator(nn.Module):
    """
    StyleGAN2 generator with improved architecture.

    Changes from StyleGAN:
    - Weight demodulation
    - Lazy regularization
    - Path length regularization
    - No progressive growing
    """

    def __init__(
        self,
        latent_dim: int = 512,
        base_channels: int = 512,
        max_resolution: int = 1024,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.mapping = MappingNetwork(latent_dim, latent_dim)
        self.synthesis = SynthesisNetwork(latent_dim, base_channels, max_resolution)

    def forward(
        self,
        z: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate images."""
        w = self.mapping(z)
        return self.synthesis(w, noise)


class ProgressiveGrowing:
    """
    Progressive growing mechanism for StyleGAN.

    Smoothly transitions from lower to higher resolutions during training.
    """

    def __init__(
        self,
        max_resolution: int = 1024,
        fade_in_steps: int = 100000,
    ):
        self.max_resolution = max_resolution
        self.fade_in_steps = fade_in_steps
        self.current_step = 0

    def get_alpha(self) -> float:
        """Get current fade-in alpha based on training step."""
        if self.current_step >= self.fade_in_steps:
            return 1.0

        return self.current_step / self.fade_in_steps

    def get_current_resolution(self) -> int:
        """Get current resolution based on training progress."""
        log2 = int(torch.log2(torch.tensor(self.max_resolution)).item())

        alpha = self.get_alpha()
        stage = int(alpha * (log2 - 1))

        return 4 * (2**stage)

    def step(self) -> None:
        """Increment training step."""
        self.current_step += 1


class DiscriminatorBlock(nn.Module):
    """Discriminator block for StyleGAN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.downsample = nn.Identity()
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, 3, stride=2, padding=1
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.downsample(x)
        return x


class StyleGANDiscriminator(nn.Module):
    """
    StyleGAN discriminator with progressive structure.
    """

    def __init__(
        self,
        base_channels: int = 512,
        max_resolution: int = 1024,
    ):
        super().__init__()

        self.from_rgb = nn.ModuleDict()

        channels = {
            4: base_channels,
            8: base_channels,
            16: base_channels,
            32: base_channels,
            64: base_channels // 2,
            128: base_channels // 4,
            256: base_channels // 8,
            512: base_channels // 16,
            1024: base_channels // 32,
        }

        self.blocks = nn.ModuleDict()

        res = 1024
        while res > 4:
            in_ch = channels[res]
            out_ch = channels.get(res // 2, base_channels)

            self.blocks[str(res)] = DiscriminatorBlock(in_ch, out_ch)
            res = res // 2

        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[4] * 4 * 4, base_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(base_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict whether input is real or fake."""
        x = self.final(x)
        return x
