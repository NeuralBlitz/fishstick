"""
Medical Image Segmentation Models
"""

import torch
from torch import nn, Tensor


class UNet3D(nn.Module):
    """3D U-Net for volumetric medical image segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: list = [32, 64, 128, 256],
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature * 2, feature))

        self.bottleneck = self._block(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []

        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)


class VNet(nn.Module):
    """V-Net for volumetric medical image segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: list = [16, 32, 64, 128],
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in features:
            self.encoder.append(self._down_block(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.decoder.append(self._up_block(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _down_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )

    def _up_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []

        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = torch.nn.functional.max_pool3d(x, kernel_size=2, stride=2)

        skip_connections = skip_connections[::-1]

        for idx, decode in enumerate(self.decoder):
            x = decode(x)
            if idx < len(skip_connections):
                x = torch.cat([x, skip_connections[idx]], dim=1)

        return self.final_conv(x)


class AttentionUNet3D(nn.Module):
    """3D Attention U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: list = [32, 64, 128, 256],
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        self.attention_blocks = nn.ModuleList()
        for feature in reversed(features):
            self.attention_blocks.append(AttentionBlock3D(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Simplified forward pass
        for encode in self.encoder:
            x = encode(x)

        return self.final_conv(x)


class AttentionBlock3D(nn.Module):
    """Attention block for 3D U-Net."""

    def __init__(self, F_g: int, F_l: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_l, kernel_size=1),
            nn.BatchNorm3d(F_l),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_l, kernel_size=1),
            nn.BatchNorm3d(F_l),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_l, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
