"""
Depth Estimation Models

Monocular depth estimation architectures.
"""

from typing import Tuple, Optional, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DepthEncoder(nn.Module):
    """
    Depth Encoder - extracts features from input image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        out_ch = base_channels
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else base_channels * (2 ** (i - 1))
            out_ch = base_channels * (2**i)

            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2 if i > 0 else 1, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            features: Multi-scale features
        """
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class DepthDecoder(nn.Module):
    """
    Depth Decoder - upsamples and predicts depth.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        num_output_channels: int = 1,
        use_skips: bool = True,
    ):
        super().__init__()

        self.use_skips = use_skips
        self.up_convs = nn.ModuleList()
        self.predict_depth = nn.ModuleList()

        channels = encoder_channels[::-1]

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            self.up_convs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

            self.predict_depth.append(nn.Conv2d(out_ch, num_output_channels, 1))

    def forward(
        self,
        encoder_features: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            encoder_features: Features from encoder

        Returns:
            upsampled_features: Upsampled features
            depth_predictions: Depth maps at multiple scales
        """
        x = encoder_features[-1]
        predictions = []

        for i, (up_conv, depth_conv) in enumerate(
            zip(self.up_convs, self.predict_depth)
        ):
            x = up_conv(x)

            if self.use_skips and i < len(encoder_features) - 1:
                skip = encoder_features[-(i + 2)]
                x = x + skip

            predictions.append(depth_conv(x))

        return x, predictions


class DepthDecoderUpconv(nn.Module):
    """
    Depth Decoder with upconv blocks.
    """

    def __init__(
        self,
        num_layers: int = 4,
        base_channels: int = 256,
    ):
        super().__init__()

        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = base_channels * (2**i) if i > 0 else base_channels
            out_ch = (
                base_channels * (2 ** (i + 1)) if i < num_layers - 1 else base_channels
            )

            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))

            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        self.depth_conv = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict depth map."""
        for upconv, conv in zip(self.upconvs, self.convs):
            x = upconv(x)
            x = conv(x)

        return self.depth_conv(x)


class MonocularDepthEstimator(nn.Module):
    """
    Complete monocular depth estimation model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        self.encoder = DepthEncoder(in_channels, base_channels)
        self.decoder = DepthDecoder(
            [base_channels * (2**i) for i in range(4)],
            num_output_channels=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            depth: Predicted depth [B, 1, H, W]
        """
        features = self.encoder(x)
        _, predictions = self.decoder(features)

        return predictions[-1]


class ResNetDepthEncoder(nn.Module):
    """
    ResNet-based depth encoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()

        import torchvision.models as models

        resnet = models.resnet18(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: Input [B, 3, H, W]

        Returns:
            Multi-scale features
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return [c1, c2, c3, c4]
