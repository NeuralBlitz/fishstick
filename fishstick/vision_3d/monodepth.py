"""
Monodepth: Monocular Depth Estimation

Implementation based on Monodepth2 paper.
"""

from typing import Tuple, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MonodepthEncoder(nn.Module):
    """
    Monodepth encoder network.
    """

    def __init__(self, num_layers: int = 18, pretrained: bool = False):
        super().__init__()

        import torchvision.models as models

        if num_layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif num_layers == 34:
            resnet = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported num_layers: {num_layers}")

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: Tensor) -> List[Tensor]:
        """Extract multi-scale features."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = [
            self.layer1(x),
            self.layer2(x),
            self.layer3(x),
            self.layer4(x),
        ]

        return features


class MonodepthDecoder(nn.Module):
    """
    Monodepth decoder with side outputs.
    """

    def __init__(self, num_ch_enc: List[int], num_output_channels: int = 1):
        super().__init__()

        self.num_output_channels = num_output_channels

        self.convs = nn.ModuleDict()
        self.upconvs = nn.ModuleDict()

        for i in range(4, 0, -1):
            skip_idx = i - 1

            in_ch = num_ch_enc[i]
            out_ch = num_ch_enc[skip_idx]

            self.upconvs[f"upconv_{i}"] = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

            self.convs[f"conv_{i}"] = nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.side_convs = nn.ModuleDict()
        for i in range(4):
            self.side_convs[f"side_conv_{i}"] = nn.Conv2d(
                num_ch_enc[i + 1], num_output_channels, 1
            )

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            features: Encoder features

        Returns:
            outputs: Multi-scale depth predictions
        """
        outputs = []

        x = features[-1]

        for i in range(4, 0, -1):
            x = self.upconvs[f"upconv_{i}"](x)

            if x.shape[2:] != features[i - 1].shape[2:]:
                x = F.interpolate(x, size=features[i - 1].shape[2:])

            x = torch.cat([x, features[i - 1]], dim=1)
            x = self.convs[f"conv_{i}"](x)

            outputs.append(self.side_convs[f"side_conv_{i - 1}"](x))

        return outputs


class DispResNet(nn.Module):
    """
    Disparity ResNet - combines encoder and decoder.
    """

    def __init__(self, num_layers: int = 18, num_output_channels: int = 1):
        super().__init__()

        self.encoder = MonodepthEncoder(num_layers)
        self.decoder = MonodepthDecoder([64, 64, 128, 256, 512], num_output_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Predict disparities at multiple scales."""
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs


class MonodepthModel(nn.Module):
    """
    Complete Monodepth model.
    """

    def __init__(self, num_layers: int = 18, num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales

        self.models = nn.ModuleDict()

        self.models["encoder"] = MonodepthEncoder(num_layers)
        self.models["decoder"] = MonodepthDecoder(
            [64, 64, 128, 256, 512], num_output_channels=1
        )

    def forward(
        self,
        x: Tensor,
    ) -> List[Tensor]:
        """
        Args:
            x: Input images [B, 3, H, W]

        Returns:
            disparities: List of disparities at different scales
        """
        features = self.models["encoder"](x)
        disparities = self.models["decoder"](features)

        return disparities

    def predict_disparity(self, x: Tensor) -> Tensor:
        """Get final disparity map."""
        disparities = self.forward(x)
        return disparities[0]
