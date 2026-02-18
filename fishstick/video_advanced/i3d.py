from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class I3DConfig:
    num_classes: int = 400
    in_channels: int = 3
    base_channels: int = 64
    dropout: float = 0.5
    use_nonlocal: bool = True
    freeze_bn: bool = False


@dataclass
class SlowFastConfig:
    num_classes: int = 400
    alpha: int = 8
    beta: int = 8
    slow_channels: int = 64
    fast_channels: int = 8
    dropout: float = 0.5
    use_nonlocal: Tuple[bool, bool] = (True, True)


class NonLocal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 2,
        mode: str = "embedded_gaussian",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mode = mode

        self.inter_channels = in_channels // reduction

        self.g = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv3d(self.inter_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        g_x = self.g(x).view(B, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(B, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(B, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.inter_channels, T, H, W)
        W_y = self.W(y)
        z = W_y + x

        return z


class NonLocal1D(NonLocal):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        g_x = self.g(x).view(B, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(B, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(B, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.inter_channels, T)
        W_y = self.W(y)
        z = W_y + x

        return z


class NonLocal2D(NonLocal):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        g_x = self.g(x).view(B, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(B, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(B, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.inter_channels, H, W)
        W_y = self.W(y)
        z = W_y + x

        return z


class Conv3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        use_bn: bool = True,
        use_relu: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class I3DStem(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        use_bn: bool = True,
    ):
        super().__init__()
        self.conv1 = Conv3DBlock(
            in_channels,
            out_channels // 2,
            kernel_size=(3, 7, 7),
            stride=(2, 2, 2),
            padding=(1, 3, 3),
            use_bn=use_bn,
        )
        self.pool1 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.conv2 = Conv3DBlock(
            out_channels // 2,
            out_channels,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            use_bn=use_bn,
        )
        self.conv3 = Conv3DBlock(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            use_bn=use_bn,
        )
        self.pool2 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        return x


class I3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
        spatial_kernel: int = 3,
        stride: Tuple[int, int, int] = (1, 1, 1),
        use_nonlocal: bool = True,
        nonlocal_reduction: int = 2,
    ):
        super().__init__()
        self.use_nonlocal = use_nonlocal

        self.bottleneck = Conv3DBlock(
            in_channels,
            out_channels // 4,
            kernel_size=(1, 1, 1),
            stride=stride,
        )
        self.conv = Conv3DBlock(
            out_channels // 4,
            out_channels // 4,
            kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
            stride=(1, 1, 1),
            padding=(
                (temporal_kernel - 1) // 2,
                (spatial_kernel - 1) // 2,
                (spatial_kernel - 1) // 2,
            ),
        )
        self.expansion = Conv3DBlock(
            out_channels // 4,
            out_channels,
            kernel_size=(1, 1, 1),
            use_relu=False,
        )

        self.relu = nn.ReLU(inplace=True)

        if use_nonlocal:
            self.nl = NonLocal(out_channels, reduction=nonlocal_reduction)
        else:
            self.nl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.bottleneck(x)
        out = self.conv(out)
        out = self.expansion(out)

        if self.nl is not None:
            out = self.nl(out)

        out = out + identity
        out = self.relu(out)

        return out


class I3DHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class I3D(nn.Module):
    def __init__(
        self,
        num_classes: int = 400,
        in_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.5,
        use_nonlocal: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.stem = I3DStem(in_channels, base_channels)

        self.layer1 = self._make_layer(
            base_channels, base_channels * 2, 3, use_nonlocal
        )
        self.layer2 = self._make_layer(
            base_channels * 2, base_channels * 4, 4, use_nonlocal
        )
        self.layer3 = self._make_layer(
            base_channels * 4, base_channels * 8, 6, use_nonlocal
        )
        self.layer4 = self._make_layer(
            base_channels * 8, base_channels * 16, 3, use_nonlocal
        )

        self.head = I3DHead(base_channels * 16, num_classes, dropout)

        self._initialize_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        use_nonlocal: bool,
    ) -> nn.Module:
        layers = []
        layers.append(
            I3DBlock(
                in_channels,
                out_channels,
                temporal_kernel=3,
                spatial_kernel=3,
                stride=(1, 2, 2),
                use_nonlocal=use_nonlocal,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                I3DBlock(
                    out_channels,
                    out_channels,
                    temporal_kernel=3,
                    spatial_kernel=3,
                    stride=(1, 1, 1),
                    use_nonlocal=use_nonlocal,
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


class SlowFastStem(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        slow_channels: int = 64,
        fast_channels: int = 8,
    ):
        super().__init__()
        self.conv1_slow = Conv3DBlock(
            in_channels,
            slow_channels // 2,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
        )
        self.conv1_fast = Conv3DBlock(
            in_channels,
            fast_channels,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_slow = self.conv1_slow(x[:, :, ::4, :, :])
        x_fast = self.conv1_fast(x[:, :, ::2, :, :])
        return x_slow, x_fast


class SlowFastPathway(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
        spatial_kernel: int = 3,
        stride: Tuple[int, int, int] = (1, 1, 1),
        use_nonlocal: bool = True,
    ):
        super().__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels // 4, kernel_size=(1, 1, 1))
        self.conv2 = Conv3DBlock(
            out_channels // 4,
            out_channels // 4,
            kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
            stride=stride,
            padding=(
                (temporal_kernel - 1) // 2,
                (spatial_kernel - 1) // 2,
                (spatial_kernel - 1) // 2,
            ),
        )
        self.conv3 = Conv3DBlock(
            out_channels // 4, out_channels, kernel_size=(1, 1, 1), use_relu=False
        )

        self.relu = nn.ReLU(inplace=True)

        if use_nonlocal:
            self.nl = NonLocal(out_channels)
        else:
            self.nl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.nl is not None:
            x = self.nl(x)

        return x


class SlowFastBlock(nn.Module):
    def __init__(
        self,
        slow_in_channels: int,
        fast_in_channels: int,
        slow_out_channels: int,
        fast_out_channels: int,
        alpha: int = 8,
        use_nonlocal: bool = True,
    ):
        super().__init__()
        self.alpha = alpha

        self.slow_path = SlowFastPathway(
            slow_in_channels, slow_out_channels, use_nonlocal=use_nonlocal
        )
        self.fast_path = SlowFastPathway(
            fast_in_channels, fast_out_channels, use_nonlocal=use_nonlocal
        )

        self.lateral_conv = nn.Conv3d(
            slow_out_channels // alpha,
            slow_out_channels + fast_out_channels,
            kernel_size=(1, 1, 1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x_slow: torch.Tensor, x_fast: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_slow = self.slow_path(x_slow)
        x_fast = self.fast_path(x_fast)

        lateral = self.lateral_conv(x_slow[:, :: self.alpha, :, :])
        lateral_slow, lateral_fast = lateral.split(
            [x_slow.shape[1], x_fast.shape[1]], dim=1
        )

        x_fast = x_fast + lateral_fast
        x_slow = x_slow + lateral_slow

        x_slow = self.relu(x_slow)
        x_fast = self.relu(x_fast)

        return x_slow, x_fast


class SlowFastHead(nn.Module):
    def __init__(
        self,
        slow_channels: int,
        fast_channels: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.slow_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fast_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_slow = nn.Linear(slow_channels, num_classes)
        self.fc_fast = nn.Linear(fast_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_slow: torch.Tensor, x_fast: torch.Tensor) -> torch.Tensor:
        x_slow = self.slow_avgpool(x_slow).flatten(1)
        x_fast = self.fast_avgpool(x_fast).flatten(1)

        x_slow = self.dropout(x_slow)
        x_fast = self.dropout(x_fast)

        x_slow = self.fc_slow(x_slow)
        x_fast = self.fc_fast(x_fast)

        return x_slow + x_fast


class SlowFast(nn.Module):
    def __init__(
        self,
        num_classes: int = 400,
        alpha: int = 8,
        beta: int = 8,
        slow_channels: int = 64,
        fast_channels: int = 8,
        dropout: float = 0.5,
        use_nonlocal: Tuple[bool, bool] = (True, True),
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.stem = SlowFastStem(3, slow_channels, fast_channels)

        self.layer1 = SlowFastBlock(
            slow_channels // 2,
            fast_channels,
            slow_channels * 2,
            fast_channels * 2,
            alpha,
            use_nonlocal[0],
        )
        self.layer2 = SlowFastBlock(
            slow_channels * 2,
            fast_channels * 2,
            slow_channels * 4,
            fast_channels * 4,
            alpha,
            use_nonlocal[0],
        )
        self.layer3 = SlowFastBlock(
            slow_channels * 4,
            fast_channels * 4,
            slow_channels * 8,
            fast_channels * 8,
            alpha,
            use_nonlocal[1],
        )
        self.layer4 = SlowFastBlock(
            slow_channels * 8,
            fast_channels * 8,
            slow_channels * 16,
            fast_channels * 16,
            alpha,
            use_nonlocal[1],
        )

        self.head = SlowFastHead(
            slow_channels * 16, fast_channels * 16, num_classes, dropout
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_slow, x_fast = self.stem(x)

        x_slow, x_fast = self.layer1(x_slow, x_fast)
        x_slow, x_fast = self.layer2(x_slow, x_fast)
        x_slow, x_fast = self.layer3(x_slow, x_fast)
        x_slow, x_fast = self.layer4(x_slow, x_fast)

        output = self.head(x_slow, x_fast)
        return output

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_slow, x_fast = self.stem(x)

        x_slow, x_fast = self.layer1(x_slow, x_fast)
        x_slow, x_fast = self.layer2(x_slow, x_fast)
        x_slow, x_fast = self.layer3(x_slow, x_fast)
        x_slow, x_fast = self.layer4(x_slow, x_fast)

        x_slow = self.head.slow_avgpool(x_slow).flatten(1)
        x_fast = self.head.fast_avgpool(x_fast).flatten(1)

        return x_slow, x_fast


def build_i3d(config: Optional[I3DConfig] = None, **kwargs) -> I3D:
    if config is None:
        config = I3DConfig(**kwargs)
    return I3D(
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        dropout=config.dropout,
        use_nonlocal=config.use_nonlocal,
    )


def build_slowfast(config: Optional[SlowFastConfig] = None, **kwargs) -> SlowFast:
    if config is None:
        config = SlowFastConfig(**kwargs)
    return SlowFast(
        num_classes=config.num_classes,
        alpha=config.alpha,
        beta=config.beta,
        slow_channels=config.slow_channels,
        fast_channels=config.fast_channels,
        dropout=config.dropout,
        use_nonlocal=config.use_nonlocal,
    )
