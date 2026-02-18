import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class WaveUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_sources: int = 2,
        channels: List[int] = [32, 64, 128, 256, 512],
        kernel_size: int = 15,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.channels = channels
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        in_ch = input_channels
        for i, out_ch in enumerate(channels):
            self.encoder.append(self._conv_block(in_ch, out_ch, kernel_size))
            if i < len(channels) - 1:
                self.skip_connections.append(
                    self._conv_block(out_ch, out_ch, kernel_size)
                )
            in_ch = out_ch

        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                channels[-1], channels[-1] * 2, kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(channels[-1] * 2) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(
                channels[-1] * 2, channels[-1], kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(channels[-1]) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )

        for i, (enc_ch, dec_ch) in enumerate(
            zip(reversed(channels), channels[1:] + [input_channels])
        ):
            self.decoder.append(self._conv_block(enc_ch * 2, dec_ch, kernel_size))

        self.output = nn.Conv1d(input_channels * 2, output_channels * num_sources, 1)

    def _conv_block(self, in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
        ]
        if self.batch_norm:
            layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        encoder_outs = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            if i < len(self.encoder) - 1:
                encoder_outs.append(x)
            x = F.avg_pool1d(x, 2)

        x = self.bottleneck(x)

        for i, dec in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
            skip_idx = len(self.skip_connections) - 1 - i
            if skip_idx >= 0:
                skip = self.skip_connections[skip_idx](encoder_outs[-(i + 1)])
                x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.output(x)
        sources = torch.split(x, x.size(1) // self.num_sources, dim=1)
        return sources


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class TasNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_sources: int = 2,
        hidden_channels: int = 512,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sources = num_sources

        self.input_conv = nn.Conv1d(input_channels, hidden_channels, 1)

        self.separator = nn.ModuleList()
        for _ in range(num_layers):
            self.separator.append(
                nn.ModuleDict(
                    {
                        "tconv": nn.ConvTranspose1d(
                            hidden_channels,
                            hidden_channels,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                        ),
                        "conv": ConvBlock(
                            hidden_channels, hidden_channels, kernel_size
                        ),
                        "norm": nn.LayerNorm(hidden_channels),
                    }
                )
            )

        self.output_conv = nn.Conv1d(hidden_channels, output_channels * num_sources, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.input_conv(x)
        x = x.transpose(1, 2)

        for layer in self.separator:
            residual = x
            x = layer["tconv"](x)
            x = x.transpose(1, 2)
            x = layer["norm"](x)
            x = x.transpose(1, 2)
            x = layer["conv"](x)
            x = x + residual
            x = self.dropout(x)

        x = x.transpose(1, 2)
        x = self.output_conv(x)
        sources = torch.split(x, x.size(1) // self.num_sources, dim=1)
        return sources


class DCCRNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2,
        use_batch_norm: bool = True,
        complex_axis: int = 0,
    ):
        super().__init__()
        self.complex_axis = complex_axis
        self.conv_real = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        if use_batch_norm:
            self.bn_real = nn.BatchNorm2d(out_channels)
            self.bn_imag = nn.BatchNorm2d(out_channels)
        else:
            self.bn_real = nn.Identity()
            self.bn_imag = nn.Identity()
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = torch.chunk(x, 2, dim=1)
        real = self.activation(self.bn_real(self.conv_real(real)))
        imag = self.activation(self.bn_imag(self.conv_imag(imag)))
        return real, imag


class DCCRN(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 5,
        use_batch_norm: bool = True,
        num_sources: int = 2,
    ):
        super().__init__()
        self.num_sources = num_sources

        self.encoder = nn.ModuleList()
        channels = [input_channels]
        for i in range(num_layers):
            in_ch = channels[-1]
            out_ch = hidden_channels * (2 ** min(i, 2))
            self.encoder.append(
                DCCRNBlock(in_ch, out_ch, kernel_size, use_batch_norm=use_batch_norm)
            )
            channels.append(out_ch)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                channels[-1] * 2,
                channels[-1] * 2,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(channels[-1] * 2) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            in_ch = channels[-(i + 1)]
            out_ch = channels[-(i + 2)] if i < num_layers - 1 else input_channels
            self.decoder.append(
                nn.ConvTranspose2d(
                    in_ch * 2,
                    out_ch * 2,
                    kernel_size,
                    stride=2,
                    padding=2,
                    output_padding=1,
                )
            )

        self.output = nn.Conv2d(input_channels * 4, num_sources * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_outs = []
        for enc in self.encoder:
            x = torch.cat(enc(x), dim=1)
            encoder_outs.append(x)

        x = self.bottleneck(x)

        for i, dec in enumerate(self.decoder):
            skip_idx = len(self.decoder) - 1 - i
            if skip_idx < len(encoder_outs):
                x = x + encoder_outs[skip_idx]
            x = dec(x)

        x = self.output(x)
        sources = torch.split(x, 2, dim=1)
        return sources


class DualPathBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        intra_channels: int,
        inter_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.intra_block = nn.Sequential(
            nn.Conv1d(in_channels, intra_channels, 1),
            nn.Conv1d(
                intra_channels,
                intra_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=intra_channels,
            ),
            nn.BatchNorm1d(intra_channels),
            nn.PReLU(),
            nn.Conv1d(intra_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
        )
        self.inter_block = nn.Sequential(
            nn.Conv1d(in_channels, inter_channels, 1),
            nn.BatchNorm1d(inter_channels),
            nn.PReLU(),
            nn.Conv1d(inter_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
        )
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.intra_block(x)
        x = x + residual
        x = self.inter_block(x)
        x = x + residual
        return self.activation(x)


class DPRNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 64,
        hidden_channels: int = 128,
        num_layers: int = 4,
        chunk_size: int = 100,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.layers = nn.ModuleList(
            [
                DualPathBlock(input_channels, hidden_channels, hidden_channels)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        chunk_count = length // self.chunk_size

        x = x[:, :, : chunk_count * self.chunk_size]
        x = x.view(batch_size, channels, chunk_count, self.chunk_size)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size * chunk_count, channels, self.chunk_size)

        for layer in self.layers(x):
            x = layer(x)

        x = x.contiguous().view(batch_size, chunk_count, channels, self.chunk_size)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, channels, chunk_count * self.chunk_size)
        return x


class SourceSeparator(nn.Module):
    def __init__(
        self,
        model_type: str = "waveunet",
        input_channels: int = 1,
        num_sources: int = 2,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__()
        self.model_type = model_type
        self.num_sources = num_sources
        self.sample_rate = sample_rate

        if model_type.lower() == "waveunet":
            self.model = WaveUNet(
                input_channels=input_channels,
                output_channels=input_channels,
                num_sources=num_sources,
                **kwargs,
            )
        elif model_type.lower() == "tasnet":
            self.model = TasNet(
                input_channels=input_channels,
                output_channels=input_channels,
                num_sources=num_sources,
                **kwargs,
            )
        elif model_type.lower() == "dccrn":
            self.model = DCCRN(
                input_channels=2,
                num_sources=num_sources,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.model(x)

    def separate(self, x: torch.Tensor) -> torch.Tensor:
        sources = self.forward(x)
        return torch.stack(sources, dim=1)


__all__ = [
    "WaveUNet",
    "TasNet",
    "DCCRN",
    "DPRNN",
    "SourceSeparator",
]
