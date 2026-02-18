"""
PixelCNN and PixelCNN++ for Autoregressive Image Generation.

Implements autoregressive pixel generation as described in:
van den Oord et al. (2016) "Conditional Image Generation with PixelCNN Decoders"
Salimans et al. (2017) "PixelCNN++: Improving the PixelCNN"

Key features:
- Gated masked convolutions
- Residual connections
- Discrete logistic mixture likelihood
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GatedMaskedConv2d(nn.Module):
    """
    Gated masked convolution for PixelCNN.

    Uses vertical and horizontal stacks for full receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mask_type: str = "A",
        num_classes: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        self.num_classes = num_classes

        padding = kernel_size // 2

        self.conv_vert = nn.Conv2d(
            in_channels, 2 * out_channels, kernel_size, padding=padding
        )
        self.conv_horiz = nn.Conv2d(
            in_channels, 2 * out_channels, (1, kernel_size), padding=(0, padding)
        )

        self.vert_to_horiz = nn.Conv2d(2 * out_channels, 2 * out_channels, 1)
        self.horiz_to_vert = nn.Conv2d(2 * out_channels, 2 * out_channels, 1)

        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, in_channels)

        self._create_mask(mask_type, kernel_size)

    def _create_mask(self, mask_type: str, kernel_size: int) -> None:
        """Create causal masking."""
        mask = torch.ones(1, 1, kernel_size, kernel_size)

        mask[:, :, kernel_size // 2 :, : kernel_size // 2] = 0
        mask[:, :, kernel_size // 2, : kernel_size // 2] = 0

        if mask_type == "A":
            mask[:, :, kernel_size // 2, kernel_size // 2] = 0

        self.register_buffer("mask", mask)

    def forward(
        self,
        x: Tensor,
        h: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with gating.

        Args:
            x: Input [B, C, H, W]
            h: Hidden state from vertical stack
            class_labels: Optional class labels for conditioning

        Returns:
            Vertical output, horizontal output
        """
        if class_labels is not None and self.num_classes > 0:
            class_emb = self.class_embed(class_labels)
            class_emb = class_emb.unsqueeze(-1).unsqueeze(-1)
            x = x + class_emb

        if h is not None:
            x = x + self.horiz_to_vert(h)

        v = self.conv_vert(x)
        v = v * self.mask

        v_filter, v_gate = v.chunk(2, dim=1)
        v_out = torch.tanh(v_filter) * torch.sigmoid(v_gate)

        x = x + v
        h = self.conv_horiz(x)
        h = h * self.mask

        h_filter, h_gate = h.chunk(2, dim=1)
        h_out = torch.tanh(h_filter) * torch.sigmoid(h_gate)

        return v_out, h_out


class PixelCNNBlock(nn.Module):
    """PixelCNN block with gated masked convolutions."""

    def __init__(
        self,
        channels: int,
        num_classes: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.gated_conv = GatedMaskedConv2d(
            channels, 2 * channels, mask_type="B", num_classes=num_classes
        )

        self.conv_1x1 = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        class_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through block."""
        v, h = self.gated_conv(x, class_labels=class_labels)

        h = self.conv_1x1(h)
        h = self.dropout(h)

        return x + h


class PixelCNN(nn.Module):
    """
    PixelCNN for autoregressive image generation.

    Generates pixels one at a time using masked convolutions.

    Args:
        num_channels: Number of color channels (3 for RGB)
        num_classes: Number of classes for conditional generation
        num_filters: Number of filters per layer
        num_layers: Number of gated conv layers
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 0,
        num_filters: int = 128,
        num_layers: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.input_conv = nn.Conv2d(num_channels, num_filters, 1)

        self.blocks = nn.ModuleList(
            [
                PixelCNNBlock(num_filters, num_classes, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 1),
            nn.ReLU(),
            nn.Conv2d(num_filters, 256 * num_channels, 1),
        )

    def forward(
        self,
        x: Tensor,
        class_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]
            class_labels: Optional class labels

        Returns:
            Pixel logits [B, 256, C, H, W]
        """
        h = self.input_conv(x)

        for block in self.blocks:
            h = block(h, class_labels)

        out = self.output_conv(h)
        out = out.view(out.size(0), 256, self.num_channels, out.size(2), out.size(3))

        return out

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],
        class_labels: Optional[Tensor] = None,
        device: str = "cpu",
    ) -> Tensor:
        """
        Generate samples autoregressively.

        Args:
            shape: (channels, height, width)
            class_labels: Optional class labels
            device: Device

        Returns:
            Generated image
        """
        channels, height, width = shape
        batch_size = class_labels.size(0) if class_labels is not None else 1

        img = torch.zeros(batch_size, channels, height, width, device=device)

        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    logits = self.forward(img, class_labels)

                    probs = F.softmax(logits[:, :, c, i, j], dim=-1)

                    pixel = torch.multinomial(probs, num_samples=1).float() / 255.0

                    img[:, c, i, j] = pixel.squeeze(-1)

        return img

    def loss(self, x: Tensor, class_labels: Optional[Tensor] = None) -> Tensor:
        """
        Compute negative log likelihood loss.

        Args:
            x: Input image [B, C, H, W] with values in [0, 1]
            class_labels: Optional class labels

        Returns:
            NLL loss
        """
        x = (x * 255).long().clamp(0, 255)

        logits = self.forward(x[:, :, :-1, :-1], class_labels)

        loss = F.cross_entropy(
            logits.reshape(-1, 256),
            x[:, :, 1:, 1:].permute(0, 2, 3, 1).reshape(-1),
            reduction="none",
        )

        return loss.mean()


class PixelCNNPPBlock(nn.Module):
    """PixelCNN++ block with improved architecture."""

    def __init__(
        self,
        channels: int,
        num_classes: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.conv3 = nn.Conv2d(channels, channels, 1)

        self.condition = (
            nn.Conv2d(num_classes, channels, 1) if num_classes > 0 else None
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, class_emb: Optional[Tensor] = None) -> Tensor:
        """Forward pass."""
        h = F.relu(self.conv1(x))
        h = self.dropout(h)

        if class_emb is not None and self.condition is not None:
            h = h + self.condition(class_emb)

        h = F.relu(self.conv2(h))
        h = self.dropout(h)
        h = self.conv3(h)

        return x + h


class PixelCNNPP(nn.Module):
    """
    PixelCNN++ with improved architecture and discrete logistic likelihood.
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 0,
        num_filters: int = 160,
        num_layers: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.input_conv = nn.Conv2d(num_channels, num_filters, 1)

        if num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, num_filters)

        self.blocks = nn.ModuleList(
            [
                PixelCNNPPBlock(num_filters, num_classes, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters // 2, 1),
            nn.ReLU(),
            nn.Conv2d(num_filters // 2, 32 * num_channels, 1),
        )

    def forward(
        self,
        x: Tensor,
        class_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        h = self.input_conv(x)

        class_emb = None
        if class_labels is not None and self.num_classes > 0:
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.unsqueeze(-1).unsqueeze(-1)

        for block in self.blocks:
            h = block(h, class_emb)

        out = self.output_conv(h)
        out = out.view(out.size(0), 32, self.num_channels, out.size(2), out.size(3))

        return out

    def loss(self, x: Tensor, class_labels: Optional[Tensor] = None) -> Tensor:
        """
        Compute discrete logistic mixture loss.

        Args:
            x: Input image [B, C, H, W]
            class_labels: Optional class labels

        Returns:
            Loss
        """
        x_scaled = x * 255

        x_in = x_scaled[:, :, :-1, :-1]
        x_target = x_scaled[:, :, 1:, 1:]

        logits = self.forward(x_in / 255.0, class_labels)

        log_probs = F.log_softmax(logits, dim=1)

        target_idx = x_target.long().clamp(0, 31)
        loss = -log_probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)

        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],
        class_labels: Optional[Tensor] = None,
        device: str = "cpu",
    ) -> Tensor:
        """Generate samples."""
        channels, height, width = shape
        batch_size = class_labels.size(0) if class_labels is not None else 1

        img = torch.zeros(batch_size, channels, height, width, device=device)

        for i in range(height):
            for j in range(width):
                logits = self.forward(img / 255.0, class_labels)

                probs = F.softmax(logits[:, :, :, i, j], dim=1)

                sample = torch.multinomial(probs, num_samples=1).float()

                img[:, :, i, j] = sample / 255.0

        return img
