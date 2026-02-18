"""
Adversarial Modality Alignment for fishstick

This module provides adversarial learning techniques for modality alignment:
- Domain adversarial training
- Cycle-consistent alignment
- Gradient reversal for modality alignment
"""

from typing import Optional, List, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GradientReversal(nn.Module):
    """Gradient reversal layer for adversarial domain adaptation."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return self._gradient_reversal(x, self.lambda_)

    @staticmethod
    def _gradient_reversal(x: Tensor, lambda_: float) -> Tensor:
        return x


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 2))
        self.discriminator = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.discriminator(x)


class AdversarialModalityAligner(nn.Module):
    """Adversarial modality alignment using gradient reversal."""

    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        classifier: nn.Module,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        lambda_adversarial: float = 1.0,
    ):
        super().__init__()
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.classifier = classifier
        self.gradient_reversal = GradientReversal(lambda_adversarial)

        self.domain_discriminator = DomainDiscriminator(embed_dim, hidden_dim)

    def forward(
        self,
        source_data: Tensor,
        target_data: Tensor,
        return_features: bool = False,
    ):
        source_features = self.source_encoder(source_data)
        target_features = self.target_encoder(target_data)

        source_class = self.classifier(source_features)
        target_class = self.classifier(target_features)

        source_aligned = self.gradient_reversal(source_features)
        target_aligned = self.gradient_reversal(target_features)

        source_domain = self.domain_discriminator(source_aligned)
        target_domain = self.domain_discriminator(target_aligned)

        if return_features:
            return source_class, target_class, source_domain, target_domain, source_features, target_features

        return source_class, target_class, source_domain, target_domain


class CycleConsistentAlignment(nn.Module):
    """Cycle-consistent modality alignment."""

    def __init__(
        self,
        encoder_a: nn.Module,
        encoder_b: nn.Module,
        decoder_a: nn.Module,
        decoder_b: nn.Module,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.decoder_a = decoder_a
        self.decoder_b = decoder_b

        self.discriminator_a = DomainDiscriminator(embed_dim)
        self.discriminator_b = DomainDiscriminator(embed_dim)

    def forward(
        self,
        data_a: Tensor,
        data_b: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        embed_a = self.encoder_a(data_a)
        embed_b = self.encoder_b(data_b)

        recon_a = self.decoder_a(embed_b)
        recon_b = self.decoder_b(embed_a)

        domain_a = self.discriminator_a(embed_a)
        domain_b = self.discriminator_b(embed_b)

        return embed_a, embed_b, recon_a, recon_b, domain_a, domain_b


class MMDAlignment(nn.Module):
    """Maximum Mean Discrepancy for modality alignment."""

    def __init__(
        self,
        kernel_type: str = "rbf",
        kernel_mul: float = 2.0,
        num_kernels: int = 5,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.num_kernels = num_kernels

    def gaussian_kernel(
        self,
        source: Tensor,
        target: Tensor,
        kernel_mul: float = 2.0,
        num_kernels: int = 5,
        fix_sigma: Optional[float] = None,
    ) -> Tensor:
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(dim=2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)

        bandwidth /= (kernel_mul ** 2)
        bandwidth_list = [bandwidth * (i + 1) for i in range(num_kernels)]

        kernel_val = [torch.exp(-L2_distance / (bw + 1e-6)) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            num_kernels=self.num_kernels
        )

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss


class CoralAlignment(nn.Module):
    """Correlation Alignment for modality alignment."""

    def __init__(self):
        super().__init__()

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)

        source_centered = source - source_mean
        target_centered = target - target_mean

        source_cov = (source_centered.T @ source_centered) / (source.size(0) - 1)
        target_cov = (target_centered.T @ target_centered) / (target.size(0) - 1)

        loss = torch.sum((source_cov - target_cov) ** 2)
        return loss


class MultiModalAdversarialNetwork(nn.Module):
    """Complete multi-modal adversarial network."""

    def __init__(
        self,
        modality_dims: dict,
        common_dim: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.common_dim = common_dim

        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, common_dim),
                nn.ReLU(),
                nn.Linear(common_dim, common_dim),
            )
            for name, dim in modality_dims.items()
        })

        self.classifier = nn.Linear(common_dim, num_classes)

        self.discriminators = nn.ModuleDict({
            name: DomainDiscriminator(common_dim)
            for name in self.modalities
        })

        self.gradient_reversal = GradientReversal()

    def encode(self, modality: str, data: Tensor) -> Tensor:
        return self.encoders[modality](data)

    def forward(
        self,
        data_dict: dict,
        target_modality: Optional[str] = None,
    ):
        features = {}
        class_preds = {}

        for modality, data in data_dict.items():
            features[modality] = self.encode(modality, data)
            class_preds[modality] = self.classifier(features[modality])

        domain_preds = {}
        for modality in self.modalities:
            aligned = self.gradient_reversal(features[modality])
            domain_preds[modality] = self.discriminators[modality](aligned)

        return features, class_preds, domain_preds


class DANNLoss(nn.Module):
    """Domain Adversarial Neural Network loss."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        class_preds: Tensor,
        class_labels: Tensor,
        domain_preds: Tensor,
        domain_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        class_loss = self.criterion(class_preds, class_labels)
       .criterion(domain_preds, domain_labels)
        return class_loss domain_loss = self, domain_loss


class InstaBoost(nn.Module):
    """Instance boosting for modality alignment."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.encoder = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        pseudo_labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        source_enc = self.encoder(source)
        target_enc = self.encoder(target)

        source_pred = self.classifier(source_enc)
        target_pred = self.classifier(target_enc)

        if pseudo_labels is not None:
            target_loss = F.cross_entropy(target_pred, pseudo_labels)
        else:
            target_loss = None

        return source_pred, target_pred, target_loss


def create_alignment_module(
    alignment_type: str = "adversarial",
    **kwargs,
) -> nn.Module:
    """Factory function to create alignment modules."""
    if alignment_type == "adversarial":
        return AdversarialModalityAligner(**kwargs)
    elif alignment_type == "cycle":
        return CycleConsistentAlignment(**kwargs)
    elif alignment_type == "mmd":
        return MMDAlignment(**kwargs)
    elif alignment_type == "coral":
        return CoralAlignment(**kwargs)
    elif alignment_type == "multimodal_adversarial":
        return MultiModalAdversarialNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown alignment type: {alignment_type}")
