import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass


@dataclass
class DANNConfig:
    """Configuration for Domain-Adversarial Neural Network (DANN)."""

    hidden_dim: int = 256
    domain_classifier_hidden: int = 128
    gradient_reversal_lambda: float = 1.0
    learning_rate: float = 1e-3


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for DANN."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer (GRL) for domain adversarial training."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DANNClassifier(nn.Module):
    """Domain-Adversarial Neural Network (DANN) for domain adaptation."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        label_classifier: nn.Module,
        domain_classifier: nn.Module,
        config: Optional[DANNConfig] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.label_classifier = label_classifier
        self.domain_classifier = domain_classifier
        self.grl = GradientReversalLayer(
            config.gradient_reversal_lambda if config else 1.0
        )
        self.config = config or DANNConfig()

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        class_logits = self.label_classifier(features)

        if return_features:
            return {"features": features, "class_logits": class_logits}

        return {"class_logits": class_logits}

    def compute_domain_loss(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        source_domain_labels = torch.zeros(
            source_features.size(0), dtype=torch.long, device=source_features.device
        )
        target_domain_labels = torch.ones(
            target_features.size(0), dtype=torch.long, device=target_features.device
        )

        source_domain_logits = self.domain_classifier(self.grl(source_features))
        target_domain_logits = self.domain_classifier(self.grl(target_features))

        source_domain_loss = F.cross_entropy(source_domain_logits, source_domain_labels)
        target_domain_loss = F.cross_entropy(target_domain_logits, target_domain_labels)

        return source_domain_loss + target_domain_loss


@dataclass
class ADDAConfig:
    """Configuration for Adversarial Discriminative Domain Adaptation (ADDA)."""

    hidden_dim: int = 256
    discriminator_hidden: int = 128
    learning_rate: float = 1e-4
    momentum: float = 0.999


class ADDAModel(nn.Module):
    """Adversarial Discriminative Domain Adaptation (ADDA)."""

    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        classifier: nn.Module,
        discriminator: nn.Module,
        config: Optional[ADDAConfig] = None,
    ):
        super().__init__()
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.classifier = classifier
        self.discriminator = discriminator
        self.config = config or ADDAConfig()
        self._init_target_encoder()

    def _init_target_encoder(self):
        """Initialize target encoder with source encoder weights."""
        self.target_encoder.load_state_dict(self.source_encoder.state_dict())

    def forward(
        self, x: torch.Tensor, use_target: bool = True, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        encoder = self.target_encoder if use_target else self.source_encoder
        features = encoder(x)
        logits = self.classifier(features)

        if return_features:
            return {"features": features, "logits": logits}
        return {"logits": logits}

    def compute_adversarial_loss(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        source_domain_score = self.discriminator(source_features)
        target_domain_score = self.discriminator(target_features)

        source_labels = torch.zeros(
            source_features.size(0), device=source_features.device
        )
        target_labels = torch.ones(
            target_features.size(0), device=target_features.device
        )

        source_loss = F.binary_cross_entropy_with_logits(
            source_domain_score, source_labels
        )
        target_loss = F.binary_cross_entropy_with_logits(
            target_domain_score, target_labels
        )

        return source_loss + target_loss


@dataclass
class MCDConfig:
    """Configuration for Minimum Classifier Discrepancy (MCD)."""

    num_iterations: int = 4
    hidden_dim: int = 256
    learning_rate: float = 1e-4


class MCDModel(nn.Module):
    """Minimum Classifier Discrepancy (MCD) for domain adaptation."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifiers: List[nn.Module],
        config: Optional[MCDConfig] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifiers = nn.ModuleList(classifiers)
        self.config = config or MCDConfig()

    def forward(
        self, x: torch.Tensor, return_all_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        logits_list = [cls(features) for cls in self.classifiers]

        if return_all_logits:
            return {"features": features, "logits_list": logits_list}

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        return {"logits": avg_logits}

    def compute_discrepancy_loss(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Compute discrepancy loss between classifier predictions."""
        logits_stack = torch.stack(logits_list, dim=0)
        mean_logits = logits_stack.mean(dim=0)
        discrepancy = torch.mean(
            F.softmax(logits_stack, dim=-1).sum(dim=0) / len(logits_list)
        )
        return discrepancy


@dataclass
class CycleGANLikeConfig:
    """Configuration for CycleGAN-like domain adaptation."""

    hidden_dim: int = 256
    latent_dim: int = 128
    learning_rate: float = 2e-4
    cycle_weight: float = 10.0
    identity_weight: float = 5.0


class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """Generator for CycleGAN-like domain adaptation."""

    def __init__(
        self, input_dim: int = 3, hidden_dim: int = 256, num_residual_blocks: int = 9
    ):
        super().__init__()

        initial_conv = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, hidden_dim, kernel_size=7, padding=0),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        self.initial_conv = nn.Sequential(*initial_conv)

        downsampling = []
        in_dim = hidden_dim
        for _ in range(2):
            out_dim = in_dim * 2
            downsampling.extend(
                [
                    nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_dim = out_dim
        self.downsampling = nn.Sequential(*downsampling)

        residual_blocks = [ResidualBlock(in_dim) for _ in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*residual_blocks)

        upsampling = []
        for _ in range(2):
            out_dim = in_dim // 2
            upsampling.extend(
                [
                    nn.ConvTranspose2d(
                        in_dim,
                        out_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.InstanceNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_dim = out_dim
        self.upsampling = nn.Sequential(*upsampling)

        output_conv = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_dim, input_dim, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.output_conv = nn.Sequential(*output_conv)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        x = self.output_conv(x)
        return x


class CycleGANDiscriminator(nn.Module):
    """PatchGAN discriminator for CycleGAN."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()

        layers = [
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = [hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8]
        for i, out_ch in enumerate(channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(channels[i], out_ch, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        layers.append(nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CycleGANLikeModel(nn.Module):
    """CycleGAN-like domain adaptation model."""

    def __init__(self, config: Optional[CycleGANLikeConfig] = None):
        super().__init__()
        self.config = config or CycleGANLikeConfig()

        self.gen_a2b = CycleGANGenerator(hidden_dim=self.config.hidden_dim)
        self.gen_b2a = CycleGANGenerator(hidden_dim=self.config.hidden_dim)

        self.disc_a = CycleGANDiscriminator()
        self.disc_b = CycleGANDiscriminator()

    def forward(self, x: torch.Tensor, direction: str = "a2b") -> torch.Tensor:
        if direction == "a2b":
            return self.gen_a2b(x)
        return self.gen_b2a(x)

    def compute_cycle_consistency_loss(
        self, original_a: torch.Tensor, original_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        recon_a = self.gen_b2a(self.gen_a2b(original_a))
        recon_b = self.gen_a2b(self.gen_b2a(original_b))

        cycle_loss_a = F.l1_loss(recon_a, original_a)
        cycle_loss_b = F.l1_loss(recon_b, original_b)

        return cycle_loss_a, cycle_loss_b

    def compute_identity_loss(
        self, original_a: torch.Tensor, original_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        identity_a = self.gen_b2a(original_a)
        identity_b = self.gen_a2b(original_b)

        identity_loss_a = F.l1_loss(identity_a, original_a)
        identity_loss_b = F.l1_loss(identity_b, original_b)

        return identity_loss_a, identity_loss_b


def create_dann(
    feature_extractor: nn.Module, num_classes: int, config: Optional[DANNConfig] = None
) -> DANNClassifier:
    """Factory function to create a DANN model."""
    config = config or DANNConfig()

    label_classifier = nn.Sequential(
        nn.Linear(512, config.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(config.hidden_dim, num_classes),
    )

    domain_classifier = nn.Sequential(
        nn.Linear(512, config.domain_classifier_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(config.domain_classifier_hidden, 2),
    )

    return DANNClassifier(
        feature_extractor=feature_extractor,
        label_classifier=label_classifier,
        domain_classifier=domain_classifier,
        config=config,
    )


def create_adda(
    source_encoder: nn.Module,
    classifier: nn.Module,
    config: Optional[ADDAConfig] = None,
) -> ADDAModel:
    """Factory function to create an ADDA model."""
    config = config or ADDAConfig()

    target_encoder = type(source_encoder)()
    target_encoder.load_state_dict(source_encoder.state_dict())

    discriminator = nn.Sequential(
        nn.Linear(512, config.discriminator_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(config.discriminator_hidden, 1),
    )

    return ADDAModel(
        source_encoder=source_encoder,
        target_encoder=target_encoder,
        classifier=classifier,
        discriminator=discriminator,
        config=config,
    )


def create_mcd(
    feature_extractor: nn.Module,
    num_classes: int,
    num_classifiers: int = 2,
    config: Optional[MCDConfig] = None,
) -> MCDModel:
    """Factory function to create an MCD model."""
    config = config or MCDConfig()

    def create_classifier():
        return nn.Sequential(
            nn.Linear(512, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_dim, num_classes),
        )

    classifiers = [create_classifier() for _ in range(num_classifiers)]

    return MCDModel(
        feature_extractor=feature_extractor, classifiers=classifiers, config=config
    )
