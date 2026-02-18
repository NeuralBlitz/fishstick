"""
Adversarial Domain Adaptation Module for Fishstick.

This module provides adversarial domain adaptation methods including DANN (Domain-Adversarial
Neural Network), ADDA (Adversarial Discriminative Domain Adaptation), MCDAN (Multi-Channel
Domain Adversarial Neural Network), and CDAN (Conditional Domain Adversarial Network).

Example:
    >>> from fishstick.domain_adaptation.adversarial import DANN
    >>> dann = DANN(feature_dim=512, num_classes=10)
    >>> features, domain_preds, class_preds = dann(x, alpha=1.0)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module

T = TypeVar("T")


class GradientReversalLayer(Function):
    """Gradient Reversal Layer for adversarial domain adaptation.

    Forward pass: identity transformation
    Backward pass: multiplies gradient by -lambda (reverses gradient)

    Reference:
        Ganin et al. "Domain-Adversarial Training of Neural Networks" JMLR 2016
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversal(Module):
    """Wrapper module for Gradient Reversal Layer (GRL).

    Args:
        lambda_init: Initial value for gradient reversal coefficient.
            Can be dynamically updated during training via `set_lambda()`.

    Example:
        >>> grl = GradientReversal(lambda_init=0.5)
        >>> grl.set_lambda(1.0)  # Increase lambda during training
        >>> reversed_features = grl(features)
    """

    def __init__(self, lambda_init: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_init

    def set_lambda(self, lambda_: float) -> None:
        """Update the gradient reversal coefficient."""
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalLayer.apply(x, self.lambda_)


class DomainDiscriminator(Module):
    """Binary domain discriminator for adversarial DA.

    Architecture: MLP with hidden layers and sigmoid output for binary classification.

    Args:
        in_features: Dimension of input features.
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability (default: 0.5).

    Example:
        >>> discriminator = DomainDiscriminator(in_features=256, hidden_dims=[128, 64])
        >>> domain_pred = discriminator(features)  # [0, 1] probabilities
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.discriminator(x))


class DANN(Module):
    """Domain-Adversarial Neural Network (DANN).

    Combines feature extractor, label classifier, and domain classifier in an
    adversarial training framework. The domain classifier is trained to maximize
    domain classification loss while the feature extractor learns domain-invariant
    representations by minimizing this same loss.

    Reference:
        Ganin & Lempitsky "Unsupervised Domain Adaptation by Backpropagation" ICML 2015

    Args:
        feature_dim: Dimension of input features.
        num_classes: Number of classes for label classification.
        hidden_dims: Hidden layer dimensions for feature extractor.
        domain_hidden_dims: Hidden dimensions for domain discriminator.
        dropout: Dropout probability.

    Example:
        >>> dann = DANN(feature_dim=512, num_classes=10)
        >>> features, domain_preds, class_preds = dann(x, alpha=1.0)
        >>> # features: domain-invariant features
        >>> # domain_preds: domain classification (source/target)
        >>> # class_preds: class predictions
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        domain_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        if domain_hidden_dims is None:
            domain_hidden_dims = [128, 64]

        self.feature_extractor = nn.Sequential()
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            self.feature_extractor.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.feature_dim = prev_dim

        self.label_classifier = nn.Linear(self.feature_dim, num_classes)
        self.domain_classifier = nn.Sequential(
            GradientReversal(lambda_init=1.0),
            DomainDiscriminator(self.feature_dim, domain_hidden_dims, dropout),
        )

    def set_lambda(self, lambda_: float) -> None:
        """Set the gradient reversal strength."""
        self.domain_classifier[0].set_lambda(lambda_)

    def forward(
        self,
        x: Tensor,
        alpha: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        features = self.feature_extractor(x)
        class_preds = self.label_classifier(features)
        self.domain_classifier[0].set_lambda(alpha)
        domain_preds = self.domain_classifier(features)
        return features, domain_preds, class_preds


class ADDA(Module):
    """Adversarial Discriminative Domain Adaptation (ADDA).

    Pre-trains a source encoder and classifier, then learns a target encoder
    that maps target samples to the source feature space using adversarial training.

    Reference:
        Tzeng et al. "Adversarial Discriminative Domain Adaptation" CVPR 2017

    Args:
        feature_dim: Dimension of input features.
        num_classes: Number of classes.
        hidden_dims: Hidden layer dimensions for encoder.
        domain_hidden_dims: Hidden dimensions for domain discriminator.
        dropout: Dropout probability.

    Example:
        >>> adda = ADDA(feature_dim=512, num_classes=10)
        >>> # Phase 1: Train source encoder
        >>> source_features = adda.source_encoder(source_data)
        >>> # Phase 2: Adapt target encoder
        >>> target_features = adda.target_encoder(target_data)
        >>> domain_preds = adda.domain_discriminator(target_features)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        domain_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        if domain_hidden_dims is None:
            domain_hidden_dims = [128, 64]

        self.feature_dim = hidden_dims[-1] if hidden_dims else feature_dim

        self.source_encoder = self._build_encoder(feature_dim, hidden_dims, dropout)
        self.target_encoder = self._build_encoder(feature_dim, hidden_dims, dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(
            self.feature_dim, domain_hidden_dims, dropout
        )

    def _build_encoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float,
    ) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward_source(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.source_encoder(x)
        class_preds = self.classifier(features)
        return features, class_preds

    def forward_target(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.target_encoder(x)
        class_preds = self.classifier(features)
        return features, class_preds

    def discriminate(self, features: Tensor) -> Tensor:
        return self.domain_discriminator(features)


class MCDAN(Module):
    """Multi-Channel Domain Adversarial Neural Network.

    Extends DANN by using multiple domain classifiers (channels) to capture
    different aspects of domain shift and improve adaptation.

    Reference:
        Rostam et al. "Multi-Adapt: Adaptive Multi-Channel Domain Adversarial Neural Network" 2020

    Args:
        feature_dim: Dimension of input features.
        num_classes: Number of classes.
        num_channels: Number of domain classifier channels.
        hidden_dims: Hidden layer dimensions for feature extractor.
        channel_hidden_dims: Hidden dimensions for each domain classifier.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_channels: int = 3,
        hidden_dims: Optional[List[int]] = None,
        channel_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        if channel_hidden_dims is None:
            channel_hidden_dims = [64]

        self.feature_extractor = nn.Sequential()
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            self.feature_extractor.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.feature_dim = prev_dim

        self.label_classifier = nn.Linear(self.feature_dim, num_classes)
        self.domain_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    GradientReversal(lambda_init=1.0),
                    DomainDiscriminator(self.feature_dim, channel_hidden_dims, dropout),
                )
                for _ in range(num_channels)
            ]
        )
        self.num_channels = num_channels

    def set_lambda(self, lambda_: float) -> None:
        for classifier in self.domain_classifiers:
            classifier[0].set_lambda(lambda_)

    def forward(
        self,
        x: Tensor,
        alpha: float = 1.0,
    ) -> Tuple[Tensor, List[Tensor], Tensor]:
        features = self.feature_extractor(x)
        class_preds = self.label_classifier(features)

        domain_preds = []
        for classifier in self.domain_classifiers:
            classifier[0].set_lambda(alpha)
            domain_preds.append(classifier(features))

        return features, domain_preds, class_preds


class CDAN(Module):
    """Conditional Domain Adversarial Network.

    Extends DANN by conditioning the domain classifier on both features and
    predictions, which provides finer-grained adaptation.

    Reference:
        Long et al. "Conditional Adversarial Domain Adaptation" NIPS 2017

    Args:
        feature_dim: Dimension of input features.
        num_classes: Number of classes.
        hidden_dims: Hidden layer dimensions for feature extractor.
        domain_hidden_dims: Hidden dimensions for domain discriminator.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        domain_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        if domain_hidden_dims is None:
            domain_hidden_dims = [128, 64]

        self.feature_extractor = nn.Sequential()
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            self.feature_extractor.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.feature_dim = prev_dim

        self.label_classifier = nn.Linear(self.feature_dim, num_classes)

        combined_dim = self.feature_dim * num_classes
        self.domain_classifier = nn.Sequential(
            GradientReversal(lambda_init=1.0),
            DomainDiscriminator(combined_dim, domain_hidden_dims, dropout),
        )

    def set_lambda(self, lambda_: float) -> None:
        self.domain_classifier[0].set_lambda(lambda_)

    def forward(
        self,
        x: Tensor,
        alpha: float = 1.0,
        use_softmax: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        features = self.feature_extractor(x)
        class_preds = self.label_classifier(features)

        if use_softmax:
            class_probs = F.softmax(class_preds, dim=1)
        else:
            class_probs = class_preds

        combined = torch.bmm(class_probs.unsqueeze(2), features.unsqueeze(1)).view(
            features.size(0), -1
        )

        self.domain_classifier[0].set_lambda(alpha)
        domain_preds = self.domain_classifier(combined)

        return features, domain_preds, class_preds


class EntropyLoss(nn.Module):
    """Entropy loss for target domain minimization.

    Encourages low-confidence predictions on target domain, promoting
    cluster assumption in semi-supervised adaptation.

    Reference:
        Grandvalet & Bengio "Semi-Supervised Learning by Entropy Minimization" NIPS 2005
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions: Tensor) -> Tensor:
        if predictions.dim() > 1:
            epsilon = 1e-10
            entropy = -torch.sum(predictions * torch.log(predictions + epsilon), dim=1)
            return torch.mean(entropy)
        return torch.tensor(0.0, device=predictions.device)
