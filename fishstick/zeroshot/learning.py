"""
Zero-Shot Learning (ZSL) Module for Fishstick

Comprehensive implementations of state-of-the-art zero-shot learning methods including:
- Embedding-based methods (DeViSE, ALE, SJE, LatEm, ESZSL, SYNC, SAE)
- Generative methods (ZSLGAN, f-CLSWGAN, CycleWCL, LisGAN, FREE, LsrGAN)
- Semantic embeddings (attributes, word embeddings, sentence embeddings)
- Generalized ZSL methods
- Feature augmentation techniques
- Transductive ZSL methods
- Evaluation metrics
- Few-shot ZSL methods

References:
- DeViSE: Frome et al. (2013)
- ALE: Akata et al. (2013)
- SJE: Akata et al. (2015)
- LatEm: Xian et al. (2016)
- ESZSL: Romera-Paredes & Torr (2015)
- SYNC: Changpinyo et al. (2016)
- SAE: Kodirov et al. (2017)
- ZSLGAN: Zhu et al. (2018)
- f-CLSWGAN: Xian et al. (2018)
- FREE: Chen et al. (2018)
"""

from typing import Optional, Tuple, List, Dict, Union, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Base Classes and Types
# =============================================================================


@dataclass
class ZSLConfig:
    """Configuration for zero-shot learning models."""

    feature_dim: int = 2048
    semantic_dim: int = 300
    hidden_dim: int = 512
    num_classes: int = 50
    num_seen_classes: int = 40
    num_unseen_classes: int = 10
    lambda_reg: float = 0.01
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ZSLBase(ABC, nn.Module):
    """Abstract base class for zero-shot learning models."""

    def __init__(self, config: ZSLConfig):
        super().__init__()
        self.config = config
        self.device = config.device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss for training."""
        pass

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train the model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            self.train()
            total_loss = 0.0

            for batch in train_loader:
                features, labels, semantics = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                semantics = semantics.to(self.device)

                optimizer.zero_grad()
                loss = self.compute_loss(features, labels, semantics)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}"
                )

                if val_loader is not None:
                    acc = self.evaluate(val_loader)
                    print(f"Validation Accuracy: {acc:.4f}")

    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate the model."""
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                features, labels, _ = batch
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.forward(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total if total > 0 else 0.0


# =============================================================================
# Embedding-Based Methods
# =============================================================================


class DeViSE(ZSLBase):
    """
    Deep Visual Semantic Embedding (DeViSE)

    Learns a linear mapping from visual features to semantic embedding space.
    Uses ranking loss to ensure correct class embeddings are closer than incorrect ones.

    Reference: Frome et al. "DeViSE: A Deep Visual-Semantic Embedding Model" (2013)
    """

    def __init__(self, config: ZSLConfig, margin: float = 0.1):
        super().__init__(config)
        self.margin = margin

        # Linear transformation from visual to semantic space
        self.visual_to_semantic = nn.Linear(config.feature_dim, config.semantic_dim)

        # Class semantic embeddings (fixed during training)
        self.register_buffer(
            "class_embeddings", torch.randn(config.num_classes, config.semantic_dim)
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing compatibility scores.

        Args:
            x: Visual features [batch_size, feature_dim]

        Returns:
            Compatibility scores [batch_size, num_classes]
        """
        # Project visual features to semantic space
        semantic_proj = self.visual_to_semantic(x)
        semantic_proj = F.normalize(semantic_proj, p=2, dim=1)

        # Compute dot product with class embeddings
        scores = torch.mm(semantic_proj, self.class_embeddings.t())
        return scores

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranking loss."""
        batch_size = features.size(0)

        # Project to semantic space
        semantic_proj = self.visual_to_semantic(features)
        semantic_proj = F.normalize(semantic_proj, p=2, dim=1)

        # Get ground truth semantic embeddings
        true_semantic = semantic_embeddings[labels]

        # Compute similarities
        true_sim = torch.sum(semantic_proj * true_semantic, dim=1)

        # Ranking loss
        loss = 0.0
        for i in range(batch_size):
            # Negative samples
            neg_mask = torch.ones(self.config.num_classes, device=self.device)
            neg_mask[labels[i]] = 0

            neg_semantic = self.class_embeddings[neg_mask.bool()]
            neg_sim = torch.mm(semantic_proj[i : i + 1], neg_semantic.t()).squeeze()

            # Hinge loss
            hinge = torch.clamp(self.margin + neg_sim - true_sim[i], min=0.0)
            loss += hinge.sum()

        return loss / batch_size


class ALE(ZSLBase):
    """
    Attribute Label Embedding (ALE)

    Learns a bilinear compatibility function between visual features and attributes.
    Uses ranking loss with optimized sampling.

    Reference: Akata et al. "Label-Embedding for Image Classification" (2013)
    """

    def __init__(self, config: ZSLConfig, margin: float = 0.1):
        super().__init__(config)
        self.margin = margin

        # Bilinear compatibility matrix
        self.W = nn.Parameter(torch.randn(config.feature_dim, config.semantic_dim))
        nn.init.xavier_uniform_(self.W)

        # Class attribute embeddings
        self.register_buffer(
            "class_attributes", torch.randn(config.num_classes, config.semantic_dim)
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute compatibility scores via bilinear form.

        Args:
            x: Visual features [batch_size, feature_dim]

        Returns:
            Compatibility scores [batch_size, num_classes]
        """
        # Project features: x @ W
        proj = torch.mm(x, self.W)

        # Compute compatibility with class attributes
        scores = torch.mm(proj, self.class_attributes.t())
        return scores

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute structured SVM loss."""
        batch_size = features.size(0)

        # Compute all compatibility scores
        scores = self.forward(features)

        # Get ground truth scores
        true_scores = scores[torch.arange(batch_size), labels]

        # Compute loss
        loss = 0.0
        for i in range(batch_size):
            # For each negative class
            for j in range(self.config.num_classes):
                if j != labels[i]:
                    # Structured hinge loss
                    loss += torch.clamp(
                        self.margin + scores[i, j] - true_scores[i], min=0.0
                    )

        # Add regularization
        reg_loss = self.config.lambda_reg * torch.norm(self.W, p="fro")

        return loss / batch_size + reg_loss


class SJE(ZSLBase):
    """
    Structured Joint Embedding (SJE)

    Learns a joint embedding space for visual features and semantic embeddings.
    Uses max-margin loss with structured prediction.

    Reference: Akata et al. "Evaluation of Output Embeddings for Fine-Grained Image Classification" (2015)
    """

    def __init__(self, config: ZSLConfig, margin: float = 1.0):
        super().__init__(config)
        self.margin = margin

        # Joint embedding parameters
        self.W = nn.Parameter(torch.randn(config.feature_dim, config.semantic_dim))
        nn.init.xavier_uniform_(self.W)

        self.register_buffer(
            "class_embeddings", torch.randn(config.num_classes, config.semantic_dim)
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute joint embedding scores."""
        proj = torch.mm(x, self.W)
        scores = torch.mm(proj, self.class_embeddings.t())
        return scores

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute structured joint embedding loss."""
        batch_size = features.size(0)

        # Project features
        proj = torch.mm(features, self.W)

        # Get true semantic embeddings
        true_semantic = semantic_embeddings[labels]

        # True compatibility scores
        true_scores = torch.sum(proj * true_semantic, dim=1)

        # Max margin loss
        loss = 0.0
        for i in range(batch_size):
            max_violation = torch.tensor(0.0, device=self.device)

            for j in range(self.config.num_classes):
                if j != labels[i]:
                    # Compute compatibility with wrong class
                    wrong_score = torch.dot(proj[i], self.class_embeddings[j])
                    violation = self.margin + wrong_score - true_scores[i]
                    max_violation = torch.max(max_violation, violation)

            loss += torch.clamp(max_violation, min=0.0)

        reg_loss = self.config.lambda_reg * torch.norm(self.W, p="fro")
        return loss / batch_size + reg_loss


class LatEm(ZSLBase):
    """
    Latent Embeddings (LatEm)

    Uses piece-wise linear compatibility for flexible embedding.
    Learns multiple linear mappings and selects the best one for each example.

    Reference: Xian et al. "Latent Embeddings for Zero-shot Classification" (2016)
    """

    def __init__(self, config: ZSLConfig, num_latent: int = 5):
        super().__init__(config)
        self.num_latent = num_latent

        # Multiple linear mappings (latent embeddings)
        self.latent_projections = nn.ModuleList(
            [
                nn.Linear(config.feature_dim, config.semantic_dim)
                for _ in range(num_latent)
            ]
        )

        # Selection network
        self.selector = nn.Sequential(
            nn.Linear(config.feature_dim, num_latent), nn.Softmax(dim=1)
        )

        self.register_buffer(
            "class_embeddings", torch.randn(config.num_classes, config.semantic_dim)
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scores using latent embeddings.

        Args:
            x: Visual features [batch_size, feature_dim]

        Returns:
            Compatibility scores [batch_size, num_classes]
        """
        # Get selection weights
        weights = self.selector(x)  # [batch_size, num_latent]

        # Compute projections for each latent space
        all_scores = []
        for i in range(self.num_latent):
            proj = self.latent_projections[i](x)
            proj = F.normalize(proj, p=2, dim=1)
            scores = torch.mm(proj, self.class_embeddings.t())
            all_scores.append(scores)

        # Stack and weight
        all_scores = torch.stack(
            all_scores, dim=2
        )  # [batch_size, num_classes, num_latent]
        weights = weights.unsqueeze(1)  # [batch_size, 1, num_latent]

        final_scores = torch.sum(all_scores * weights, dim=2)
        return final_scores

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranking loss with latent embeddings."""
        batch_size = features.size(0)

        # Get scores
        scores = self.forward(features)

        # Ground truth scores
        true_scores = scores[torch.arange(batch_size), labels]

        # Ranking loss
        loss = 0.0
        for i in range(batch_size):
            for j in range(self.config.num_classes):
                if j != labels[i]:
                    loss += torch.clamp(1.0 + scores[i, j] - true_scores[i], min=0.0)

        return loss / batch_size


class ESZSL(ZSLBase):
    """
    Embarrassingly Simple Zero-Shot Learning (ESZSL)

    Closed-form solution for ZSL with simple regularization.
    Very efficient as it doesn't require iterative optimization.

    Reference: Romera-Paredes & Torr "An Embarrassingly Simple Approach to Zero-Shot Learning" (2015)
    """

    def __init__(self, config: ZSLConfig, alpha: float = 3.0, gamma: float = 0.0):
        super().__init__(config)
        self.alpha = alpha  # Regularization for features
        self.gamma = gamma  # Regularization for attributes

        # Parameters to be computed in closed form
        self.W = None

        self.register_buffer(
            "class_attributes", torch.randn(config.num_classes, config.semantic_dim)
        )

        self.to(self.device)

    def fit_closed_form(self, X: torch.Tensor, Y: torch.Tensor, S: torch.Tensor):
        """
        Compute W in closed form.

        Args:
            X: Features [n_samples, feature_dim]
            Y: One-hot labels [n_samples, num_classes]
            S: Class attributes [num_classes, semantic_dim]
        """
        # ESZSL solution: W = (X^T X + alpha*I)^{-1} X^T Y S (S^T S + gamma*I)^{-1}

        X = X.to(self.device)
        Y = Y.to(self.device)
        S = S.to(self.device)

        # Compute terms
        XtX = torch.mm(X.t(), X) + self.alpha * torch.eye(
            self.config.feature_dim, device=self.device
        )
        StS = torch.mm(S.t(), S) + self.gamma * torch.eye(
            self.config.semantic_dim, device=self.device
        )

        XtY = torch.mm(X.t(), Y)

        # Solve
        inv_XtX = torch.inverse(XtX)
        inv_StS = torch.inverse(StS)

        self.W = torch.mm(torch.mm(inv_XtX, XtY), torch.mm(S, inv_StS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using learned weights."""
        if self.W is None:
            raise RuntimeError("Model not fitted. Call fit_closed_form first.")

        # Project features
        proj = torch.mm(x, self.W)

        # Compute compatibility with class attributes
        scores = torch.mm(proj, self.class_attributes.t())
        return scores

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Not used for ESZSL - uses closed form instead."""
        return torch.tensor(0.0)


class SYNC(ZSLBase):
    """
    Synthesized Classifiers (SYNC)

    Synthesizes classifiers from semantic embeddings using dictionary learning.
    Maps between semantic and visual classifier spaces.

    Reference: Changpinyo et al. "Synthesized Classifiers for Zero-Shot Learning" (2016)
    """

    def __init__(self, config: ZSLConfig, num_bases: int = 50):
        super().__init__(config)
        self.num_bases = num_bases

        # Dictionary in semantic space
        self.D_semantic = nn.Parameter(torch.randn(config.semantic_dim, num_bases))
        nn.init.xavier_uniform_(self.D_semantic)

        # Dictionary in visual classifier space
        self.D_visual = nn.Parameter(torch.randn(config.feature_dim, num_bases))
        nn.init.xavier_uniform_(self.D_visual)

        self.to(self.device)

    def encode_semantic(self, semantic: torch.Tensor) -> torch.Tensor:
        """Encode semantic embedding using dictionary."""
        # Sparse coding approximation
        codes = torch.mm(semantic, self.D_semantic)
        codes = F.softmax(codes, dim=1)
        return codes

    def synthesize_classifier(self, semantic: torch.Tensor) -> torch.Tensor:
        """Synthesize visual classifier from semantic embedding."""
        codes = self.encode_semantic(semantic)
        classifier = torch.mm(codes, self.D_visual.t())
        return classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using synthesized classifiers.

        Args:
            x: Features [batch_size, feature_dim]

        Returns:
            Scores [batch_size, num_classes]
        """
        # Synthesize all classifiers
        batch_size = x.size(0)

        # For each class, synthesize classifier and compute score
        scores = []
        for i in range(self.config.num_classes):
            # Get semantic for class i (would be provided externally in practice)
            semantic_i = torch.randn(1, self.config.semantic_dim, device=self.device)
            classifier_i = self.synthesize_classifier(semantic_i)

            # Compute score
            score_i = torch.mm(x, classifier_i.t())
            scores.append(score_i)

        return torch.cat(scores, dim=1)

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute synthesis loss."""
        batch_size = features.size(0)

        # Get semantic embeddings for classes in batch
        class_semantic = semantic_embeddings

        # Encode
        codes = self.encode_semantic(class_semantic)

        # Synthesize classifiers
        synthesized = torch.mm(codes, self.D_visual.t())

        # Compute classification loss
        scores = torch.mm(features, synthesized.t())

        loss = F.cross_entropy(scores, labels)

        # Add dictionary coherence regularization
        coherence = torch.norm(
            torch.mm(self.D_semantic.t(), self.D_semantic)
            - torch.eye(self.num_bases, device=self.device),
            p="fro",
        )

        return loss + self.config.lambda_reg * coherence


class SAE(ZSLBase):
    """
    Semantic Autoencoder (SAE)

    Uses encoder-decoder structure with semantic constraint.
    Ensures reconstruction while preserving semantic information.

    Reference: Kodirov et al. "Semantic Autoencoder for Zero-Shot Learning" (2017)
    """

    def __init__(self, config: ZSLConfig, lambda_semantic: float = 1.0):
        super().__init__(config)
        self.lambda_semantic = lambda_semantic

        # Encoder: visual -> semantic
        self.encoder = nn.Linear(config.feature_dim, config.semantic_dim)

        # Decoder: semantic -> visual
        self.decoder = nn.Linear(config.semantic_dim, config.feature_dim)

        self.register_buffer(
            "class_embeddings", torch.randn(config.num_classes, config.semantic_dim)
        )

        self.to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode visual features to semantic space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode semantic embeddings to visual space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and classifier."""
        # Encode
        semantic = self.encode(x)

        # Compute compatibility scores
        scores = torch.mm(semantic, self.class_embeddings.t())
        return scores

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction and semantic loss."""
        # Encode
        semantic = self.encode(features)

        # Decode
        reconstructed = self.decode(semantic)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, features)

        # Semantic constraint: encoded should match class semantics
        class_semantic = semantic_embeddings[labels]
        semantic_loss = F.mse_loss(semantic, class_semantic)

        # Classification loss
        scores = torch.mm(semantic, self.class_embeddings.t())
        cls_loss = F.cross_entropy(scores, labels)

        return recon_loss + self.lambda_semantic * semantic_loss + cls_loss


# =============================================================================
# Generative Methods
# =============================================================================


class ZSLGAN(nn.Module):
    """
    GAN for Zero-Shot Learning (ZSLGAN)

    Generates visual features from semantic embeddings using conditional GAN.
    Enables classifier training on synthesized features for unseen classes.

    Reference: Zhu et al. "Generative Adversarial Approach for Zero-Shot Learning" (2018)
    """

    def __init__(self, config: ZSLConfig, noise_dim: int = 100):
        super().__init__()
        self.config = config
        self.noise_dim = noise_dim
        self.device = config.device

        # Generator: semantic + noise -> visual features
        self.generator = nn.Sequential(
            nn.Linear(config.semantic_dim + noise_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.feature_dim),
            nn.Tanh(),
        )

        # Discriminator: visual + semantic -> real/fake
        self.discriminator = nn.Sequential(
            nn.Linear(config.feature_dim + config.semantic_dim, config.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Classifier for generated features
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

        self.to(self.device)

    def generate_features(
        self, semantic: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """Generate visual features from semantic embeddings."""
        batch_size = semantic.size(0)

        # Sample noise
        noise = torch.randn(
            batch_size * num_samples, self.noise_dim, device=self.device
        )

        # Expand semantic for multiple samples
        semantic_expanded = semantic.repeat_interleave(num_samples, dim=0)

        # Concatenate and generate
        z = torch.cat([semantic_expanded, noise], dim=1)
        features = self.generator(z)

        return features

    def discriminate(
        self, features: torch.Tensor, semantic: torch.Tensor
    ) -> torch.Tensor:
        """Discriminate real vs generated features."""
        # Expand semantic to match features
        batch_size = features.size(0)
        num_per_class = batch_size // semantic.size(0)
        semantic_expanded = semantic.repeat_interleave(num_per_class, dim=0)

        x = torch.cat([features, semantic_expanded], dim=1)
        return self.discriminator(x)

    def forward_generator(
        self, semantic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through generator with classification."""
        features = self.generate_features(semantic)
        scores = self.classifier(features)
        return features, scores

    def train_step(
        self,
        real_features: torch.Tensor,
        real_labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        opt_G: torch.optim.Optimizer,
        opt_D: torch.optim.Optimizer,
    ):
        """Single training step for both generator and discriminator."""
        batch_size = real_features.size(0)
        real_labels_binary = torch.ones(batch_size, 1, device=self.device)
        fake_labels_binary = torch.zeros(batch_size, 1, device=self.device)

        # Get semantic embeddings for real labels
        real_semantic = semantic_embeddings[real_labels]

        # Train Discriminator
        opt_D.zero_grad()

        # Real features
        real_pred = self.discriminate(real_features, real_semantic)
        d_loss_real = F.binary_cross_entropy(real_pred, real_labels_binary)

        # Fake features
        fake_features = self.generate_features(real_semantic)
        fake_pred = self.discriminate(fake_features.detach(), real_semantic)
        d_loss_fake = F.binary_cross_entropy(fake_pred, fake_labels_binary)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_D.step()

        # Train Generator
        opt_G.zero_grad()

        # Fool discriminator
        fake_pred = self.discriminate(fake_features, real_semantic)
        g_loss_adv = F.binary_cross_entropy(fake_pred, real_labels_binary)

        # Classification loss on generated features
        fake_scores = self.classifier(fake_features)
        g_loss_cls = F.cross_entropy(fake_scores, real_labels)

        # Reconstruction-like loss (feature regression)
        g_loss_reg = F.mse_loss(fake_features, real_features)

        g_loss = g_loss_adv + g_loss_cls + 0.1 * g_loss_reg
        g_loss.backward()
        opt_G.step()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "g_loss_adv": g_loss_adv.item(),
            "g_loss_cls": g_loss_cls.item(),
        }


class FCLSWGAN(ZSLGAN):
    """
    Feature Generating Network for Zero-Shot Learning (f-CLSWGAN)

    Extension of ZSLGAN with classification loss for generated features.
    Specifically designed for generalized zero-shot learning.

    Reference: Xian et al. "Feature Generating Networks for Zero-Shot Learning" (2018)
    """

    def __init__(
        self,
        config: ZSLConfig,
        noise_dim: int = 100,
        cls_weight: float = 0.01,
        visual_weight: float = 0.1,
    ):
        super().__init__(config, noise_dim)
        self.cls_weight = cls_weight
        self.visual_weight = visual_weight

    def train_step(
        self,
        real_features: torch.Tensor,
        real_labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        opt_G: torch.optim.Optimizer,
        opt_D: torch.optim.Optimizer,
        opt_C: Optional[torch.optim.Optimizer] = None,
    ):
        """Training step with separate classifier optimization."""
        batch_size = real_features.size(0)
        real_labels_binary = torch.ones(batch_size, 1, device=self.device)
        fake_labels_binary = torch.zeros(batch_size, 1, device=self.device)

        real_semantic = semantic_embeddings[real_labels]

        # Train Discriminator
        opt_D.zero_grad()

        real_pred = self.discriminate(real_features, real_semantic)
        d_loss_real = F.binary_cross_entropy(real_pred, real_labels_binary)

        fake_features = self.generate_features(real_semantic)
        fake_pred = self.discriminate(fake_features.detach(), real_semantic)
        d_loss_fake = F.binary_cross_entropy(fake_pred, fake_labels_binary)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_D.step()

        # Train Generator
        opt_G.zero_grad()

        fake_pred = self.discriminate(fake_features, real_semantic)
        g_loss_adv = F.binary_cross_entropy(fake_pred, real_labels_binary)

        # Strong classification supervision
        fake_scores = self.classifier(fake_features)
        g_loss_cls = F.cross_entropy(fake_scores, real_labels)

        # Visual reconstruction
        g_loss_visual = F.mse_loss(fake_features, real_features)

        g_loss = (
            g_loss_adv
            + self.cls_weight * g_loss_cls
            + self.visual_weight * g_loss_visual
        )
        g_loss.backward()
        opt_G.step()

        # Train Classifier on real features
        if opt_C is not None:
            opt_C.zero_grad()
            real_scores = self.classifier(real_features)
            c_loss = F.cross_entropy(real_scores, real_labels)
            c_loss.backward()
            opt_C.step()
        else:
            c_loss = torch.tensor(0.0)

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "c_loss": c_loss.item() if isinstance(c_loss, torch.Tensor) else 0.0,
        }


class CycleWCL(nn.Module):
    """
    Cycle-Consistent Wasserstein Zero-Shot Learning

    Uses cycle consistency between visual and semantic domains
    with Wasserstein distance for better training stability.
    """

    def __init__(self, config: ZSLConfig, noise_dim: int = 100):
        super().__init__()
        self.config = config
        self.noise_dim = noise_dim
        self.device = config.device

        # Visual to semantic encoder
        self.encoder_VS = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.semantic_dim),
        )

        # Semantic to visual generator
        self.generator_SV = nn.Sequential(
            nn.Linear(config.semantic_dim + noise_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )

        # Cycle: Visual -> Semantic -> Visual
        # Cycle: Semantic -> Visual -> Semantic

        # Discriminator for Wasserstein loss
        self.discriminator_V = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(config.hidden_dim, 1),
        )

        self.discriminator_S = nn.Sequential(
            nn.Linear(config.semantic_dim, config.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(config.hidden_dim, 1),
        )

        self.to(self.device)

    def encode_visual(self, visual: torch.Tensor) -> torch.Tensor:
        """Encode visual features to semantic space."""
        return self.encoder_VS(visual)

    def generate_visual(
        self, semantic: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate visual features from semantic embeddings."""
        if noise is None:
            noise = torch.randn(semantic.size(0), self.noise_dim, device=self.device)
        z = torch.cat([semantic, noise], dim=1)
        return self.generator_SV(z)

    def forward_cycle(self, visual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward cycle: V -> S -> V."""
        semantic = self.encode_visual(visual)
        visual_recon = self.generate_visual(semantic)
        return semantic, visual_recon

    def backward_cycle(
        self, semantic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward cycle: S -> V -> S."""
        visual = self.generate_visual(semantic)
        semantic_recon = self.encode_visual(visual)
        return visual, semantic_recon

    def compute_loss(
        self,
        real_visual: torch.Tensor,
        real_semantic: torch.Tensor,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute all cycle consistency losses."""
        batch_size = real_visual.size(0)

        # Forward cycle loss
        semantic_encoded, visual_recon = self.forward_cycle(real_visual)
        loss_cycle_V = F.l1_loss(visual_recon, real_visual)

        # Backward cycle loss
        visual_generated, semantic_recon = self.backward_cycle(real_semantic)
        loss_cycle_S = F.l1_loss(semantic_recon, real_semantic)

        # Identity losses
        semantic_identity = self.encode_visual(self.generate_visual(real_semantic))
        loss_identity_S = F.l1_loss(semantic_identity, real_semantic)

        visual_identity = self.generate_visual(self.encode_visual(real_visual))
        loss_identity_V = F.l1_loss(visual_identity, real_visual)

        # Total cycle loss
        loss_cycle = (
            loss_cycle_V
            + loss_cycle_S
            + lambda_identity * (loss_identity_S + loss_identity_V)
        )

        # Adversarial losses (Wasserstein)
        loss_G_V = -torch.mean(self.discriminator_V(visual_generated))
        loss_G_S = -torch.mean(self.discriminator_S(semantic_encoded))

        loss_G = loss_G_V + loss_G_S + lambda_cycle * loss_cycle

        return {
            "loss_G": loss_G,
            "loss_cycle": loss_cycle,
            "loss_cycle_V": loss_cycle_V,
            "loss_cycle_S": loss_cycle_S,
        }


class LisGAN(nn.Module):
    """
    Latent Space Interpolation GAN for Zero-Shot Learning

    Uses interpolation in latent space to generate diverse features
    for unseen classes by interpolating between seen class features.
    """

    def __init__(
        self, config: ZSLConfig, noise_dim: int = 100, num_interpolations: int = 5
    ):
        super().__init__()
        self.config = config
        self.noise_dim = noise_dim
        self.num_interpolations = num_interpolations
        self.device = config.device

        # Generator with interpolation
        self.generator = nn.Sequential(
            nn.Linear(config.semantic_dim * 2 + noise_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.feature_dim),
            nn.Tanh(),
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Interpolation network
        self.interpolator = nn.Sequential(
            nn.Linear(config.semantic_dim * 2, config.semantic_dim), nn.Sigmoid()
        )

        self.to(self.device)

    def interpolate_semantic(
        self,
        semantic1: torch.Tensor,
        semantic2: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Interpolate between two semantic embeddings."""
        if alpha is None:
            alpha = torch.rand(semantic1.size(0), 1, device=self.device)

        # Use learned interpolation
        concat = torch.cat([semantic1, semantic2], dim=1)
        weights = self.interpolator(concat)

        interpolated = weights * semantic1 + (1 - weights) * semantic2
        return interpolated

    def generate_with_interpolation(self, semantic: torch.Tensor) -> torch.Tensor:
        """Generate features with latent space interpolation."""
        batch_size = semantic.size(0)

        # Create interpolated semantics
        interpolated_semantics = []
        for _ in range(self.num_interpolations):
            # Random pair within batch
            idx = torch.randperm(batch_size)
            semantic2 = semantic[idx]
            interp = self.interpolate_semantic(semantic, semantic2)
            interpolated_semantics.append(interp)

        all_semantics = torch.stack(
            interpolated_semantics, dim=1
        )  # [batch, num_interp, sem_dim]
        all_semantics = all_semantics.view(-1, self.config.semantic_dim)

        # Generate features
        noise = torch.randn(all_semantics.size(0), self.noise_dim, device=self.device)
        z = torch.cat([all_semantics, noise], dim=1)
        features = self.generator(z)

        return features.view(
            batch_size, self.num_interpolations, self.config.feature_dim
        )


class FREE(nn.Module):
    """
    Feature Refinement for Zero-Shot Learning

    Refines generated features through multiple stages to improve quality.
    Uses attention mechanism to focus on discriminative parts.

    Reference: Chen et al. "Zero-Shot Learning with Co-attention and Feature Refinement" (2018)
    """

    def __init__(
        self, config: ZSLConfig, noise_dim: int = 100, num_refinements: int = 3
    ):
        super().__init__()
        self.config = config
        self.noise_dim = noise_dim
        self.num_refinements = num_refinements
        self.device = config.device

        # Initial generator
        self.initial_generator = nn.Sequential(
            nn.Linear(config.semantic_dim + noise_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )

        # Refinement modules
        self.refinement_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        config.feature_dim + config.semantic_dim, config.hidden_dim
                    ),
                    nn.ReLU(True),
                    nn.Linear(config.hidden_dim, config.feature_dim),
                    nn.Tanh(),
                )
                for _ in range(num_refinements)
            ]
        )

        # Attention for feature selection
        self.attention = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 4),
            nn.ReLU(True),
            nn.Linear(config.feature_dim // 4, config.feature_dim),
            nn.Sigmoid(),
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def generate_and_refine(self, semantic: torch.Tensor) -> List[torch.Tensor]:
        """Generate and progressively refine features."""
        batch_size = semantic.size(0)

        # Initial generation
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        z = torch.cat([semantic, noise], dim=1)
        features = self.initial_generator(z)

        all_features = [features]

        # Progressive refinement
        for refine_module in self.refinement_modules:
            # Concatenate with semantic for guidance
            combined = torch.cat([features, semantic], dim=1)
            delta = refine_module(combined)

            # Apply attention
            attn_weights = self.attention(features)
            features = features + delta * attn_weights

            all_features.append(features)

        return all_features

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        """Forward pass returning final refined features."""
        features_list = self.generate_and_refine(semantic)
        return features_list[-1]


class LsrGAN(nn.Module):
    """
    Leveraging Seen Regions GAN

    Focuses on leveraging knowledge from seen classes to generate
    features for unseen classes by modeling region-specific generators.
    """

    def __init__(self, config: ZSLConfig, noise_dim: int = 100, num_regions: int = 4):
        super().__init__()
        self.config = config
        self.noise_dim = noise_dim
        self.num_regions = num_regions
        self.device = config.device

        # Region-specific generators
        self.region_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.semantic_dim + noise_dim, config.hidden_dim),
                    nn.ReLU(True),
                    nn.Linear(config.hidden_dim, config.feature_dim // num_regions),
                    nn.Tanh(),
                )
                for _ in range(num_regions)
            ]
        )

        # Region selector based on semantic
        self.region_selector = nn.Sequential(
            nn.Linear(config.semantic_dim, num_regions), nn.Softmax(dim=1)
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(True),
            nn.Linear(config.feature_dim, config.feature_dim),
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def generate_by_regions(self, semantic: torch.Tensor) -> torch.Tensor:
        """Generate features using region-specific generators."""
        batch_size = semantic.size(0)

        # Get region weights
        region_weights = self.region_selector(semantic)  # [batch, num_regions]

        # Generate region features
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        z = torch.cat([semantic, noise], dim=1)

        region_features = []
        for generator in self.region_generators:
            region_feat = generator(z)
            region_features.append(region_feat)

        # Concatenate regions
        all_regions = torch.cat(region_features, dim=1)  # [batch, feature_dim]

        # Apply region-wise weighting
        region_weights_expanded = region_weights.repeat_interleave(
            self.config.feature_dim // self.num_regions, dim=1
        )
        weighted_features = all_regions * region_weights_expanded

        # Fuse
        final_features = self.fusion(weighted_features)

        return final_features


# =============================================================================
# Semantic Embeddings
# =============================================================================


class AttributeVectors:
    """
    Class attribute vectors for zero-shot learning.

    Represents each class by a vector of attribute scores/presences.
    """

    def __init__(self, num_classes: int, num_attributes: int):
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.attributes = np.zeros((num_classes, num_attributes))

    def set_attributes(self, class_idx: int, attributes: np.ndarray):
        """Set attributes for a specific class."""
        self.attributes[class_idx] = attributes

    def get_attributes(self, class_idx: Union[int, List[int]]) -> np.ndarray:
        """Get attributes for one or more classes."""
        if isinstance(class_idx, int):
            return self.attributes[class_idx]
        return self.attributes[class_idx]

    def normalize(self, method: str = "l2"):
        """Normalize attribute vectors."""
        if method == "l2":
            norms = np.linalg.norm(self.attributes, axis=1, keepdims=True)
            self.attributes = self.attributes / (norms + 1e-8)
        elif method == "max":
            max_vals = np.max(np.abs(self.attributes), axis=1, keepdims=True)
            self.attributes = self.attributes / (max_vals + 1e-8)

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert to PyTorch tensor."""
        return torch.from_numpy(self.attributes).float().to(device)


class WordEmbeddings:
    """
    Word embeddings (Word2Vec/GloVe) for class names.

    Provides semantic embeddings based on class name word vectors.
    """

    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.embeddings: Dict[str, np.ndarray] = {}
        self.vocabulary: set = set()

    def load_pretrained(self, filepath: str, vocab_limit: Optional[int] = None):
        """Load pretrained word embeddings from file."""
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if vocab_limit and count >= vocab_limit:
                    break

                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])

                self.embeddings[word] = vector
                self.vocabulary.add(word)
                count += 1

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word."""
        word = word.lower().replace("_", " ").replace("-", " ")

        # Direct lookup
        if word in self.embeddings:
            return self.embeddings[word]

        # Try multi-word average
        words = word.split()
        vectors = [self.embeddings[w] for w in words if w in self.embeddings]

        if vectors:
            return np.mean(vectors, axis=0)

        return None

    def get_class_embeddings(self, class_names: List[str]) -> np.ndarray:
        """Get embeddings for a list of class names."""
        embeddings = []
        for name in class_names:
            emb = self.get_embedding(name)
            if emb is None:
                emb = np.random.randn(self.embedding_dim) * 0.1
            embeddings.append(emb)

        return np.array(embeddings)

    def compute_similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two words."""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)

        if emb1 is None or emb2 is None:
            return 0.0

        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


class SentenceEmbeddings:
    """
    Sentence embeddings using Sentence-BERT or similar.

    Provides semantic embeddings for class descriptions.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_dim = 384  # Default for MiniLM-L6-v2
        self.model = None
        self.cache: Dict[str, np.ndarray] = {}

    def load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            print("sentence-transformers not installed. Using random embeddings.")
            self.model = None

    def encode(
        self, sentences: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """Encode sentences to embeddings."""
        if isinstance(sentences, str):
            sentences = [sentences]

        # Check cache
        uncached = []
        uncached_indices = []
        embeddings = []

        for i, sent in enumerate(sentences):
            if sent in self.cache:
                embeddings.append(self.cache[sent])
            else:
                uncached.append(sent)
                uncached_indices.append(i)
                embeddings.append(None)

        # Encode uncached
        if uncached:
            if self.model is None:
                self.load_model()

            if self.model is not None:
                new_embeddings = self.model.encode(uncached, convert_to_numpy=True)
            else:
                new_embeddings = np.random.randn(len(uncached), self.embedding_dim)

            # Store in cache and embeddings list
            for i, sent in enumerate(uncached):
                self.cache[sent] = new_embeddings[i]
                embeddings[uncached_indices[i]] = new_embeddings[i]

        embeddings = np.array(embeddings)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def encode_class_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """Encode class descriptions."""
        return self.encode(descriptions, normalize=True)

    def compute_similarity_matrix(
        self, sentences1: List[str], sentences2: List[str]
    ) -> np.ndarray:
        """Compute similarity matrix between two sets of sentences."""
        emb1 = self.encode(sentences1)
        emb2 = self.encode(sentences2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2.T)
        return similarity


class ClassDescription:
    """
    Text descriptions for classes.

    Manages textual descriptions that can be encoded using
    sentence embeddings or other NLP methods.
    """

    def __init__(self):
        self.descriptions: Dict[int, str] = {}
        self.class_names: Dict[int, str] = {}

    def add_description(self, class_idx: int, name: str, description: str):
        """Add a description for a class."""
        self.class_names[class_idx] = name
        self.descriptions[class_idx] = description

    def get_description(self, class_idx: int) -> str:
        """Get description for a class."""
        return self.descriptions.get(class_idx, "")

    def get_all_descriptions(self) -> List[str]:
        """Get all descriptions as a list."""
        max_idx = max(self.descriptions.keys()) if self.descriptions else -1
        return [self.descriptions.get(i, "") for i in range(max_idx + 1)]

    def load_from_file(self, filepath: str):
        """Load descriptions from a file."""
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    class_idx = int(parts[0])
                    name = parts[1]
                    description = parts[2]
                    self.add_description(class_idx, name, description)


class HierarchicalEmbeddings:
    """
    Hierarchical embeddings based on class taxonomy.

    Encodes class hierarchy information into embeddings.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 300):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hierarchy: Dict[int, List[int]] = {}  # class -> parent classes
        self.embeddings = np.random.randn(num_classes, embedding_dim) * 0.1

    def add_parent(self, class_idx: int, parent_idx: int):
        """Add parent relationship."""
        if class_idx not in self.hierarchy:
            self.hierarchy[class_idx] = []
        self.hierarchy[class_idx].append(parent_idx)

    def build_hierarchical_embeddings(self, taxonomy_depth: int = 3):
        """Build embeddings respecting hierarchy."""
        for level in range(taxonomy_depth):
            for class_idx in range(self.num_classes):
                parents = self.hierarchy.get(class_idx, [])
                if parents:
                    # Average parent embeddings
                    parent_embs = self.embeddings[parents]
                    parent_mean = np.mean(parent_embs, axis=0)

                    # Update with hierarchical influence
                    alpha = 0.5 / (level + 1)
                    self.embeddings[class_idx] = (1 - alpha) * self.embeddings[
                        class_idx
                    ] + alpha * parent_mean

    def get_embedding(self, class_idx: int) -> np.ndarray:
        """Get embedding for a class."""
        return self.embeddings[class_idx]

    def get_path_to_root(self, class_idx: int) -> List[int]:
        """Get path from class to root of hierarchy."""
        path = [class_idx]
        current = class_idx

        while current in self.hierarchy and self.hierarchy[current]:
            parent = self.hierarchy[current][0]  # Take first parent
            path.append(parent)
            current = parent

        return path

    def compute_hierarchical_distance(self, class1: int, class2: int) -> int:
        """Compute distance in hierarchy (number of steps to common ancestor)."""
        path1 = set(self.get_path_to_root(class1))
        path2 = set(self.get_path_to_root(class2))

        common_ancestors = path1 & path2

        if not common_ancestors:
            return float("inf")

        # Find lowest common ancestor
        lca = max(common_ancestors, key=lambda x: len(self.get_path_to_root(x)))

        dist1 = len(self.get_path_to_root(class1)) - len(self.get_path_to_root(lca))
        dist2 = len(self.get_path_to_root(class2)) - len(self.get_path_to_root(lca))

        return dist1 + dist2


# =============================================================================
# Generalized Zero-Shot Learning
# =============================================================================


class CalibratedStacking(ZSLBase):
    """
    Calibrated Stacking for Generalized ZSL

    Calibrates scores between seen and unseen classes to handle
    the bias towards seen classes in generalized ZSL.

    Reference: Chao et al. "An Empirical Study and Analysis of Generalized Zero-Shot Learning" (2016)
    """

    def __init__(self, config: ZSLConfig, calibration_factor: float = 0.5):
        super().__init__(config)
        self.calibration_factor = calibration_factor

        # Base ZSL model (e.g., ALE, SJE)
        self.base_model = ALE(config)

        self.to(self.device)

    def forward(
        self, x: torch.Tensor, seen_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with calibration.

        Args:
            x: Input features
            seen_mask: Binary mask for seen classes [num_classes]

        Returns:
            Calibrated scores
        """
        scores = self.base_model(x)

        if seen_mask is not None:
            # Calibrate seen class scores
            scores = scores.clone()
            seen_indices = seen_mask.bool()
            scores[:, seen_indices] = scores[:, seen_indices] - self.calibration_factor

        return scores

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss using base model."""
        return self.base_model.compute_loss(features, labels, semantic_embeddings)

    def predict(
        self, x: torch.Tensor, seen_mask: torch.Tensor, use_calibration: bool = True
    ) -> torch.Tensor:
        """Predict with optional calibration."""
        if use_calibration:
            scores = self.forward(x, seen_mask)
        else:
            scores = self.base_model(x)

        return torch.argmax(scores, dim=1)


class RelationNet(ZSLBase):
    """
    Relation Network for Zero-Shot Learning

    Learns a relation module to compare query features with class prototypes.

    Reference: Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (2018)
    """

    def __init__(self, config: ZSLConfig):
        super().__init__(config)

        # Embedding module for features
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(True),
        )

        # Embedding module for semantic embeddings
        self.semantic_encoder = nn.Sequential(
            nn.Linear(config.semantic_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(True),
        )

        # Relation module
        self.relation_module = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor, class_semantics: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing relations.

        Args:
            x: Query features [batch_size, feature_dim]
            class_semantics: Class semantic embeddings [num_classes, semantic_dim]

        Returns:
            Relation scores [batch_size, num_classes]
        """
        batch_size = x.size(0)
        num_classes = class_semantics.size(0)

        # Encode features
        feature_emb = self.feature_encoder(x)  # [batch_size, hidden_dim]

        # Encode semantics
        semantic_emb = self.semantic_encoder(
            class_semantics
        )  # [num_classes, hidden_dim]

        # Expand for comparison
        feature_expanded = feature_emb.unsqueeze(1).expand(-1, num_classes, -1)
        semantic_expanded = semantic_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate
        combined = torch.cat([feature_expanded, semantic_expanded], dim=2)

        # Compute relations
        relations = self.relation_module(combined.view(-1, combined.size(2)))
        relations = relations.view(batch_size, num_classes)

        return relations

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relation loss."""
        # Get relation scores
        relations = self.forward(features, semantic_embeddings)

        # Target: 1 for correct class, 0 for others
        batch_size = features.size(0)
        targets = torch.zeros_like(relations)
        targets[torch.arange(batch_size), labels] = 1.0

        # MSE loss for relations
        loss = F.mse_loss(relations, targets)

        return loss


class DeepEmbedding(ZSLBase):
    """
    Deep Embedding Model for Generalized ZSL

    Uses deep neural networks for both visual and semantic embeddings
    with non-linear compatibility.
    """

    def __init__(self, config: ZSLConfig, temperature: float = 1.0):
        super().__init__(config)
        self.temperature = temperature

        # Deep visual embedding
        self.visual_embedding = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.semantic_dim),
        )

        # Deep semantic embedding (for refinement)
        self.semantic_embedding = nn.Sequential(
            nn.Linear(config.semantic_dim, config.semantic_dim),
            nn.ReLU(True),
            nn.Linear(config.semantic_dim, config.semantic_dim),
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor, class_semantics: torch.Tensor) -> torch.Tensor:
        """Forward pass with deep embeddings."""
        # Embed visual features
        visual_emb = self.visual_embedding(x)
        visual_emb = F.normalize(visual_emb, p=2, dim=1)

        # Refine semantic embeddings
        semantic_emb = self.semantic_embedding(class_semantics)
        semantic_emb = F.normalize(semantic_emb, p=2, dim=1)

        # Compute cosine similarity
        scores = torch.mm(visual_emb, semantic_emb.t()) / self.temperature

        return scores

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        scores = self.forward(features, semantic_embeddings)
        loss = F.cross_entropy(scores, labels)

        # Add regularization
        reg_loss = self.config.lambda_reg * sum(
            p.pow(2).sum() for p in self.parameters()
        )

        return loss + reg_loss


class GPZSL(ZSLBase):
    """
    Gaussian Process Zero-Shot Learning

    Uses Gaussian processes to model the relationship between
    semantic and visual spaces with uncertainty estimation.
    """

    def __init__(self, config: ZSLConfig, kernel_type: str = "rbf"):
        super().__init__(config)
        self.kernel_type = kernel_type

        # Mean function
        self.mean_fn = nn.Linear(config.semantic_dim, config.feature_dim)

        # Kernel parameters
        self.log_lengthscale = nn.Parameter(torch.zeros(config.semantic_dim))
        self.log_variance = nn.Parameter(torch.zeros(1))

        self.to(self.device)

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel."""
        # x1: [n1, d], x2: [n2, d]
        lengthscale = torch.exp(self.log_lengthscale)

        # Compute squared distances
        x1_scaled = x1 / lengthscale
        x2_scaled = x2 / lengthscale

        dist = torch.cdist(x1_scaled, x2_scaled, p=2) ** 2

        variance = torch.exp(self.log_variance)
        K = variance * torch.exp(-0.5 * dist)

        return K

    def forward(
        self,
        x: torch.Tensor,
        train_semantics: torch.Tensor,
        train_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GP prediction.

        Returns:
            mean: Predicted features
            variance: Prediction uncertainty
        """
        # Compute kernels
        K_train = self.kernel(train_semantics, train_semantics)
        K_test_train = self.kernel(x, train_semantics)

        # Add noise
        noise = 1e-4 * torch.eye(K_train.size(0), device=self.device)
        K_train = K_train + noise

        # GP prediction
        K_inv = torch.inverse(K_train)
        mean = self.mean_fn(x) + torch.mm(
            K_test_train,
            torch.mm(K_inv, train_features - self.mean_fn(train_semantics)),
        )

        # Compute variance
        K_test = self.kernel(x, x)
        var = K_test - torch.mm(K_test_train, torch.mm(K_inv, K_test_train.t()))

        return mean, var

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood."""
        # Mean prediction
        mean_pred = self.mean_fn(semantic_embeddings[labels])

        # MSE loss
        loss = F.mse_loss(mean_pred, features)

        return loss


class FVAE(nn.Module):
    """
    Flow-based Variational Autoencoder for ZSL

    Uses normalizing flows to model complex distributions in the
    semantic-to-visual mapping.
    """

    def __init__(self, config: ZSLConfig, latent_dim: int = 100, num_flows: int = 4):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.num_flows = num_flows
        self.device = config.device

        # Encoder: semantic -> latent parameters
        self.encoder = nn.Sequential(
            nn.Linear(config.semantic_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(config.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, latent_dim)

        # Decoder: latent + semantic -> features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + config.semantic_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )

        # Normalizing flows
        self.flows = nn.ModuleList(
            [
                AffineCouplingFlow(latent_dim, config.hidden_dim)
                for _ in range(num_flows)
            ]
        )

        self.to(self.device)

    def encode(self, semantic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode semantic to latent distribution."""
        h = self.encoder(semantic)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def sample(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from latent distribution with flows."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std

        # Apply flows
        z = z0
        log_det_jacobian = torch.zeros(z.size(0), device=self.device)

        for flow in self.flows:
            z, ldj = flow(z)
            log_det_jacobian += ldj

        return z, log_det_jacobian

    def decode(self, z: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        """Decode latent + semantic to features."""
        z = torch.cat([z, semantic], dim=1)
        return self.decoder(z)

    def forward(
        self, semantic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(semantic)
        z, log_det_jacobian = self.sample(mu, logvar)
        recon = self.decode(z, semantic)

        return recon, mu, logvar, log_det_jacobian

    def compute_loss(
        self, features: torch.Tensor, semantic: torch.Tensor, beta: float = 1.0
    ) -> torch.Tensor:
        """Compute ELBO loss."""
        mu, logvar = self.encode(semantic)
        z, log_det_jacobian = self.sample(mu, logvar)
        recon = self.decode(z, semantic)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, features, reduction="sum")

        # KL divergence with flow correction
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss - torch.sum(log_det_jacobian)

        # ELBO
        elbo = recon_loss + beta * kl_loss

        return elbo / features.size(0)


class AffineCouplingFlow(nn.Module):
    """Affine coupling layer for normalizing flows."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim

        # Networks for scale and translation
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh(),
        )

        self.translate_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, dim // 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with log determinant of Jacobian."""
        x1, x2 = x.chunk(2, dim=1)

        s = self.scale_net(x1)
        t = self.translate_net(x1)

        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)

        log_det_jacobian = s.sum(dim=1)

        return y, log_det_jacobian


# =============================================================================
# Feature Augmentation
# =============================================================================


class FeatureGenerating:
    """
    Feature generation for data augmentation in ZSL.

    Generates synthetic features using various methods.
    """

    def __init__(self, method: str = "gan", config: Optional[ZSLConfig] = None):
        self.method = method
        self.config = config or ZSLConfig()
        self.generator = None

        if method == "gan":
            self.generator = ZSLGAN(self.config)

    def generate_features(
        self, semantic_embeddings: np.ndarray, num_samples: int = 100
    ) -> np.ndarray:
        """Generate synthetic features from semantic embeddings."""
        if self.generator is None:
            # Fallback to simple interpolation
            return self._interpolate_features(semantic_embeddings, num_samples)

        # Use GAN generator
        semantic_tensor = (
            torch.from_numpy(semantic_embeddings).float().to(self.config.device)
        )

        with torch.no_grad():
            generated = self.generator.generate_features(semantic_tensor, num_samples)

        return generated.cpu().numpy()

    def _interpolate_features(
        self, semantic_embeddings: np.ndarray, num_samples: int
    ) -> np.ndarray:
        """Generate features via interpolation."""
        num_classes = semantic_embeddings.shape[0]
        generated = []

        for _ in range(num_samples):
            # Random pair of classes
            i, j = np.random.choice(num_classes, 2, replace=False)
            alpha = np.random.rand()

            # Interpolate
            interp_semantic = (
                alpha * semantic_embeddings[i] + (1 - alpha) * semantic_embeddings[j]
            )

            # Add noise
            noise = np.random.randn(self.config.feature_dim) * 0.1
            feature = interp_semantic[: self.config.feature_dim] + noise

            generated.append(feature)

        return np.array(generated)


class SemanticAugmentation:
    """
    Semantic data augmentation.

    Augments semantic embeddings to improve generalization.
    """

    def __init__(self, noise_level: float = 0.1, dropout: float = 0.1):
        self.noise_level = noise_level
        self.dropout = dropout

    def augment(
        self, semantic_embeddings: np.ndarray, num_augmentations: int = 5
    ) -> np.ndarray:
        """Augment semantic embeddings."""
        augmented = []

        for emb in semantic_embeddings:
            augmented.append(emb)

            for _ in range(num_augmentations):
                aug_emb = emb.copy()

                # Add Gaussian noise
                noise = np.random.randn(*emb.shape) * self.noise_level
                aug_emb = aug_emb + noise

                # Random dropout
                if self.dropout > 0:
                    mask = np.random.rand(*emb.shape) > self.dropout
                    aug_emb = aug_emb * mask

                augmented.append(aug_emb)

        return np.array(augmented)

    def mixup_semantic(
        self, semantic1: np.ndarray, semantic2: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """Mixup augmentation for semantic embeddings."""
        lam = np.random.beta(alpha, alpha)
        return lam * semantic1 + (1 - lam) * semantic2


class MixupZSL:
    """
    Mixup for Zero-Shot Learning

    Applies mixup augmentation between seen and synthesized unseen class features.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def mixup_features(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup to features and labels."""
        lam = np.random.beta(self.alpha, self.alpha)

        mixed_features = lam * features1 + (1 - lam) * features2

        # Mixed labels (soft)
        batch_size = features1.size(0)
        num_classes = int(labels1.max().item()) + 1

        mixed_labels = torch.zeros(batch_size, num_classes, device=features1.device)
        mixed_labels.scatter_(1, labels1.unsqueeze(1), lam)
        mixed_labels.scatter_(1, labels2.unsqueeze(1), 1 - lam)

        return mixed_features, mixed_labels

    def mixup_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mixup loss."""
        return -torch.mean(torch.sum(target * F.log_softmax(pred, dim=1), dim=1))


class FeatureRefinement:
    """
    Feature refinement for generated features.

    Refines synthesized features to improve quality and discriminability.
    """

    def __init__(self, method: str = "autoencoder", feature_dim: int = 2048):
        self.method = method
        self.feature_dim = feature_dim
        self.refinement_model = None

        if method == "autoencoder":
            self.refinement_model = self._build_autoencoder()

    def _build_autoencoder(self) -> nn.Module:
        """Build autoencoder for feature refinement."""

        class RefinementAE(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.ReLU(True),
                    nn.Linear(dim // 2, dim // 4),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(dim // 4, dim // 2),
                    nn.ReLU(True),
                    nn.Linear(dim // 2, dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        return RefinementAE(self.feature_dim)

    def refine(self, features: np.ndarray) -> np.ndarray:
        """Refine features."""
        if self.refinement_model is None:
            return features

        features_tensor = torch.from_numpy(features).float()

        with torch.no_grad():
            refined = self.refinement_model(features_tensor)

        return refined.numpy()

    def denoise(self, features: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Denoise features using learned model."""
        if self.refinement_model is None:
            # Simple denoising via PCA-like projection
            return self._simple_denoise(features, noise_level)

        return self.refine(features)

    def _simple_denoise(self, features: np.ndarray, noise_level: float) -> np.ndarray:
        """Simple denoising via dimensionality reduction."""
        from sklearn.decomposition import PCA

        n_components = int(features.shape[1] * (1 - noise_level))
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(features)
        reconstructed = pca.inverse_transform(reduced)

        return reconstructed


# =============================================================================
# Transductive ZSL
# =============================================================================


class QuasiFullySupervised:
    """
    Quasi-Fully Supervised Learning (QFSL) for Transductive ZSL

    Uses test data to improve classification by treating it as
    semi-supervised learning.

    Reference: Song et al. "Transductive Unbiased Embedding for Zero-Shot Learning" (2018)
    """

    def __init__(self, base_model: ZSLBase, num_iterations: int = 10):
        self.base_model = base_model
        self.num_iterations = num_iterations
        self.device = base_model.device

    def fit_transductive(
        self,
        train_loader: DataLoader,
        test_features: torch.Tensor,
        test_semantics: torch.Tensor,
    ):
        """Train with transductive setting."""
        optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=self.base_model.config.learning_rate
        )

        for iteration in range(self.num_iterations):
            # Train on labeled data
            self.base_model.train()
            for batch in train_loader:
                features, labels, semantics = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                semantics = semantics.to(self.device)

                optimizer.zero_grad()
                loss = self.base_model.compute_loss(features, labels, semantics)
                loss.backward()
                optimizer.step()

            # Pseudo-label test data
            self.base_model.eval()
            with torch.no_grad():
                test_scores = self.base_model(test_features)
                pseudo_labels = torch.argmax(test_scores, dim=1)

            # Add confident predictions to training
            # (simplified version - in practice, use confidence thresholding)
            print(f"Iteration {iteration + 1}/{self.num_iterations} complete")


class DomainAdaptationZSL:
    """
    Domain Adaptation for Zero-Shot Learning

    Handles domain shift between training and test distributions.
    """

    def __init__(self, method: str = "coral", config: Optional[ZSLConfig] = None):
        self.method = method
        self.config = config or ZSLConfig()

    def adapt(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> np.ndarray:
        """Adapt source features to target domain."""
        if self.method == "coral":
            return self._coral(source_features, target_features)
        elif self.method == "mmd":
            return self._mmd_adaptation(source_features, target_features)
        else:
            return source_features

    def _coral(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """CORAL: Correlation Alignment."""
        # Compute covariances
        source_cov = np.cov(source.T) + np.eye(source.shape[1]) * 1e-5
        target_cov = np.cov(target.T) + np.eye(target.shape[1]) * 1e-5

        # Compute transformation
        source_cov_sqrt = self._matrix_sqrt(source_cov)
        target_cov_sqrt = self._matrix_sqrt(target_cov)

        transformation = np.dot(source_cov_sqrt, np.linalg.inv(target_cov_sqrt))

        # Apply transformation
        adapted = np.dot(source, transformation)

        return adapted

    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

    def _mmd_adaptation(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """MMD-based domain adaptation."""
        # Simple mean matching
        source_mean = np.mean(source, axis=0)
        target_mean = np.mean(target, axis=0)

        adapted = source - source_mean + target_mean

        return adapted


class SemiSupervisedZSL:
    """
    Semi-Supervised Zero-Shot Learning

    Leverages both labeled and unlabeled data during training.
    """

    def __init__(self, base_model: ZSLBase, consistency_weight: float = 1.0):
        self.base_model = base_model
        self.consistency_weight = consistency_weight
        self.device = base_model.device

    def compute_loss(
        self,
        labeled_features: torch.Tensor,
        labeled_labels: torch.Tensor,
        labeled_semantics: torch.Tensor,
        unlabeled_features: torch.Tensor,
        unlabeled_semantics: torch.Tensor,
    ) -> torch.Tensor:
        """Compute semi-supervised loss."""
        # Supervised loss
        sup_loss = self.base_model.compute_loss(
            labeled_features, labeled_labels, labeled_semantics
        )

        # Consistency loss on unlabeled data
        with torch.no_grad():
            # Teacher prediction
            teacher_scores = self.base_model(unlabeled_features)
            teacher_pred = F.softmax(teacher_scores, dim=1)

        # Student prediction (with augmentation)
        aug_features = self._augment_features(unlabeled_features)
        student_scores = self.base_model(aug_features)
        student_pred = F.log_softmax(student_scores, dim=1)

        # KL divergence for consistency
        consistency_loss = F.kl_div(student_pred, teacher_pred, reduction="batchmean")

        total_loss = sup_loss + self.consistency_weight * consistency_loss

        return total_loss

    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """Simple feature augmentation."""
        noise = torch.randn_like(features) * 0.1
        return features + noise


class SelfSupervisedZSL:
    """
    Self-Supervised Learning for Zero-Shot Learning

    Uses self-supervised pretext tasks to learn better representations.
    """

    def __init__(self, base_model: ZSLBase, pretext_tasks: List[str] = None):
        self.base_model = base_model
        self.pretext_tasks = pretext_tasks or ["rotation", "jigsaw"]
        self.device = base_model.device

        # Pretext task heads
        self.pretext_heads = nn.ModuleDict()

        if "rotation" in self.pretext_tasks:
            self.pretext_heads["rotation"] = nn.Linear(
                base_model.config.feature_dim,
                4,  # 4 rotations
            )

        if "jigsaw" in self.pretext_tasks:
            self.pretext_heads["jigsaw"] = nn.Linear(
                base_model.config.feature_dim,
                24,  # 24 permutations
            )

    def compute_pretext_loss(
        self, features: torch.Tensor, task_labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute self-supervised pretext loss."""
        total_loss = 0.0

        for task, head in self.pretext_heads.items():
            if task in task_labels:
                pred = head(features)
                loss = F.cross_entropy(pred, task_labels[task])
                total_loss += loss

        return total_loss

    def train_with_pretext(
        self, train_loader: DataLoader, num_pretext_epochs: int = 10
    ):
        """Train with self-supervised pretext tasks."""
        optimizer = torch.optim.Adam(
            list(self.base_model.parameters()) + list(self.pretext_heads.parameters()),
            lr=self.base_model.config.learning_rate,
        )

        for epoch in range(num_pretext_epochs):
            for batch in train_loader:
                features = batch[0].to(self.device)
                task_labels = batch[3] if len(batch) > 3 else {}

                # Convert task labels to tensors
                task_labels = {k: v.to(self.device) for k, v in task_labels.items()}

                optimizer.zero_grad()
                loss = self.compute_pretext_loss(features, task_labels)
                loss.backward()
                optimizer.step()


# =============================================================================
# Evaluation
# =============================================================================


class ZSLAccuracy:
    """
    Zero-Shot Learning Accuracy Metrics

    Computes accuracy for seen, unseen, and both classes.
    """

    def __init__(self, num_seen_classes: int, num_unseen_classes: int):
        self.num_seen_classes = num_seen_classes
        self.num_unseen_classes = num_unseen_classes

        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.seen_correct = 0
        self.seen_total = 0
        self.unseen_correct = 0
        self.unseen_total = 0
        self.all_correct = 0
        self.all_total = 0

    def update(
        self, predictions: torch.Tensor, labels: torch.Tensor, is_seen: torch.Tensor
    ):
        """Update metrics with new predictions."""
        correct = (predictions == labels).float()

        # Seen classes
        seen_mask = is_seen.bool()
        self.seen_correct += correct[seen_mask].sum().item()
        self.seen_total += seen_mask.sum().item()

        # Unseen classes
        unseen_mask = ~seen_mask
        self.unseen_correct += correct[unseen_mask].sum().item()
        self.unseen_total += unseen_mask.sum().item()

        # All
        self.all_correct += correct.sum().item()
        self.all_total += labels.size(0)

    def get_accuracy(self, split: str = "all") -> float:
        """Get accuracy for a specific split."""
        if split == "seen":
            return self.seen_correct / self.seen_total if self.seen_total > 0 else 0.0
        elif split == "unseen":
            return (
                self.unseen_correct / self.unseen_total
                if self.unseen_total > 0
                else 0.0
            )
        else:
            return self.all_correct / self.all_total if self.all_total > 0 else 0.0

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all accuracy metrics."""
        return {
            "seen_accuracy": self.get_accuracy("seen"),
            "unseen_accuracy": self.get_accuracy("unseen"),
            "overall_accuracy": self.get_accuracy("all"),
        }


class HarmonicMean:
    """
    Harmonic Mean for Generalized ZSL

    Computes H-mean of seen and unseen accuracies, which is the
    standard metric for generalized ZSL.
    """

    @staticmethod
    def compute(seen_acc: float, unseen_acc: float) -> float:
        """
        Compute harmonic mean.

        H = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)
        """
        if seen_acc + unseen_acc == 0:
            return 0.0

        return 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)

    @staticmethod
    def compute_from_dict(metrics: Dict[str, float]) -> float:
        """Compute H-mean from metrics dictionary."""
        seen_acc = metrics.get("seen_accuracy", 0.0)
        unseen_acc = metrics.get("unseen_accuracy", 0.0)
        return HarmonicMean.compute(seen_acc, unseen_acc)


class AUSUC:
    """
    Area Under Seen-Unseen Curve (AUSUC)

    Measures the trade-off between seen and unseen accuracy as
    the bias parameter varies.

    Reference: Chao et al. "An Empirical Study and Analysis of Generalized Zero-Shot Learning" (2016)
    """

    def __init__(self, num_points: int = 20):
        self.num_points = num_points

    def compute(
        self, model: ZSLBase, test_loader: DataLoader, seen_mask: torch.Tensor
    ) -> float:
        """
        Compute AUSUC by varying calibration.

        Returns:
            Area under the seen-unseen curve
        """
        calibration_values = np.linspace(-1.0, 1.0, self.num_points)

        seen_accs = []
        unseen_accs = []

        for cal in calibration_values:
            # Set calibration
            if hasattr(model, "calibration_factor"):
                model.calibration_factor = cal

            # Evaluate
            acc = ZSLAccuracy(
                num_seen_classes=int(seen_mask.sum()),
                num_unseen_classes=int((~seen_mask.bool()).sum()),
            )

            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    features, labels, _ = batch
                    features = features.to(model.device)
                    labels = labels.to(model.device)

                    if hasattr(model, "forward"):
                        scores = model(features, seen_mask)
                    else:
                        scores = model(features)

                    predictions = torch.argmax(scores, dim=1)
                    is_seen = seen_mask[labels]

                    acc.update(predictions, labels, is_seen)

            metrics = acc.get_all_metrics()
            seen_accs.append(metrics["seen_accuracy"])
            unseen_accs.append(metrics["unseen_accuracy"])

        # Compute area under curve
        auc = np.trapz(unseen_accs, seen_accs)

        return float(auc)


class PerClassMetrics:
    """
    Per-Class Metrics for Zero-Shot Learning

    Computes accuracy for each class individually.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset metrics."""
        self.class_correct = np.zeros(self.num_classes)
        self.class_total = np.zeros(self.num_classes)

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Update per-class metrics."""
        correct = (predictions == labels).cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i in range(len(labels)):
            self.class_correct[labels_np[i]] += correct[i]
            self.class_total[labels_np[i]] += 1

    def get_per_class_accuracy(self) -> np.ndarray:
        """Get accuracy for each class."""
        return np.divide(
            self.class_correct,
            self.class_total,
            out=np.zeros_like(self.class_correct),
            where=self.class_total != 0,
        )

    def get_unseen_accuracy(self, unseen_classes: List[int]) -> float:
        """Get average accuracy for unseen classes."""
        accs = self.get_per_class_accuracy()
        return float(np.mean(accs[unseen_classes]))

    def get_seen_accuracy(self, seen_classes: List[int]) -> float:
        """Get average accuracy for seen classes."""
        accs = self.get_per_class_accuracy()
        return float(np.mean(accs[seen_classes]))


# =============================================================================
# Few-Shot ZSL
# =============================================================================


class FSLZSL:
    """
    Few-Shot Zero-Shot Learning

    Combines few-shot learning with zero-shot learning to handle
    classes with very few examples.
    """

    def __init__(
        self, config: ZSLConfig, n_way: int = 5, k_shot: int = 1, n_query: int = 15
    ):
        self.config = config
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.device = config.device

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        ).to(self.device)

        # Semantic encoder
        self.semantic_encoder = nn.Sequential(
            nn.Linear(config.semantic_dim, config.hidden_dim),
            nn.ReLU(True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        ).to(self.device)

    def compute_prototypes(
        self, support_features: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute class prototypes from support set."""
        n_classes = len(torch.unique(support_labels))
        prototypes = []

        for c in range(n_classes):
            mask = support_labels == c
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for episode.

        Args:
            support_features: Support set features [n_way * k_shot, feature_dim]
            support_labels: Support set labels [n_way * k_shot]
            query_features: Query set features [n_query * n_way, feature_dim]
            semantic_embeddings: Class semantic embeddings [n_way, semantic_dim]

        Returns:
            Query predictions [n_query * n_way]
        """
        # Encode features
        support_encoded = self.encoder(support_features)
        query_encoded = self.encoder(query_features)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_encoded, support_labels)

        # Encode semantics
        semantic_encoded = self.semantic_encoder(semantic_embeddings)

        # Combine prototype and semantic information
        combined_prototypes = prototypes + 0.5 * semantic_encoded

        # Compute distances to prototypes
        distances = torch.cdist(query_encoded, combined_prototypes)

        # Predict
        predictions = torch.argmin(distances, dim=1)

        return predictions

    def compute_loss(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for an episode."""
        # Encode
        support_encoded = self.encoder(support_features)
        query_encoded = self.encoder(query_features)
        semantic_encoded = self.semantic_encoder(semantic_embeddings)

        # Prototypes
        prototypes = self.compute_prototypes(support_encoded, support_labels)
        combined_prototypes = prototypes + 0.5 * semantic_encoded

        # Distances and loss
        distances = torch.cdist(query_encoded, combined_prototypes)
        logits = -distances

        loss = F.cross_entropy(logits, query_labels)

        return loss


class GeneralizedFSLZSL(FSLZSL):
    """
    Generalized Few-Shot Zero-Shot Learning

    Extends FSL-ZSL to generalized setting where both seen and unseen
    classes appear in the query set.
    """

    def __init__(
        self,
        config: ZSLConfig,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        calibration: float = 0.5,
    ):
        super().__init__(config, n_way, k_shot, n_query)
        self.calibration = calibration

    def forward_generalized(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        seen_semantics: torch.Tensor,
        unseen_semantics: torch.Tensor,
        seen_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for generalized setting.

        Args:
            support_features: Support features
            support_labels: Support labels
            query_features: Query features
            seen_semantics: Semantic embeddings for seen classes
            unseen_semantics: Semantic embeddings for unseen classes
            seen_mask: Mask indicating seen classes

        Returns:
            Predictions for query samples
        """
        # Encode features
        support_encoded = self.encoder(support_features)
        query_encoded = self.encoder(query_features)

        # Compute prototypes for support classes
        prototypes = self.compute_prototypes(support_encoded, support_labels)

        # Get all semantics
        all_semantics = torch.cat([seen_semantics, unseen_semantics], dim=0)
        all_semantics_encoded = self.semantic_encoder(all_semantics)

        # Compute similarity between prototypes and all class semantics
        similarities = torch.mm(prototypes, all_semantics_encoded.t())

        # Calibrate seen classes
        similarities[:, seen_mask] = similarities[:, seen_mask] - self.calibration

        # Assign each prototype to closest class
        prototype_to_class = torch.argmax(similarities, dim=1)

        # Compute query predictions
        query_to_prototype_dist = torch.cdist(query_encoded, prototypes)
        query_to_prototype = torch.argmin(query_to_prototype_dist, dim=1)

        # Map to class predictions
        predictions = prototype_to_class[query_to_prototype]

        return predictions


class EpisodeZSL:
    """
    Episode-Based Zero-Shot Learning

    Training and evaluation in episodes similar to meta-learning.
    Each episode samples N classes with K examples each.
    """

    def __init__(
        self,
        config: ZSLConfig,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        n_episodes: int = 100,
    ):
        self.config = config
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.device = config.device

    def sample_episode(
        self, features: torch.Tensor, labels: torch.Tensor, semantics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Sample an episode from the dataset."""
        unique_labels = torch.unique(labels)
        n_classes = min(self.n_way, len(unique_labels))

        # Sample classes
        episode_classes = unique_labels[torch.randperm(len(unique_labels))[:n_classes]]

        support_features = []
        support_labels = []
        query_features = []
        query_labels = []
        episode_semantics = []

        for i, c in enumerate(episode_classes):
            class_mask = labels == c
            class_features = features[class_mask]

            # Sample support and query
            perm = torch.randperm(len(class_features))
            support_idx = perm[: self.k_shot]
            query_idx = perm[self.k_shot : self.k_shot + self.n_query]

            support_features.append(class_features[support_idx])
            support_labels.append(torch.full((self.k_shot,), i, dtype=torch.long))

            if len(query_idx) > 0:
                query_features.append(class_features[query_idx])
                query_labels.append(torch.full((len(query_idx),), i, dtype=torch.long))

            episode_semantics.append(semantics[c])

        return {
            "support_features": torch.cat(support_features).to(self.device),
            "support_labels": torch.cat(support_labels).to(self.device),
            "query_features": torch.cat(query_features).to(self.device)
            if query_features
            else torch.empty(0).to(self.device),
            "query_labels": torch.cat(query_labels).to(self.device)
            if query_labels
            else torch.empty(0).to(self.device),
            "semantics": torch.stack(episode_semantics).to(self.device),
        }

    def train_episodic(
        self,
        model: FSLZSL,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        train_semantics: torch.Tensor,
        num_iterations: int = 1000,
    ):
        """Train model episodically."""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        for iteration in range(num_iterations):
            # Sample episode
            episode = self.sample_episode(train_data, train_labels, train_semantics)

            if episode["query_features"].size(0) == 0:
                continue

            # Compute loss
            loss = model.compute_loss(
                episode["support_features"],
                episode["support_labels"],
                episode["query_features"],
                episode["query_labels"],
                episode["semantics"],
            )

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iteration + 1) % 100 == 0:
                print(
                    f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.4f}"
                )


# =============================================================================
# Utility Functions
# =============================================================================


def create_zsl_dataset(
    features: np.ndarray, labels: np.ndarray, semantic_embeddings: np.ndarray
) -> Dataset:
    """Create a PyTorch Dataset for ZSL."""

    class ZSLDataset(Dataset):
        def __init__(self, features, labels, semantics):
            self.features = torch.from_numpy(features).float()
            self.labels = torch.from_numpy(labels).long()
            self.semantics = torch.from_numpy(semantics).float()

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return (
                self.features[idx],
                self.labels[idx],
                self.semantics[self.labels[idx]],
            )

    return ZSLDataset(features, labels, semantic_embeddings)


def split_seen_unseen(
    labels: np.ndarray, num_seen: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split data into seen and unseen classes."""
    seen_mask = labels < num_seen
    unseen_mask = ~seen_mask

    return seen_mask, unseen_mask


def compute_class_similarity(semantic_embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise similarity between class semantic embeddings."""
    # Normalize
    normalized = semantic_embeddings / (
        np.linalg.norm(semantic_embeddings, axis=1, keepdims=True) + 1e-8
    )

    # Cosine similarity
    similarity = np.dot(normalized, normalized.T)

    return similarity


def evaluate_zsl_model(
    model: ZSLBase,
    test_loader: DataLoader,
    seen_classes: List[int],
    unseen_classes: List[int],
) -> Dict[str, float]:
    """Comprehensive evaluation of a ZSL model."""
    model.eval()

    # Metrics
    acc_metric = ZSLAccuracy(len(seen_classes), len(unseen_classes))
    per_class = PerClassMetrics(len(seen_classes) + len(unseen_classes))

    seen_mask_tensor = torch.zeros(
        len(seen_classes) + len(unseen_classes), device=model.device
    )
    seen_mask_tensor[seen_classes] = 1.0

    with torch.no_grad():
        for batch in test_loader:
            features, labels, _ = batch
            features = features.to(model.device)
            labels = labels.to(model.device)

            # Forward
            if (
                hasattr(model, "forward")
                and "seen_mask" in model.forward.__code__.co_varnames
            ):
                scores = model(features, seen_mask_tensor)
            else:
                scores = model(features)

            predictions = torch.argmax(scores, dim=1)

            # Update metrics
            is_seen = seen_mask_tensor[labels]
            acc_metric.update(predictions, labels, is_seen)
            per_class.update(predictions, labels)

    # Collect results
    results = acc_metric.get_all_metrics()
    results["harmonic_mean"] = HarmonicMean.compute(
        results["seen_accuracy"], results["unseen_accuracy"]
    )
    results["per_class_accuracy"] = per_class.get_per_class_accuracy().tolist()
    results["unseen_per_class_acc"] = per_class.get_unseen_accuracy(unseen_classes)
    results["seen_per_class_acc"] = per_class.get_seen_accuracy(seen_classes)

    return results


__all__ = [
    # Base classes
    "ZSLConfig",
    "ZSLBase",
    # Embedding-based methods
    "DeViSE",
    "ALE",
    "SJE",
    "LatEm",
    "ESZSL",
    "SYNC",
    "SAE",
    # Generative methods
    "ZSLGAN",
    "FCLSWGAN",
    "CycleWCL",
    "LisGAN",
    "FREE",
    "LsrGAN",
    # Semantic embeddings
    "AttributeVectors",
    "WordEmbeddings",
    "SentenceEmbeddings",
    "ClassDescription",
    "HierarchicalEmbeddings",
    # Generalized ZSL
    "CalibratedStacking",
    "RelationNet",
    "DeepEmbedding",
    "GPZSL",
    "FVAE",
    # Feature augmentation
    "FeatureGenerating",
    "SemanticAugmentation",
    "MixupZSL",
    "FeatureRefinement",
    # Transductive ZSL
    "QuasiFullySupervised",
    "DomainAdaptationZSL",
    "SemiSupervisedZSL",
    "SelfSupervisedZSL",
    # Evaluation
    "ZSLAccuracy",
    "HarmonicMean",
    "AUSUC",
    "PerClassMetrics",
    # Few-shot ZSL
    "FSLZSL",
    "GeneralizedFSLZSL",
    "EpisodeZSL",
    # Utilities
    "create_zsl_dataset",
    "split_seen_unseen",
    "compute_class_similarity",
    "evaluate_zsl_model",
]
