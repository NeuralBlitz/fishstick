"""
Multiple Instance Learning for Medical Imaging

MIL pooling and attention-based MIL for whole slide imaging.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MILPool(nn.Module):
    """Multiple Instance Learning pooling.
    
    Aggregates instance-level predictions into bag-level predictions.
    """

    def __init__(
        self,
        aggregation: str = "max",
    ):
        super().__init__()
        self.aggregation = aggregation

    def forward(
        self,
        instances: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if instances.ndim == 2:
            instances = instances.unsqueeze(0)
        
        if mask is not None:
            instances = instances * mask.unsqueeze(-1)
        
        if self.aggregation == "max":
            pooled, indices = instances.max(dim=1)
        elif self.aggregation == "mean":
            pooled = instances.mean(dim=1)
        elif self.aggregation == "attention":
            raise NotImplementedError("Use AttentionMIL for attention pooling")
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return pooled, indices if self.aggregation == "max" else None


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning.
    
    Uses learned attention weights to aggregate instances.
    
    Example:
        >>> mil = AttentionMIL(feature_dim=512, num_classes=2)
        >>> bag_output = mil(instance_features)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = feature_dim // 2
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        instances: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if instances.ndim == 2:
            instances = instances.unsqueeze(0)
        
        b, n, d = instances.shape
        
        attention_scores = self.attention(instances).squeeze(-1)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        weighted_instances = instances * attention_weights.unsqueeze(-1)
        
        bag_features = weighted_instances.sum(dim=1)
        
        logits = self.classifier(bag_features)
        
        return logits, {
            "attention_weights": attention_weights,
            "bag_features": bag_features,
        }

    def predict(
        self,
        instances: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        logits, info = self.forward(instances, mask)
        
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        
        attention = info["attention_weights"]
        
        return preds, attention


class DeepMIL(nn.Module):
    """Deep Multiple Instance Learning with feature extraction.
    
    End-to-end MIL with convolutional feature extraction.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        
        self.projection = nn.Linear(128, feature_dim)
        
        self.mil = AttentionMIL(feature_dim, num_classes)

    def forward(
        self,
        patches: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        b, n, c, d, h, w = patches.shape
        
        patches = patches.view(b * n, c, d, h, w)
        
        features = self.feature_extractor(patches)
        features = features.squeeze(-1).squeeze(-1).squeeze(-1)
        
        features = self.projection(features)
        
        features = features.view(b, n, -1)
        
        return self.mil(features, mask)


class InstanceSelectorMIL(nn.Module):
    """MIL with learned instance selection."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        k: int = 10,
    ):
        super().__init__()
        
        self.k = k
        
        self.selector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        instances: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if instances.ndim == 2:
            instances = instances.unsqueeze(0)
        
        b, n, d = instances.shape
        
        scores = self.selector(instances).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        topk_scores, topk_indices = torch.topk(scores, min(self.k, n), dim=1)
        
        selected_instances = torch.gather(
            instances,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, d),
        )
        
        bag_features = selected_instances.mean(dim=1)
        
        logits = self.classifier(bag_features)
        
        attention_weights = F.softmax(scores, dim=1)
        
        return logits, {
            "attention_weights": attention_weights,
            "selected_indices": topk_indices,
            "bag_features": bag_features,
        }
