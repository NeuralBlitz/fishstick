"""
Disease Classification Frameworks

Multi-task learning and ensemble classification for medical images.
"""

from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiTaskClassifier(nn.Module):
    """Multi-task classifier for disease classification with auxiliary tasks.
    
    Example:
        >>> model = MultiTaskClassifier(backbone, task_heads={'main': 2, 'severity': 5})
    """

    def __init__(
        self,
        backbone: nn.Module,
        task_heads: Dict[str, int],
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.task_heads = nn.ModuleDict()
        
        feature_dim = backbone.get_feature_dim()
        
        for task_name, num_classes in task_heads.items():
            self.task_heads[task_name] = nn.Linear(feature_dim, num_classes)
        
        if task_weights is None:
            task_weights = {name: 1.0 for name in task_heads}
        
        self.task_weights = task_weights

    def forward(
        self,
        x: Tensor,
        task: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        features = self.backbone.extract_features(x)
        
        outputs = {}
        
        if task is not None:
            if task in self.task_heads:
                outputs[task] = self.task_heads[task](features)
            return outputs
        
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(features)
        
        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss = 0.0
        task_losses = {}
        
        criterion = nn.CrossEntropyLoss()
        
        for task_name, output in outputs.items():
            if task_name in targets:
                task_loss = criterion(output, targets[task_name])
                task_losses[task_name] = task_loss.item()
                
                weight = self.task_weights.get(task_name, 1.0)
                loss = loss + weight * task_loss
        
        return loss, task_losses


class EnsembleClassifier(nn.Module):
    """Ensemble classifier combining multiple models.
    
    Combines predictions from multiple backbones or models.
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        fusion: str = "average",
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.weights = weights
        self.fusion = fusion

    def forward(self, x: Tensor) -> Tensor:
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        if self.fusion == "average":
            weighted_sum = sum(p * w for p, w in zip(predictions, self.weights))
            return weighted_sum / sum(self.weights)
        
        return predictions[0]


class DiseaseClassifier(nn.Module):
    """Disease classification framework with interpretability.
    
    Complete pipeline for disease classification with attention maps.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        use_attention: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        feature_dim = backbone.get_feature_dim()
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 4, 1),
            )

    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        features = self.backbone.extract_features(x)
        
        features = self.dropout(features)
        
        logits = self.classifier(features)
        
        output = {"logits": logits}
        
        if self.use_attention:
            attention_scores = self.attention(features)
            attention_weights = F.softmax(attention_scores, dim=0)
            output["attention"] = attention_weights
            
            weighted_features = features * attention_weights
            output["weighted_logits"] = self.classifier(weighted_features)
        
        return output

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.forward(x, return_attention=True)
        
        logits = output.get("weighted_logits", output["logits"])
        
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        
        attention = output.get("attention", None)
        
        return preds, attention


class TransferLearningWrapper(nn.Module):
    """Wrapper for transfer learning from pretrained models."""

    def __init__(
        self,
        pretrained_backbone: nn.Module,
        num_classes: int,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.backbone = pretrained_backbone
        self.num_classes = num_classes
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        feature_dim = self.backbone.get_feature_dim()
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone.extract_features(x)
        return self.classifier(features)
