"""
ANIL (Almost No Inner Loop) implementation.

ANIL is a variant of MAML where the inner loop is applied only to the
classifier head, not the feature encoder. This reduces computational
cost while maintaining competitive performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List

from .config import MAMLConfig
from .types import FewShotTask, AdaptationResult


class ANIL(nn.Module):
    """Almost No Inner Loop (ANIL) meta-learner.

    ANIL freezes the encoder during inner loop adaptation and only
    adapts the classifier head.

    Args:
        encoder: Feature encoder network (frozen during adaptation)
        num_classes: Number of output classes
        inner_lr: Learning rate for classifier adaptation
        num_inner_steps: Number of inner loop steps (typically fewer than MAML)

    Example:
        >>> encoder = ResNet18(pretrained=False)
        >>> anil = ANIL(encoder, num_classes=5, inner_lr=0.1, num_inner_steps=1)
        >>> task = FewShotTask(support_x, support_y, query_x, query_y, 5, 5, 15)
        >>> result = anil.adapt(task)

    References:
        Raghu et al. "Rapid Learning or Feature Reuse? Towards Understanding
        the Effectiveness of MAML" (ICLR 2020)
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        inner_lr: float = 0.1,
        num_inner_steps: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        self._freeze_encoder()

        self.classifier = nn.Linear(self._get_embedding_dim(), num_classes)

    def _freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        """Unfreeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension by forward pass."""
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = self.encoder(dummy)
        self.encoder.train()
        return out.view(out.size(0), -1).size(1)

    def forward(
        self, x: Tensor, classifier_params: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """Forward pass through encoder and classifier.

        Args:
            x: Input tensor
            classifier_params: Optional classifier parameters for adaptation

        Returns:
            Logits [batch_size, num_classes]
        """
        with torch.no_grad():
            features = self.encoder(x).view(x.size(0), -1)

        if classifier_params is None:
            logits = self.classifier(features)
        else:
            logits = F.linear(
                features, classifier_params["weight"], classifier_params["bias"]
            )

        return logits

    def get_classifier_params(self) -> Dict[str, Tensor]:
        """Get current classifier parameters."""
        return {
            "weight": self.classifier.weight.data.clone(),
            "bias": self.classifier.bias.data.clone()
            if self.classifier.bias is not None
            else None,
        }

    def adapt(self, task: FewShotTask) -> AdaptationResult:
        """Adapt classifier to the few-shot task.

        Args:
            task: Few-shot task

        Returns:
            AdaptationResult
        """
        adapted_params = self._inner_loop(task.support_x, task.support_y)

        query_logits = self.forward(task.query_x, adapted_params)
        query_loss = F.cross_entropy(query_logits, task.query_y)

        return AdaptationResult(
            adapted_model=adapted_params,
            query_logits=query_logits,
            query_loss=query_loss,
        )

    def _inner_loop(self, support_x: Tensor, support_y: Tensor) -> Dict[str, Tensor]:
        """Inner loop adaptation - only adapts classifier."""
        adapted_params = self.get_classifier_params()

        for _ in range(self.num_inner_steps):
            logits = self.forward(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)

            grads = torch.autograd.grad(
                loss,
                [adapted_params["weight"], adapted_params["bias"]],
                create_graph=False,
            )

            adapted_params["weight"] = (
                adapted_params["weight"] - self.inner_lr * grads[0]
            )
            if adapted_params["bias"] is not None and grads[1] is not None:
                adapted_params["bias"] = (
                    adapted_params["bias"] - self.inner_lr * grads[1]
                )

        return adapted_params

    def meta_train_step(
        self,
        tasks: List[FewShotTask],
        outer_optimizer: torch.optim.Optimizer,
    ) -> tuple:
        """Perform meta-training step."""
        outer_optimizer.zero_grad()

        meta_loss = 0.0
        metrics = {"query_loss": 0.0}

        for task in tasks:
            result = self.adapt(task)
            meta_loss = meta_loss + result.query_loss
            metrics["query_loss"] += result.query_loss.item()

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()

        outer_optimizer.step()

        metrics["query_loss"] /= len(tasks)

        return meta_loss, metrics


class BOIL(nn.Module):
    """Bootstrapped Inner Loop (BOIL) meta-learner.

    BOIL is similar to ANIL but uses a learned initialization for
    the classifier and can optionally perform gradient-based updates.

    References:
        Liu et al. "Bootstrapped Meta-Learning" (ICLR 2022)
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        inner_lr: float = 0.1,
        num_inner_steps: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        self._freeze_encoder()
        self.classifier = nn.Linear(self._get_embedding_dim(), num_classes)

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _get_embedding_dim(self) -> int:
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = self.encoder(dummy)
        self.encoder.train()
        return out.view(out.size(0), -1).size(1)

    def forward(
        self, x: Tensor, classifier_params: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        with torch.no_grad():
            features = self.encoder(x).view(x.size(0), -1)

        if classifier_params is None:
            logits = self.classifier(features)
        else:
            logits = F.linear(
                features, classifier_params["weight"], classifier_params.get("bias")
            )

        return logits

    def get_classifier_params(self) -> Dict[str, Tensor]:
        return {
            "weight": self.classifier.weight.data.clone(),
            "bias": self.classifier.bias.data.clone()
            if self.classifier.bias is not None
            else None,
        }

    def adapt(self, task: FewShotTask) -> AdaptationResult:
        """Adapt to task - zero steps means use meta-learned initialization."""
        if self.num_inner_steps == 0:
            adapted_params = self.get_classifier_params()
        else:
            adapted_params = self._inner_loop(task.support_x, task.support_y)

        query_logits = self.forward(task.query_x, adapted_params)
        query_loss = F.cross_entropy(query_logits, task.query_y)

        return AdaptationResult(
            adapted_model=adapted_params,
            query_logits=query_logits,
            query_loss=query_loss,
        )

    def _inner_loop(self, support_x: Tensor, support_y: Tensor) -> Dict[str, Tensor]:
        adapted_params = self.get_classifier_params()

        for _ in range(self.num_inner_steps):
            logits = self.forward(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)

            grads = torch.autograd.grad(
                loss,
                [adapted_params["weight"], adapted_params["bias"]],
                create_graph=False,
            )

            adapted_params["weight"] = (
                adapted_params["weight"] - self.inner_lr * grads[0]
            )
            if adapted_params["bias"] is not None:
                adapted_params["bias"] = (
                    adapted_params["bias"] - self.inner_lr * grads[1]
                )

        return adapted_params

    def meta_train_step(self, tasks: List, outer_optimizer):
        outer_optimizer.zero_grad()
        meta_loss = 0.0
        for task in tasks:
            result = self.adapt(task)
            meta_loss = meta_loss + result.query_loss
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        outer_optimizer.step()
        return meta_loss, {"query_loss": meta_loss.item()}
