"""
First-Order MAML (FOMAML) implementation.

FOMAML is a computationally efficient variant of MAML that ignores
second-order gradients during meta-update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict

from .maml import MAML
from .config import MAMLConfig
from .types import FewShotTask, AdaptationResult


class FOMAML(MAML):
    """First-Order Model-Agnostic Meta-Learning (FOMAML).

    FOMAML is computationally more efficient than MAML by ignoring
    second-order derivatives in the meta-update.

    Args:
        encoder: Feature encoder network
        num_classes: Number of output classes
        config: MAML configuration (first_order=True by default)

    References:
        Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
        Nichol et al. "On First-Order Meta-Learning Algorithms"
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        config: Optional[MAMLConfig] = None,
    ):
        if config is None:
            config = MAMLConfig(first_order=True)
        else:
            config.first_order = True

        super().__init__(encoder, num_classes, config)

    def meta_train_step(
        self,
        tasks: list,
        outer_optimizer: torch.optim.Optimizer,
    ) -> tuple:
        """Perform FOMAML meta-training step.

        Uses first-order approximation for efficiency.
        """
        outer_optimizer.zero_grad()

        meta_loss = 0.0
        metrics = {"query_loss": 0.0}

        for task in tasks:
            adapted_params = self._inner_loop(task.support_x, task.support_y)

            query_logits = self.forward(task.query_x, adapted_params)
            query_loss = F.cross_entropy(query_logits, task.query_y)

            meta_loss = meta_loss + query_loss
            metrics["query_loss"] += query_loss.item()

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()

        outer_optimizer.step()

        metrics["query_loss"] /= len(tasks)

        return meta_loss, metrics
