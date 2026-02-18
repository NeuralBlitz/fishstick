"""
Task-Agnostic Continual Learning Methods.

Methods that work without explicit task boundaries or task IDs.

Classes:
- TaskAgnosticContinualLearner: Base task-agnostic learner
- MetaLearningAgnostic: Meta-learning approach
- OMLearnable: Online Meta-Learning variant
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import copy


class TaskAgnosticContinualLearner(nn.Module):
    """
    Base class for Task-Agnostic Continual Learning.
    
    Methods that learn continuously without explicit task boundaries.
    
    Args:
        model: Neural network
        device: Device for computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.model = model
        self.device = device
        
        self.steps = 0
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(x)
    
    def update(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """
        Update on a single batch.
        
        Args:
            batch: (inputs, targets) tuple
            
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        return {"steps": self.steps}


class MetaLearningAgnostic(TaskAgnosticContinualLearner):
    """
    Meta-Learning for Task-Agnostic Continual Learning.
    
    Uses meta-learning to quickly adapt to new experiences
    while maintaining knowledge of past tasks.
    
    Args:
        model: Neural network
        meta_lr: Meta-learning rate
        inner_lr: Inner loop learning rate
        device: Device for computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 1e-3,
        inner_lr: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def inner_update(
        self,
        support_x: Tensor,
        support_y: Tensor,
    ) -> nn.Module:
        """
        Perform inner loop update on support set.
        
        Args:
            support_x: Support inputs
            support_y: Support targets
            
        Returns:
            Adapted model
        """
        adapted = copy.deepcopy(self.model)
        
        adapted.eval()
        
        for _ in range(5):
            logits = adapted(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            grads = torch.autograd.grad(
                loss,
                adapted.parameters(),
                create_graph=True,
                allow_unused=True,
            )
            
            with torch.no_grad():
                for (name, param), grad in zip(adapted.named_parameters(), grads):
                    if grad is not None:
                        param -= self.inner_lr * grad
                        
        return adapted
    
    def outer_update(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        query_y: Tensor,
    ) -> Dict[str, float]:
        """
        Perform outer loop update using query set.
        
        Args:
            support_x: Support inputs
            support_y: Support targets  
            query_x: Query inputs
            query_y: Query targets
            
        Returns:
            Dictionary of losses
        """
        adapted = self.inner_update(support_x, support_y)
        
        adapted.eval()
        
        query_logits = adapted(query_x)
        query_loss = F.cross_entropy(query_logits, query_y)
        
        self.meta_optimizer.zero_grad()
        
        self.model.train()
        
        logits = self.model(support_x)
        support_loss = F.cross_entropy(logits, support_y)
        
        total_loss = support_loss + query_loss
        
        total_loss.backward()
        self.meta_optimizer.step()
        
        self.steps += 1
        
        return {
            "support_loss": support_loss.item(),
            "query_loss": query_loss.item(),
            "total_loss": total_loss.item(),
        }
    
    def update(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """Update on batch."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        batch_size = x.size(0)
        
        split = batch_size // 2
        
        support_x = x[:split]
        support_y = y[:split]
        query_x = x[split:]
        query_y = y[split:]
        
        return self.outer_update(support_x, support_y, query_x, query_y)


class OMLearnable(TaskAgnosticContinualLearner):
    """
    Online Meta-Learning (OML) for Continual Learning.
    
    Learns representations that can quickly adapt to new tasks
    while maintaining stability.
    
    Reference:
        Javed & White, "Meta-Learning Representations for Continual Learning", NeurIPS 2019
        
    Args:
        model: Feature extractor
        classifier: Task-specific classifier
        meta_lr: Meta-learning rate
        device: Device for computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        classifier: Optional[nn.Module] = None,
        meta_lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        
        self.classifier = classifier or nn.Linear(512, 10)
        self.meta_lr = meta_lr
        
        self.feature_dim = 512
        
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.classifier.parameters()),
            lr=meta_lr,
        )
        
    def forward(self, x: Tensor, task_id: Optional[int] = None) -> Tensor:
        """Forward with feature extraction."""
        features = self.model(x)
        
        if isinstance(features, tuple):
            features = features[0]
            
        return self.classifier(features)
    
    def update(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """Update on batch."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        
        features = self.model(x)
        
        if isinstance(features, tuple):
            features = features[0]
            
        logits = self.classifier(features)
        
        loss = F.cross_entropy(logits, y)
        
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        
        return {"loss": loss.item()}
    
    def compute_meta_loss(
        self,
        current_x: Tensor,
        current_y: Tensor,
        prev_features: Tensor,
    ) -> Tensor:
        """
        Compute meta-regularization to preserve representations.
        
        Args:
            current_x: Current inputs
            current_y: Current targets
            prev_features: Features from previous step
            
        Returns:
            Meta regularization loss
        """
        current_features = self.model(current_x)
        
        if isinstance(current_features, tuple):
            current_features = current_features[0]
            
        meta_loss = F.mse_loss(current_features, prev_features.detach())
        
        return meta_loss


class Implicit gradients(TaskAgnosticContinualLearner):
    """
    Implicit Gradient Descent for Continual Learning.
    
    Uses implicit differentiation to prevent catastrophic forgetting
    without storing past data.
    
    Args:
        model: Neural network
        memory_size: Size of implicit memory
        device: Device for computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 100,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        
        self.memory_size = memory_size
        
        self.memory_x: List[Tensor] = []
        self.memory_y: List[Tensor] = []
        
    def add_to_memory(self, x: Tensor, y: Tensor) -> None:
        """Add samples to implicit memory."""
        self.memory_x.append(x.detach().cpu())
        self.memory_y.append(y.detach().cpu())
        
        if len(self.memory_x) > self.memory_size:
            self.memory_x = self.memory_x[-self.memory_size:]
            self.memory_y = self.memory_y[-self.memory_size:]
            
    def update(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """Update with implicit regularization."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        self.add_to_memory(x, y)
        
        self.model.zero_grad()
        
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        loss.backward()
        
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.add_(p.data.clone() * 0.001)
                
        self.steps += 1
        
        return {"loss": loss.item()}


class BootstrappedContinual(TaskAgnosticContinualLearner):
    """
    Bootstrapped Continual Learning.
    
    Uses bootstrapped predictions as pseudo-labels to maintain
    past knowledge.
    
    Args:
        model: Neural network
        alpha: Bootstrap ratio
        device: Device for computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        
        self.alpha = alpha
        
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        
        for param in self.target_model.parameters():
            param.requires_grad = False
            
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    def update_target(self) -> None:
        """Update target model with exponential moving average."""
        alpha = self.alpha
        
        with torch.no_grad():
            for (name, param), (target_name, target_param) in zip(
                self.model.named_parameters(),
                self.target_model.named_parameters(),
            ):
                target_param.data.mul_(alpha).add_(param.data, mul=1 - alpha)
                
    def update(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """Update with bootstrapped targets."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        
        current_logits = self.model(x)
        
        with torch.no_grad():
            target_logits = self.target_model(x)
            
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        
        task_loss = F.cross_entropy(current_logits, y)
        distill_loss = F.kl_div(current_log_probs, target_probs, reduction='batchmean')
        
        loss = task_loss + 0.1 * distill_loss
        
        loss.backward()
        self.optimizer.step()
        
        self.update_target()
        
        self.steps += 1
        
        return {
            "task_loss": task_loss.item(),
            "distill_loss": distill_loss.item(),
        }
