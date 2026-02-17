"""
Comprehensive Meta-Learning Module

This module implements state-of-the-art meta-learning and few-shot learning algorithms
including MAML, Reptile, Prototypical Networks, Matching Networks, and Relation Networks.
It provides a complete framework for few-shot learning with utilities for task sampling,
episode generation, inner loop optimization, and comprehensive evaluation.

References:
    - MAML: Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (ICML 2017)
    - Reptile: Nichol et al. "On First-Order Meta-Learning Algorithms" (2018)
    - Prototypical Networks: Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
    - Matching Networks: Vinyals et al. "Matching Networks for One Shot Learning" (NeurIPS 2016)
    - Relation Networks: Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
"""

import abc
import copy
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, Type
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler


# =============================================================================
# Data Structures and Types
# =============================================================================


@dataclass
class Task:
    """Represents a single few-shot learning task.

    Attributes:
        support_x: Support set inputs [n_way * n_shot, ...]
        support_y: Support set labels [n_way * n_shot]
        query_x: Query set inputs [n_way * n_query, ...]
        query_y: Query set labels [n_way * n_query]
        n_way: Number of classes in task
        n_shot: Number of examples per class in support set
        n_query: Number of examples per class in query set
    """

    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor
    n_way: int
    n_shot: int
    n_query: int


@dataclass
class Episode:
    """Represents a meta-learning episode containing multiple tasks.

    Attributes:
        tasks: List of tasks in the episode
        meta_batch_size: Number of tasks per meta-batch
    """

    tasks: List[Task]
    meta_batch_size: int


@dataclass
class MetaLearningState:
    """Tracks the state of meta-learning training.

    Attributes:
        epoch: Current epoch number
        iteration: Current iteration number
        inner_losses: History of inner loop losses
        outer_losses: History of outer loop losses
        validation_accuracies: History of validation accuracies
    """

    epoch: int = 0
    iteration: int = 0
    inner_losses: List[float] = field(default_factory=list)
    outer_losses: List[float] = field(default_factory=list)
    validation_accuracies: List[float] = field(default_factory=list)


# =============================================================================
# Meta-Learning Algorithms
# =============================================================================


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) implementation.

    MAML is a gradient-based meta-learning algorithm that learns good parameter
    initializations that can be quickly adapted to new tasks with a few gradient steps.

    Supports both first-order (FOMAML) and second-order (full MAML) approximations.

    Args:
        model: Base neural network model to meta-train
        inner_lr: Learning rate for inner loop adaptation
        num_inner_steps: Number of gradient steps in inner loop
        first_order: If True, use first-order approximation (FOMAML)
        inner_optimizer: Type of optimizer for inner loop ('sgd' or 'adam')

    Example:
        >>> model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
        >>> maml = MAML(model, inner_lr=0.01, num_inner_steps=5)
        >>> query_logits = maml(support_x, support_y, query_x)

    References:
        Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (ICML 2017)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        first_order: bool = False,
        inner_optimizer: str = "sgd",
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        self.inner_optimizer = inner_optimizer

    def forward(
        self, support_x: Tensor, support_y: Tensor, query_x: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """Forward pass with inner loop adaptation.

        Args:
            support_x: Support set inputs [n_support, ...]
            support_y: Support set labels [n_support]
            query_x: Query set inputs [n_query, ...]

        Returns:
            query_logits: Predictions on query set [n_query, n_classes]
            adapted_params: Adapted parameters after inner loop
            inner_loss: Final inner loop loss
        """
        adapted_params = self._inner_loop(support_x, support_y)
        query_logits = self._forward_with_params(query_x, adapted_params)

        # Compute final inner loss for monitoring
        with torch.no_grad():
            inner_logits = self._forward_with_params(support_x, adapted_params)
            inner_loss = F.cross_entropy(inner_logits, support_y)

        return query_logits, adapted_params, inner_loss

    def _inner_loop(self, support_x: Tensor, support_y: Tensor) -> Dict[str, Tensor]:
        """Perform inner loop adaptation on support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels

        Returns:
            adapted_params: Adapted parameters after inner loop
        """
        # Clone model parameters
        adapted_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        for step in range(self.num_inner_steps):
            # Forward pass with adapted parameters
            logits = self._forward_with_params(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients
            create_graph = not self.first_order
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=create_graph,
                allow_unused=True,
            )

            # Update adapted parameters using SGD
            adapted_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
                if grad is not None
            }

        return adapted_params

    def _forward_with_params(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """Forward pass with custom parameters.

        This implements functional forward pass using torch.nn.functional
        equivalents for common layers.

        Args:
            x: Input tensor
            params: Custom parameters to use

        Returns:
            Output logits
        """
        return self._functional_forward(x, params, self.model)

    def _functional_forward(
        self, x: Tensor, params: Dict[str, Tensor], model: nn.Module, prefix: str = ""
    ) -> Tensor:
        """Recursively apply functional forward pass.

        Args:
            x: Input tensor
            params: Parameter dictionary
            model: Model module
            prefix: Parameter name prefix

        Returns:
            Output tensor
        """
        for name, module in model.named_children():
            full_name = f"{prefix}{name}"

            if isinstance(module, nn.Linear):
                weight = params[f"{full_name}.weight"]
                bias = params.get(f"{full_name}.bias")
                x = F.linear(x, weight, bias)
            elif isinstance(module, nn.Conv2d):
                weight = params[f"{full_name}.weight"]
                bias = params.get(f"{full_name}.bias")
                x = F.conv2d(
                    x,
                    weight,
                    bias,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                )
            elif isinstance(module, nn.BatchNorm2d):
                weight = params.get(
                    f"{full_name}.weight", torch.ones(module.num_features)
                )
                bias = params.get(f"{full_name}.bias", torch.zeros(module.num_features))
                x = F.batch_norm(
                    x,
                    None,
                    None,
                    weight,
                    bias,
                    training=True,
                    momentum=module.momentum,
                    eps=module.eps,
                )
            elif isinstance(module, nn.ReLU):
                x = F.relu(x, inplace=module.inplace)
            elif isinstance(module, nn.MaxPool2d):
                x = F.max_pool2d(x, module.kernel_size, module.stride, module.padding)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                x = F.adaptive_avg_pool2d(x, module.output_size)
            elif isinstance(module, nn.Flatten):
                x = torch.flatten(x, start_dim=1)
            elif isinstance(module, nn.Sequential):
                x = self._functional_forward(x, params, module, prefix=f"{full_name}.")
            else:
                # For other modules, try to use the module directly
                x = module(x)

        return x


class Reptile(nn.Module):
    """Reptile: Simple meta-learning algorithm.

    Reptile is a first-order meta-learning algorithm that repeatedly samples a task,
    trains on it, and moves the meta-parameters towards the trained parameters.
    It is simpler than MAML and often achieves comparable performance.

    Args:
        model: Base neural network model
        inner_lr: Learning rate for inner loop task adaptation
        num_inner_steps: Number of gradient steps on each task
        meta_lr: Meta learning rate (step size towards adapted params)

    Example:
        >>> model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
        >>> reptile = Reptile(model, inner_lr=0.01, num_inner_steps=5)
        >>> loss = reptile.meta_step(support_x, support_y, query_x, query_y)

    References:
        Nichol et al. "On First-Order Meta-Learning Algorithms" (2018)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        meta_lr: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.meta_lr = meta_lr

    def meta_step(
        self, support_x: Tensor, support_y: Tensor, query_x: Tensor, query_y: Tensor
    ) -> Tensor:
        """Perform one meta-learning step.

        Samples a task, performs inner loop adaptation, and computes meta-gradient.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels

        Returns:
            query_loss: Loss on query set after adaptation
        """
        # Store original parameters
        original_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        # Inner loop adaptation on the task
        self._inner_loop_adaptation(support_x, support_y, query_x, query_y)

        # Get adapted parameters
        adapted_params = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

        # Compute Reptile meta-gradient: move towards adapted params
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                meta_gradient = adapted_params[name] - original_params[name]
                param.data = original_params[name] + self.meta_lr * meta_gradient

        # Compute query loss for monitoring
        query_logits = self.model(query_x)
        query_loss = F.cross_entropy(query_logits, query_y)

        return query_loss

    def _inner_loop_adaptation(
        self, support_x: Tensor, support_y: Tensor, query_x: Tensor, query_y: Tensor
    ) -> None:
        """Perform inner loop SGD adaptation.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

        for _ in range(self.num_inner_steps):
            optimizer.zero_grad()

            # Combine support and query for Reptile training
            x = torch.cat([support_x, query_x], dim=0)
            y = torch.cat([support_y, query_y], dim=0)

            logits = self.model(x)
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()


class PrototypicalNetworks(nn.Module):
    """Prototypical Networks for few-shot learning.

    Prototypical Networks learn a metric space where classification is performed
    by computing distances to prototype representations of each class. The prototype
    for each class is the mean of its support set embeddings.

    Args:
        encoder: Feature encoder network (maps inputs to embeddings)
        distance: Distance metric ('euclidean' or 'cosine')

    Example:
        >>> encoder = CNNEncoder(input_channels=3, hidden_dim=64, output_dim=64)
        >>> protonet = PrototypicalNetworks(encoder, distance='euclidean')
        >>> logits = protonet(support_x, support_y, query_x, n_way=5, n_shot=1)

    References:
        Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
    """

    def __init__(self, encoder: nn.Module, distance: str = "euclidean"):
        super().__init__()
        self.encoder = encoder
        self.distance = distance

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tensor:
        """Compute class probabilities for query samples.

        Args:
            support_x: Support set inputs [n_way * n_shot, ...]
            support_y: Support set labels [n_way * n_shot]
            query_x: Query set inputs [n_query, ...]
            n_way: Number of classes
            n_shot: Number of examples per class

        Returns:
            logits: Negative distances to prototypes [n_query, n_way]
        """
        # Encode support and query sets
        support_embeddings = self.encoder(support_x)  # [n_way * n_shot, embed_dim]
        query_embeddings = self.encoder(query_x)  # [n_query, embed_dim]

        # Compute prototypes (mean of support embeddings per class)
        prototypes = self._compute_prototypes(
            support_embeddings, support_y, n_way, n_shot
        )  # [n_way, embed_dim]

        # Compute distances from queries to prototypes
        distances = self._compute_distances(query_embeddings, prototypes)

        # Convert distances to logits (negative distances as logits)
        logits = -distances

        return logits

    def _compute_prototypes(
        self, support_embeddings: Tensor, support_y: Tensor, n_way: int, n_shot: int
    ) -> Tensor:
        """Compute prototype for each class as mean of support embeddings.

        Args:
            support_embeddings: Support set embeddings
            support_y: Support set labels
            n_way: Number of classes
            n_shot: Number of examples per class

        Returns:
            prototypes: Class prototypes [n_way, embed_dim]
        """
        embed_dim = support_embeddings.size(-1)
        prototypes = torch.zeros(n_way, embed_dim, device=support_embeddings.device)

        for class_idx in range(n_way):
            class_mask = support_y == class_idx
            class_embeddings = support_embeddings[class_mask]
            prototypes[class_idx] = class_embeddings.mean(dim=0)

        return prototypes

    def _compute_distances(
        self, query_embeddings: Tensor, prototypes: Tensor
    ) -> Tensor:
        """Compute distances between queries and prototypes.

        Args:
            query_embeddings: Query embeddings [n_query, embed_dim]
            prototypes: Class prototypes [n_way, embed_dim]

        Returns:
            distances: Pairwise distances [n_query, n_way]
        """
        if self.distance == "euclidean":
            # Euclidean distance: ||x - y||^2
            distances = torch.cdist(query_embeddings, prototypes, p=2).pow(2)
        elif self.distance == "cosine":
            # Cosine distance: 1 - cosine_similarity
            query_norm = F.normalize(query_embeddings, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            cosine_sim = torch.mm(query_norm, proto_norm.t())
            distances = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        return distances


class MatchingNetworks(nn.Module):
    """Matching Networks for one-shot and few-shot learning.

    Matching Networks use an attention mechanism to classify query samples based on
    their similarity to support set samples. It includes fully-conditional embeddings
    (FCE) that condition the embeddings on the entire support set.

    Args:
        encoder: Feature encoder network
        use_fce: Whether to use Fully-Conditional Embeddings
        fce_hidden_dim: Hidden dimension for FCE LSTM

    Example:
        >>> encoder = CNNEncoder(input_channels=3, hidden_dim=64, output_dim=64)
        >>> matching_net = MatchingNetworks(encoder, use_fce=True)
        >>> logits = matching_net(support_x, support_y, query_x)

    References:
        Vinyals et al. "Matching Networks for One Shot Learning" (NeurIPS 2016)
    """

    def __init__(
        self, encoder: nn.Module, use_fce: bool = True, fce_hidden_dim: int = 32
    ):
        super().__init__()
        self.encoder = encoder
        self.use_fce = use_fce

        if use_fce:
            # Fully-Conditional Embeddings with bidirectional LSTM
            self.fce_lstm = nn.LSTM(
                input_size=encoder.output_dim if hasattr(encoder, "output_dim") else 64,
                hidden_size=fce_hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )

            # Attention LSTM for query embeddings
            self.attention_lstm = nn.LSTMCell(
                input_size=encoder.output_dim if hasattr(encoder, "output_dim") else 64,
                hidden_size=fce_hidden_dim,
            )

    def forward(self, support_x: Tensor, support_y: Tensor, query_x: Tensor) -> Tensor:
        """Compute class probabilities for query samples.

        Args:
            support_x: Support set inputs [n_support, ...]
            support_y: Support set labels [n_support]
            query_x: Query set inputs [n_query, ...]

        Returns:
            logits: Class probabilities [n_query, n_classes]
        """
        # Encode support and query
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)

        if self.use_fce:
            # Apply Fully-Conditional Embeddings
            support_embeddings = self._apply_fce_support(support_embeddings)
            query_embeddings = self._apply_fce_query(
                query_embeddings, support_embeddings
            )

        # Compute attention weights (cosine similarity)
        attention = self._compute_attention(query_embeddings, support_embeddings)

        # Convert support labels to one-hot
        n_classes = support_y.max().item() + 1
        support_y_onehot = F.one_hot(support_y, num_classes=n_classes).float()

        # Weighted sum of support labels
        logits = torch.mm(attention, support_y_onehot)

        return logits

    def _apply_fce_support(self, support_embeddings: Tensor) -> Tensor:
        """Apply fully-conditional embedding to support set.

        Args:
            support_embeddings: Raw support embeddings [n_support, embed_dim]

        Returns:
            fce_embeddings: FCE-enhanced embeddings [n_support, embed_dim]
        """
        # Add batch dimension for LSTM
        support_embeddings = support_embeddings.unsqueeze(
            0
        )  # [1, n_support, embed_dim]
        fce_output, _ = self.fce_lstm(support_embeddings)
        fce_output = fce_output.squeeze(0)  # [n_support, embed_dim]

        # Residual connection
        return support_embeddings.squeeze(0) + fce_output

    def _apply_fce_query(
        self, query_embeddings: Tensor, support_embeddings: Tensor
    ) -> Tensor:
        """Apply attention-based FCE to query embeddings.

        Args:
            query_embeddings: Raw query embeddings [n_query, embed_dim]
            support_embeddings: FCE-enhanced support embeddings [n_support, embed_dim]

        Returns:
            enhanced_queries: Enhanced query embeddings [n_query, embed_dim]
        """
        n_query = query_embeddings.size(0)
        hidden_dim = self.attention_lstm.hidden_size

        # Initialize hidden state
        h = torch.zeros(n_query, hidden_dim, device=query_embeddings.device)
        c = torch.zeros(n_query, hidden_dim, device=query_embeddings.device)

        # Process through attention LSTM
        outputs = []
        for _ in range(3):  # Number of processing steps
            h, c = self.attention_lstm(query_embeddings, (h, c))

            # Compute attention over support set
            attention = F.softmax(torch.mm(h, support_embeddings.t()), dim=-1)
            read = torch.mm(attention, support_embeddings)

            outputs.append(read)

        # Combine all outputs
        enhanced = query_embeddings + sum(outputs) / len(outputs)

        return enhanced

    def _compute_attention(
        self, query_embeddings: Tensor, support_embeddings: Tensor
    ) -> Tensor:
        """Compute attention weights between queries and support.

        Args:
            query_embeddings: Query embeddings [n_query, embed_dim]
            support_embeddings: Support embeddings [n_support, embed_dim]

        Returns:
            attention: Attention weights [n_query, n_support]
        """
        # Cosine similarity
        query_norm = F.normalize(query_embeddings, dim=-1)
        support_norm = F.normalize(support_embeddings, dim=-1)
        similarities = torch.mm(query_norm, support_norm.t())

        # Softmax over support set
        attention = F.softmax(similarities, dim=-1)

        return attention


class RelationNetworks(nn.Module):
    """Relation Networks for few-shot learning.

    Relation Networks learn a deep distance metric to compare query samples with
    support samples. Unlike Prototypical Networks which use fixed distance metrics,
    Relation Networks learn a non-linear similarity metric through a relation module.

    Args:
        encoder: Feature encoder network
        relation_module: Module that computes similarity between embeddings
        fusion: How to combine query and support embeddings ('concat' or 'sum')

    Example:
        >>> encoder = CNNEncoder(input_channels=3, hidden_dim=64, output_dim=64)
        >>> relation_mod = RelationModule(input_dim=128, hidden_dim=64)
        >>> relation_net = RelationNetworks(encoder, relation_mod)
        >>> logits = relation_net(support_x, support_y, query_x, n_way=5, n_shot=1)

    References:
        Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
    """

    def __init__(
        self, encoder: nn.Module, relation_module: nn.Module, fusion: str = "concat"
    ):
        super().__init__()
        self.encoder = encoder
        self.relation_module = relation_module
        self.fusion = fusion

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        query_x: Tensor,
        n_way: int,
        n_shot: int,
    ) -> Tensor:
        """Compute relation scores for query samples.

        Args:
            support_x: Support set inputs [n_way * n_shot, ...]
            support_y: Support set labels [n_way * n_shot]
            query_x: Query set inputs [n_query, ...]
            n_way: Number of classes
            n_shot: Number of examples per class

        Returns:
            relation_scores: Relation scores [n_query, n_way]
        """
        # Encode support and query
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)

        n_query = query_embeddings.size(0)
        embed_dim = support_embeddings.size(-1)

        # Compute prototypes (mean of support embeddings per class)
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_y == class_idx
            class_embeddings = support_embeddings[class_mask]
            prototypes.append(class_embeddings.mean(dim=0))
        prototypes = torch.stack(prototypes)  # [n_way, embed_dim]

        # Create query-prototype pairs
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, n_way, -1)
        prototypes_expanded = prototypes.unsqueeze(0).expand(n_query, -1, -1)

        # Fuse embeddings
        if self.fusion == "concat":
            pairs = torch.cat([query_expanded, prototypes_expanded], dim=-1)
        elif self.fusion == "sum":
            pairs = query_expanded + prototypes_expanded
        elif self.fusion == "diff":
            pairs = torch.abs(query_expanded - prototypes_expanded)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

        # Reshape for relation module
        pairs = pairs.view(n_query * n_way, -1)

        # Compute relation scores
        relation_scores = self.relation_module(pairs)
        relation_scores = relation_scores.view(n_query, n_way)

        return relation_scores


class RelationModule(nn.Module):
    """Relation module for computing similarity between embeddings.

    Args:
        input_dim: Input dimension (typically 2 * embed_dim for concatenation)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (typically 1 for similarity score)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Relation score in [0, 1]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute relation scores.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            scores: Relation scores [batch, output_dim]
        """
        return self.net(x)


# =============================================================================
# Few-Shot Learning Utilities
# =============================================================================


class TaskSampler(Sampler):
    """Sampler for generating N-way K-shot tasks from a dataset.

    This sampler creates episodes for meta-learning by sampling classes and
    then sampling examples within each class.

    Args:
        dataset: Dataset with labels accessible via dataset.targets or dataset.labels
        n_way: Number of classes per task
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        n_tasks: Number of tasks to sample per epoch

    Example:
        >>> dataset = datasets.Omniglot(root='./data', download=True)
        >>> sampler = TaskSampler(dataset, n_way=5, n_shot=1, n_query=15, n_tasks=100)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self, dataset: Dataset, n_way: int, n_shot: int, n_query: int, n_tasks: int
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        # Organize dataset by class
        self.classes = self._get_classes()
        self.class_to_indices = self._create_class_to_indices()

    def _get_classes(self) -> List[int]:
        """Get list of unique classes in dataset."""
        if hasattr(self.dataset, "targets"):
            targets = self.dataset.targets
        elif hasattr(self.dataset, "labels"):
            targets = self.dataset.labels
        elif hasattr(self.dataset, "y"):
            targets = self.dataset.y
        else:
            raise ValueError("Dataset must have 'targets', 'labels', or 'y' attribute")

        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

        return list(set(targets))

    def _create_class_to_indices(self) -> Dict[int, List[int]]:
        """Create mapping from class label to sample indices."""
        class_to_indices = {cls: [] for cls in self.classes}

        if hasattr(self.dataset, "targets"):
            targets = self.dataset.targets
        elif hasattr(self.dataset, "labels"):
            targets = self.dataset.labels
        elif hasattr(self.dataset, "y"):
            targets = self.dataset.y

        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

        for idx, label in enumerate(targets):
            class_to_indices[label].append(idx)

        return class_to_indices

    def __len__(self) -> int:
        """Return number of tasks per epoch."""
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over tasks, yielding indices for each task.

        Yields:
            indices: List of indices for support and query sets
        """
        for _ in range(self.n_tasks):
            # Sample n_way classes
            task_classes = torch.randperm(len(self.classes))[: self.n_way]
            task_classes = [self.classes[i] for i in task_classes.tolist()]

            indices = []

            for cls in task_classes:
                # Sample support and query examples for this class
                class_indices = self.class_to_indices[cls]
                sampled = torch.randperm(len(class_indices))

                support_idx = sampled[: self.n_shot]
                query_idx = sampled[self.n_shot : self.n_shot + self.n_query]

                # Map back to actual dataset indices
                support_actual = [class_indices[i] for i in support_idx.tolist()]
                query_actual = [class_indices[i] for i in query_idx.tolist()]

                indices.extend(support_actual)
                indices.extend(query_actual)

            yield indices


class EpisodeDataset(Dataset):
    """Dataset wrapper that generates episodes for meta-learning.

    This dataset generates episodes on-the-fly, where each episode consists
    of multiple N-way K-shot tasks.

    Args:
        dataset: Base dataset
        n_way: Number of classes per task
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        n_tasks: Number of tasks per episode
        transform: Optional transform to apply to samples

    Example:
        >>> base_dataset = datasets.Omniglot(root='./data', download=True)
        >>> episode_dataset = EpisodeDataset(base_dataset, n_way=5, n_shot=1, n_query=15)
        >>> task = episode_dataset[0]
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int = 1000,
        transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.transform = transform

        # Create task sampler
        self.sampler = TaskSampler(dataset, n_way, n_shot, n_query, n_tasks=1)

    def __len__(self) -> int:
        """Return number of tasks."""
        return self.n_tasks

    def __getitem__(self, idx: int) -> Task:
        """Get a single task.

        Args:
            idx: Task index

        Returns:
            task: Task object containing support and query sets
        """
        # Sample task classes
        task_classes = torch.randperm(len(self.sampler.classes))[: self.n_way]
        task_classes = [self.sampler.classes[i] for i in task_classes.tolist()]

        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, cls in enumerate(task_classes):
            class_indices = self.sampler.class_to_indices[cls]
            sampled = torch.randperm(len(class_indices))

            # Sample support examples
            support_idx = sampled[: self.n_shot]
            for i in support_idx:
                idx_in_dataset = class_indices[i.item()]
                x, _ = self.dataset[idx_in_dataset]
                if self.transform:
                    x = self.transform(x)
                support_x.append(x)
                support_y.append(new_label)

            # Sample query examples
            query_idx = sampled[self.n_shot : self.n_shot + self.n_query]
            for i in query_idx:
                idx_in_dataset = class_indices[i.item()]
                x, _ = self.dataset[idx_in_dataset]
                if self.transform:
                    x = self.transform(x)
                query_x.append(x)
                query_y.append(new_label)

        # Stack tensors
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y, dtype=torch.long)

        return Task(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            n_way=self.n_way,
            n_shot=self.n_shot,
            n_query=self.n_query,
        )


def create_episode(
    dataset: Dataset,
    n_way: int,
    n_shot: int,
    n_query: int,
    device: Optional[torch.device] = None,
) -> Task:
    """Create a single episode with support and query sets.

    This is a utility function for quickly generating a single task without
    creating a full EpisodeDataset.

    Args:
        dataset: Dataset to sample from
        n_way: Number of classes
        n_shot: Support examples per class
        n_query: Query examples per class
        device: Device to place tensors on

    Returns:
        task: Task with support and query sets

    Example:
        >>> task = create_episode(omniglot_dataset, n_way=5, n_shot=1, n_query=15)
        >>> print(f"Support shape: {task.support_x.shape}")
    """
    # Get class information
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    elif hasattr(dataset, "y"):
        targets = dataset.y
    else:
        raise ValueError("Dataset must have 'targets', 'labels', or 'y' attribute")

    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()

    classes = list(set(targets))

    # Organize by class
    class_to_indices = {cls: [] for cls in classes}
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    # Sample task classes
    task_classes = torch.randperm(len(classes))[:n_way]
    task_classes = [classes[i] for i in task_classes.tolist()]

    support_x, support_y = [], []
    query_x, query_y = [], []

    for new_label, cls in enumerate(task_classes):
        class_indices = class_to_indices[cls]
        sampled = torch.randperm(len(class_indices))

        # Sample support
        for i in sampled[:n_shot]:
            x, _ = dataset[class_indices[i.item()]]
            support_x.append(x)
            support_y.append(new_label)

        # Sample query
        for i in sampled[n_shot : n_shot + n_query]:
            x, _ = dataset[class_indices[i.item()]]
            query_x.append(x)
            query_y.append(new_label)

    # Stack and move to device
    support_x = torch.stack(support_x)
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_x = torch.stack(query_x)
    query_y = torch.tensor(query_y, dtype=torch.long)

    if device:
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

    return Task(
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
    )


def split_support_query(
    x: Tensor, y: Tensor, n_shot: int, shuffle: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split data into support and query sets.

    Args:
        x: Input data [n_samples, ...]
        y: Labels [n_samples]
        n_shot: Number of examples per class for support set
        shuffle: Whether to shuffle data before splitting

    Returns:
        support_x: Support inputs
        support_y: Support labels
        query_x: Query inputs
        query_y: Query labels

    Example:
        >>> support_x, support_y, query_x, query_y = split_support_query(x, y, n_shot=5)
    """
    if shuffle:
        perm = torch.randperm(len(x))
        x = x[perm]
        y = y[perm]

    # Organize by class
    classes = torch.unique(y)
    support_x, support_y = [], []
    query_x, query_y = [], []

    for cls in classes:
        mask = y == cls
        class_x = x[mask]
        class_y = y[mask]

        support_x.append(class_x[:n_shot])
        support_y.append(class_y[:n_shot])
        query_x.append(class_x[n_shot:])
        query_y.append(class_y[n_shot:])

    support_x = torch.cat(support_x, dim=0)
    support_y = torch.cat(support_y, dim=0)
    query_x = torch.cat(query_x, dim=0)
    query_y = torch.cat(query_y, dim=0)

    return support_x, support_y, query_x, query_y


# =============================================================================
# Meta-Learners
# =============================================================================


class MetaLearner(abc.ABC, nn.Module):
    """Abstract base class for meta-learners.

    Meta-learners implement different strategies for learning from multiple tasks.
    This base class defines the interface that all meta-learners must implement.

    Args:
        model: Base model to meta-train
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abc.abstractmethod
    def adapt(self, support_x: Tensor, support_y: Tensor) -> Dict[str, Any]:
        """Adapt model to a new task using support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels

        Returns:
            adapted_state: State of adapted model (e.g., parameters, embeddings)
        """
        pass

    @abc.abstractmethod
    def predict(self, query_x: Tensor, adapted_state: Dict[str, Any]) -> Tensor:
        """Make predictions on query set using adapted model.

        Args:
            query_x: Query set inputs
            adapted_state: Adapted model state from adapt()

        Returns:
            predictions: Model predictions
        """
        pass

    def meta_update(
        self, tasks: List[Task], meta_optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform meta-update across multiple tasks.

        Args:
            tasks: List of tasks
            meta_optimizer: Optimizer for meta-parameters

        Returns:
            metrics: Dictionary of training metrics
        """
        meta_optimizer.zero_grad()

        total_loss = 0.0
        total_acc = 0.0

        for task in tasks:
            # Adapt to task
            adapted_state = self.adapt(task.support_x, task.support_y)

            # Predict on query
            predictions = self.predict(task.query_x, adapted_state)

            # Compute loss
            loss = F.cross_entropy(predictions, task.query_y)
            loss.backward()

            total_loss += loss.item()

            # Compute accuracy
            acc = (predictions.argmax(dim=-1) == task.query_y).float().mean()
            total_acc += acc.item()

        meta_optimizer.step()

        n_tasks = len(tasks)
        return {"loss": total_loss / n_tasks, "accuracy": total_acc / n_tasks}


class GradientBasedMetaLearner(MetaLearner):
    """Meta-learner using gradient-based adaptation (MAML, Reptile).

    This meta-learner performs task adaptation through gradient descent on the
    support set, similar to MAML and Reptile.

    Args:
        model: Base model
        inner_lr: Learning rate for task adaptation
        num_inner_steps: Number of gradient steps per task
        first_order: Whether to use first-order approximation
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        first_order: bool = False,
    ):
        super().__init__(model)
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

    def adapt(self, support_x: Tensor, support_y: Tensor) -> Dict[str, Any]:
        """Adapt model using gradient descent on support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels

        Returns:
            adapted_state: Dictionary with adapted parameters
        """
        # Clone current parameters
        adapted_params = {
            name: param.clone().requires_grad_(True)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Inner loop adaptation
        for _ in range(self.num_inner_steps):
            logits = self._forward_with_params(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)

            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            # Update parameters
            adapted_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
                if grad is not None
            }

        return {"params": adapted_params}

    def predict(self, query_x: Tensor, adapted_state: Dict[str, Any]) -> Tensor:
        """Make predictions using adapted parameters.

        Args:
            query_x: Query inputs
            adapted_state: Adapted state from adapt()

        Returns:
            predictions: Model predictions
        """
        params = adapted_state["params"]
        return self._forward_with_params(query_x, params)

    def _forward_with_params(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """Forward pass with custom parameters."""
        # This is a simplified version - in practice would use functional API
        # Save original parameters
        original_params = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

        # Set adapted parameters
        for name, param in self.model.named_parameters():
            if name in params:
                param.data = params[name]

        # Forward pass
        output = self.model(x)

        # Restore original parameters
        for name, param in self.model.named_parameters():
            if name in original_params:
                param.data = original_params[name]

        return output


class MetricBasedMetaLearner(MetaLearner):
    """Meta-learner using metric learning (Prototypical, Matching Networks).

    This meta-learner performs task adaptation by computing embeddings and
    using distance-based classification.

    Args:
        encoder: Feature encoder
        distance_metric: Distance metric ('euclidean', 'cosine', 'learned')

    Example:
        >>> encoder = CNNEncoder(input_channels=3, hidden_dim=64, output_dim=64)
        >>> meta_learner = MetricBasedMetaLearner(encoder, distance_metric='euclidean')
    """

    def __init__(self, encoder: nn.Module, distance_metric: str = "euclidean"):
        super().__init__(encoder)
        self.encoder = encoder
        self.distance_metric = distance_metric

    def adapt(self, support_x: Tensor, support_y: Tensor) -> Dict[str, Any]:
        """Adapt by computing class prototypes from support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels

        Returns:
            adapted_state: Dictionary with prototypes and support info
        """
        # Encode support set
        support_embeddings = self.encoder(support_x)

        # Compute prototypes (class centroids)
        classes = torch.unique(support_y)
        prototypes = []

        for cls in classes:
            mask = support_y == cls
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        return {
            "prototypes": prototypes,
            "classes": classes,
            "support_embeddings": support_embeddings,
            "support_y": support_y,
        }

    def predict(self, query_x: Tensor, adapted_state: Dict[str, Any]) -> Tensor:
        """Predict by comparing query embeddings to prototypes.

        Args:
            query_x: Query inputs
            adapted_state: Adapted state from adapt()

        Returns:
            predictions: Class logits (negative distances)
        """
        query_embeddings = self.encoder(query_x)
        prototypes = adapted_state["prototypes"]

        # Compute distances
        if self.distance_metric == "euclidean":
            distances = torch.cdist(query_embeddings, prototypes).pow(2)
        elif self.distance_metric == "cosine":
            query_norm = F.normalize(query_embeddings, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            cosine_sim = torch.mm(query_norm, proto_norm.t())
            distances = 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Negative distances as logits
        return -distances


class MemoryAugmentedMetaLearner(MetaLearner):
    """Meta-learner with external memory for task adaptation.

    This meta-learner uses an external memory module to store and retrieve
    information from support sets, similar to Neural Turing Machines or
    Memory-Augmented Neural Networks (MANN).

    Args:
        model: Base model
        memory_size: Size of external memory
        memory_dim: Dimension of memory vectors
        num_reads: Number of memory reads per query

    Example:
        >>> model = LSTMEncoder(input_dim=64, hidden_dim=128)
        >>> meta_learner = MemoryAugmentedMetaLearner(model, memory_size=128, memory_dim=64)
    """

    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 128,
        memory_dim: int = 64,
        num_reads: int = 4,
    ):
        super().__init__(model)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_reads = num_reads

        # External memory
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))

        # Memory controller
        self.read_head = nn.Linear(
            model.output_dim if hasattr(model, "output_dim") else 64, memory_size
        )
        self.write_head = nn.Linear(memory_dim, memory_dim)

    def adapt(self, support_x: Tensor, support_y: Tensor) -> Dict[str, Any]:
        """Adapt by writing support set to memory.

        Args:
            support_x: Support set inputs
            support_y: Support set labels

        Returns:
            adapted_state: Dictionary with memory state
        """
        # Encode support set
        support_embeddings = self.model(support_x)

        # Write to memory
        batch_size = support_embeddings.size(0)
        memory_update = self.write_head(support_embeddings)

        # Simple content-based addressing
        similarities = torch.mm(support_embeddings, self.memory.t())
        attention = F.softmax(similarities, dim=-1)

        # Update memory (simplified write)
        for i in range(batch_size):
            self.memory.data += attention[i].unsqueeze(-1) * memory_update[i]

        return {
            "memory": self.memory.clone(),
            "support_embeddings": support_embeddings,
            "support_y": support_y,
        }

    def predict(self, query_x: Tensor, adapted_state: Dict[str, Any]) -> Tensor:
        """Predict by reading from memory.

        Args:
            query_x: Query inputs
            adapted_state: Adapted state from adapt()

        Returns:
            predictions: Model predictions
        """
        query_embeddings = self.model(query_x)
        memory = adapted_state["memory"]

        # Read from memory
        read_weights = F.softmax(self.read_head(query_embeddings), dim=-1)
        read_vectors = torch.mm(read_weights, memory)

        # Combine query embedding with memory read
        combined = query_embeddings + read_vectors

        # Simple linear classifier
        if not hasattr(self, "classifier"):
            self.classifier = nn.Linear(
                combined.size(-1), adapted_state["support_y"].max().item() + 1
            )
            self.classifier = self.classifier.to(combined.device)

        return self.classifier(combined)


# =============================================================================
# Inner Loop Optimization
# =============================================================================


class InnerLoopOptimizer:
    """Optimizer for inner loop adaptation in meta-learning.

    This optimizer handles the gradient-based adaptation on support sets,
    supporting both SGD and Adam update rules.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        optimizer_type: Type of optimizer ('sgd' or 'adam')
        momentum: Momentum for SGD or betas for Adam
        weight_decay: Weight decay coefficient

    Example:
        >>> params = dict(model.named_parameters())
        >>> optimizer = InnerLoopOptimizer(params, lr=0.01, optimizer_type='sgd')
        >>> adapted_params = optimizer.step(loss, params)
    """

    def __init__(
        self,
        params: Dict[str, Tensor],
        lr: float = 0.01,
        optimizer_type: str = "sgd",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.optimizer_type = optimizer_type.lower()
        self.momentum = momentum
        self.weight_decay = weight_decay

        # State for momentum/Adam
        self.state = {}
        if self.optimizer_type == "adam":
            for name, param in params.items():
                self.state[name] = {
                    "exp_avg": torch.zeros_like(param),
                    "exp_avg_sq": torch.zeros_like(param),
                    "step": 0,
                }
        elif self.optimizer_type == "sgd":
            for name, param in params.items():
                self.state[name] = {"momentum_buffer": torch.zeros_like(param)}

    def step(
        self, loss: Tensor, params: Dict[str, Tensor], create_graph: bool = False
    ) -> Dict[str, Tensor]:
        """Perform one optimization step.

        Args:
            loss: Loss tensor to minimize
            params: Current parameters
            create_graph: Whether to create computation graph for higher-order grads

        Returns:
            updated_params: Updated parameters
        """
        # Compute gradients
        grads = torch.autograd.grad(
            loss, params.values(), create_graph=create_graph, allow_unused=True
        )

        updated_params = {}

        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                updated_params[name] = param
                continue

            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            if self.optimizer_type == "sgd":
                updated_params[name] = self._sgd_update(name, param, grad)
            elif self.optimizer_type == "adam":
                updated_params[name] = self._adam_update(name, param, grad)
            else:
                raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        return updated_params

    def _sgd_update(self, name: str, param: Tensor, grad: Tensor) -> Tensor:
        """Apply SGD update with momentum."""
        if self.momentum > 0:
            buf = self.state[name]["momentum_buffer"]
            buf.mul_(self.momentum).add_(grad)
            grad = buf

        return param - self.lr * grad

    def _adam_update(self, name: str, param: Tensor, grad: Tensor) -> Tensor:
        """Apply Adam update."""
        state = self.state[name]
        state["step"] += 1

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = (
            (self.momentum, 0.999)
            if isinstance(self.momentum, float)
            else self.momentum
        )

        # Decay first and second moment
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        step_size = self.lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)

        return param - step_size * exp_avg / denom


class MetaOptimizer:
    """Meta-optimizer that operates at the meta-level across tasks.

    Unlike the InnerLoopOptimizer which adapts to individual tasks, the MetaOptimizer
    updates the meta-parameters based on performance across multiple tasks.

    Args:
        params: Meta-parameters to optimize
        lr: Meta learning rate
        optimizer_type: Type of optimizer

    Example:
        >>> meta_opt = MetaOptimizer(model.parameters(), lr=1e-3, optimizer_type='adam')
        >>> meta_opt.meta_step(tasks, meta_learner)
    """

    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float = 1e-3,
        optimizer_type: str = "adam",
        **kwargs,
    ):
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, **kwargs)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=lr, **kwargs)
        elif optimizer_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=lr, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def meta_step(
        self, tasks: List[Task], meta_learner: MetaLearner
    ) -> Dict[str, float]:
        """Perform one meta-optimization step.

        Args:
            tasks: List of tasks
            meta_learner: Meta-learner to optimize

        Returns:
            metrics: Dictionary of metrics
        """
        self.optimizer.zero_grad()

        total_loss = 0.0
        correct = 0
        total = 0

        for task in tasks:
            # Adapt to task
            adapted_state = meta_learner.adapt(task.support_x, task.support_y)

            # Predict on query
            predictions = meta_learner.predict(task.query_x, adapted_state)

            # Compute loss
            loss = F.cross_entropy(predictions, task.query_y)
            loss.backward()

            total_loss += loss.item()

            # Compute accuracy
            pred_labels = predictions.argmax(dim=-1)
            correct += (pred_labels == task.query_y).sum().item()
            total += task.query_y.size(0)

        self.optimizer.step()

        return {
            "loss": total_loss / len(tasks),
            "accuracy": correct / total if total > 0 else 0.0,
        }

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Step the optimizer."""
        self.optimizer.step()


def learnable_learning_rates(
    model: nn.Module, init_lr: float = 0.01, per_layer: bool = True
) -> nn.ParameterDict:
    """Create learnable per-parameter learning rates.

    This allows the meta-learner to learn optimal learning rates for each
    parameter or layer, improving adaptation speed and stability.

    Args:
        model: Model to create learning rates for
        init_lr: Initial learning rate value
        per_layer: If True, create one LR per layer; otherwise per parameter

    Returns:
        lr_params: ParameterDict of learnable learning rates

    Example:
        >>> model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
        >>> lrs = learnable_learning_rates(model, init_lr=0.01)
        >>> print(lrs)
    """
    lr_params = nn.ParameterDict()

    if per_layer:
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and any(
                isinstance(module, t) for t in [nn.Linear, nn.Conv2d, nn.Conv1d]
            ):
                lr_params[name] = nn.Parameter(torch.tensor(init_lr))
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                lr_params[name] = nn.Parameter(torch.tensor(init_lr))

    return lr_params


# =============================================================================
# Meta-Learning Tasks and Data Loaders
# =============================================================================


class OmniglotTaskLoader:
    """Task loader for Omniglot dataset.

    Omniglot is a standard benchmark for few-shot learning containing 1623
    handwritten characters from 50 different alphabets.

    Args:
        root: Root directory for dataset
        n_way: Number of classes per task
        n_shot: Support examples per class
        n_query: Query examples per class
        download: Whether to download dataset
        transform: Optional transform to apply

    Example:
        >>> loader = OmniglotTaskLoader(root='./data', n_way=5, n_shot=1, n_query=15)
        >>> for task in loader:
        ...     print(f"Support: {task.support_x.shape}, Query: {task.query_x.shape}")
    """

    def __init__(
        self,
        root: str = "./data",
        n_way: int = 5,
        n_shot: int = 1,
        n_query: int = 15,
        download: bool = True,
        transform: Optional[Callable] = None,
        num_tasks: int = 1000,
    ):
        try:
            from torchvision import datasets
        except ImportError:
            raise ImportError("torchvision is required for OmniglotTaskLoader")

        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.num_tasks = num_tasks

        # Load datasets
        self.train_dataset = datasets.Omniglot(
            root=root, background=True, download=download, transform=transform
        )

        self.test_dataset = datasets.Omniglot(
            root=root, background=False, download=download, transform=transform
        )

    def __iter__(self) -> Iterator[Task]:
        """Iterate over tasks."""
        for _ in range(self.num_tasks):
            yield create_episode(
                self.train_dataset, self.n_way, self.n_shot, self.n_query
            )

    def __len__(self) -> int:
        """Return number of tasks."""
        return self.num_tasks

    def get_test_task(self) -> Task:
        """Get a task from the test set."""
        return create_episode(self.test_dataset, self.n_way, self.n_shot, self.n_query)


class MiniImageNetTaskLoader:
    """Task loader for Mini-ImageNet dataset.

    Mini-ImageNet is a standard few-shot learning benchmark derived from
    ImageNet with 100 classes (64 train, 16 val, 20 test), 600 images per class.

    Args:
        root: Root directory for dataset
        n_way: Number of classes per task
        n_shot: Support examples per class
        n_query: Query examples per class
        split: Dataset split ('train', 'val', or 'test')
        transform: Optional transform to apply
        num_tasks: Number of tasks to generate

    Example:
        >>> loader = MiniImageNetTaskLoader(root='./data', n_way=5, n_shot=5, split='train')
        >>> for task in loader:
        ...     train_on_task(task)
    """

    def __init__(
        self,
        root: str = "./data/mini-imagenet",
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
        split: str = "train",
        transform: Optional[Callable] = None,
        num_tasks: int = 1000,
    ):
        self.root = root
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.split = split
        self.num_tasks = num_tasks

        # Load dataset
        self.dataset = self._load_dataset(transform)

    def _load_dataset(self, transform: Optional[Callable]) -> Dataset:
        """Load Mini-ImageNet dataset from CSV files."""
        import os
        from PIL import Image

        class MiniImageNetDataset(Dataset):
            def __init__(self, root, split, transform):
                self.root = root
                self.split = split
                self.transform = transform

                # Load CSV
                csv_path = os.path.join(root, f"{split}.csv")
                self.data = []
                self.targets = []

                class_to_idx = {}

                if os.path.exists(csv_path):
                    with open(csv_path, "r") as f:
                        next(f)  # Skip header
                        for line in f:
                            parts = line.strip().split(",")
                            if len(parts) >= 2:
                                filename, label = parts[0], parts[1]
                                if label not in class_to_idx:
                                    class_to_idx[label] = len(class_to_idx)

                                self.data.append(os.path.join(root, "images", filename))
                                self.targets.append(class_to_idx[label])

                self.classes = list(class_to_idx.keys())
                self.class_to_idx = class_to_idx

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                img_path = self.data[idx]
                label = self.targets[idx]

                if os.path.exists(img_path):
                    image = Image.open(img_path).convert("RGB")
                else:
                    # Return dummy image if file doesn't exist
                    image = Image.new("RGB", (84, 84))

                if self.transform:
                    image = self.transform(image)

                return image, label

        return MiniImageNetDataset(self.root, self.split, transform)

    def __iter__(self) -> Iterator[Task]:
        """Iterate over tasks."""
        for _ in range(self.num_tasks):
            yield create_episode(self.dataset, self.n_way, self.n_shot, self.n_query)

    def __len__(self) -> int:
        """Return number of tasks."""
        return self.num_tasks


class CIFAR_FSTaskLoader:
    """Task loader for CIFAR-FS (CIFAR Few-Shot) dataset.

    CIFAR-FS is a few-shot learning benchmark derived from CIFAR-100 with
    100 classes (64 train, 16 val, 20 test).

    Args:
        root: Root directory for dataset
        n_way: Number of classes per task
        n_shot: Support examples per class
        n_query: Query examples per class
        split: Dataset split ('train', 'val', or 'test')
        download: Whether to download dataset
        transform: Optional transform to apply
        num_tasks: Number of tasks to generate

    Example:
        >>> loader = CIFAR_FSTaskLoader(root='./data', n_way=5, n_shot=5, split='train')
        >>> for task in loader:
        ...     train_on_task(task)
    """

    def __init__(
        self,
        root: str = "./data",
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
        split: str = "train",
        download: bool = True,
        transform: Optional[Callable] = None,
        num_tasks: int = 1000,
    ):
        try:
            from torchvision import datasets
        except ImportError:
            raise ImportError("torchvision is required for CIFAR_FSTaskLoader")

        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.split = split
        self.num_tasks = num_tasks

        # Load full CIFAR-100
        train_dataset = datasets.CIFAR100(
            root=root, train=True, download=download, transform=transform
        )

        test_dataset = datasets.CIFAR100(
            root=root, train=False, download=download, transform=transform
        )

        # Combine train and test
        all_data = train_dataset.data
        all_targets = train_dataset.targets

        # Split classes: 64 train, 16 val, 20 test
        all_classes = list(range(100))
        if split == "train":
            split_classes = all_classes[:64]
        elif split == "val":
            split_classes = all_classes[64:80]
        else:  # test
            split_classes = all_classes[80:100]

        # Filter dataset
        indices = [i for i, t in enumerate(all_targets) if t in split_classes]
        self.dataset = torch.utils.data.Subset(train_dataset, indices)

        # Remap labels to 0-n_way
        self.label_map = {old: new for new, old in enumerate(split_classes)}

    def __iter__(self) -> Iterator[Task]:
        """Iterate over tasks."""
        for _ in range(self.num_tasks):
            yield create_episode(self.dataset, self.n_way, self.n_shot, self.n_query)

    def __len__(self) -> int:
        """Return number of tasks."""
        return self.num_tasks


class CustomTaskLoader:
    """Task loader for custom datasets.

    This loader allows using any PyTorch dataset for few-shot learning,
    automatically organizing samples by class.

    Args:
        dataset: Custom dataset with labels
        n_way: Number of classes per task
        n_shot: Support examples per class
        n_query: Query examples per class
        num_tasks: Number of tasks to generate
        shuffle: Whether to shuffle classes

    Example:
        >>> my_dataset = MyCustomDataset()
        >>> loader = CustomTaskLoader(my_dataset, n_way=5, n_shot=1, n_query=15)
        >>> for task in loader:
        ...     process_task(task)
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        num_tasks: int = 1000,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.num_tasks = num_tasks
        self.shuffle = shuffle

        # Organize by class
        self._organize_by_class()

    def _organize_by_class(self) -> None:
        """Organize dataset samples by class."""
        # Get labels
        if hasattr(self.dataset, "targets"):
            labels = self.dataset.targets
        elif hasattr(self.dataset, "labels"):
            labels = self.dataset.labels
        elif hasattr(self.dataset, "y"):
            labels = self.dataset.y
        else:
            # Try to infer from dataset
            labels = []
            for i in range(len(self.dataset)):
                _, label = self.dataset[i]
                labels.append(label)

        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        # Organize
        self.classes = list(set(labels))
        self.class_to_indices = {cls: [] for cls in self.classes}

        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        # Filter classes with enough samples
        min_samples = self.n_shot + self.n_query
        self.classes = [
            cls
            for cls in self.classes
            if len(self.class_to_indices[cls]) >= min_samples
        ]

    def __iter__(self) -> Iterator[Task]:
        """Iterate over tasks."""
        for _ in range(self.num_tasks):
            # Sample classes
            if self.shuffle:
                classes = torch.randperm(len(self.classes))[: self.n_way]
                task_classes = [self.classes[i] for i in classes.tolist()]
            else:
                task_classes = self.classes[: self.n_way]

            support_x, support_y = [], []
            query_x, query_y = [], []

            for new_label, cls in enumerate(task_classes):
                class_indices = self.class_to_indices[cls]
                sampled = torch.randperm(len(class_indices))

                # Support
                for i in sampled[: self.n_shot]:
                    x, _ = self.dataset[class_indices[i.item()]]
                    support_x.append(x)
                    support_y.append(new_label)

                # Query
                for i in sampled[self.n_shot : self.n_shot + self.n_query]:
                    x, _ = self.dataset[class_indices[i.item()]]
                    query_x.append(x)
                    query_y.append(new_label)

            yield Task(
                support_x=torch.stack(support_x),
                support_y=torch.tensor(support_y, dtype=torch.long),
                query_x=torch.stack(query_x),
                query_y=torch.tensor(query_y, dtype=torch.long),
                n_way=self.n_way,
                n_shot=self.n_shot,
                n_query=self.n_query,
            )

    def __len__(self) -> int:
        """Return number of tasks."""
        return self.num_tasks


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_few_shot(
    model: Union[nn.Module, MetaLearner],
    task_loader: Iterator[Task],
    num_tasks: int = 600,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Standard few-shot evaluation protocol.

    Evaluates a model on multiple few-shot tasks and computes average accuracy
    and confidence intervals.

    Args:
        model: Model or meta-learner to evaluate
        task_loader: Iterator over tasks
        num_tasks: Number of evaluation tasks
        device: Device to run evaluation on

    Returns:
        metrics: Dictionary with mean accuracy, std, and confidence intervals

    Example:
        >>> protonet = PrototypicalNetworks(encoder)
        >>> loader = OmniglotTaskLoader(root='./data', n_way=5, n_shot=1, n_query=15)
        >>> metrics = evaluate_few_shot(protonet, loader, num_tasks=600)
        >>> print(f"Accuracy: {metrics['mean']:.2%} ± {metrics['ci95']:.2%}")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    accuracies = []

    with torch.no_grad():
        for i, task in enumerate(task_loader):
            if i >= num_tasks:
                break

            # Move to device
            support_x = task.support_x.to(device)
            support_y = task.support_y.to(device)
            query_x = task.query_x.to(device)
            query_y = task.query_y.to(device)

            # Get predictions
            if isinstance(model, MetaLearner):
                adapted_state = model.adapt(support_x, support_y)
                predictions = model.predict(query_x, adapted_state)
            elif isinstance(model, PrototypicalNetworks):
                predictions = model(
                    support_x, support_y, query_x, task.n_way, task.n_shot
                )
            elif isinstance(model, MAML):
                predictions, _, _ = model(support_x, support_y, query_x)
            else:
                # Generic model - assumes it handles support/query internally
                predictions = model(support_x, support_y, query_x)

            # Compute accuracy
            pred_labels = predictions.argmax(dim=-1)
            accuracy = (pred_labels == query_y).float().mean().item()
            accuracies.append(accuracy)

    accuracies = torch.tensor(accuracies)

    mean = accuracies.mean().item()
    std = accuracies.std().item()
    ci95 = 1.96 * std / math.sqrt(len(accuracies))

    return {"mean": mean, "std": std, "ci95": ci95, "accuracies": accuracies.tolist()}


def cross_domain_evaluation(
    model: Union[nn.Module, MetaLearner],
    source_loader: Iterator[Task],
    target_loader: Iterator[Task],
    num_tasks: int = 600,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on different domain (cross-domain few-shot learning).

    Tests how well the meta-learned model generalizes to a different data
    distribution than it was trained on.

    Args:
        model: Model to evaluate
        source_loader: Source domain task loader (for adaptation)
        target_loader: Target domain task loader (for evaluation)
        num_tasks: Number of tasks to evaluate
        device: Device to run on

    Returns:
        results: Dictionary with source and target domain metrics

    Example:
        >>> results = cross_domain_evaluation(
        ...     model,
        ...     omniglot_loader,
        ...     miniimagenet_loader,
        ...     num_tasks=600
        ... )
    """
    # Evaluate on source domain (in-domain)
    source_metrics = evaluate_few_shot(model, source_loader, num_tasks, device)

    # Evaluate on target domain (cross-domain)
    target_metrics = evaluate_few_shot(model, target_loader, num_tasks, device)

    return {
        "source": source_metrics,
        "target": target_metrics,
        "transfer_gap": source_metrics["mean"] - target_metrics["mean"],
    }


def meta_train(
    meta_learner: MetaLearner,
    train_loader: Iterator[Task],
    val_loader: Optional[Iterator[Task]],
    meta_optimizer: MetaOptimizer,
    num_iterations: int = 60000,
    eval_interval: int = 1000,
    log_interval: int = 100,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> MetaLearningState:
    """Full meta-training pipeline.

    Trains a meta-learner on few-shot tasks with periodic validation and
    checkpoint saving.

    Args:
        meta_learner: Meta-learner to train
        train_loader: Training task loader
        val_loader: Validation task loader
        meta_optimizer: Meta-optimizer
        num_iterations: Number of training iterations
        eval_interval: Evaluate every N iterations
        log_interval: Log every N iterations
        device: Device to train on
        save_path: Path to save best model

    Returns:
        state: MetaLearningState with training history

    Example:
        >>> maml = GradientBasedMetaLearner(model, inner_lr=0.01, num_inner_steps=5)
        >>> optimizer = MetaOptimizer(maml.parameters(), lr=1e-3)
        >>> loader = OmniglotTaskLoader(root='./data', n_way=5, n_shot=1, n_query=15)
        >>> state = meta_train(maml, loader, None, optimizer, num_iterations=60000)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_learner.to(device)
    meta_learner.train()

    state = MetaLearningState()
    best_val_acc = 0.0

    train_iter = iter(train_loader)

    for iteration in range(num_iterations):
        # Sample tasks for this meta-batch
        tasks = []
        for _ in range(4):  # Meta-batch size of 4
            try:
                task = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                task = next(train_iter)

            # Move to device
            task = Task(
                support_x=task.support_x.to(device),
                support_y=task.support_y.to(device),
                query_x=task.query_x.to(device),
                query_y=task.query_y.to(device),
                n_way=task.n_way,
                n_shot=task.n_shot,
                n_query=task.n_query,
            )
            tasks.append(task)

        # Meta-update
        meta_optimizer.zero_grad()

        total_loss = 0.0
        total_acc = 0.0

        for task in tasks:
            # Adapt and predict
            adapted_state = meta_learner.adapt(task.support_x, task.support_y)
            predictions = meta_learner.predict(task.query_x, adapted_state)

            loss = F.cross_entropy(predictions, task.query_y)
            loss.backward()

            total_loss += loss.item()
            acc = (predictions.argmax(dim=-1) == task.query_y).float().mean()
            total_acc += acc.item()

        meta_optimizer.step()

        # Logging
        avg_loss = total_loss / len(tasks)
        avg_acc = total_acc / len(tasks)
        state.outer_losses.append(avg_loss)

        if iteration % log_interval == 0:
            print(
                f"Iteration {iteration}/{num_iterations}: "
                f"Loss = {avg_loss:.4f}, Acc = {avg_acc:.4f}"
            )

        # Validation
        if val_loader is not None and iteration % eval_interval == 0:
            val_metrics = meta_validate(meta_learner, val_loader, device=device)
            print(
                f"Validation: Acc = {val_metrics['mean']:.4f} "
                f"± {val_metrics['ci95']:.4f}"
            )

            state.validation_accuracies.append(val_metrics["mean"])

            # Save best model
            if val_metrics["mean"] > best_val_acc and save_path:
                best_val_acc = val_metrics["mean"]
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": meta_learner.state_dict(),
                        "optimizer_state_dict": meta_optimizer.optimizer.state_dict(),
                        "val_acc": best_val_acc,
                    },
                    save_path,
                )
                print(f"Saved best model with val acc: {best_val_acc:.4f}")

        state.iteration = iteration

    return state


def meta_validate(
    meta_learner: MetaLearner,
    val_loader: Iterator[Task],
    num_tasks: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Validate meta-learner on validation tasks.

    Args:
        meta_learner: Meta-learner to validate
        val_loader: Validation task loader
        num_tasks: Number of validation tasks
        device: Device to validate on

    Returns:
        metrics: Validation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_learner.to(device)
    meta_learner.eval()

    accuracies = []

    with torch.no_grad():
        for i, task in enumerate(val_loader):
            if i >= num_tasks:
                break

            # Move to device
            support_x = task.support_x.to(device)
            support_y = task.support_y.to(device)
            query_x = task.query_x.to(device)
            query_y = task.query_y.to(device)

            # Adapt and predict
            adapted_state = meta_learner.adapt(support_x, support_y)
            predictions = meta_learner.predict(query_x, adapted_state)

            acc = (predictions.argmax(dim=-1) == query_y).float().mean().item()
            accuracies.append(acc)

    accuracies = torch.tensor(accuracies)

    return {
        "mean": accuracies.mean().item(),
        "std": accuracies.std().item(),
        "ci95": 1.96 * accuracies.std().item() / math.sqrt(len(accuracies)),
    }


def meta_test(
    meta_learner: MetaLearner,
    test_loader: Iterator[Task],
    num_tasks: int = 600,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Final test evaluation of meta-learner.

    Args:
        meta_learner: Trained meta-learner
        test_loader: Test task loader
        num_tasks: Number of test tasks
        device: Device to test on

    Returns:
        metrics: Test metrics with 95% confidence intervals
    """
    return evaluate_few_shot(meta_learner, test_loader, num_tasks, device)


# =============================================================================
# Utility Classes and Functions
# =============================================================================


class CNNEncoder(nn.Module):
    """CNN encoder for few-shot learning.

    Standard 4-layer convolutional encoder commonly used in few-shot learning.

    Args:
        input_channels: Number of input channels
        hidden_dim: Number of channels in hidden layers
        output_dim: Output embedding dimension
        image_size: Input image size (assumes square)

    Example:
        >>> encoder = CNNEncoder(input_channels=3, hidden_dim=64, output_dim=64)
        >>> embeddings = encoder(images)  # [batch, 64]
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 64,
        image_size: int = 84,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            flattened_size = self.features(dummy).view(1, -1).size(1)

        self.fc = nn.Linear(flattened_size, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input images [batch, channels, height, width]

        Returns:
            embeddings: Output embeddings [batch, output_dim]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetEncoder(nn.Module):
    """ResNet-based encoder for few-shot learning.

    Uses a ResNet backbone for better feature extraction on more complex datasets.

    Args:
        input_channels: Number of input channels
        output_dim: Output embedding dimension
        backbone: ResNet variant ('resnet10', 'resnet18', 'resnet34')
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim

        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision is required for ResNetEncoder")

        # Load ResNet
        if backbone == "resnet10":
            # Custom small ResNet
            self.backbone = self._make_resnet10(input_channels)
            backbone_dim = 512
        elif backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            if input_channels != 3:
                resnet.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            backbone_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.backbone = resnet
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            if input_channels != 3:
                resnet.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            backbone_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.backbone = resnet
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.fc = nn.Linear(backbone_dim, output_dim)

    def _make_resnet10(self, input_channels: int) -> nn.Module:
        """Create a small ResNet-10 architecture."""
        return nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            # Residual blocks
            self._make_res_block(64, 64),
            self._make_res_block(64, 128, stride=2),
            self._make_res_block(128, 256, stride=2),
            self._make_res_block(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )

    def _make_res_block(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.backbone(x)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def compute_prototypical_accuracy(
    support_x: Tensor,
    support_y: Tensor,
    query_x: Tensor,
    query_y: Tensor,
    encoder: nn.Module,
    distance: str = "euclidean",
) -> float:
    """Compute accuracy using prototypical network approach.

    Args:
        support_x: Support inputs
        support_y: Support labels
        query_x: Query inputs
        query_y: Query labels
        encoder: Feature encoder
        distance: Distance metric

    Returns:
        accuracy: Classification accuracy
    """
    with torch.no_grad():
        support_emb = encoder(support_x)
        query_emb = encoder(query_x)

        # Compute prototypes
        classes = torch.unique(support_y)
        prototypes = []
        for cls in classes:
            mask = support_y == cls
            prototypes.append(support_emb[mask].mean(dim=0))
        prototypes = torch.stack(prototypes)

        # Compute distances
        if distance == "euclidean":
            distances = torch.cdist(query_emb, prototypes).pow(2)
        else:  # cosine
            query_norm = F.normalize(query_emb, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            distances = 1 - torch.mm(query_norm, proto_norm.t())

        # Predict
        predictions = distances.argmin(dim=-1)
        accuracy = (predictions == query_y).float().mean().item()

    return accuracy


def compute_confidence_interval(
    accuracies: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute mean and confidence interval for accuracies.

    Args:
        accuracies: List of accuracy values
        confidence: Confidence level (default 0.95)

    Returns:
        mean: Mean accuracy
        ci: Confidence interval half-width
    """
    import scipy.stats

    accuracies = torch.tensor(accuracies)
    mean = accuracies.mean().item()
    std = accuracies.std().item()
    n = len(accuracies)

    # t-distribution critical value
    t_crit = scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_crit * std / math.sqrt(n)

    return mean, ci


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data structures
    "Task",
    "Episode",
    "MetaLearningState",
    # Meta-learning algorithms
    "MAML",
    "Reptile",
    "PrototypicalNetworks",
    "MatchingNetworks",
    "RelationNetworks",
    "RelationModule",
    # Few-shot utilities
    "TaskSampler",
    "EpisodeDataset",
    "create_episode",
    "split_support_query",
    # Meta-learners
    "MetaLearner",
    "GradientBasedMetaLearner",
    "MetricBasedMetaLearner",
    "MemoryAugmentedMetaLearner",
    # Inner loop optimization
    "InnerLoopOptimizer",
    "MetaOptimizer",
    "learnable_learning_rates",
    # Task loaders
    "OmniglotTaskLoader",
    "MiniImageNetTaskLoader",
    "CIFAR_FSTaskLoader",
    "CustomTaskLoader",
    # Evaluation
    "evaluate_few_shot",
    "cross_domain_evaluation",
    "meta_train",
    "meta_validate",
    "meta_test",
    # Utilities
    "CNNEncoder",
    "ResNetEncoder",
    "compute_prototypical_accuracy",
    "compute_confidence_interval",
]
