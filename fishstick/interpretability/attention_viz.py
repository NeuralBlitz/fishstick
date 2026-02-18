"""
Attention Visualization Methods

Implements various attention visualization techniques:
- Attention Rollout
- Attention Flow
- Head Importance Analysis
- Attention Pattern Analysis
"""

from typing import Optional, List, Dict, Tuple, Union, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math


class AttentionVisualizerBase(ABC):
    """Base class for attention visualization methods."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self._hooks = []
        self._attention_maps: Dict[str, Tensor] = {}

    def _register_attention_hooks(self, attention_modules: Dict[str, nn.Module]):
        self._clear_hooks()
        self._attention_maps = {}

        def make_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1:
                    attn = out[1]
                    if isinstance(attn, Tensor):
                        self._attention_maps[name] = attn.detach()
                elif isinstance(out, Tensor):
                    if out.dim() == 4:
                        self._attention_maps[name] = out.detach()

            return hook

        for name, module in attention_modules.items():
            h = module.register_forward_hook(make_hook(name))
            self._hooks.append(h)

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._attention_maps = {}

    @abstractmethod
    def visualize(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def __del__(self):
        self._clear_hooks()


class AttentionRollout(AttentionVisualizerBase):
    """Attention Rollout for Vision Transformers.

    Computes attention rollout by recursively multiplying attention matrices.

    Args:
        model: Vision Transformer model
        attention_layers: Dict mapping layer names to attention modules
        discard_ratio: Ratio of lowest attention to discard
        head_fusion: How to fuse heads ('mean', 'max', 'min')
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
        discard_ratio: float = 0.0,
        head_fusion: str = "mean",
    ):
        super().__init__(model)
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

        if attention_layers is not None:
            self._register_attention_hooks(attention_layers)

        self.attention_layers = attention_layers

    def set_attention_layers(self, layers: Dict[str, nn.Module]):
        self.attention_layers = layers
        self._register_attention_hooks(layers)

    def _fuse_heads(self, attention: Tensor) -> Tensor:
        B, H, N, _ = attention.shape

        if self.discard_ratio > 0:
            flat = attention.view(B, H, -1)
            _, indices = flat.topk(
                int(flat.size(-1) * self.discard_ratio), -1, largest=False
            )
            flat.scatter_(-1, indices, 0)
            attention = flat.view(B, H, N, N)

        if self.head_fusion == "mean":
            return attention.mean(dim=1)
        elif self.head_fusion == "max":
            return attention.max(dim=1)[0]
        elif self.head_fusion == "min":
            return attention.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown fusion: {self.head_fusion}")

    def visualize(
        self,
        x: Tensor,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
        start_layer: int = 0,
    ) -> Tensor:
        if attention_layers is not None:
            self.set_attention_layers(attention_layers)

        if self.attention_layers is None:
            raise ValueError("Attention layers must be provided")

        self._attention_maps = {}

        with torch.no_grad():
            _ = self.model(x)

        layer_names = sorted(self._attention_maps.keys())

        if len(layer_names) == 0:
            raise RuntimeError("No attention maps captured")

        first_attn = self._attention_maps[layer_names[0]]
        batch_size = first_attn.size(0)
        num_tokens = first_attn.size(-1)

        result = (
            torch.eye(num_tokens, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        for i, name in enumerate(layer_names):
            if i < start_layer:
                continue

            attn = self._attention_maps[name]

            attn_fused = self._fuse_heads(attn)

            attn_with_identity = attn_fused + torch.eye(
                num_tokens, device=x.device
            ).unsqueeze(0)
            attn_norm = attn_with_identity / attn_with_identity.sum(
                dim=-1, keepdim=True
            )

            result = torch.bmm(attn_norm, result)

        mask = result[:, 0, 1:]

        num_patches = int(math.sqrt(mask.size(-1)))
        if num_patches * num_patches == mask.size(-1):
            mask = mask.reshape(batch_size, num_patches, num_patches)

        return mask


class AttentionFlow(AttentionVisualizerBase):
    """Attention Flow using flow-based aggregation.

    Computes attention flow considering token interactions.

    Args:
        model: Vision Transformer model
        attention_layers: Dict mapping layer names to attention modules
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__(model)

        if attention_layers is not None:
            self._register_attention_hooks(attention_layers)

        self.attention_layers = attention_layers

    def set_attention_layers(self, layers: Dict[str, nn.Module]):
        self.attention_layers = layers
        self._register_attention_hooks(layers)

    def visualize(
        self,
        x: Tensor,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
    ) -> Tensor:
        if attention_layers is not None:
            self.set_attention_layers(attention_layers)

        if self.attention_layers is None:
            raise ValueError("Attention layers must be provided")

        self._attention_maps = {}

        with torch.no_grad():
            _ = self.model(x)

        layer_names = sorted(self._attention_maps.keys())

        if len(layer_names) == 0:
            raise RuntimeError("No attention maps captured")

        first_attn = self._attention_maps[layer_names[0]]
        batch_size = first_attn.size(0)
        num_tokens = first_attn.size(-1)

        flow = (
            torch.eye(num_tokens, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        for name in layer_names:
            attn = self._attention_maps[name]
            attn_mean = attn.mean(dim=1)

            attn_norm = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-8)

            flow = torch.bmm(attn_norm, flow)
            flow = flow / (flow.sum(dim=-2, keepdim=True) + 1e-8)

        mask = flow[:, 0, 1:]

        num_patches = int(math.sqrt(mask.size(-1)))
        if num_patches * num_patches == mask.size(-1):
            mask = mask.reshape(batch_size, num_patches, num_patches)

        return mask


class HeadImportance(AttentionVisualizerBase):
    """Head Importance Analysis for Transformers.

    Computes importance scores for each attention head.

    Args:
        model: Transformer model
        attention_layers: Dict mapping layer names to attention modules
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__(model)

        if attention_layers is not None:
            self._register_attention_hooks(attention_layers)

        self.attention_layers = attention_layers
        self._head_importance: Dict[str, Tensor] = {}

    def set_attention_layers(self, layers: Dict[str, nn.Module]):
        self.attention_layers = layers
        self._register_attention_hooks(layers)

    def compute_importance(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
    ) -> Dict[str, Tensor]:
        if attention_layers is not None:
            self.set_attention_layers(attention_layers)

        if self.attention_layers is None:
            raise ValueError("Attention layers must be provided")

        self._attention_maps = {}
        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        target_output = output.gather(1, target_indices.unsqueeze(1)).squeeze(1)

        self.model.zero_grad()
        target_output.sum().backward()

        self._head_importance = {}

        for name, attn in self._attention_maps.items():
            B, H, N, _ = attn.shape

            head_importance = attn.abs().mean(dim=(0, 2, 3))

            self._head_importance[name] = head_importance

        return self._head_importance

    def visualize(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
    ) -> Tensor:
        importance = self.compute_importance(x, target, attention_layers)

        all_importance = torch.cat(list(importance.values()))

        return all_importance

    def get_head_ranking(self) -> List[Tuple[str, int, float]]:
        if not self._head_importance:
            raise RuntimeError("Compute importance first")

        rankings = []
        for layer_name, importance in self._head_importance.items():
            for head_idx, score in enumerate(importance.tolist()):
                rankings.append((layer_name, head_idx, score))

        rankings.sort(key=lambda x: x[2], reverse=True)
        return rankings

    def prune_heads(self, threshold: float = 0.1) -> Dict[str, List[int]]:
        if not self._head_importance:
            raise RuntimeError("Compute importance first")

        max_importance = max(imp.max().item() for imp in self._head_importance.values())

        heads_to_prune = {}
        for layer_name, importance in self._head_importance.items():
            normalized = importance / (max_importance + 1e-8)
            prune_mask = normalized < threshold
            heads_to_prune[layer_name] = torch.where(prune_mask)[0].tolist()

        return heads_to_prune


class AttentionPatternAnalyzer(AttentionVisualizerBase):
    """Attention Pattern Analysis.

    Analyzes patterns in attention weights.

    Args:
        model: Transformer model
        attention_layers: Dict mapping layer names to attention modules
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__(model)

        if attention_layers is not None:
            self._register_attention_hooks(attention_layers)

        self.attention_layers = attention_layers

    def set_attention_layers(self, layers: Dict[str, nn.Module]):
        self.attention_layers = layers
        self._register_attention_hooks(layers)

    def analyze_patterns(
        self, x: Tensor, attention_layers: Optional[Dict[str, nn.Module]] = None
    ) -> Dict[str, Dict[str, Tensor]]:
        if attention_layers is not None:
            self.set_attention_layers(attention_layers)

        if self.attention_layers is None:
            raise ValueError("Attention layers must be provided")

        self._attention_maps = {}

        with torch.no_grad():
            _ = self.model(x)

        analysis = {}

        for name, attn in self._attention_maps.items():
            B, H, N, _ = attn.shape

            patterns = {
                "entropy": self._compute_entropy(attn),
                "sparsity": self._compute_sparsity(attn),
                "locality": self._compute_locality(attn),
                "head_diversity": self._compute_head_diversity(attn),
            }

            analysis[name] = patterns

        return analysis

    def _compute_entropy(self, attn: Tensor) -> Tensor:
        B, H, N, _ = attn.shape
        entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)
        return entropy.mean(dim=(0, 1))

    def _compute_sparsity(self, attn: Tensor, threshold: float = 0.01) -> Tensor:
        sparse_ratio = (attn < threshold).float().mean(dim=(0, 1))
        return sparse_ratio

    def _compute_locality(self, attn: Tensor) -> Tensor:
        B, H, N, _ = attn.shape

        positions = torch.arange(N, device=attn.device).float()

        expected_dist = torch.zeros(H, device=attn.device)

        for h in range(H):
            dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
            weighted_dist = (attn[0, h] * dist_matrix).sum() / attn[0, h].sum()
            expected_dist[h] = weighted_dist

        return expected_dist / N

    def _compute_head_diversity(self, attn: Tensor) -> Tensor:
        B, H, N, _ = attn.shape

        attn_flat = attn[0].view(H, -1)

        similarity = torch.mm(attn_flat, attn_flat.T)
        attn_norm = attn_flat.norm(dim=1, keepdim=True)
        similarity = similarity / (attn_norm * attn_norm.T + 1e-8)

        off_diagonal = similarity.clone()
        off_diagonal.fill_diagonal_(0)

        diversity = 1 - off_diagonal.abs().mean()

        return diversity.unsqueeze(0)

    def visualize(
        self, x: Tensor, attention_layers: Optional[Dict[str, nn.Module]] = None
    ) -> Dict[str, Dict[str, Tensor]]:
        return self.analyze_patterns(x, attention_layers)


class AttentionGradient(AttentionVisualizerBase):
    """Attention Gradient Visualization.

    Visualizes gradients of attention weights.

    Args:
        model: Transformer model
        attention_layers: Dict mapping layer names to attention modules
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__(model)
        self._attention_grads: Dict[str, Tensor] = {}

        if attention_layers is not None:
            self._register_attention_hooks(attention_layers)

        self.attention_layers = attention_layers

    def _register_attention_hooks(self, attention_modules: Dict[str, nn.Module]):
        self._clear_hooks()
        self._attention_maps = {}
        self._attention_grads = {}

        def make_forward_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1:
                    attn = out[1]
                    if isinstance(attn, Tensor):
                        self._attention_maps[name] = attn

            return hook

        def make_backward_hook(name: str):
            def hook(module, grad_in, grad_out):
                if name in self._attention_maps:
                    self._attention_grads[name] = self._attention_maps[
                        name
                    ].grad.clone()

            return hook

        for name, module in attention_modules.items():
            h1 = module.register_forward_hook(make_forward_hook(name))
            h2 = module.register_full_backward_hook(make_backward_hook(name))
            self._hooks.extend([h1, h2])

    def set_attention_layers(self, layers: Dict[str, nn.Module]):
        self.attention_layers = layers
        self._register_attention_hooks(layers)

    def visualize(
        self,
        x: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        attention_layers: Optional[Dict[str, nn.Module]] = None,
    ) -> Dict[str, Tensor]:
        if attention_layers is not None:
            self.set_attention_layers(attention_layers)

        if self.attention_layers is None:
            raise ValueError("Attention layers must be provided")

        self._attention_maps = {}
        self._attention_grads = {}

        x = x.clone().requires_grad_(True)

        output = self.model(x)

        if target is None:
            target_indices = output.argmax(dim=-1)
        elif isinstance(target, int):
            target_indices = torch.full(
                (x.size(0),), target, dtype=torch.long, device=x.device
            )
        else:
            target_indices = target

        target_output = output.gather(1, target_indices.unsqueeze(1)).squeeze(1)

        self.model.zero_grad()
        target_output.sum().backward()

        result = {}
        for name in self._attention_maps:
            if name in self._attention_grads:
                grad = self._attention_grads[name]
                result[name] = grad.abs().mean(dim=1)

        return result


def create_attention_viz(
    method: str,
    model: nn.Module,
    attention_layers: Optional[Dict[str, nn.Module]] = None,
    **kwargs,
) -> AttentionVisualizerBase:
    """Factory function to create attention visualization methods.

    Args:
        method: Method name ('rollout', 'flow', 'importance', 'pattern', 'gradient')
        model: Transformer model
        attention_layers: Dict mapping layer names to attention modules
        **kwargs: Additional arguments

    Returns:
        Attention visualizer instance
    """
    methods = {
        "rollout": AttentionRollout,
        "flow": AttentionFlow,
        "importance": HeadImportance,
        "pattern": AttentionPatternAnalyzer,
        "gradient": AttentionGradient,
    }

    method_lower = method.lower()
    if method_lower not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    return methods[method_lower](model, attention_layers, **kwargs)
