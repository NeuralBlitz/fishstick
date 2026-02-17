"""
Attention Visualization

Tools for visualizing and analyzing attention patterns.
"""

from typing import Optional, List
import torch
from torch import Tensor, nn
import numpy as np


class AttentionVisualization:
    """Visualize attention patterns in transformer models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = []

    def get_attention(self, x: Tensor) -> List[Tensor]:
        """Extract attention weights from all layers."""
        self.attention_weights = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                attn = output[1]
            else:
                attn = output
            self.attention_weights.append(attn)

        hooks = []
        for name, module in self.model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.model(x)

        for hook in hooks:
            hook.remove()

        return self.attention_weights

    def visualize_head(self, layer: int, head: int) -> np.ndarray:
        """Visualize a specific attention head."""
        if layer >= len(self.attention_weights):
            raise ValueError(f"Layer {layer} not found")

        attn = self.attention_weights[layer]
        if head >= attn.shape[1]:
            raise ValueError(f"Head {head} not found")

        return attn[0, head].cpu().numpy()

    def average_heads(self, layer: int) -> np.ndarray:
        """Average attention across all heads."""
        if layer >= len(self.attention_weights):
            raise ValueError(f"Layer {layer} not found")

        attn = self.attention_weights[layer]
        return attn[0].mean(dim=0).cpu().numpy()

    def attention_flow(self, layer: int) -> np.ndarray:
        """Compute attention flow between tokens."""
        attn = self.attention_weights[layer][0]
        num_heads = attn.shape[0]

        flow = attn.mean(dim=0)
        return flow.cpu().numpy()


class AttentionRollout:
    """Compute attention rollout for analyzing layer-wise information flow."""

    def __init__(self, model: nn.Module, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.attention_weights = []

    def compute_rollout(self, x: Tensor) -> Tensor:
        """Compute attention rollout across all layers."""
        self.attention_weights = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                attn = output[1]
            else:
                attn = output
            self.attention_weights.append(attn)

        hooks = []
        for name, module in self.model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.model(x)

        for hook in hooks:
            hook.remove()

        rollout = torch.eye(x.shape[1]).unsqueeze(0).to(x.device)

        for attn in self.attention_weights:
            attn = attn[0].mean(dim=0)
            attn = attn + torch.eye(attn.shape[0]).to(attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(attn, rollout)

        return rollout.squeeze(0)


class AttentionPatternAnalysis:
    """Analyze attention patterns for insights."""

    @staticmethod
    def compute_entropy(attention: Tensor) -> Tensor:
        """Compute attention entropy for each head."""
        epsilon = 1e-8
        entropy = -(attention * torch.log(attention + epsilon)).sum(dim=-1)
        return entropy.mean()

    @staticmethod
    def compute_attention_density(attention: Tensor) -> Tensor:
        """Compute how focused attention is."""
        return (attention > 0.1).float().mean()

    @staticmethod
    def find_attention_heads(attention: Tensor, threshold: float = 0.5) -> List[tuple]:
        """Find significant attention connections."""
        connections = []
        batch, n_heads, seq, _ = attention.shape

        for b in range(batch):
            for h in range(n_heads):
                attn = attention[b, h]
                for i in range(seq):
                    for j in range(seq):
                        if attn[i, j] > threshold:
                            connections.append((h, i, j, attn[i, j].item()))

        return connections
