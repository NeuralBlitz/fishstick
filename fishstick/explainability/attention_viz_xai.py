"""
Attention Visualization for Explainable AI

Advanced attention visualization tools specifically designed for explainability.
Extracts, analyzes, and visualizes attention patterns in transformer-based models
to provide human-interpretable explanations of model behavior.

Includes:
- TransformerAttentionExtractor: Extract multi-head attention weights
- AttentionHeadAnalyzer: Analyze and score attention heads
- AttentionPatternClusterer: Cluster similar attention patterns
- CrossAttentionVisualizer: Visualize cross-attention in encoder-decoder models
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

import torch
from torch import Tensor, nn
import numpy as np


class AttentionType(Enum):
    """Types of attention patterns."""

    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD = "multi_head"
    CAUSAL = "causal"


@dataclass
class AttentionWeights:
    """Container for extracted attention weights."""

    attention_probs: Tensor
    query: Optional[Tensor] = None
    key: Optional[Tensor] = None
    value: Optional[Tensor] = None
    layer: int = 0
    head: int = 0
    attention_type: AttentionType = AttentionType.SELF_ATTENTION


class TransformerAttentionExtractor:
    """Extract attention weights from transformer models.

    Provides a unified interface for extracting attention weights from
    various transformer architectures (BERT, GPT, ViT, etc.).

    Args:
        model: Transformer model to extract attention from
        layer_indices: Specific layers to extract (default: all)
        include_qkv: Whether to extract Q, K, V projections

    Example:
        >>> extractor = TransformerAttentionExtractor(model)
        >>> weights = extractor.extract(input_ids, attention_mask)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
        include_qkv: bool = True,
    ):
        self.model = model
        self.layer_indices = layer_indices
        self.include_qkv = include_qkv
        self._attention_weights: Dict[str, List[AttentionWeights]] = {}
        self._hooks = []

    def extract(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Dict[str, List[AttentionWeights]]:
        """Extract attention weights from the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for BERT-style models)

        Returns:
            Dictionary mapping layer names to attention weights
        """
        self._attention_weights.clear()
        self._register_hooks()

        self.model.eval()
        with torch.no_grad():
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            _ = self.model(**kwargs)

        self._remove_hooks()
        return self._attention_weights

    def _register_hooks(self):
        """Register forward hooks on attention layers."""

        def create_hook(name: str, layer_idx: int):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attn_probs = output[1] if len(output) > 1 else None
                    if attn_probs is not None:
                        weights = AttentionWeights(
                            attention_probs=attn_probs,
                            layer=layer_idx,
                            attention_type=AttentionType.SELF_ATTENTION,
                        )
                        if name not in self._attention_weights:
                            self._attention_weights[name] = []
                        self._attention_weights[name].append(weights)

            return hook

        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                if self.layer_indices is not None:
                    layer_num = self._extract_layer_number(name)
                    if layer_num not in self.layer_indices:
                        continue

                hook_fn = create_hook(name, self._extract_layer_number(name))
                handle = module.register_forward_hook(hook_fn)
                self._hooks.append(handle)

    def _extract_layer_number(self, name: str) -> int:
        """Extract layer number from module name."""
        parts = name.split(".")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


class AttentionHeadAnalyzer:
    """Analyze and score attention heads.

    Provides metrics for evaluating the importance and behavior of
    individual attention heads, including:
    - Attention entropy (spread of attention)
    - Attention sparsity
    - Head importance scores
    - Pattern analysis

    Args:
        model: Transformer model
        num_layers: Number of layers
        num_heads: Number of attention heads

    Example:
        >>> analyzer = AttentionHeadAnalyzer(model, num_layers=12, num_heads=12)
        >>> analysis = analyzer.analyze_attention_heads(input_tensor)
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int = 12,
        num_heads: int = 12,
    ):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.extractor = TransformerAttentionExtractor(model)

    def analyze_attention_heads(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """Analyze attention heads and compute metrics.

        Returns:
            Dictionary with entropy, sparsity, importance scores
        """
        attention_weights = self.extractor.extract(inputs, attention_mask)

        all_weights = []
        for name, weights_list in attention_weights.items():
            all_weights.extend(weights_list)

        if not all_weights:
            warnings.warn("No attention weights extracted")
            return {}

        stacked = torch.stack([w.attention_probs for w in all_weights])

        entropy = self._compute_entropy(stacked)
        sparsity = self._compute_sparsity(stacked)
        importance = self._compute_importance(stacked)

        return {
            "entropy": entropy,
            "sparsity": sparsity,
            "importance": importance,
        }

    def _compute_entropy(self, weights: Tensor) -> np.ndarray:
        """Compute attention entropy per head."""
        weights = weights.clamp(min=1e-8)
        entropy = -(weights * torch.log(weights)).sum(dim=-1)
        return entropy.mean(dim=(0, 1)).cpu().numpy()

    def _compute_sparsity(self, weights: Tensor) -> np.ndarray:
        """Compute attention sparsity (focus on few tokens)."""
        top_k = 3
        batch_size, num_heads, seq_len, _ = weights.shape

        _, top_indices = weights.topk(top_k, dim=-1)
        top_weights = weights.gather(-1, top_indices)

        sparsity = 1 - (top_weights.sum(dim=-1) / weights.sum(dim=-1))
        return sparsity.mean(dim=(0, 1)).cpu().numpy()

    def _compute_importance(self, weights: Tensor) -> np.ndarray:
        """Compute importance based on attention to [CLS] or first token."""
        cls_attention = weights[..., 0]
        importance = cls_attention.mean(dim=(0, 1))
        return importance.cpu().numpy()

    def rank_heads(
        self,
        inputs: Tensor,
        metric: str = "importance",
    ) -> List[Tuple[int, int, float]]:
        """Rank attention heads by a specific metric.

        Args:
            inputs: Input tensor
            metric: Metric to rank by ('importance', 'entropy', 'sparsity')

        Returns:
            List of (layer, head, score) tuples sorted by score
        """
        analysis = self.analyze_attention_heads(inputs)

        if metric not in analysis:
            raise ValueError(f"Unknown metric: {metric}")

        scores = analysis[metric]
        rankings = []

        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                idx = layer * self.num_heads + head
                if idx < len(scores):
                    rankings.append((layer, head, float(scores[idx])))

        rankings.sort(key=lambda x: x[2], reverse=(metric != "entropy"))
        return rankings


class AttentionPatternClusterer:
    """Cluster similar attention patterns.

    Groups attention heads with similar patterns to identify functional
    redundancy and discover specialized heads (e.g., syntactic, semantic).

    Args:
        n_clusters: Number of clusters
        metric: Distance metric for clustering

    Example:
        >>> clusterer = AttentionPatternClusterer(n_clusters=5)
        >>> clusters = clusterer.fit_cluster(attention_weights)
    """

    def __init__(
        self,
        n_clusters: int = 5,
        metric: str = "cosine",
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit_cluster(
        self,
        attention_weights: Dict[str, List[AttentionWeights]],
    ) -> np.ndarray:
        """Cluster attention patterns.

        Args:
            attention_weights: Dictionary of attention weights

        Returns:
            Cluster labels for each head
        """
        features = self._extract_features(attention_weights)

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels_ = kmeans.fit_predict(features)
        self.cluster_centers_ = kmeans.cluster_centers_

        return self.labels_

    def _extract_features(
        self,
        attention_weights: Dict[str, List[AttentionWeights]],
    ) -> np.ndarray:
        """Extract feature vectors from attention weights."""
        all_weights = []
        for name, weights_list in attention_weights.items():
            for weights in weights_list:
                probs = weights.attention_probs
                features = self._compute_pattern_features(probs)
                all_weights.append(features)

        if not all_weights:
            return np.array([])

        return np.stack(all_weights)

    def _compute_pattern_features(self, probs: Tensor) -> np.ndarray:
        """Compute features from attention probability distribution."""
        batch_mean = probs.mean(dim=0)

        mean_attn = batch_mean.mean(dim=-1).cpu().numpy()
        max_attn = batch_mean.max(dim=-1)[0].cpu().numpy()
        entropy = -(batch_mean * torch.log(batch_mean + 1e-8)).sum(-1).cpu().numpy()

        return np.concatenate([mean_attn, max_attn, entropy])


class CrossAttentionVisualizer:
    """Visualize cross-attention in encoder-decoder models.

    Extracts and analyzes cross-attention weights to understand how
    decoder attends to encoder representations.

    Args:
        model: Encoder-decoder model
        encoder_layer_indices: Layers to extract from encoder
        decoder_layer_indices: Layers to extract from decoder

    Example:
        >>> cross_viz = CrossAttentionVisualizer(model)
        >>> cross_attn = cross_viz.extract_cross_attention(encoder_out, decoder_input)
    """

    def __init__(
        self,
        model: nn.Module,
        encoder_layer_indices: Optional[List[int]] = None,
        decoder_layer_indices: Optional[List[int]] = None,
    ):
        self.model = model
        self.encoder_layer_indices = encoder_layer_indices
        self.decoder_layer_indices = decoder_layer_indices
        self._cross_attention: Optional[Tensor] = None

    def extract_cross_attention(
        self,
        encoder_output: Tensor,
        decoder_input: Tensor,
    ) -> Tensor:
        """Extract cross-attention weights.

        Args:
            encoder_output: Encoder hidden states
            decoder_input: Decoder input

        Returns:
            Cross-attention weights (batch, heads, decoder_len, encoder_len)
        """
        self._cross_attention = None

        self._register_cross_hooks()

        with torch.no_grad():
            _ = self.model(encoder_outputs=encoder_output, input_ids=decoder_input)

        self._remove_hooks()

        if self._cross_attention is None:
            warnings.warn("Cross-attention not captured")
            return torch.zeros(1, 1, decoder_input.size(1), encoder_output.size(1))

        return self._cross_attention

    def _register_cross_hooks(self):
        """Register hooks for cross-attention layers."""

        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                self._cross_attention = output[1]

        for name, module in self.model.named_modules():
            if "cross" in name.lower() or "encoder_decoder" in name.lower():
                handle = module.register_forward_hook(hook_fn)
                self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove hooks."""
        for handle in getattr(self, "_hooks", []):
            handle.remove()
        self._hooks = []

    def compute_attention_flow(
        self,
        cross_attention: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute aggregated attention flow metrics.

        Returns:
            Dictionary with aggregated attention flows
        """
        attn_mean = cross_attention.mean(dim=1)
        attn_max = cross_attention.max(dim=1)[0]

        source_importance = cross_attention.sum(dim=2)
        target_importance = cross_attention.sum(dim=3)

        return {
            "mean": attn_mean,
            "max": attn_max,
            "source_importance": source_importance,
            "target_importance": target_importance,
        }


class AttentionGradientExtractor:
    """Extract gradients through attention for counterfactual analysis.

    Computes how changes in attention weights affect outputs, enabling
    understanding of causal attention mechanisms.

    Args:
        model: Transformer model

    Example:
        >>> grad_extractor = AttentionGradientExtractor(model)
        >>> gradients = grad_extractor.extract_gradients(input_tensor, target=0)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attention_grads: List[Tensor] = []

    def extract_gradients(
        self,
        inputs: Tensor,
        target: int,
    ) -> List[Tensor]:
        """Extract gradients through attention weights.

        Args:
            inputs: Input tensor
            target: Target class index

        Returns:
            List of gradient tensors per layer
        """
        self._attention_grads.clear()
        hooks = self._register_hooks()

        self.model.eval()
        inputs.requires_grad_(True)

        output = self.model(inputs)
        score = output[0, target] if output.dim() > 1 else output[0]
        score.backward()

        self._remove_hooks(hooks)

        return self._attention_grads

    def _register_hooks(self):
        """Register backward hooks on attention layers."""
        hooks = []

        def create_hook():
            def hook(module, grad_input, grad_output):
                self._attention_grads.append(grad_output[0])

            return hook

        for module in self.model.modules():
            if "attention" in str(type(module)).lower():
                handle = module.register_full_backward_hook(create_hook())
                hooks.append(handle)

        return hooks

    def _remove_hooks(self, hooks: List):
        """Remove registered hooks."""
        for handle in hooks:
            handle.remove()


def create_attention_explainer(
    model: nn.Module,
    explainer_type: str = "standard",
    **kwargs,
) -> Union[
    TransformerAttentionExtractor,
    AttentionHeadAnalyzer,
    AttentionPatternClusterer,
    CrossAttentionVisualizer,
]:
    """Factory function to create attention explainers.

    Args:
        model: Model to analyze
        explainer_type: Type of explainer
        **kwargs: Additional arguments

    Returns:
        Configured attention explainer

    Example:
        >>> explainer = create_attention_explainer(model, 'analyzer', num_heads=12)
    """
    if explainer_type == "standard":
        return TransformerAttentionExtractor(model, **kwargs)
    elif explainer_type == "analyzer":
        return AttentionHeadAnalyzer(model, **kwargs)
    elif explainer_type == "clusterer":
        return AttentionPatternClusterer(**kwargs)
    elif explainer_type == "cross":
        return CrossAttentionVisualizer(model, **kwargs)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
