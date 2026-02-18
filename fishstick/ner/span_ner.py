"""
Span-based NER module for Named Entity Recognition.

Provides span extraction models that identify entity spans directly
rather than doing token-by-token classification. Includes:
- Span extractor
- Boundary detector
- Entity type classifier
- Merge and label approach
- Token reduction for long sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Set
import math


class BoundaryDetector(nn.Module):
    """Boundary detector for span-based NER.
    
    Identifies start and end positions of potential entities.
    
    Args:
        hidden_dim: Dimension of hidden states
        num_layers: Number of detection layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.start_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        self.end_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect start and end positions.
        
        Args:
            hidden_states: Encoder hidden states (batch_size, seq_len, hidden_dim)
            
        Returns:
            Tuple of (start_logits, end_logits)
        """
        start_logits = self.start_detector(hidden_states).squeeze(-1)
        end_logits = self.end_detector(hidden_states).squeeze(-1)
        
        return start_logits, end_logits


class EntityTypeClassifier(nn.Module):
    """Entity type classifier for span-based NER.
    
    Classifies entity types for given span representations.
    
    Args:
        hidden_dim: Dimension of hidden states
        num_types: Number of entity types
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_types: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_types = num_types
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_types),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        start_ids: torch.Tensor,
        end_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Classify entity types for spans.
        
        Args:
            hidden_states: Encoder hidden states
            start_ids: Start position indices
            end_ids: End position indices
            
        Returns:
            Type logits (batch_size, num_spans, num_types)
        """
        batch_size = hidden_states.size(0)
        
        start_hidden = hidden_states.gather(1, start_ids.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        end_hidden = hidden_states.gather(1, end_ids.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        
        span_repr = torch.cat([start_hidden, end_hidden, start_hidden * end_hidden], dim=-1)
        
        type_logits = self.classifier(span_repr)
        
        return type_logits


class SpanExtractor(nn.Module):
    """Span extractor for entity extraction.
    
    Extracts span representations using attention or pooling.
    
    Args:
        hidden_dim: Dimension of hidden states
        span_hidden: Dimension of span representations
        pooling_type: Type of pooling ('mean', 'max', 'attention')
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        span_hidden: int = 256,
        pooling_type: str = "attention",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.span_hidden = span_hidden
        self.pooling_type = pooling_type
        
        if pooling_type == "attention":
            self.attention = nn.Linear(hidden_dim, 1)
        elif pooling_type == "boundary":
            self.start_projection = nn.Linear(hidden_dim, span_hidden)
            self.end_projection = nn.Linear(hidden_dim, span_hidden)
        else:
            self.projection = nn.Linear(hidden_dim, span_hidden)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        start_ids: torch.Tensor,
        end_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract span representations.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            start_ids: (batch_size, num_spans)
            end_ids: (batch_size, num_spans)
            
        Returns:
            Span representations (batch_size, num_spans, span_hidden)
        """
        batch_size = hidden_states.size(0)
        num_spans = start_ids.size(1)
        
        start_hidden = hidden_states.gather(
            1, start_ids.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )
        end_hidden = hidden_states.gather(
            1, end_ids.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )
        
        if self.pooling_type == "boundary":
            span_hidden = self.start_projection(start_hidden) + self.end_projection(end_hidden)
        elif self.pooling_type == "attention":
            span_masks = self._create_span_masks(start_ids, end_ids, hidden_states.size(1))
            attention_weights = self.attention(hidden_states).squeeze(-1)
            attention_weights = attention_weights.masked_fill(span_masks == 0, float("-inf"))
            attention_weights = F.softmax(attention_weights, dim=1)
            attended = (hidden_states * attention_weights.unsqueeze(-1)).sum(dim=1)
            attended = attended.unsqueeze(1).expand(-1, num_spans, -1)
            span_hidden = torch.cat([start_hidden, end_hidden, attended], dim=-1)
        else:
            span_repr = torch.stack([start_hidden, end_hidden], dim=2)
            if self.pooling_type == "mean":
                span_hidden = span_repr.mean(dim=2)
            elif self.pooling_type == "max":
                span_hidden = span_repr.max(dim=2)[0]
            else:
                span_hidden = self.projection(span_repr.mean(dim=2))
        
        return span_hidden
    
    def _create_span_masks(
        self,
        start_ids: torch.Tensor,
        end_ids: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Create attention masks for spans."""
        batch_size, num_spans = start_ids.shape
        
        positions = torch.arange(seq_len, device=start_ids.device)
        positions = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_spans, -1)
        
        start_ids_exp = start_ids.unsqueeze(-1).expand(-1, -1, seq_len)
        end_ids_exp = end_ids.unsqueeze(-1).expand(-1, -1, seq_len)
        
        masks = (positions >= start_ids_exp) & (positions <= end_ids_exp)
        
        return masks.float()


class SpanNERModel(nn.Module):
    """Complete span-based NER model.
    
    A span-based named entity recognition model that:
    1. Encodes input sequence
    2. Detects entity boundaries
    3. Extracts span representations
    4. Classifies entity types
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden states
        num_entity_types: Number of entity types
        max_span_length: Maximum length of entity spans
        encoder_type: Encoder type ('lstm' or 'transformer')
        num_layers: Number of encoder layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_entity_types: int = 9,
        max_span_length: int = 16,
        encoder_type: str = "lstm",
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_entity_types = num_entity_types
        self.max_span_length = max_span_length
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        
        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            encoder_output_dim = hidden_dim * 2
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            encoder_output_dim = embedding_dim
        
        self.boundary_detector = BoundaryDetector(
            hidden_dim=encoder_output_dim,
            dropout=dropout,
        )
        
        self.span_extractor = SpanExtractor(
            hidden_dim=encoder_output_dim,
            pooling_type="boundary",
        )
        
        self.entity_classifier = EntityTypeClassifier(
            hidden_dim=encoder_output_dim,
            num_types=num_entity_types,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.max_span_length = max_span_length
    
    def forward(
        self,
        input_ids: torch.Tensor,
        span_starts: Optional[torch.Tensor] = None,
        span_ends: Optional[torch.Tensor] = None,
        entity_types: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training or inference.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            span_starts: Start positions of candidate spans (batch_size, num_candidates)
            span_ends: End positions of candidate spans (batch_size, num_candidates)
            entity_types: Entity type labels for training
            mask: Attention mask
            
        Returns:
            Dictionary with boundary logits and type logits
        """
        if mask is None:
            mask = input_ids != 0
        
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        if isinstance(self.encoder, nn.LSTM):
            encoded, _ = self.encoder(embeddings)
        else:
            encoded = self.encoder(embeddings, src_key_padding_mask=~mask)
        
        encoded = self.dropout(encoded)
        
        start_logits, end_logits = self.boundary_detector(encoded)
        
        outputs: Dict[str, torch.Tensor] = {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        
        if span_starts is not None and span_ends is not None:
            span_repr = self.span_extractor(encoded, span_starts, span_ends)
            
            type_logits = self.entity_classifier(
                encoded, span_starts, span_ends
            )
            outputs["type_logits"] = type_logits
            
            if entity_types is not None and self.training:
                loss = F.cross_entropy(
                    type_logits.view(-1, self.num_entity_types),
                    entity_types.view(-1),
                )
                outputs["loss"] = loss
        
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        top_k: int = 100,
    ) -> List[List[Tuple[int, int, str, float]]]:
        """Predict entity spans.
        
        Args:
            input_ids: Input token IDs
            mask: Attention mask
            threshold: Confidence threshold
            top_k: Number of top candidates to consider
            
        Returns:
            List of predictions per batch, each as list of (start, end, type, score)
        """
        if mask is None:
            mask = input_ids != 0
        
        outputs = self.forward(input_ids, mask=mask)
        
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        
        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)
        
        batch_size, seq_len = start_probs.shape
        device = input_ids.device
        
        predictions: List[List[Tuple[int, int, str, float]]] = []
        
        for b in range(batch_size):
            valid_start = mask[b].nonzero(as_tuple=True)[0]
            valid_end = mask[b].nonzero(as_tuple=True)[0]
            
            candidates: List[Tuple[int, int, float]] = []
            
            for start_idx in valid_start[:top_k]:
                for end_idx in valid_end:
                    if end_idx >= start_idx and (end_idx - start_idx) <= self.max_span_length:
                        score = start_probs[b, start_idx] * end_probs[b, end_idx]
                        if score > threshold:
                            candidates.append((start_idx.item(), end_idx.item(), score.item()))
            
            candidates.sort(key=lambda x: x[2], reverse=True)
            candidates = candidates[:top_k]
            
            non_overlapping = []
            for start, end, score in candidates:
                overlap = False
                for s, e, _, _ in non_overlapping:
                    if not (end < s or start > e):
                        overlap = True
                        break
                if not overlap:
                    non_overlapping.append((start, end, "ENTITY", score))
            
            predictions.append(non_overlapping)
        
        return predictions


class MergeAndLabel(nn.Module):
    """Merge-and-label span-based NER.
    
    An efficient approach that first identifies candidate spans,
    then classifies them all at once using a merged representation.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden states
        num_entity_types: Number of entity types
        max_span_length: Maximum span length to consider
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_entity_types: int = 9,
        max_span_length: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_entity_types = num_entity_types
        self.max_span_length = max_span_length
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        
        self.span_projection = nn.Linear(hidden_dim * 2 * 3, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_entity_types),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        span_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            span_starts: (batch_size, num_spans)
            span_ends: (batch_size, num_spans)
            span_labels: (batch_size, num_spans)
            
        Returns:
            Loss or logits
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        encoded, _ = self.encoder(embeddings)
        encoded = self.dropout(encoded)
        
        batch_size, seq_len, hidden = encoded.shape
        
        span_reprs = self._get_span_representations(
            encoded, span_starts, span_ends
        )
        
        span_logits = self.classifier(span_reprs)
        
        if self.training and span_labels is not None:
            loss = F.cross_entropy(
                span_logits.view(-1, self.num_entity_types),
                span_labels.view(-1),
            )
            return loss
        
        return span_logits
    
    def _get_span_representations(
        self,
        encoded: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
    ) -> torch.Tensor:
        """Get representations for spans."""
        batch_size, seq_len, hidden = encoded.shape
        num_spans = span_starts.size(1)
        
        start_hidden = encoded.gather(
            1, span_starts.unsqueeze(-1).expand(-1, -1, hidden)
        )
        end_hidden = encoded.gather(
            1, span_ends.unsqueeze(-1).expand(-1, -1, hidden)
        )
        
        span_masks = self._create_span_masks(span_starts, span_ends, seq_len)
        span_masks_expanded = span_masks.unsqueeze(-1).float()
        
        span_content = encoded.unsqueeze(1).expand(-1, num_spans, -1, -1)
        
        covered_content = span_content * span_masks_expanded
        
        covered_sum = covered_content.sum(dim=2)
        span_lengths = span_masks.sum(dim=2, keepdim=True).clamp(min=1)
        avg_hidden = covered_sum / span_lengths
        
        span_repr = torch.cat([start_hidden, end_hidden, avg_hidden], dim=-1)
        span_repr = self.span_projection(span_repr)
        
        return span_repr
    
    def _create_span_masks(
        self,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Create masks for span content."""
        batch_size, num_spans = span_starts.shape
        
        positions = torch.arange(seq_len, device=span_starts.device)
        positions = positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_spans, -1)
        
        start_exp = span_starts.unsqueeze(-1).expand(-1, -1, seq_len)
        end_exp = span_ends.unsqueeze(-1).expand(-1, -1, seq_len)
        
        masks = (positions >= start_exp) & (positions <= end_exp)
        
        return masks


class TokenReduction(nn.Module):
    """Token reduction for long sequences.
    
    Reduces sequence length before NER to handle long documents.
    Uses span-based pooling or learned reduction.
    
    Args:
        hidden_dim: Dimension of hidden states
        reduction_factor: Factor to reduce sequence by
        reduction_type: Type of reduction ('pooling', 'learned', 'span')
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        reduction_factor: int = 2,
        reduction_type: str = "pooling",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.reduction_factor = reduction_factor
        self.reduction_type = reduction_type
        
        if reduction_type == "learned":
            self.projection = nn.Linear(hidden_dim * reduction_factor, hidden_dim)
        elif reduction_type == "span":
            self.span_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reduce sequence length.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len)
            
        Returns:
            Tuple of (reduced_states, new_mask)
        """
        batch_size, seq_len, hidden = hidden_states.shape
        
        if self.reduction_type == "pooling":
            new_len = seq_len // self.reduction_factor
            hidden_states = hidden_states[:, :new_len * self.reduction_factor, :]
            hidden_states = hidden_states.view(
                batch_size, new_len, self.reduction_factor, hidden
            )
            reduced = hidden_states.mean(dim=2)
            
            if mask is not None:
                mask = mask[:, :new_len]
            else:
                mask = torch.ones(
                    batch_size, new_len,
                    dtype=torch.bool, device=hidden_states.device
                )
        
        elif self.reduction_type == "learned":
            new_len = seq_len // self.reduction_factor
            hidden_states = hidden_states[:, :new_len * self.reduction_factor, :]
            hidden_states = hidden_states.view(
                batch_size, new_len, self.reduction_factor, hidden
            )
            hidden_states = hidden_states.view(batch_size, new_len, -1)
            reduced = self.projection(hidden_states)
            
            if mask is not None:
                mask = mask[:, :new_len]
            else:
                mask = torch.ones(
                    batch_size, new_len,
                    dtype=torch.bool, device=hidden_states.device
                )
        
        elif self.reduction_type == "span":
            new_len = (seq_len + 1) // (self.reduction_factor + 1)
            reduced_list = []
            mask_list = []
            
            for i in range(new_len):
                start = i * (self.reduction_factor + 1)
                end = min(start + self.reduction_factor + 1, seq_len)
                span = hidden_states[:, start:end, :]
                
                span_mean = span.mean(dim=1)
                span_max = span.max(dim=1)[0]
                span_repr = torch.cat([span_mean, span_max], dim=-1)
                reduced_list.append(self.span_projection(span_repr))
                
                if mask is not None:
                    span_mask = mask[:, start:end].any(dim=1)
                    mask_list.append(span_mask)
            
            reduced = torch.stack(reduced_list, dim=1)
            
            if mask is not None:
                mask = torch.stack(mask_list, dim=1)
            else:
                mask = torch.ones(
                    batch_size, new_len,
                    dtype=torch.bool, device=hidden_states.device
                )
        
        else:
            reduced = hidden_states
            mask = mask if mask is not None else torch.ones_like(hidden0],_states[:, :,  dtype=torch.bool)
        
        return reduced, mask
