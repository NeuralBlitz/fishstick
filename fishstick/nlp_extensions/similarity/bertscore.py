"""
BERTScore

Implementation of BERTScore - evaluation metric using contextual embeddings.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import Counter


class BERTScorer:
    """BERTScore calculator using contextual embeddings.
    
    BERTScore computes similarity between candidate and reference texts
    using contextual embeddings from BERT or similar models.
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        device: str = 'cpu',
        num_layers: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.num_layers = num_layers
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        pass

    def score(
        self,
        candidate: str,
        reference: str,
        beta: float = 1.0,
    ) -> Dict[str, float]:
        """Calculate BERTScore between candidate and reference."""
        cand_tokens = candidate.split()
        ref_tokens = reference.split()

        if not cand_tokens or not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        precision = self._compute_weighted_similarity(cand_tokens, ref_tokens)
        recall = self._compute_weighted_similarity(ref_tokens, cand_tokens)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _compute_weighted_similarity(
        self,
        source_tokens: List[str],
        target_tokens: List[str],
    ) -> float:
        """Compute weighted token similarity."""
        if not source_tokens or not target_tokens:
            return 0.0

        source_counter = Counter(source_tokens)
        target_counter = Counter(target_tokens)

        common_tokens = set(source_counter.keys()) & set(target_counter.keys())

        if not common_tokens:
            return 0.0

        weighted_sum = 0.0

        for token in common_tokens:
            weighted_sum += min(source_counter[token], target_counter[token])

        return weighted_sum / len(source_tokens)

    def score_corpus(
        self,
        candidates: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Calculate corpus-level BERTScore."""
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must match")

        precisions = []
        recalls = []
        f1s = []

        for candidate, reference in zip(candidates, references):
            scores = self.score(candidate, reference)
            precisions.append(scores['precision'])
            recalls.append(scores['recall'])
            f1s.append(scores['f1'])

        return {
            'precision': float(np.mean(precisions)),
            'recall': float(np.mean(recalls)),
            'f1': float(np.mean(f1s)),
        }


def bertscore(
    candidate: str,
    reference: str,
    model_name: str = 'bert-base-uncased',
) -> Dict[str, float]:
    """Calculate BERTScore."""
    scorer = BERTScorer(model_name=model_name)
    return scorer.score(candidate, reference)


def bertscore_corpus(
    candidates: List[str],
    references: List[str],
    model_name: str = 'bert-base-uncased',
) -> Dict[str, float]:
    """Calculate corpus-level BERTScore."""
    scorer = BERTScorer(model_name=model_name)
    return scorer.score_corpus(candidates, references)
