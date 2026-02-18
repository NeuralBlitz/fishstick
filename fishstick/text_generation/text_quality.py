"""
Text Quality Evaluation
=======================

Metrics for evaluating text generation quality including:
- Perplexity
- BLEU score
- ROUGE score
- Diversity metrics
- Text quality scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class QualityScore:
    """Container for quality evaluation scores."""

    score: float
    metric_name: str
    details: Optional[dict] = None


class PerplexityMetric:
    """Computes perplexity for language model evaluation."""

    def __init__(
        self,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def compute(
        self,
        logits: Tensor,
        targets: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> float:
        """
        Compute perplexity.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)
            attention_mask: Optional mask for valid tokens

        Returns:
            Perplexity score
        """
        batch_size, seq_len, vocab_size = logits.size()

        log_probs = F.log_softmax(logits, dim=-1)

        targets_flat = targets.view(-1)
        log_probs_flat = log_probs.view(-1, vocab_size)

        loss = F.nll_loss(
            log_probs_flat,
            targets_flat,
            reduction="none",
        )

        if attention_mask is not None:
            mask_flat = attention_mask.view(-1).float()
            loss = (loss * mask_flat).sum() / mask_flat.sum()
        else:
            loss = loss.mean()

        perplexity = torch.exp(loss).item()
        return perplexity

    def compute_from_log_probs(
        self,
        log_probs: Tensor,
        targets: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> float:
        """Compute perplexity from log probabilities."""
        vocab_size = log_probs.size(-1)

        targets_flat = targets.view(-1)
        log_probs_flat = log_probs.view(-1, vocab_size)

        loss = F.nll_loss(
            log_probs_flat,
            targets_flat,
            reduction="none",
        )

        if attention_mask is not None:
            mask_flat = attention_mask.view(-1).float()
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-9)
        else:
            loss = loss.mean()

        perplexity = torch.exp(loss).item()
        return perplexity


class BleuScore:
    """Computes BLEU score for machine translation evaluation."""

    def __init__(
        self,
        max_n: int = 4,
        weights: Optional[list[float]] = None,
    ):
        self.max_n = max_n
        self.weights = weights or [1.0 / max_n] * max_n

    def compute(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict:
        """
        Compute BLEU score.

        Args:
            references: List of reference texts
            hypotheses: List of generated texts

        Returns:
            Dictionary with BLEU scores for each n and overall
        """
        scores = {}

        for n in range(1, self.max_n + 1):
            clipped_matches = 0
            total_ngrams = 0

            for ref, hyp in zip(references, hypotheses):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()

                ref_ngrams = self._get_ngrams(ref_tokens, n)
                hyp_ngrams = self._get_ngrams(hyp_tokens, n)

                clipped_matches += len(ref_ngrams & hyp_ngrams)
                total_ngrams += len(hyp_ngrams)

            if total_ngrams > 0:
                precision = clipped_matches / total_ngrams
            else:
                precision = 0.0

            scores[f"bleu_{n}"] = precision

        brevity_penalty = self._compute_brevity_penalty(references, hypotheses)
        scores["brevity_penalty"] = brevity_penalty

        log_precisions = [
            max(0, scores.get(f"bleu_{n}", 0)) for n in range(1, self.max_n + 1)
        ]
        if log_precisions:
            avg_log_precision = sum(
                w * max(0, torch.log(torch.tensor(p + 1e-9)).item())
                for w, p in zip(self.weights, log_precisions)
            )
            scores["bleu"] = (
                brevity_penalty * torch.exp(torch.tensor(avg_log_precision)).item()
            )
        else:
            scores["bleu"] = 0.0

        return scores

    def _get_ngrams(self, tokens: list[str], n: int) -> set:
        """Extract n-grams from token list."""
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.add(ngram)
        return ngrams

    def _compute_brevity_penalty(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> float:
        """Compute brevity penalty."""
        c = sum(len(h.split()) for h in hypotheses)
        r = sum(len(r.split()) for r in references)

        if c > r:
            return 1.0
        elif c == 0:
            return 0.0
        else:
            return torch.exp(torch.tensor(1 - r / (c + 1e-9))).item()


class RougeScore:
    """Computes ROUGE scores for summarization evaluation."""

    def __init__(
        self,
        rouge_types: Optional[list[str]] = None,
    ):
        self.rouge_types = rouge_types or ["rouge-1", "rouge-2", "rouge-l"]

    def compute(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict:
        """
        Compute ROUGE scores.

        Args:
            references: List of reference texts
            hypotheses: List of generated texts

        Returns:
            Dictionary with ROUGE scores
        """
        scores = {rt: [] for rt in self.rouge_types}

        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()

            if "rouge-1" in self.rouge_types:
                scores["rouge-1"].append(self._compute_f1(ref_tokens, hyp_tokens, n=1))

            if "rouge-2" in self.rouge_types:
                scores["rouge-2"].append(self._compute_f1(ref_tokens, hyp_tokens, n=2))

            if "rouge-l" in self.rouge_types:
                scores["rouge-l"].append(self._compute_lcs_f1(ref_tokens, hyp_tokens))

        return {
            rt: sum(vals) / len(vals) if vals else 0.0 for rt, vals in scores.items()
        }

    def _get_ngrams(self, tokens: list[str], n: int) -> list:
        """Get n-grams from tokens."""
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _compute_f1(self, ref: list[str], hyp: list[str], n: int) -> float:
        """Compute F1 score for n-gram overlap."""
        ref_ngrams = set(self._get_ngrams(ref, n))
        hyp_ngrams = set(self._get_ngrams(hyp, n))

        if not ref_ngrams or not hyp_ngrams:
            return 0.0

        overlap = len(ref_ngrams & hyp_ngrams)
        precision = overlap / len(hyp_ngrams)
        recall = overlap / len(ref_ngrams)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _compute_lcs_f1(self, ref: list[str], hyp: list[str]) -> float:
        """Compute F1 score based on longest common subsequence."""
        m, n = len(ref), len(hyp)

        if m == 0 or n == 0:
            return 0.0

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]

        precision = lcs_length / n if n > 0 else 0
        recall = lcs_length / m if m > 0 else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1


class DiversityMetric:
    """Computes diversity metrics for generated text."""

    def __init__(
        self,
        ngram_sizes: Optional[list[int]] = None,
    ):
        self.ngram_sizes = ngram_sizes or [1, 2, 3]

    def compute(
        self,
        sequences: list[str],
    ) -> dict:
        """
        Compute diversity metrics.

        Args:
            sequences: List of generated text sequences

        Returns:
            Dictionary with diversity scores
        """
        if not sequences:
            return {"diversity": 0.0}

        results = {}

        for n in self.ngram_sizes:
            unique_ngrams = set()
            total_ngrams = 0

            for seq in sequences:
                tokens = seq.split()
                ngrams = self._get_ngrams(tokens, n)
                unique_ngrams.update(ngrams)
                total_ngrams += len(ngrams)

            if total_ngrams > 0:
                diversity = len(unique_ngrams) / total_ngrams
            else:
                diversity = 0.0

            results[f"diversity_{n}gram"] = diversity

        results["diversity"] = sum(
            results[f"diversity_{n}gram"] for n in self.ngram_sizes
        ) / len(self.ngram_sizes)

        return results

    def _get_ngrams(self, tokens: list[str], n: int) -> set:
        """Get set of n-grams."""
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.add(ngram)
        return ngrams


class TextQualityEvaluator:
    """Comprehensive text quality evaluation."""

    def __init__(
        self,
        compute_perplexity: bool = True,
        compute_bleu: bool = True,
        compute_rouge: bool = True,
        compute_diversity: bool = True,
    ):
        self.compute_perplexity = compute_perplexity
        self.compute_bleu = compute_bleu
        self.compute_rouge = compute_rouge
        self.compute_diversity = compute_diversity

        self.perplexity_metric = PerplexityMetric() if compute_perplexity else None
        self.bleu_metric = BleuScore() if compute_bleu else None
        self.rouge_metric = RougeScore() if compute_rouge else None
        self.diversity_metric = DiversityMetric() if compute_diversity else None

    def evaluate(
        self,
        logits: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        hypotheses: Optional[list[str]] = None,
        references: Optional[list[str]] = None,
    ) -> dict:
        """
        Evaluate text quality with multiple metrics.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)
            hypotheses: Generated text sequences
            references: Reference text sequences

        Returns:
            Dictionary with all computed metrics
        """
        results = {}

        if self.perplexity_metric and logits is not None and targets is not None:
            results["perplexity"] = self.perplexity_metric.compute(logits, targets)

        if self.bleu_metric and hypotheses is not None and references is not None:
            results["bleu"] = self.bleu_metric.compute(references, hypotheses)

        if self.rouge_metric and hypotheses is not None and references is not None:
            results["rouge"] = self.rouge_metric.compute(references, hypotheses)

        if self.diversity_metric and hypotheses is not None:
            results["diversity"] = self.diversity_metric.compute(hypotheses)

        return results

    def evaluate_batch(
        self,
        batch_results: list[dict],
    ) -> dict:
        """Evaluate a batch of results and aggregate."""
        if not batch_results:
            return {}

        aggregated = {}
        keys = batch_results[0].keys()

        for key in keys:
            values = [r.get(key, 0) for r in batch_results]
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_std"] = self._std(values)

        return aggregated

    def _std(self, values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance**0.5
