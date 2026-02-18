"""
BLEU Score

Implementation of Bilingual Evaluation Understudy (BLEU) score
for evaluating machine translation and text generation.
"""

from typing import List, Dict, Tuple
import math
from collections import Counter


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-grams from tokens.

    Args:
        tokens: List of tokens
        n: N-gram size

    Returns:
        Counter of n-grams
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams.append(ngram)
    return Counter(ngrams)


def _clip_counts(candidate_counts: Counter, reference_counts: Counter) -> Counter:
    """Clip counts based on reference counts.

    Args:
        candidate_counts: Candidate n-gram counts
        reference_counts: Reference n-gram counts

    Returns:
        Clipped counts
    """
    clipped = Counter()
    for ngram, count in candidate_counts.items():
        clipped[ngram] = min(count, reference_counts.get(ngram, 0))
    return clipped


def _brevity_penalty(candidate_len: int, reference_len: int) -> float:
    """Calculate brevity penalty.

    Args:
        candidate_len: Length of candidate
        reference_len: Length of reference

    Returns:
        Brevity penalty
    """
    if candidate_len > reference_len:
        return 1.0

    if candidate_len == 0:
        return 0.0

    return math.exp(1 - reference_len / candidate_len)


def bleu_score(
    candidate: str,
    reference: str,
    max_n: int = 4,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """Calculate BLEU score.

    BLEU (Bilingual Evaluation Understudy) measures the similarity
    between candidate and reference texts based on n-gram precision.

    Args:
        candidate: Candidate text
        reference: Reference text
        max_n: Maximum n-gram size
        weights: Weights for each n-gram precision

    Returns:
        BLEU score between 0 and 1
    """
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    if not candidate_tokens or not reference_tokens:
        return 0.0

    candidate_len = len(candidate_tokens)
    reference_len = len(reference_tokens)

    bp = _brevity_penalty(candidate_len, reference_len)

    precisions = []

    for i in range(1, max_n + 1):
        if i > candidate_len:
            precisions.append(0.0)
            continue

        candidate_ngrams = _get_ngrams(candidate_tokens, i)
        reference_ngrams = _get_ngrams(reference_tokens, i)

        clipped = _clip_counts(candidate_ngrams, reference_ngrams)

        total_clipped = sum(clipped.values())
        total_candidate = sum(candidate_ngrams.values())

        if total_candidate == 0:
            precisions.append(0.0)
        else:
            precisions.append(total_clipped / total_candidate)

    log_precisions = []
    for i, precision in enumerate(precisions):
        if precision == 0:
            log_precisions.append(0.0)
        else:
            log_precisions.append(precision * weights[i])

    if all(p == 0 for p in precisions):
        return 0.0

    log_sum = sum(log_precisions)

    score = bp * math.exp(log_sum)

    return score


def sentence_bleu(
    candidate: str,
    reference: str,
    max_n: int = 4,
) -> float:
    """Calculate sentence-level BLEU score.

    Args:
        candidate: Candidate text
        reference: Reference text
        max_n: Maximum n-gram size

    Returns:
        Sentence BLEU score
    """
    return bleu_score(candidate, reference, max_n)


def corpus_bleu(
    candidates: List[str],
    references: List[str],
    max_n: int = 4,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """Calculate corpus-level BLEU score.

    Args:
        candidates: List of candidate texts
        references: List of reference texts
        max_n: Maximum n-gram size
        weights: Weights for each n-gram precision

    Returns:
        Corpus BLEU score
    """
    if len(candidates) != len(references):
        raise ValueError("Number of candidates and references must match")

    if not candidates:
        return 0.0

    candidate_lengths = []
    reference_lengths = []

    for candidate, reference in zip(candidates, references):
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()

        candidate_lengths.append(len(candidate_tokens))
        reference_lengths.append(len(reference_tokens))

    total_candidate_len = sum(candidate_lengths)
    total_reference_len = sum(reference_lengths)

    avg_candidate_len = total_candidate_len / len(candidates)
    avg_reference_len = total_reference_len / len(references)

    bp = _brevity_penalty(
        int(avg_candidate_len),
        int(avg_reference_len),
    )

    precisions = []

    for i in range(1, max_n + 1):
        total_clipped = 0
        total_candidate = 0

        for candidate, reference in zip(candidates, references):
            candidate_tokens = candidate.split()
            reference_tokens = reference.split()

            if i > len(candidate_tokens):
                continue

            candidate_ngrams = _get_ngrams(candidate_tokens, i)
            reference_ngrams = _get_ngrams(reference_tokens, i)

            clipped = _clip_counts(candidate_ngrams, reference_ngrams)

            total_clipped += sum(clipped.values())
            total_candidate += sum(candidate_ngrams.values())

        if total_candidate == 0:
            precisions.append(0.0)
        else:
            precisions.append(total_clipped / total_candidate)

    log_precisions = []
    for i, precision in enumerate(precisions):
        if precision == 0:
            log_precisions.append(0.0)
        else:
            log_precisions.append(precision * weights[i])

    if all(p == 0 for p in precisions):
        return 0.0

    log_sum = sum(log_precisions)

    score = bp * math.exp(log_sum)

    return score


class BLEUScorer:
    """BLEU score calculator with caching and batch processing."""

    def __init__(
        self,
        max_n: int = 4,
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    ):
        self.max_n = max_n
        self.weights = weights

    def score(
        self,
        candidate: str,
        reference: str,
    ) -> float:
        """Calculate BLEU score.

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            BLEU score
        """
        return bleu_score(candidate, reference, self.max_n, self.weights)

    def score_sentence(
        self,
        candidate: str,
        reference: str,
    ) -> float:
        """Calculate sentence-level BLEU score."""
        return sentence_bleu(candidate, reference, self.max_n)

    def score_corpus(
        self,
        candidates: List[str],
        references: List[str],
    ) -> float:
        """Calculate corpus-level BLEU score."""
        return corpus_bleu(candidates, references, self.max_n, self.weights)


def compute_bleu_details(
    candidate: str,
    reference: str,
    max_n: int = 4,
) -> Dict[str, float]:
    """Compute detailed BLEU scores for each n-gram level.

    Args:
        candidate: Candidate text
        reference: Reference text
        max_n: Maximum n-gram size

    Returns:
        Dictionary with detailed scores
    """
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    details = {}

    candidate_len = len(candidate_tokens)
    reference_len = len(reference_tokens)

    details["brevity_penalty"] = _brevity_penalty(candidate_len, reference_len)
    details["candidate_length"] = candidate_len
    details["reference_length"] = reference_len

    for i in range(1, max_n + 1):
        if i > candidate_len:
            details[f"precision_{i}"] = 0.0
            continue

        candidate_ngrams = _get_ngrams(candidate_tokens, i)
        reference_ngrams = _get_ngrams(reference_tokens, i)

        clipped = _clip_counts(candidate_ngrams, reference_ngrams)

        total_clipped = sum(clipped.values())
        total_candidate = sum(candidate_ngrams.values())

        if total_candidate == 0:
            details[f"precision_{i}"] = 0.0
        else:
            details[f"precision_{i}"] = total_clipped / total_candidate

    details["bleu"] = bleu_score(candidate, reference, max_n)

    return details
