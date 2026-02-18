"""
ROUGE Score

Implementation of Recall-Oriented Understudy for Gisting Evaluation (ROUGE)
for evaluating text summarization and generation.
"""

from typing import List, Dict, Set
from collections import Counter
import math


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-grams from tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams.append(ngram)
    return Counter(ngrams)


def rouge_1_score(candidate: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE-1 (unigram) score.

    Args:
        candidate: Candidate text
        reference: Reference text

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    if not candidate_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidate_unigrams = Counter(candidate_tokens)
    reference_unigrams = Counter(reference_tokens)

    overlap = sum((candidate_unigrams & reference_unigrams).values())

    precision = overlap / len(candidate_tokens) if candidate_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_2_score(candidate: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE-2 (bigram) score.

    Args:
        candidate: Candidate text
        reference: Reference text

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    if len(candidate_tokens) < 2 or len(reference_tokens) < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidate_bigrams = _get_ngrams(candidate_tokens, 2)
    reference_bigrams = _get_ngrams(reference_tokens, 2)

    overlap = sum((candidate_bigrams & reference_bigrams).values())

    precision = overlap / (len(candidate_tokens) - 1)
    recall = overlap / (len(reference_tokens) - 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l_score(candidate: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE-L (longest common subsequence) score.

    Args:
        candidate: Candidate text
        reference: Reference text

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    m = len(candidate_tokens)
    n = len(reference_tokens)

    lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate_tokens[i - 1] == reference_tokens[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])

    lcs_length = lcs_matrix[m][n]

    precision = lcs_length / m if m > 0 else 0.0
    recall = lcs_length / n if n > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_lsum_score(candidate: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE-Lsum (summary-level LCS) score.

    Splits into sentences and computes LCS for each.

    Args:
        candidate: Candidate text
        reference: Reference text

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    candidate_sentences = candidate.split(". ")
    reference_sentences = reference.split(". ")

    if not candidate_sentences or not reference_sentences:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    total_lcs = 0
    total_candidate = 0
    total_reference = 0

    for cand_sent in candidate_sentences:
        cand_tokens = cand_sent.split()
        total_candidate += len(cand_tokens)

        best_lcs = 0
        for ref_sent in reference_sentences:
            ref_tokens = ref_sent.split()
            total_reference += len(ref_tokens)

            lcs_len = _lcs_length(cand_tokens, ref_tokens)
            best_lcs = max(best_lcs, lcs_len)

        total_lcs += best_lcs

    precision = total_lcs / total_candidate if total_candidate > 0 else 0.0
    recall = total_lcs / total_reference if total_reference > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(tokens1: List[str], tokens2: List[str]) -> int:
    """Calculate longest common subsequence length."""
    m, n = len(tokens1), len(tokens2)

    if m == 0 or n == 0:
        return 0

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def rouge_score(
    candidate: str,
    reference: str,
    rouge_types: List[str] = ["1", "2", "l"],
) -> Dict[str, Dict[str, float]]:
    """Calculate multiple ROUGE scores.

    Args:
        candidate: Candidate text
        reference: Reference text
        rouge_types: List of ROUGE types ('1', '2', 'l', 'lsum')

    Returns:
        Dictionary mapping ROUGE types to their scores
    """
    results = {}

    if "1" in rouge_types:
        results["rouge_1"] = rouge_1_score(candidate, reference)

    if "2" in rouge_types:
        results["rouge_2"] = rouge_2_score(candidate, reference)

    if "l" in rouge_types:
        results["rouge_l"] = rouge_l_score(candidate, reference)

    if "lsum" in rouge_types:
        results["rouge_lsum"] = rouge_lsum_score(candidate, reference)

    return results


class ROUGEScorer:
    """ROUGE score calculator."""

    def __init__(self, rouge_types: List[str] = ["1", "2", "l"]):
        self.rouge_types = rouge_types

    def score(
        self,
        candidate: str,
        reference: str,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE scores."""
        return rouge_score(candidate, reference, self.rouge_types)

    def score_corpus(
        self,
        candidates: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate corpus-level ROUGE scores."""
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must match")

        agg_scores = {
            rt: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for rt in self.rouge_types
        }

        for candidate, reference in zip(candidates, references):
            scores = rouge_score(candidate, reference, self.rouge_types)

            for rt, score_dict in scores.items():
                for metric, value in score_dict.items():
                    agg_scores[rt][metric] += value

        n = len(candidates)
        for rt in agg_scores:
            for metric in agg_scores[rt]:
                agg_scores[rt][metric] /= n

        return agg_scores


def rouge_n_score(candidate: str, reference: str, n: int) -> Dict[str, float]:
    """Calculate ROUGE-N score.

    Args:
        candidate: Candidate text
        reference: Reference text
        n: N-gram size

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    if len(candidate_tokens) < n or len(reference_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidate_ngrams = _get_ngrams(candidate_tokens, n)
    reference_ngrams = _get_ngrams(reference_tokens, n)

    overlap = sum((candidate_ngrams & reference_ngrams).values())

    precision = overlap / (len(candidate_tokens) - n + 1)
    recall = overlap / (len(reference_tokens) - n + 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge_details(
    candidate: str,
    reference: str,
) -> Dict[str, Dict[str, float]]:
    """Compute detailed ROUGE scores for all standard metrics."""
    return rouge_score(candidate, reference, ["1", "2", "l", "lsum"])
