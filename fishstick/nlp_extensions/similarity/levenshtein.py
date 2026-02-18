"""
Levenshtein Distance

Implementation of Levenshtein (edit) distance for string similarity.
"""

from typing import List


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change
    one string into the other.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]

        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0 and 1
    """
    if not s1 and not s2:
        return 1.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    distance = levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute Levenshtein ratio (similar to fuzzywuzzy ratio).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio between 0 and 1
    """
    distance = levenshtein_distance(s1, s2)
    len_sum = len(s1) + len(s2)

    if len_sum == 0:
        return 1.0

    return 1.0 - (distance / len_sum)


def partial_ratio(s1: str, s2: str) -> float:
    """Compute partial ratio (best partial match).

    Useful for finding if one string is contained in another.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Partial ratio score
    """
    if len(s1) <= len(s2):
        shorter, longer = s1, s2
    else:
        shorter, longer = s2, s1

    m = len(shorter)
    ratio = 0.0

    for i in range(len(longer) - m + 1):
        substring = longer[i : i + m]
        distance = levenshtein_distance(shorter, substring)
        current_ratio = 1.0 - (distance / m)
        ratio = max(ratio, current_ratio)

    return ratio


def token_sort_ratio(s1: str, s2: str) -> float:
    """Compute token sort ratio.

    Sorts tokens alphabetically before computing similarity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Token sort ratio score
    """
    sorted_s1 = " ".join(sorted(s1.split()))
    sorted_s2 = " ".join(sorted(s2.split()))

    return levenshtein_ratio(sorted_s1, sorted_s2)


def token_set_ratio(s1: str, s2: str) -> float:
    """Compute token set ratio.

    Splits strings into tokens and computes similarity
    based on token overlap.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Token set ratio score
    """
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())

    intersection = tokens1 & tokens2
    diff1 = tokens1 - tokens2
    diff2 = tokens2 - tokens1

    sorted_sect = " ".join(sorted(intersection))
    sorted_s1 = " ".join(sorted(intersection | diff1))
    sorted_s2 = " ".join(sorted(intersection | diff2))

    ratios = [
        levenshtein_ratio(sorted_sect, sorted_s1),
        levenshtein_ratio(sorted_sect, sorted_s2),
        levenshtein_ratio(sorted_s1, sorted_s2),
    ]

    return max(ratios)


class LevenshteinScorer:
    """Levenshtein distance scorer for string comparison."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def score(self, s1: str, s2: str) -> float:
        """Compute similarity score between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity or distance score
        """
        if self.normalize:
            return normalized_levenshtein(s1, s2)
        return float(levenshtein_distance(s1, s2))

    def batch_score(self, pairs: List[tuple]) -> List[float]:
        """Compute scores for multiple string pairs.

        Args:
            pairs: List of (s1, s2) tuples

        Returns:
            List of scores
        """
        return [self.score(s1, s2) for s1, s2 in pairs]


class FuzzyMatcher:
    """Fuzzy string matcher using multiple Levenshtein-based metrics."""

    def __init__(self):
        pass

    def match(self, s1: str, s2: str) -> float:
        """Compute fuzzy match score using multiple metrics.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Best fuzzy match score
        """
        ratio = levenshtein_ratio(s1, s2)
        partial = partial_ratio(s1, s2)
        token_sort = token_sort_ratio(s1, s2)
        token_set = token_set_ratio(s1, s2)

        return max(ratio, partial, token_sort, token_set)

    def find_best_match(
        self,
        query: str,
        candidates: List[str],
    ) -> tuple:
        """Find best matching candidate for a query.

        Args:
            query: Query string
            candidates: List of candidate strings

        Returns:
            Tuple of (best_match, best_score)
        """
        best_match = None
        best_score = -1

        for candidate in candidates:
            score = self.match(query, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match, best_score
