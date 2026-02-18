"""
Cosine Similarity

Implementation of cosine similarity for text and vector comparison.
"""

import numpy as np
from typing import List, Union, Optional
from numpy.typing import NDArray


def cosine_similarity(
    a: Union[List[float], NDArray],
    b: Union[List[float], NDArray],
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    a = np.array(a)
    b = np.array(b)

    dot_product = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def cosine_distance(
    a: Union[List[float], NDArray],
    b: Union[List[float], NDArray],
) -> float:
    """Compute cosine distance between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine distance (1 - cosine_similarity)
    """
    return 1.0 - cosine_similarity(a, b)


def batch_cosine_similarity(
    vectors: Union[List[List[float]], NDArray],
) -> NDArray:
    """Compute pairwise cosine similarity for a batch of vectors.

    Args:
        vectors: Batch of vectors

    Returns:
        Pairwise cosine similarity matrix
    """
    vectors = np.array(vectors)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)

    normalized = vectors / norms

    return np.dot(normalized, normalized.T)


class CosineSimilarityScorer:
    """Cosine similarity scorer for text comparison.

    Can be used with embeddings to compute text similarity.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def score(
        self,
        embedding_a: Union[List[float], NDArray],
        embedding_b: Union[List[float], NDArray],
    ) -> float:
        """Compute similarity score between embeddings.

        Args:
            embedding_a: First embedding
            embedding_b: Second embedding

        Returns:
            Similarity score
        """
        if self.normalize:
            return cosine_similarity(embedding_a, embedding_b)
        else:
            a = np.array(embedding_a)
            b = np.array(embedding_b)
            return float(np.dot(a, b))

    def batch_score(
        self,
        embeddings: Union[List[List[float]], NDArray],
    ) -> NDArray:
        """Compute pairwise similarity for multiple embeddings.

        Args:
            embeddings: List of embeddings

        Returns:
            Similarity matrix
        """
        return batch_cosine_similarity(embeddings)


def sentence_cosine_similarity(
    sentences: List[str],
    embeddings_fn: Optional[callable] = None,
) -> NDArray:
    """Compute cosine similarity between sentences using embeddings.

    Args:
        sentences: List of sentences
        embeddings_fn: Function to compute embeddings

    Returns:
        Similarity matrix
    """
    if embeddings_fn is None:
        return np.zeros((len(sentences), len(sentences)))

    embeddings = embeddings_fn(sentences)

    return batch_cosine_similarity(embeddings)


class SemanticSimilarity:
    """Semantic similarity using cosine similarity on embeddings."""

    def __init__(self, model=None):
        self.model = model

    def compute_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if self.model is None:
            return 0.0

        emb_a = self.model.encode(text_a)
        emb_b = self.model.encode(text_b)

        return (cosine_similarity(emb_a, emb_b) + 1) / 2

    def compute_similarities(
        self,
        text: str,
        candidates: List[str],
    ) -> List[float]:
        """Compute similarity between one text and multiple candidates.

        Args:
            text: Reference text
            candidates: List of candidate texts

        Returns:
            List of similarity scores
        """
        if self.model is None:
            return [0.0] * len(candidates)

        emb_text = self.model.encode(text)
        emb_candidates = self.model.encode(candidates)

        similarities = []
        for emb in emb_candidates:
            sim = (cosine_similarity(emb_text, emb) + 1) / 2
            similarities.append(sim)

        return similarities
