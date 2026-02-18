"""
Sentiment analysis utilities.

Provides common utilities for sentiment analysis including preprocessing,
lexicons, evaluation metrics, and helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import re


class SentimentLexicon:
    """Dictionary-based sentiment lexicon for rule-based sentiment analysis."""

    def __init__(self, lexicon_dict: Optional[Dict[str, float]] = None):
        self.lexicon = lexicon_dict or {}
        self.default_score = 0.0

    def add_word(self, word: str, score: float) -> None:
        self.lexicon[word.lower()] = score

    def get_score(self, word: str) -> float:
        return self.lexicon.get(word.lower(), self.default_score)

    def score_text(self, tokens: List[str]) -> float:
        scores = [self.get_score(token) for token in tokens]
        return sum(scores) / len(scores) if scores else 0.0

    def score_text_weighted(self, tokens: List[str]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for i, token in enumerate(tokens):
            weight = 1.0
            weighted_sum += self.get_score(token) * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class SentimentPreprocessor:
    """Text preprocessing utilities for sentiment analysis."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        handle_negation: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.handle_negation = handle_negation

        self.stopwords: Set[str] = set()
        self.negation_words = {
            "not",
            "no",
            "never",
            "n't",
            "cannot",
            "without",
            "none",
            "nothing",
            "neither",
            "nobody",
            "nowhere",
        }

    def preprocess(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        tokens = text.split()

        if self.handle_negation:
            tokens = self._handle_negation(tokens)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        return tokens

    def _handle_negation(self, tokens: List[str]) -> List[str]:
        result = []
        negation_active = False

        for token in tokens:
            if token.lower() in self.negation_words:
                negation_active = True
                result.append(token)
            elif negation_active and token.isalpha():
                result.append(f"NOT_{token}")
                negation_active = False
            else:
                result.append(token)

        return result


class SentimentMetrics:
    """Evaluation metrics for sentiment analysis."""

    @staticmethod
    def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0

    @staticmethod
    def compute_precision_recall_f1(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
    ) -> Dict[str, float]:
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for c in range(num_classes):
            tp = ((predictions == c) & (targets == c)).sum().item()
            fp = ((predictions == c) & (targets != c)).sum().item()
            fn = ((predictions != c) & (targets == c)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        return {
            "precision": sum(precision_per_class) / len(precision_per_class),
            "recall": sum(recall_per_class) / len(recall_per_class),
            "f1": sum(f1_per_class) / len(f1_per_class),
        }

    @staticmethod
    def compute_confusion_matrix(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for pred, target in zip(predictions, targets):
            confusion[target, pred] += 1
        return confusion


class AspectTermExtractor:
    """Rule-based aspect term extraction using dependency parsing patterns."""

    def __init__(self):
        self.aspect_patterns = [
            r"\b(NN|NNS|NNP|NNPS)\b",
            r"\b(JJ)\s+(NN|NNS|NNP|NNPS)\b",
        ]

    def extract_aspects(
        self, tokens: List[str], pos_tags: List[str]
    ) -> List[Tuple[int, int]]:
        aspects = []
        current_aspect = None

        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            if pos in ("NN", "NNS", "NNP", "NNPS"):
                if current_aspect is None:
                    current_aspect = (i, i + 1)
                else:
                    current_aspect = (current_aspect[0], i + 1)
            else:
                if current_aspect is not None:
                    aspects.append(current_aspect)
                    current_aspect = None

        if current_aspect is not None:
            aspects.append(current_aspect)

        return aspects


class NegationHandler:
    """Handles negation in sentiment analysis."""

    def __init__(self):
        self.negation_words = {
            "not",
            "no",
            "never",
            "n't",
            "cannot",
            "without",
            "none",
            "nothing",
            "neither",
            "nobody",
            "nowhere",
            "hardly",
            "barely",
            "scarcely",
        }
        self.negation_scope = 3

    def apply_negation(self, tokens: List[str]) -> List[str]:
        result = []
        negation_scope_count = 0

        for token in tokens:
            if token.lower() in self.negation_words:
                negation_scope_count = self.negation_scope
                result.append(token)
            elif negation_scope_count > 0:
                result.append(f"NOT_{token}")
                negation_scope_count -= 1
            else:
                result.append(token)

        return result


class SentiWordNetLexicon:
    """SentiWordNet-based lexicon for sentiment analysis."""

    def __init__(self):
        self.pos_scores: Dict[str, float] = {}
        self.neg_scores: Dict[str, float] = {}

    def add_synset(self, synset_id: str, pos_score: float, neg_score: float) -> None:
        self.pos_scores[synset_id] = pos_score
        self.neg_scores[synset_id] = neg_score

    def get_sentiment(self, word: str) -> Tuple[float, float]:
        pos = sum(v for k, v in self.pos_scores.items() if word in k) / max(
            len(self.pos_scores), 1
        )
        neg = sum(v for k, v in self.neg_scores.items() if word in k) / max(
            len(self.neg_scores), 1
        )
        return pos, neg


class VADELexicon:
    """Valence-Arousal-Dominance lexicon."""

    def __init__(self):
        self.scores: Dict[str, Dict[str, float]] = {}

    def add_word(
        self, word: str, valence: float, arousal: float, dominance: float
    ) -> None:
        self.scores[word.lower()] = {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
        }

    def get_vad(self, word: str) -> Dict[str, float]:
        return self.scores.get(
            word.lower(), {"valence": 0.5, "arousal": 0.5, "dominance": 0.5}
        )


class EmoLexicon:
    """EmoLex (NRC Emotion Lexicon) for emotion detection."""

    def __init__(self):
        self.emotion_scores: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.emotions = [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "trust",
        ]

    def add_word(self, word: str, emotion: str, score: float) -> None:
        self.emotion_scores[word.lower()][emotion] = score

    def get_emotions(self, word: str) -> Dict[str, float]:
        return dict(self.emotion_scores.get(word.lower(), {}))

    def get_dominant_emotion(self, word: str) -> Optional[str]:
        emotions = self.get_emotions(word)
        if emotions:
            return max(emotions, key=emotions.get)
        return None


class SentimentAugmenter:
    """Data augmentation for sentiment analysis."""

    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        self.synonym_dict = synonym_dict or {}

    def synonym_replacement(self, tokens: List[str], n: int = 1) -> List[str]:
        result = tokens.copy()
        replaceable_indices = [
            i for i, t in enumerate(tokens) if t in self.synonym_dict
        ]

        if not replaceable_indices:
            return result

        import random

        indices_to_replace = random.sample(
            replaceable_indices, min(n, len(replaceable_indices))
        )

        for idx in indices_to_replace:
            token = tokens[idx]
            synonyms = self.synonym_dict.get(token, [])
            if synonyms:
                result[idx] = random.choice(synonyms)

        return result

    def random_deletion(self, tokens: List[str], p: float = 0.1) -> List[str]:
        import random

        if len(tokens) <= 1:
            return tokens
        return [t for t in tokens if random.random() > p]

    def random_swap(self, tokens: List[str], n: int = 1) -> List[str]:
        import random

        result = tokens.copy()
        length = len(result)

        for _ in range(n):
            if length < 2:
                break
            idx1, idx2 = random.sample(range(length), 2)
            result[idx1], result[idx2] = result[idx2], result[idx1]

        return result


class SentimentVocabulary:
    """Vocabulary manager for sentiment models."""

    def __init__(self, max_size: int = 50000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts: Counter = Counter()

    def add_word(self, word: str) -> None:
        self.word_counts[word] += 1

    def build_vocab(self) -> None:
        valid_words = [
            word
            for word, count in self.word_counts.most_common(self.max_size)
            if count >= self.min_freq
        ]

        for word in valid_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(i, "<UNK>") for i in indices]

    def __len__(self) -> int:
        return len(self.word2idx)


def build_default_lexicon() -> SentimentLexicon:
    """Build a default sentiment lexicon with common words."""
    lexicon = SentimentLexicon()

    positive_words = {
        "good": 1.0,
        "great": 1.0,
        "excellent": 1.0,
        "amazing": 1.0,
        "wonderful": 1.0,
        "fantastic": 1.0,
        "awesome": 1.0,
        "best": 1.0,
        "love": 1.0,
        "happy": 1.0,
        "perfect": 1.0,
        "beautiful": 1.0,
        "brilliant": 1.0,
        "outstanding": 1.0,
        "superb": 1.0,
    }

    negative_words = {
        "bad": -1.0,
        "terrible": -1.0,
        "horrible": -1.0,
        "worst": -1.0,
        "hate": -1.0,
        "awful": -1.0,
        "poor": -1.0,
        "disappointing": -1.0,
        "boring": -1.0,
        "useless": -1.0,
        "pathetic": -1.0,
        "dreadful": -1.0,
        "appalling": -1.0,
        "mediocre": -1.0,
        "disgusting": -1.0,
    }

    for word, score in positive_words.items():
        lexicon.add_word(word, score)

    for word, score in negative_words.items():
        lexicon.add_word(word, score)

    return lexicon
