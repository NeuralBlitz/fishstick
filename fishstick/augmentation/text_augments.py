"""Text augmentation techniques."""

import torch
import numpy as np
from typing import List, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import random


class TextAugment(ABC):
    """Base class for text augmentation."""

    @abstractmethod
    def __call__(self, text: str) -> str:
        pass


class SynonymReplacement(TextAugment):
    """Replace words with synonyms."""

    def __init__(self, n: int = 1, synonyms_dict: Optional[dict] = None):
        self.n = n
        self.synonyms_dict = synonyms_dict or {
            "good": ["great", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "minor"],
            "fast": ["quick", "rapid", "swift"],
            "slow": ["gradual", "unhurried"],
            "happy": ["joyful", "pleased", "glad"],
            "sad": ["unhappy", "sorrowful", "down"],
        }

    def __call__(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        n_replace = min(self.n, len(words))
        indices = random.sample(range(len(words)), n_replace)

        for idx in indices:
            word = words[idx].lower()
            if word in self.synonyms_dict:
                synonyms = self.synonyms_dict[word]
                words[idx] = random.choice(synonyms)

        return " ".join(words)


class RandomInsertion(TextAugment):
    """Randomly insert words into text."""

    def __init__(self, n: int = 1, stopwords: Optional[List[str]] = None):
        self.n = n
        self.stopwords = set(
            stopwords
            or [
                "a",
                "an",
                "the",
                "is",
                "are",
                "was",
                "were",
                "to",
                "of",
                "in",
                "for",
                "on",
                "with",
            ]
        )

    def __call__(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(self.n):
            idx = random.randint(0, len(words))
            filler = random.choice(["really", "very", "quite", "somewhat"])
            words.insert(idx, filler)

        return " ".join(words)


class RandomDeletion(TextAugment):
    """Randomly delete words from text."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        kept_words = [w for w in words if random.random() > self.p]
        return " ".join(kept_words) if kept_words else random.choice(words)


class RandomSwap(TextAugment):
    """Randomly swap adjacent words."""

    def __init__(self, n: int = 1):
        self.n = n

    def __call__(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(self.n):
            if len(words) < 2:
                break
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return " ".join(words)


class BackTranslation(TextAugment):
    """Placeholder for back-translation augmentation."""

    def __init__(self, intermediate_lang: str = "fr", mock: bool = True):
        self.intermediate_lang = intermediate_lang
        self.mock = mock

    def __call__(self, text: str) -> str:
        if self.mock:
            return text
        raise NotImplementedError("Back translation requires translation API")


class CharacterLevelAugment(TextAugment):
    """Character-level augmentations."""

    def __init__(self, p: float = 0.1, keyboard_distance: bool = True):
        self.p = p
        self.keyboard_distance = keyboard_distance

        self.keyboard_neighbors = {
            "a": ["q", "w", "s", "z"],
            "b": ["v", "g", "h", "n"],
            "c": ["x", "d", "f", "v"],
            "d": ["s", "e", "r", "f", "c", "x"],
            "e": ["w", "s", "d", "r"],
            "f": ["d", "r", "t", "g", "v", "c"],
            "g": ["f", "t", "y", "h", "b", "v"],
            "h": ["g", "y", "u", "j", "n", "b"],
            "i": ["u", "j", "k", "o"],
            "j": ["h", "u", "i", "k", "m", "n"],
            "k": ["j", "i", "o", "l", "m"],
            "l": ["k", "o", "p"],
            "m": ["n", "j", "k"],
            "n": ["b", "h", "j", "m"],
            "o": ["i", "k", "l", "p"],
            "p": ["o", "l"],
            "q": ["w", "a"],
            "r": ["e", "d", "f", "t"],
            "s": ["a", "w", "e", "d", "x", "z"],
            "t": ["r", "f", "g", "y"],
            "u": ["y", "h", "j", "i"],
            "v": ["c", "f", "g", "b"],
            "w": ["q", "a", "s", "e"],
            "x": ["z", "s", "d", "c"],
            "y": ["t", "g", "h", "u"],
            "z": ["a", "s", "x"],
        }

    def __call__(self, text: str) -> str:
        chars = list(text)
        for i in range(len(chars)):
            if random.random() > self.p:
                continue

            char = chars[i].lower()
            if self.keyboard_distance and char in self.keyboard_neighbors:
                chars[i] = random.choice(self.keyboard_neighbors[char])
            elif random.random() < 0.5 and char.isalpha():
                chars[i] = ""

        return "".join(chars)


class WordDropout(TextAugment):
    """Randomly dropout words during training."""

    def __init__(self, p: float = 0.15):
        self.p = p

    def __call__(self, text: str) -> str:
        words = text.split()
        kept = [w for w in words if random.random() > self.p]
        return " ".join(kept) if kept else text


class EDA(TextAugment):
    """Easy Data Augmentation combining multiple techniques."""

    def __init__(
        self,
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        p_rd: float = 0.1,
    ):
        self.sr = SynonymReplacement(n=1)
        self.ri = RandomInsertion(n=1)
        self.rs = RandomSwap(n=1)
        self.rd = RandomDeletion(p=p_rd)
        self.alpha = alpha_sr

    def __call__(self, text: str) -> str:
        words = text.split()
        n_words = max(1, int(len(words) * self.alpha))

        aug_type = random.choice(["sr", "ri", "rs", "rd"])

        if aug_type == "sr":
            self.sr.n = n_words
            return self.sr(text)
        elif aug_type == "ri":
            self.ri.n = n_words
            return self.ri(text)
        elif aug_type == "rs":
            self.rs.n = n_words
            return self.rs(text)
        else:
            return self.rd(text)


def create_text_augment(augment_type: str, **kwargs) -> TextAugment:
    """Factory function to create text augmentations."""
    augments = {
        "synonym": SynonymReplacement,
        "insertion": RandomInsertion,
        "deletion": RandomDeletion,
        "swap": RandomSwap,
        "back_translation": BackTranslation,
        "char_level": CharacterLevelAugment,
        "dropout": WordDropout,
        "eda": EDA,
    }

    if augment_type not in augments:
        raise ValueError(f"Unknown augment: {augment_type}")

    return augments[augment_type](**kwargs)
