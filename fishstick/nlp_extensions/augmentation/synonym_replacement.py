"""
Synonym Replacement Augmentation

Text augmentation through synonym replacement using wordnet or
other synonym dictionaries.
"""

from typing import List, Set, Optional, Dict, Callable
import random


class SynonymReplacement:
    """Synonym replacement augmentation.

    Replaces words with their synonyms to create augmented examples.

    Attributes:
        synonym_dict: Dictionary mapping words to their synonyms
        n_augmentations: Number of augmentations to generate
    """

    def __init__(
        self,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        n_augmentations: int = 1,
    ):
        self.synonym_dict = synonym_dict or self._default_synonyms()
        self.n_augmentations = n_augmentations

    def _default_synonyms(self) -> Dict[str, List[str]]:
        """Provide default synonyms for common words."""
        return {
            "good": ["great", "excellent", "fine", "nice"],
            "bad": ["poor", "terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous", "vast"],
            "small": ["tiny", "little", "miniature", "compact"],
            "fast": ["quick", "rapid", "swift", "speedy"],
            "slow": ["gradual", "unhurried", "leisurely"],
            "happy": ["glad", "joyful", "delighted", "pleased"],
            "sad": ["unhappy", "sorrowful", "melancholy"],
            "beautiful": ["lovely", "gorgeous", "stunning", "attractive"],
            "ugly": ["unattractive", "unsightly", "hideous"],
            "old": ["ancient", "aged", "antique"],
            "new": ["fresh", "modern", "recent"],
            "important": ["significant", "crucial", "vital", "essential"],
            "interesting": ["fascinating", "intriguing", "engaging"],
            "difficult": ["hard", "challenging", "tough", "complex"],
            "easy": ["simple", "effortless", "straightforward"],
            "smart": ["intelligent", "clever", "bright", "wise"],
            "stupid": ["foolish", "silly", "dumb", "idiotic"],
            "rich": ["wealthy", "affluent", "prosperous"],
            "poor": ["impoverished", "destitute", "needy"],
        }

    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word."""
        word_lower = word.lower()
        return self.synonym_dict.get(word_lower, [])

    def _should_replace(self, word: str) -> bool:
        """Determine if a word should be replaced."""
        return bool(self._get_synonyms(word))

    def augment(
        self,
        text: str,
        n_replacements: Optional[int] = None,
    ) -> List[str]:
        """Generate augmented examples through synonym replacement.

        Args:
            text: Input text to augment
            n_replacements: Number of words to replace

        Returns:
            List of augmented texts
        """
        words = text.split()

        if not words:
            return [text]

        augmented = []
        n_replacements = n_replacements or min(3, len(words) // 10 + 1)

        for _ in range(self.n_augmentations):
            indices_to_replace = random.sample(
                range(len(words)),
                min(n_replacements, len(words)),
            )

            new_words = words.copy()

            for idx in indices_to_replace:
                word = words[idx]
                synonyms = self._get_synonyms(word)

                if synonyms:
                    new_word = random.choice(synonyms)

                    if word[0].isupper():
                        new_word = new_word.capitalize()

                    new_words[idx] = new_word

            augmented.append(" ".join(new_words))

        return augmented

    def augment_batch(
        self,
        texts: List[str],
    ) -> List[str]:
        """Augment a batch of texts."""
        augmented = []

        for text in texts:
            augmented.extend(self.augment(text))

        return augmented


class EasyDataAugmentation(SynonymReplacement):
    """Easy Data Augmentation (EDA) techniques.

    Implements the EDA techniques: synonym replacement,
    random insertion, random swap, and random deletion.

    Attributes:
        alpha_sr: Percentage of words to replace with synonyms
        alpha_ri: Percentage of words to randomly insert
        alpha_rs: Percentage of words to randomly swap
        alpha_rd: Percentage of words to randomly delete
    """

    def __init__(
        self,
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        alpha_rd: float = 0.1,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(synonym_dict)
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.alpha_rd = alpha_rd

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
    ) -> List[str]:
        """Generate augmented examples using all EDA techniques."""
        augmented = []

        for _ in range(n_augmentations):
            technique = random.choice(["sr", "ri", "rs", "rd"])

            if technique == "sr":
                result = self._synonym_replacement(text)
            elif technique == "ri":
                result = self._random_insertion(text)
            elif technique == "rs":
                result = self._random_swap(text)
            else:
                result = self._random_deletion(text)

            augmented.append(result)

        return augmented

    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        n = max(1, int(len(words) * self.alpha_sr))

        indices = random.sample(range(len(words)), min(n, len(words)))

        for idx in indices:
            word = words[idx]
            synonyms = self._get_synonyms(word)
            if synonyms:
                new_word = random.choice(synonyms)
                if word[0].isupper():
                    new_word = new_word.capitalize()
                words[idx] = new_word

        return " ".join(words)

    def _random_insertion(self, text: str) -> str:
        """Randomly insert synonyms into the text."""
        words = text.split()
        n = max(1, int(len(words) * self.alpha_ri))

        for _ in range(n):
            synonyms_found = False
            for idx in range(len(words)):
                word = words[idx]
                synonyms = self._get_synonyms(word)
                if synonyms:
                    synonym = random.choice(synonyms)
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    words.insert(idx, synonym)
                    synonyms_found = True
                    break

            if not synonyms_found:
                break

        return " ".join(words)

    def _random_swap(self, text: str) -> str:
        """Randomly swap adjacent words."""
        words = text.split()
        n = max(1, int(len(words) * self.alpha_rs))

        for _ in range(n):
            if len(words) < 2:
                break

            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return " ".join(words)

    def _random_deletion(self, text: str) -> str:
        """Randomly delete words from the text."""
        words = text.split()

        if len(words) == 1:
            return text

        n = max(1, int(len(words) * self.alpha_rd))

        indices_to_delete = random.sample(
            range(len(words)),
            min(n, len(words) - 1),
        )

        new_words = [w for i, w in enumerate(words) if i not in indices_to_delete]

        return " ".join(new_words) if new_words else words[0]


class ContextualSynonymReplacement(SynonymReplacement):
    """Context-aware synonym replacement using language models.

    Uses a language model to find contextually appropriate synonyms.
    """

    def __init__(
        self,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        mask_model: Optional[Callable[[str, int], str]] = None,
    ):
        super().__init__(synonym_dict)
        self.mask_model = mask_model

    def _get_contextual_synonym(
        self,
        text: str,
        position: int,
    ) -> Optional[str]:
        """Get a contextually appropriate synonym."""
        words = text.split()

        if position >= len(words):
            return None

        word = words[position]
        synonyms = self._get_synonyms(word)

        if not synonyms:
            return None

        if self.mask_model:
            masked_text = text.replace(word, "[MASK]", 1)
            return self.mask_model(masked_text, position)

        return random.choice(synonyms)

    def augment(
        self,
        text: str,
        n_replacements: Optional[int] = None,
    ) -> List[str]:
        """Generate augmented examples with contextual synonyms."""
        words = text.split()

        if not words:
            return [text]

        augmented = []
        n_replacements = n_replacements or min(3, len(words) // 10 + 1)

        for _ in range(self.n_augmentations):
            indices = random.sample(
                range(len(words)),
                min(n_replacements, len(words)),
            )

            new_words = words.copy()

            for idx in indices:
                synonym = self._get_contextual_synonym(text, idx)
                if synonym:
                    if words[idx][0].isupper():
                        synonym = synonym.capitalize()
                    new_words[idx] = synonym

            augmented.append(" ".join(new_words))

        return augmented
