"""
Random Edits Augmentation

Text augmentation through random insertions, deletions, and swaps.
Implements the EDA (Easy Data Augmentation) techniques.
"""

from typing import List, Optional
import random


class RandomInsertion:
    """Random insertion augmentation.

    Randomly inserts words into the text.

    Attributes:
        n_insertions: Number of words to insert
    """

    def __init__(self, n_insertions: int = 1):
        self.n_insertions = n_insertions

    def augment(self, text: str) -> str:
        """Generate augmented text through random insertion.

        Args:
            text: Input text to augment

        Returns:
            Augmented text
        """
        words = text.split()

        if not words:
            return text

        new_words = words.copy()

        for _ in range(self.n_insertions):
            if not words:
                break

            idx = random.randint(0, len(words) - 1)
            new_words.insert(idx, random.choice(words))

        return " ".join(new_words)


class RandomDeletion:
    """Random deletion augmentation.

    Randomly deletes words from the text.

    Attributes:
        p: Probability of deleting each word
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def augment(self, text: str) -> str:
        """Generate augmented text through random deletion.

        Args:
            text: Input text to augment

        Returns:
            Augmented text
        """
        words = text.split()

        if not words:
            return text

        new_words = [w for w in words if random.random() > self.p]

        if not new_words:
            return random.choice(words)

        return " ".join(new_words)


class RandomSwap:
    """Random swap augmentation.

    Randomly swaps adjacent words in the text.

    Attributes:
        n_swaps: Number of swaps to perform
    """

    def __init__(self, n_swaps: int = 1):
        self.n_swaps = n_swaps

    def augment(self, text: str) -> str:
        """Generate augmented text through random swapping.

        Args:
            text: Input text to augment

        Returns:
            Augmented text
        """
        words = text.split()

        if len(words) < 2:
            return text

        new_words = words.copy()

        for _ in range(self.n_swaps):
            idx = random.randint(0, len(new_words) - 2)
            new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]

        return " ".join(new_words)


class TextAugmenter:
    """Complete text augmentation combining multiple techniques.

    Provides a unified interface for various text augmentation
    techniques including insertion, deletion, swap, and more.

    Attributes:
        techniques: List of augmentation techniques to apply
        n_augmentations: Number of augmented examples to generate
    """

    def __init__(
        self,
        n_augmentations: int = 1,
        use_insertion: bool = True,
        use_deletion: bool = True,
        use_swap: bool = True,
        p_insertion: float = 0.1,
        p_deletion: float = 0.1,
        n_swaps: int = 1,
    ):
        self.n_augmentations = n_augmentations
        self.use_insertion = use_insertion
        self.use_deletion = use_deletion
        self.use_swap = use_swap

        self.insertion = RandomInsertion(int(p_insertion * 10))
        self.deletion = RandomDeletion(p_deletion)
        self.swap = RandomSwap(n_swaps)

    def augment(self, text: str) -> List[str]:
        """Generate augmented texts using all techniques.

        Args:
            text: Input text to augment

        Returns:
            List of augmented texts
        """
        augmented = []

        for _ in range(self.n_augmentations):
            technique = random.choice(["insertion", "deletion", "swap"])

            if technique == "insertion" and self.use_insertion:
                result = self.insertion.augment(text)
            elif technique == "deletion" and self.use_deletion:
                result = self.deletion.augment(text)
            elif technique == "swap" and self.use_swap:
                result = self.swap.augment(text)
            else:
                result = text

            augmented.append(result)

        return augmented

    def augment_batch(self, texts: List[str]) -> List[str]:
        """Augment a batch of texts."""
        augmented = []

        for text in texts:
            augmented.extend(self.augment(text))

        return augmented


class CharacterLevelAugmentation:
    """Character-level text augmentation.

    Applies augmentation at the character level including
    keyboard typos, random character changes, and more.
    """

    def __init__(
        self,
        char_error_prob: float = 0.05,
        keyboard_neighbors: Optional[dict] = None,
    ):
        self.char_error_prob = char_error_prob

        self.keyboard_neighbors = keyboard_neighbors or {
            "a": ["q", "w", "s", "z"],
            "b": ["v", "g", "h", "n"],
            "c": ["x", "d", "f", "v"],
            "d": ["s", "e", "r", "f", "c", "x"],
            "e": ["w", "s", "d", "r"],
            "f": ["d", "r", "t", "g", "v", "c"],
            "g": ["f", "t", "y", "h", "v", "b"],
            "h": ["g", "y", "u", "j", "b", "n"],
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

    def _keyboard_typo(self, char: str) -> str:
        """Generate a keyboard-based typo."""
        char_lower = char.lower()

        if char_lower in self.keyboard_neighbors:
            return random.choice(self.keyboard_neighbors[char_lower])

        return char

    def augment(self, text: str) -> str:
        """Generate augmented text with character-level errors.

        Args:
            text: Input text to augment

        Returns:
            Augmented text
        """
        chars = list(text)

        for i in range(len(chars)):
            if random.random() < self.char_error_prob:
                error_type = random.choice(["typo", "delete", "swap", "replace"])

                if error_type == "typo":
                    chars[i] = self._keyboard_typo(chars[i])
                elif error_type == "delete" and len(chars) > 1:
                    chars[i] = ""
                elif error_type == "swap" and i < len(chars) - 1:
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                elif error_type == "replace":
                    chars[i] = chr(random.randint(ord("a"), ord("z")))

        return "".join(c for c in chars if c)


class BackRandomEdits:
    """Combined random edits augmentation.

    Provides back-translation compatibility by implementing
    similar augmentation patterns.
    """

    def __init__(
        self,
        p_insertion: float = 0.05,
        p_deletion: float = 0.05,
        p_swap: float = 0.05,
    ):
        self.insertion = RandomInsertion(int(p_insertion * 20))
        self.deletion = RandomDeletion(p_deletion)
        self.swap = RandomSwap(int(p_swap * 10))

    def augment(self, text: str) -> str:
        """Apply all random edit augmentations."""
        result = self.insertion.augment(text)
        result = self.deletion.augment(result)
        result = self.swap.augment(result)

        return result
