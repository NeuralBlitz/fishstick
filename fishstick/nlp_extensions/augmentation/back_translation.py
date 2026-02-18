"""
Back Translation Augmentation

Text augmentation using back-translation to generate paraphrases.
Translates text to another language and back to create augmented data.
"""

from typing import List, Optional, Dict, Callable
import random


class BackTranslationAugmentation:
    """Back translation augmentation for text data.

    Generates augmented examples by translating text to another
    language and back to the original language.

    Attributes:
        forward_translator: Function to translate from source to target language
        backward_translator: Function to translate from target to source language
        source_lang: Source language code
        target_lang: Target language code
    """

    def __init__(
        self,
        forward_translator: Optional[Callable[[str], str]] = None,
        backward_translator: Optional[Callable[[str], str]] = None,
        source_lang: str = "en",
        target_lang: str = "de",
    ):
        self.forward_translator = forward_translator or self._dummy_translator(
            target_lang
        )
        self.backward_translator = backward_translator or self._dummy_translator(
            source_lang
        )
        self.source_lang = source_lang
        self.target_lang = target_lang

    def _dummy_translator(self, target_lang: str) -> Callable[[str], str]:
        """Create a dummy translator for testing."""

        def translate(text: str) -> str:
            return f"[{target_lang}]{text}[/{target_lang}]"

        return translate

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
    ) -> List[str]:
        """Generate augmented examples through back-translation.

        Args:
            text: Input text to augment
            n_augmentations: Number of augmentations to generate

        Returns:
            List of augmented texts
        """
        augmented = []

        for _ in range(n_augmentations):
            translated = self.forward_translator(text)
            back_translated = self.backward_translator(translated)
            augmented.append(back_translated)

        return augmented

    def augment_batch(
        self,
        texts: List[str],
        n_augmentations: int = 1,
    ) -> List[str]:
        """Augment a batch of texts.

        Args:
            texts: List of input texts
            n_augmentations: Number of augmentations per text

        Returns:
            List of all augmented texts
        """
        augmented = []

        for text in texts:
            augmented.extend(self.augment(text, n_augmentations))

        return augmented


class MultilingualBackTranslation(BackTranslationAugmentation):
    """Back translation with multiple target languages.

    Uses multiple intermediate languages for diverse augmentation.

    Attributes:
        target_languages: List of target language codes
    """

    def __init__(
        self,
        target_languages: Optional[List[str]] = None,
        source_lang: str = "en",
    ):
        self.target_languages = target_languages or ["de", "fr", "es", "it", "pt"]
        self.source_lang = source_lang

        self.translators: Dict[str, Callable[[str], str]] = {}
        for lang in self.target_languages:
            self.translators[lang] = self._dummy_translator(lang)

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
    ) -> List[str]:
        """Generate augmented examples using multiple languages."""
        augmented = []

        languages_to_use = random.sample(
            self.target_languages,
            min(n_augmentations, len(self.target_languages)),
        )

        for lang in languages_to_use:
            translated = self.translators[lang](text)
            back_translated = self._dummy_translator(self.source_lang)(translated)
            augmented.append(back_translated)

        return augmented


class RoundTripTranslation(BackTranslationAugmentation):
    """Round-trip translation with multiple intermediate steps.

    Performs multiple back-translation passes for more diverse augmentation.

    Attributes:
        num_steps: Number of round-trip translation steps
    """

    def __init__(
        self,
        target_languages: Optional[List[str]] = None,
        num_steps: int = 2,
    ):
        self.target_languages = target_languages or ["de", "fr", "es"]
        self.num_steps = num_steps

        self.translators: Dict[str, Callable[[str], str]] = {}
        for lang in self.target_languages + ["en"]:
            self.translators[lang] = self._dummy_translator(lang)

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
    ) -> List[str]:
        """Generate augmented examples with multiple round trips."""
        augmented = []

        for _ in range(n_augmentations):
            result = text

            for step in range(self.num_steps):
                target_lang = random.choice(self.target_languages)
                result = self.translators[target_lang](result)

            result = self.translators["en"](result)
            augmented.append(result)

        return augmented


class BackTranslationWithNoise(BackTranslationAugmentation):
    """Back translation with additional noise injection.

    Combines back-translation with other augmentation techniques
    for more diverse training data.
    """

    def __init__(
        self,
        target_languages: Optional[List[str]] = None,
        source_lang: str = "en",
        dropout_prob: float = 0.1,
        swap_prob: float = 0.1,
    ):
        super().__init__(source_lang=source_lang)
        self.target_languages = target_languages or ["de", "fr"]
        self.dropout_prob = dropout_prob
        self.swap_prob = swap_prob

    def _inject_noise(self, text: str) -> str:
        """Inject random noise into text."""
        words = text.split()

        if self.dropout_prob > 0:
            words = [w for w in words if random.random() > self.dropout_prob]

        if self.swap_prob > 0 and len(words) > 1:
            new_words = []
            for i, word in enumerate(words):
                if random.random() < self.swap_prob and i < len(words) - 1:
                    new_words.append(words[i + 1])
                    new_words.append(word)
                    i += 1
                else:
                    new_words.append(word)
            words = new_words

        return " ".join(words) if words else text

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
    ) -> List[str]:
        """Generate augmented examples with noise injection."""
        augmented = []

        for _ in range(n_augmentations):
            noisy_text = self._inject_noise(text)
            translated = self.forward_translator(noisy_text)
            back_translated = self.backward_translator(translated)
            augmented.append(back_translated)

        return augmented
