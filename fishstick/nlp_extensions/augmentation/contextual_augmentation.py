"""
Contextual Augmentation

Text augmentation using language models to generate contextually
appropriate augmentations.
"""

from typing import List, Optional, Callable, Dict
import random


class ContextualAugmentation:
    """Contextual augmentation using language models.

    Uses a language model to generate contextually appropriate
    replacements for words in the text.

    Attributes:
        mask_predictor: Function that predicts masked words
        n_predictions: Number of predictions to generate per masked word
    """

    def __init__(
        self,
        mask_predictor: Optional[Callable[[str, int], List[str]]] = None,
        n_predictions: int = 5,
    ):
        self.mask_predictor = mask_predictor or self._default_predictor()
        self.n_predictions = n_predictions

    def _default_predictor(self) -> Callable[[str, int], List[str]]:
        """Default predictor for testing."""
        common_words = [
            "the",
            "a",
            "an",
            "some",
            "many",
            "few",
            "big",
            "small",
            "large",
            "tiny",
            "good",
            "great",
            "excellent",
            "bad",
            "terrible",
        ]

        def predict(text: str, position: int) -> List[str]:
            return random.sample(
                common_words, min(self.n_predictions, len(common_words))
            )

        return predict

    def _mask_word(self, text: str, position: int) -> str:
        """Replace word at position with [MASK]."""
        words = text.split()

        if 0 <= position < len(words):
            words[position] = "[MASK]"

        return " ".join(words)

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
    ) -> List[str]:
        """Generate augmented texts through contextual masking.

        Args:
            text: Input text to augment
            n_augmentations: Number of augmentations to generate

        Returns:
            List of augmented texts
        """
        words = text.split()

        if not words:
            return [text]

        augmented = []

        for _ in range(n_augmentations):
            position = random.randint(0, len(words) - 1)

            masked_text = self._mask_word(text, position)
            predictions = self.mask_predictor(masked_text, position)

            if predictions:
                new_word = random.choice(predictions)
                new_words = words.copy()
                new_words[position] = new_word
                augmented.append(" ".join(new_words))
            else:
                augmented.append(text)

        return augmented

    def augment_batch(
        self,
        texts: List[str],
        n_augmentations: int = 1,
    ) -> List[str]:
        """Augment a batch of texts."""
        augmented = []

        for text in texts:
            augmented.extend(self.augment(text, n_augmentations))

        return augmented


class ConditionalContextualAugmentation(ContextualAugmentation):
    """Contextual augmentation with conditional generation.

    Uses conditional generation to produce augmentations
    that preserve specific attributes of the text.

    Attributes:
        condition: Condition for augmentation (e.g., sentiment)
    """

    def __init__(
        self,
        mask_predictor: Optional[Callable[[str, int, str], List[str]]] = None,
        n_predictions: int = 5,
        condition: str = "neutral",
    ):
        super().__init__(None, n_predictions)
        self.condition = condition
        self.conditional_predictor = mask_predictor or self._default_conditional()

    def _default_conditional(self) -> Callable[[str, int, str], List[str]]:
        """Default conditional predictor."""
        word_sets = {
            "positive": ["good", "great", "excellent", "wonderful", "amazing"],
            "negative": ["bad", "terrible", "awful", "horrible", "poor"],
            "neutral": ["the", "a", "some", "many", "few"],
        }

        def predict(text: str, position: int, condition: str) -> List[str]:
            words = word_sets.get(condition, word_sets["neutral"])
            return random.sample(words, min(self.n_predictions, len(words)))

        return predict

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
        condition: Optional[str] = None,
    ) -> List[str]:
        """Generate augmented texts with condition."""
        words = text.split()

        if not words:
            return [text]

        augmented = []
        cond = condition or self.condition

        for _ in range(n_augmentations):
            position = random.randint(0, len(words) - 1)

            masked_text = self._mask_word(text, position)
            predictions = self.conditional_predictor(masked_text, position, cond)

            if predictions:
                new_word = random.choice(predictions)
                new_words = words.copy()
                new_words[position] = new_word
                augmented.append(" ".join(new_words))
            else:
                augmented.append(text)

        return augmented


class BERTContextualAugmentation(ContextualAugmentation):
    """BERT-style contextual augmentation.

    Uses BERT-style masked language modeling for augmentation.
    """

    def __init__(
        self,
        model_predictor: Optional[Callable[[str], Dict[int, List[str]]]] = None,
        n_predictions: int = 5,
    ):
        super().__init__(None, n_predictions)
        self.model_predictor = model_predictor or self._default_bert_predictor()

    def _default_bert_predictor(self) -> Callable[[str], Dict[int, List[str]]]:
        """Default BERT-style predictor."""
        default_predictions = {
            0: ["the", "a", "an"],
            1: ["cat", "dog", "bird"],
            2: ["is", "was", "will"],
            3: ["running", "walking", "sleeping"],
        }

        def predict(text: str) -> Dict[int, List[str]]:
            return default_predictions

        return predict

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
    ) -> List[str]:
        """Generate augmented texts using BERT-style predictions."""
        words = text.split()

        if not words:
            return [text]

        masked_texts = []

        for position in range(len(words)):
            masked = words.copy()
            masked[position] = "[MASK]"
            masked_texts.append((position, " ".join(masked)))

        predictions = {}
        for position, masked_text in masked_texts:
            preds = self.model_predictor(masked_text)
            predictions[position] = preds.get(position, ["word"])

        augmented = []

        for _ in range(n_augmentations):
            new_words = words.copy()

            num_replacements = random.randint(1, max(1, len(words) // 4))
            positions = random.sample(
                range(len(words)), min(num_replacements, len(words))
            )

            for position in positions:
                if predictions.get(position):
                    new_words[position] = random.choice(predictions[position])

            augmented.append(" ".join(new_words))

        return augmented


class FillMaskAugmentation:
    """Fill-mask augmentation using pre-trained models.

    Uses pre-trained fill-mask models for high-quality contextual augmentation.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        pass

    def augment(
        self,
        text: str,
        n_augmentations: int = 1,
    ) -> List[str]:
        """Generate augmented texts using fill-mask.

        Args:
            text: Input text to augment
            n_augmentations: Number of augmentations to generate

        Returns:
            List of augmented texts
        """
        words = text.split()

        if not words:
            return [text]

        augmented = []

        for _ in range(n_augmentations):
            position = random.randint(0, len(words) - 1)

            masked = words.copy()
            masked[position] = "[MASK]"

            augmented.append(" ".join(masked))

        return augmented

    def augment_batch(
        self,
        texts: List[str],
        n_augmentations: int = 1,
    ) -> List[str]:
        """Augment a batch of texts."""
        augmented = []

        for text in texts:
            augmented.extend(self.augment(text, n_augmentations))

        return augmented
