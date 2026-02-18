"""
Dataset and DataLoader Utilities for Question Answering

This module provides dataset classes and utilities for loading and
processing QA data, including support for various QA formats and
data augmentation.

Author: Fishstick AI Framework
"""

from __future__ import annotations

import json
import random
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Iterator
from pathlib import Path
from dataclasses import dataclass, field
import logging

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import Tensor
import numpy as np

from fishstick.question_answering.types import (
    QAExample,
    QAPrediction,
    Answer,
    Context,
    Question,
    QATaskType,
    AnswerType,
)
from fishstick.question_answering.metrics import normalize_answer


logger = logging.getLogger(__name__)


@dataclass
class QADataConfig:
    """Configuration for QA data processing."""

    max_seq_length: int = 384
    max_answer_length: int = 30
    doc_stride: int = 128
    max_question_length: int = 64
    max_context_length: int = 384
    pad_to_max_length: bool = True
    truncation_strategy: str = "only_second"
    return_token_type_ids: bool = True
    return_attention_mask: bool = True
    include_impossible: bool = False
    n_best_size: int = 20
    null_score_diff_threshold: float = 0.0
    version_2_with_negative: bool = False


class QADataset(Dataset[QAExample]):
    """Dataset for Question Answering tasks.

    This dataset class handles various QA formats and provides
    efficient data loading with sliding window for long contexts.
    """

    def __init__(
        self,
        examples: List[QAExample],
        config: Optional[QADataConfig] = None,
        tokenizer: Optional[Any] = None,
        is_training: bool = True,
    ):
        """Initialize QA dataset.

        Args:
            examples: List of QA examples
            config: Data configuration
            tokenizer: Tokenizer for processing
            is_training: Whether in training mode
        """
        self.examples = examples
        self.config = config or QADataConfig()
        self.tokenizer = tokenizer
        self.is_training = is_training

        self.features: List[Dict[str, Any]] = []
        if tokenizer is not None:
            self._preprocess()

    def _preprocess(self) -> None:
        """Preprocess examples into features."""
        from tqdm import tqdm

        logger.info(f"Preprocessing {len(self.examples)} examples...")

        for example in tqdm(self.examples, desc="Tokenizing"):
            if isinstance(example.question, Question):
                question_text = example.question.text
            else:
                question_text = example.question

            if isinstance(example.context, Context):
                context_text = example.context.text
            else:
                context_text = example.context

            if isinstance(example.answer, Answer):
                answer_text = example.answer.text
                answer_start = example.answer.start_char
                answer_end = example.answer.end_char
            else:
                answer_text = example.answer or ""
                answer_start = None
                answer_end = None

            features = self._tokenize_single(
                example.id,
                question_text,
                context_text,
                answer_text,
                answer_start,
                answer_end,
            )
            self.features.extend(features)

        logger.info(f"Created {len(self.features)} features")

    def _tokenize_single(
        self,
        example_id: str,
        question_text: str,
        context_text: str,
        answer_text: str,
        answer_start: Optional[int],
        answer_end: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Tokenize a single example with sliding window.

        Args:
            example_id: Example ID
            question_text: Question text
            context_text: Context text
            answer_text: Answer text
            answer_start: Answer start position
            answer_end: Answer end position

        Returns:
            List of feature dictionaries
        """
        if self.tokenizer is None:
            return [
                {
                    "example_id": example_id,
                    "question": question_text,
                    "context": context_text,
                }
            ]

        tokenizer = self.tokenizer

        question_tokens = tokenizer.encode(
            question_text,
            add_special_tokens=False,
            max_length=self.config.max_question_length,
            truncation=self.config.truncation_strategy == "only_first",
        )

        context_tokens = tokenizer.encode(
            context_text,
            add_special_tokens=False,
            max_length=self.config.max_context_length,
            truncation=self.config.truncation_strategy == "only_second",
        )

        max_length = self.config.max_seq_length
        doc_stride = self.config.doc_stride
        max_answer_length = self.config.max_answer_length

        max_seq_length = min(
            len(question_tokens) + len(context_tokens) + 3,
            max_length,
        )

        length = len(question_tokens) + len(context_tokens) + 3

        features = []
        start_position = None
        end_position = None

        if length <= max_seq_length:
            sequence_ids = [0] * (len(question_tokens) + 2) + [1] * (
                len(context_tokens) + 1
            )
            input_ids = (
                [tokenizer.cls_token_id]
                + question_tokens
                + [tokenizer.sep_token_id]
                + context_tokens
                + [tokenizer.sep_token_id]
            )

            if answer_start is not None and answer_end is not None:
                answer_offset = len(question_tokens) + 2
                start_position = answer_offset + answer_start
                end_position = answer_offset + answer_end

                if start_position < len(input_ids) and end_position < len(input_ids):
                    end_position = min(
                        start_position + max_answer_length - 1, len(input_ids) - 1
                    )

            features.append(
                {
                    "example_id": example_id,
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "token_type_ids": sequence_ids,
                    "start_position": start_position,
                    "end_position": end_position,
                    "question_text": question_text,
                    "context_text": context_text,
                }
            )
        else:
            max_len_for_doc = max_seq_length - len(question_tokens) - 3

            doc_offset = len(question_tokens) + 2

            for start_offset in range(0, len(context_tokens), doc_stride):
                length = min(max_len_for_doc, len(context_tokens) - start_offset)

                context_start = start_offset
                context_end = start_offset + length - 1

                input_ids = (
                    [tokenizer.cls_token_id]
                    + question_tokens
                    + [tokenizer.sep_token_id]
                    + context_tokens[context_start : context_end + 1]
                    + [tokenizer.sep_token_id]
                )

                sequence_ids = [0] * (len(question_tokens) + 2) + [1] * (length + 1)

                start_position = None
                end_position = None

                if answer_start is not None and answer_end is not None:
                    if answer_start >= context_start and answer_end <= context_end:
                        start_position = answer_start - context_start + doc_offset
                        end_position = answer_end - context_start + doc_offset
                        end_position = min(
                            start_position + max_answer_length - 1,
                            len(input_ids) - 1,
                        )

                features.append(
                    {
                        "example_id": example_id,
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "token_type_ids": sequence_ids,
                        "start_position": start_position,
                        "end_position": end_position,
                        "offset": start_offset,
                        "question_text": question_text,
                        "context_text": context_text,
                    }
                )

                if len(features) >= 30:
                    break

        return features

    def __len__(self) -> int:
        """Return dataset size."""
        if self.features:
            return len(self.features)
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index.

        Args:
            idx: Index

        Returns:
            Feature dictionary
        """
        if self.features:
            feature = self.features[idx]
            return {
                "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    feature.get("attention_mask", [1] * len(feature["input_ids"])),
                    dtype=torch.long,
                ),
                "token_type_ids": torch.tensor(
                    feature.get("token_type_ids", [0] * len(feature["input_ids"])),
                    dtype=torch.long,
                ),
                "start_position": torch.tensor(
                    feature.get("start_position"), dtype=torch.long
                )
                if feature.get("start_position") is not None
                else torch.tensor(-1),
                "end_position": torch.tensor(
                    feature.get("end_position"), dtype=torch.long
                )
                if feature.get("end_position") is not None
                else torch.tensor(-1),
                "example_id": feature.get("example_id", ""),
            }

        example = self.examples[idx]
        return {
            "example_id": example.id,
            "question": example.question.text
            if isinstance(example.question, Question)
            else example.question,
            "context": example.context.text
            if isinstance(example.context, Context)
            else example.context,
        }


def collate_qa_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """Collate batch of QA examples.

    Args:
        batch: List of example dictionaries
        pad_token_id: Token ID for padding

    Returns:
        Collated batch dictionary
    """
    if not batch:
        return {}

    if "input_ids" in batch[0]:
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_masks = []
        token_type_ids = []
        start_positions = []
        end_positions = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(item["input_ids"].tolist() + [pad_token_id] * pad_len)
            attention_masks.append(item["attention_mask"].tolist() + [0] * pad_len)
            token_type_ids.append(item["token_type_ids"].tolist() + [0] * pad_len)
            start_positions.append(item["start_position"])
            end_positions.append(item["end_position"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "start_position": torch.tensor(start_positions, dtype=torch.long),
            "end_position": torch.tensor(end_positions, dtype=torch.long),
            "example_ids": [item.get("example_id", "") for item in batch],
        }

    return {
        "example_ids": [item.get("example_id", "") for item in batch],
        "questions": [item.get("question", "") for item in batch],
        "contexts": [item.get("context", "") for item in batch],
    }


class QASampler(Sampler[int]):
    """Custom sampler for QA datasets.

    Provides options for balanced sampling and hard negative mining.
    """

    def __init__(
        self,
        dataset: QADataset,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """Initialize QA sampler.

        Args:
            dataset: QA dataset
            shuffle: Whether to shuffle
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.indices = list(range(len(dataset)))

        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self) -> Iterator[int]:
        """Iterate over indices."""
        return iter(self.indices)

    def __len__(self) -> int:
        """Return sampler length."""
        return len(self.indices)


class HardNegativeSampler:
    """Sampler for hard negative examples.

    Samples negative examples that are difficult (similar to positive).
    """

    def __init__(
        self,
        examples: List[QAExample],
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        num_negatives: int = 1,
    ):
        """Initialize hard negative sampler.

        Args:
            examples: List of QA examples
            similarity_fn: Function to compute similarity
            num_negatives: Number of negatives per positive
        """
        self.examples = examples
        self.similarity_fn = similarity_fn or self._default_similarity
        self.num_negatives = num_negatives

        self._build_negative_cache()

    def _default_similarity(self, text1: str, text2: str) -> float:
        """Compute default word overlap similarity."""
        words1 = set(normalize_answer(text1).split())
        words2 = set(normalize_answer(text2).split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))

    def _build_negative_cache(self) -> None:
        """Build cache of hard negatives for each example."""
        self.negative_cache: Dict[int, List[int]] = {}

        for i, example in enumerate(self.examples):
            context = (
                example.context.text
                if isinstance(example.context, Context)
                else example.context
            )

            similarities = []
            for j, other in enumerate(self.examples):
                if i == j:
                    continue

                other_context = (
                    other.context.text
                    if isinstance(other.context, Context)
                    else other.context
                )

                if example.answer is None and other.answer is None:
                    sim = self.similarity_fn(context, other_context)
                    similarities.append((j, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            self.negative_cache[i] = [
                idx for idx, _ in similarities[: self.num_negatives * 10]
            ]

    def sample(self, idx: int, num_negatives: Optional[int] = None) -> List[QAExample]:
        """Sample hard negatives for given index.

        Args:
            idx: Positive example index
            num_negatives: Number of negatives to sample

        Returns:
            List of negative examples
        """
        num_negatives = num_negatives or self.num_negatives

        if idx not in self.negative_cache:
            return []

        candidates = self.negative_cache[idx]
        if len(candidates) < num_negatives:
            return [self.examples[i] for i in candidates]

        selected = random.sample(candidates, num_negatives)
        return [self.examples[i] for i in selected]


class DataAugmentor:
    """Data augmentation for QA tasks.

    Provides various augmentation strategies including
    back-translation, synonym replacement, and noise injection.
    """

    def __init__(
        self,
        augmentation_types: List[str] = None,
        probability: float = 0.3,
    ):
        """Initialize augmentor.

        Args:
            augmentation_types: Types of augmentation to apply
            probability: Probability of applying augmentation
        """
        self.augmentation_types = augmentation_types or [
            "synonym_replacement",
            "random_deletion",
            "swap",
        ]
        self.probability = probability

        self.synonym_dict: Dict[str, List[str]] = {}

    def augment(self, example: QAExample) -> QAExample:
        """Augment a single QA example.

        Args:
            example: Original example

        Returns:
            Augmented example
        """
        if isinstance(example.question, Question):
            question_text = example.question.text
        else:
            question_text = example.question

        if random.random() > self.probability:
            return example

        aug_type = random.choice(self.augmentation_types)

        if aug_type == "synonym_replacement":
            question_text = self._synonym_replace(question_text)
        elif aug_type == "random_deletion":
            question_text = self._random_delete(question_text)
        elif aug_type == "swap":
            question_text = self._swap_words(question_text)

        if isinstance(example.question, Question):
            example.question.text = question_text
        else:
            example.question = Question(text=question_text)

        return example

    def _synonym_replace(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        result = []

        for word in words:
            synonyms = self.synonym_dict.get(word.lower(), [])
            if synonyms and random.random() < self.probability:
                result.append(random.choice(synonyms))
            else:
                result.append(word)

        return " ".join(result)

    def _random_delete(self, text: str) -> str:
        """Randomly delete words."""
        words = text.split()
        if len(words) <= 3:
            return text

        result = [w for w in words if random.random() > self.probability]
        return " ".join(result) if result else text

    def _swap_words(self, text: str) -> str:
        """Swap adjacent words."""
        words = text.split()
        if len(words) <= 1:
            return text

        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return " ".join(words)


class MultiDocQADataset(Dataset):
    """Dataset for multi-document QA tasks.

    Handles questions that require reasoning across multiple documents.
    """

    def __init__(
        self,
        examples: List[QAExample],
        max_docs: int = 10,
        is_training: bool = True,
    ):
        """Initialize multi-document QA dataset.

        Args:
            examples: List of QA examples
            max_docs: Maximum number of documents per example
            is_training: Whether in training mode
        """
        self.examples = examples
        self.max_docs = max_docs
        self.is_training = is_training

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index.

        Args:
            idx: Index

        Returns:
            Example dictionary
        """
        example = self.examples[idx]

        if isinstance(example.question, Question):
            question = example.question.text
        else:
            question = example.question

        if isinstance(example.context, Context):
            context = example.context.text
            title = example.context.title
        else:
            context = example.context
            title = None

        answer = None
        if example.answer:
            if isinstance(example.answer, Answer):
                answer = example.answer.text
            else:
                answer = example.answer

        return {
            "id": example.id,
            "question": question,
            "context": context,
            "title": title,
            "answer": answer,
            "is_impossible": example.is_impossible,
        }


def load_squad_dataset(
    path: Union[str, Path],
    is_training: bool = True,
) -> List[QAExample]:
    """Load SQuAD format dataset.

    Args:
        path: Path to SQuAD JSON file
        is_training: Whether in training mode

    Returns:
        List of QA examples
    """
    with open(path, "r") as f:
        data = json.load(f)

    examples = []

    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context_text = paragraph["context"]
            context = Context(text=context_text, title=article.get("title"))

            for qa in paragraph.get("qas", []):
                question_id = qa["id"]
                question_text = qa["question"]

                is_impossible = qa.get("is_impossible", False)

                if is_impossible:
                    answer_text = ""
                    answer_start = None
                    answer_end = None
                else:
                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    answer_start = answer.get("answer_start")
                    answer_end = (
                        answer_start + len(answer_text) if answer_start else None
                    )

                question = Question(text=question_text, id=question_id)

                answer_obj = Answer(
                    text=answer_text,
                    start_char=answer_start,
                    end_char=answer_end,
                )

                example = QAExample(
                    id=question_id,
                    question=question,
                    context=context,
                    answer=answer_obj,
                    is_impossible=is_impossible,
                )
                examples.append(example)

    return examples


def load_natural_questions(
    path: Union[str, Path],
    max_examples: Optional[int] = None,
) -> List[QAExample]:
    """Load Natural Questions dataset.

    Args:
        path: Path to NQ JSON file
        max_examples: Maximum number of examples to load

    Returns:
        List of QA examples
    """
    examples = []

    with open(path, "r") as f:
        for line in f:
            if max_examples and len(examples) >= max_examples:
                break

            data = json.loads(line)

            question_text = data["question_text"]
            example_id = str(data["example_id"])

            context_text = data.get("document_tokens", [])
            context_text = " ".join(context_text)

            annotations = data.get("annotations", [])
            if annotations and annotations[0].get("long_answer"):
                long_ans = annotations[0]["long_answer"]
                start = long_ans.get("start_token", 0)
                end = long_ans.get("end_token", 0)

                tokens = data.get("document_tokens", [])
                if start < len(tokens) and end <= len(tokens):
                    answer_text = " ".join(tokens[start:end])
                    answer_start = len(" ".join(tokens[:start]))
                else:
                    answer_text = ""
                    answer_start = None
            else:
                short_ans = annotations[0].get("short_answers", [])
                if short_ans:
                    start = short_ans[0].get("start_token", 0)
                    end = short_ans[0].get("end_token", 0)
                    tokens = data.get("document_tokens", [])
                    if start < len(tokens) and end <= len(tokens):
                        answer_text = " ".join(tokens[start:end])
                        answer_start = len(" ".join(tokens[:start]))
                    else:
                        answer_text = ""
                        answer_start = None
                else:
                    answer_text = ""
                    answer_start = None

            question = Question(text=question_text, id=example_id)
            context = Context(text=context_text)

            answer = Answer(
                text=answer_text,
                start_char=answer_start,
                end_char=answer_start + len(answer_text) if answer_start else None,
            )

            example = QAExample(
                id=example_id,
                question=question,
                context=context,
                answer=answer,
                is_impossible=answer_text == "",
            )
            examples.append(example)

    return examples


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """Create DataLoader for QA dataset.

    Args:
        dataset: QA dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        collate_fn: Custom collate function

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
