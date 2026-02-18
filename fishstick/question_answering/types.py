"""
Question Answering Types and Data Structures

This module provides comprehensive type definitions for the QA framework,
including data classes for examples, predictions, retrieval results,
and various configuration types.

Author: Fishstick AI Framework
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple, Sequence
from enum import Enum, auto
from pathlib import Path
import json


class QATaskType(Enum):
    """Types of QA tasks supported by the framework."""

    EXTRACTIVE = auto()
    GENERATIVE = auto()
    MULTI_HOP = auto()
    READING_COMPREHENSION = auto()
    DOMAIN_SPECIFIC = auto()
    OPEN_DOMAIN = auto()
    CONVERSATIONAL = auto()
    KNOWLEDGE_BASE = auto()


class AnswerType(Enum):
    """Types of answers that QA systems can produce."""

    SPAN = auto()
    YES_NO = auto()
    COUNT = auto()
    FREE_FORM = auto()
    ENTITY = auto()
    MULTIPLE_CHOICE = auto()


class RetrievalMethod(Enum):
    """Methods for document retrieval."""

    DENSE = auto()
    SPARSE = auto()
    HYBRID = auto()
    ITERATIVE = auto()
    BM25 = auto()


@dataclass
class Question:
    """Represents a question with metadata."""

    text: str
    id: Optional[str] = None
    type: Optional[str] = None
    category: Optional[str] = None
    domain: Optional[str] = None
    ambiguity_level: float = 0.0
    requires_reasoning: bool = False
    num_hops_required: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "id": self.id,
            "type": self.type,
            "category": self.category,
            "domain": self.domain,
            "ambiguity_level": self.ambiguity_level,
            "requires_reasoning": self.requires_reasoning,
            "num_hops_required": self.num_hops_required,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Question:
        """Create Question from dictionary."""
        return cls(
            text=data["text"],
            id=data.get("id"),
            type=data.get("type"),
            category=data.get("category"),
            domain=data.get("domain"),
            ambiguity_level=data.get("ambiguity_level", 0.0),
            requires_reasoning=data.get("requires_reasoning", False),
            num_hops_required=data.get("num_hops_required", 1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Context:
    """Represents context/document passage for QA."""

    text: str
    id: Optional[str] = None
    title: Optional[str] = None
    source: Optional[str] = None
    tokens: Optional[List[str]] = None
    embeddings: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the length of the context in characters."""
        return len(self.text)

    def get_passages(
        self, passage_length: int = 200, overlap: int = 50
    ) -> List[Tuple[str, int, int]]:
        """Split context into overlapping passages.

        Args:
            passage_length: Maximum length of each passage in characters
            overlap: Number of overlapping characters between passages

        Yields:
            Tuple of (passage_text, start_char, end_char)
        """
        words = self.text.split()
        current_passage = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > passage_length:
                if current_passage:
                    passage = " ".join(current_passage)
                    start = self.text.find(passage)
                    end = start + len(passage)
                    yield passage, start, end
                current_passage = [word]
                current_length = word_length
            else:
                current_passage.append(word)
                current_length += word_length

        if current_passage:
            passage = " ".join(current_passage)
            start = self.text.find(passage)
            end = start + len(passage)
            yield passage, start, end


@dataclass
class Answer:
    """Represents an answer with optional evidence."""

    text: str
    type: AnswerType = AnswerType.SPAN
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    confidence: float = 1.0
    evidence_spans: List[Tuple[int, int]] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "type": self.type.name,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "evidence_spans": self.evidence_spans,
            "reasoning_chain": self.reasoning_chain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Answer:
        """Create Answer from dictionary."""
        return cls(
            text=data["text"],
            type=AnswerType[data.get("type", "SPAN")],
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            confidence=data.get("confidence", 1.0),
            evidence_spans=data.get("evidence_spans", []),
            reasoning_chain=data.get("reasoning_chain", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QAExample:
    """Complete QA example with question, context, and answer.

    This is the primary data structure for training and evaluating
    QA models. It supports various task types and includes rich
    metadata for analysis.
    """

    id: str
    question: Union[str, Question]
    context: Union[str, Context]
    answer: Optional[Union[str, Answer]] = None
    is_impossible: bool = False
    negative_contexts: List[Union[str, Context]] = field(default_factory=list)
    task_type: QATaskType = QATaskType.EXTRACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure question and context are proper types."""
        if isinstance(self.question, str):
            self.question = Question(text=self.question)
        if isinstance(self.context, str):
            self.context = Context(text=self.context)
        if isinstance(self.answer, str):
            self.answer = Answer(text=self.answer)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "question": (
                self.question.to_dict()
                if isinstance(self.question, Question)
                else self.question
            ),
            "context": (
                self.context.text if isinstance(self.context, Context) else self.context
            ),
            "answer": (
                self.answer.to_dict()
                if isinstance(self.answer, Answer)
                else self.answer
            ),
            "is_impossible": self.is_impossible,
            "task_type": self.task_type.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QAExample:
        """Create QAExample from dictionary."""
        question = data.get("question", "")
        if isinstance(question, dict):
            question = Question.from_dict(question)

        context = data.get("context", "")
        if isinstance(context, str):
            context = Context(text=context)

        answer = data.get("answer")
        if isinstance(answer, dict):
            answer = Answer.from_dict(answer)

        return cls(
            id=data["id"],
            question=question,
            context=context,
            answer=answer,
            is_impossible=data.get("is_impossible", False),
            task_type=QATaskType[data.get("task_type", "EXTRACTIVE")],
            metadata=data.get("metadata", {}),
        )


@dataclass
class QAPrediction:
    """Model prediction for a QA example."""

    id: str
    question: str
    answer: Answer
    context_used: str
    retrieved_documents: List[str] = field(default_factory=list)
    attention_weights: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def answer_text(self) -> str:
        """Get the answer text."""
        return self.answer.text

    @property
    def confidence(self) -> float:
        """Get the confidence score."""
        return self.answer.confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer.to_dict(),
            "context_used": self.context_used,
            "retrieved_documents": self.retrieved_documents,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QAPrediction:
        """Create QAPrediction from dictionary."""
        answer = data["answer"]
        if isinstance(answer, dict):
            answer = Answer.from_dict(answer)

        return cls(
            id=data["id"],
            question=data["question"],
            answer=answer,
            context_used=data.get("context_used", ""),
            retrieved_documents=data.get("retrieved_documents", []),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """Document retrieval result with score and metadata."""

    document_id: str
    document: str
    score: float
    rank: int = 0
    title: Optional[str] = None
    passage: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: RetrievalResult) -> bool:
        """Compare by score for sorting."""
        return self.score < other.score

    def __eq__(self, other: RetrievalResult) -> bool:
        """Check equality based on document_id."""
        if not isinstance(other, RetrievalResult):
            return False
        return self.document_id == other.document_id

    def __hash__(self) -> int:
        """Hash based on document_id."""
        return hash(self.document_id)


@dataclass
class EvaluationResult:
    """Results from evaluating a QA model."""

    example_id: str
    exact_match: bool
    f1_score: float
    precision: float
    recall: float
    prediction: Optional[QAPrediction] = None
    gold_answer: Optional[Answer] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_correct(self) -> bool:
        """Check if prediction is correct."""
        return self.exact_match


@dataclass
class MultiHopStep:
    """Single reasoning step in multi-hop QA."""

    step_number: int
    question: str
    context: str
    answer: str
    evidence: str
    reasoning_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """Complete reasoning chain for multi-hop questions."""

    id: str
    question: str
    hops: List[MultiHopStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    supporting_facts: List[str] = field(default_factory=list)

    def add_hop(
        self,
        question: str,
        context: str,
        answer: str,
        evidence: str,
        reasoning_type: str,
    ) -> MultiHopStep:
        """Add a reasoning hop to the chain."""
        hop = MultiHopStep(
            step_number=len(self.hops) + 1,
            question=question,
            context=context,
            answer=answer,
            evidence=evidence,
            reasoning_type=reasoning_type,
        )
        self.hops.append(hop)
        return hop


@dataclass
class DomainConfig:
    """Configuration for domain-specific QA."""

    domain: str
    vocabulary: Optional[List[str]] = None
    entity_types: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    schema: Optional[Dict[str, Any]] = None
    preprocessing_rules: List[str] = field(default_factory=list)
    postprocessing_rules: List[str] = field(default_factory=list)


@dataclass
class QAConfig:
    """Main configuration for QA systems."""

    model_name: str
    task_type: QATaskType
    max_seq_length: int = 384
    max_answer_length: int = 30
    doc_stride: int = 128
    n_best_size: int = 20
    null_score_diff_threshold: float = 0.0
    device: str = "cuda"
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    seed: int = 42
    fp16: bool = True
    use_amp: bool = False
    dataloader_num_workers: int = 4
    overwrite_cache: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type.name,
            "max_seq_length": self.max_seq_length,
            "max_answer_length": self.max_answer_length,
            "doc_stride": self.doc_stride,
            "n_best_size": self.n_best_size,
            "null_score_diff_threshold": self.null_score_diff_threshold,
            "device": self.device,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "seed": self.seed,
            "fp16": self.fp16,
            "use_amp": self.use_amp,
            "dataloader_num_workers": self.dataloader_num_workers,
            "overwrite_cache": self.overwrite_cache,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QAConfig:
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            task_type=QATaskType[data.get("task_type", "EXTRACTIVE")],
            max_seq_length=data.get("max_seq_length", 384),
            max_answer_length=data.get("max_answer_length", 30),
            doc_stride=data.get("doc_stride", 128),
            n_best_size=data.get("n_best_size", 20),
            null_score_diff_threshold=data.get("null_score_diff_threshold", 0.0),
            device=data.get("device", "cuda"),
            batch_size=data.get("batch_size", 8),
            learning_rate=data.get("learning_rate", 3e-5),
            num_epochs=data.get("num_epochs", 3),
            warmup_ratio=data.get("warmup_ratio", 0.1),
            weight_decay=data.get("weight_decay", 0.01),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 1),
            max_grad_norm=data.get("max_grad_norm", 1.0),
            logging_steps=data.get("logging_steps", 100),
            save_steps=data.get("save_steps", 1000),
            eval_steps=data.get("eval_steps", 1000),
            seed=data.get("seed", 42),
            fp16=data.get("fp16", True),
            use_amp=data.get("use_amp", False),
            dataloader_num_workers=data.get("dataloader_num_workers", 4),
            overwrite_cache=data.get("overwrite_cache", True),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> QAConfig:
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_jsonl(path: Union[str, Path]) -> List[QAExample]:
    """Load QA examples from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of QAExample objects
    """
    examples = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            examples.append(QAExample.from_dict(data))
    return examples


def save_jsonl(examples: List[QAExample], path: Union[str, Path]) -> None:
    """Save QA examples to JSONL file.

    Args:
        examples: List of QAExample objects
        path: Path to output JSONL file
    """
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict()) + "\n")
