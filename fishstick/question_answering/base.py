"""
Abstract Base Classes for Question Answering Systems

This module provides abstract base classes that define the interface
for all QA systems in the framework. These classes ensure consistent
APIs across different QA implementations.

Author: Fishstick AI Framework
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    TypeVar,
    Generic,
    Callable,
)
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor

from fishstick.question_answering.types import (
    QAExample,
    QAPrediction,
    Answer,
    Context,
    Question,
    EvaluationResult,
    RetrievalResult,
    QAConfig,
)


T = TypeVar("T")
ModelType = TypeVar("ModelType", bound=nn.Module)


class QABase(ABC, Generic[ModelType]):
    """Abstract base class for all QA systems.

    This class defines the common interface that all QA implementations
    must follow. Subclasses must implement the core prediction methods.
    """

    def __init__(self, config: QAConfig):
        """Initialize the QA system.

        Args:
            config: Configuration for the QA system
        """
        self.config = config
        self.device = torch.device(config.device)
        self.model: Optional[ModelType] = None
        self.is_trained = False

    @abstractmethod
    def forward(
        self,
        question: Union[str, Question],
        context: Union[str, Context],
    ) -> Answer:
        """Forward pass to generate answer.

        Args:
            question: The question to answer
            context: The context to extract answer from

        Returns:
            Answer object with the predicted answer
        """
        pass

    @abstractmethod
    def predict(
        self,
        examples: List[QAExample],
    ) -> List[QAPrediction]:
        """Generate predictions for a batch of examples.

        Args:
            examples: List of QA examples to predict

        Returns:
            List of predictions
        """
        pass

    @abstractmethod
    def train_model(
        self,
        train_examples: List[QAExample],
        eval_examples: Optional[List[QAExample]] = None,
    ) -> Dict[str, Any]:
        """Train the QA model.

        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples

        Returns:
            Training history dictionary
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load the model from
        """
        pass

    def evaluate(
        self,
        examples: List[QAExample],
        predictions: List[QAPrediction],
    ) -> Dict[str, float]:
        """Evaluate predictions against gold answers.

        Args:
            examples: Gold examples
            predictions: Model predictions

        Returns:
            Dictionary of evaluation metrics
        """
        from fishstick.question_answering.metrics import compute_exact_match
        from fishstick.question_answering.metrics import compute_f1

        exact_matches = []
        f1_scores = []

        for example, prediction in zip(examples, predictions):
            if example.answer is None:
                continue

            gold_answer = (
                example.answer.text
                if isinstance(example.answer, Answer)
                else example.answer
            )
            pred_answer = prediction.answer.text

            exact_matches.append(compute_exact_match(pred_answer, gold_answer))
            f1_scores.append(compute_f1(pred_answer, gold_answer))

        return {
            "exact_match": sum(exact_matches) / len(exact_matches)
            if exact_matches
            else 0.0,
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        }

    def to(self, device: Union[str, torch.device]) -> QABase:
        """Move model to device.

        Args:
            device: Target device

        Returns:
            Self for method chaining
        """
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def eval_mode(self) -> QABase:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self

    def train_mode(self) -> QABase:
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
        return self


class ExtractiveQABase(QABase[nn.Module]):
    """Abstract base class for extractive QA systems.

    Extractive QA systems select answer spans directly from the input context.
    """

    def __init__(self, config: QAConfig):
        """Initialize extractive QA base.

        Args:
            config: Configuration for the QA system
        """
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.max_answer_length = config.max_answer_length
        self.doc_stride = config.doc_stride

    @abstractmethod
    def extract_spans(
        self,
        question_hidden: Tensor,
        context_hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Extract answer spans from hidden states.

        Args:
            question_hidden: Hidden states for question tokens
            context_hidden: Hidden states for context tokens
            attention_mask: Attention mask for context

        Returns:
            Tuple of (start_logits, end_logits)
        """
        pass

    @abstractmethod
    def predict_spans(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        context_tokens: List[str],
        n_best_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Predict answer spans from logits.

        Args:
            start_logits: Logits for start positions
            end_logits: Logits for end positions
            context_tokens: List of context tokens
            n_best_size: Number of best spans to return

        Returns:
            List of best spans with scores
        """
        pass


class GenerativeQABase(QABase[nn.Module]):
    """Abstract base class for generative QA systems.

    Generative QA systems generate free-form answers rather than
    extracting spans from the context.
    """

    def __init__(self, config: QAConfig):
        """Initialize generative QA base.

        Args:
            config: Configuration for the QA system
        """
        super().__init__(config)
        self.max_new_tokens = config.max_answer_length
        self.num_beams = 4

    @abstractmethod
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Tensor:
        """Generate answer tokens.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search

        Returns:
            Generated token IDs
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        pass


class MultiHopQABase(QABase[nn.Module]):
    """Abstract base class for multi-hop QA systems.

    Multi-hop QA systems require reasoning across multiple pieces
    of evidence to answer a question.
    """

    def __init__(self, config: QAConfig):
        """Initialize multi-hop QA base.

        Args:
            config: Configuration for the QA system
        """
        super().__init__(config)
        self.max_hops = config.metadata.get("max_hops", 3)
        self.retrieval_threshold = config.metadata.get("retrieval_threshold", 0.5)

    @abstractmethod
    def decompose_question(self, question: Question) -> List[str]:
        """Decompose question into sub-questions.

        Args:
            question: The complex question

        Returns:
            List of sub-questions
        """
        pass

    @abstractmethod
    def retrieve_for_hop(
        self,
        question: str,
        evidence_so_far: List[str],
    ) -> List[RetrievalResult]:
        """Retrieve evidence for a reasoning hop.

        Args:
            question: Current sub-question
            evidence_so_far: Accumulated evidence

        Returns:
            List of retrieved documents
        """
        pass

    @abstractmethod
    def aggregate_hop_answers(
        self,
        hop_answers: List[str],
        reasoning_chain: List[str],
    ) -> str:
        """Aggregate answers from multiple hops.

        Args:
            hop_answers: Answers from each hop
            reasoning_chain: Reasoning chain

        Returns:
            Final answer
        """
        pass


class ReadingComprehensionBase(QABase[nn.Module]):
    """Abstract base class for reading comprehension systems.

    Reading comprehension extends extractive QA with deeper
    understanding of document structure and coherence.
    """

    def __init__(self, config: QAConfig):
        """Initialize reading comprehension base.

        Args:
            config: Configuration for the RC system
        """
        super().__init__(config)
        self.use_coherence = config.metadata.get("use_coherence", True)
        self.use_argument_mining = config.metadata.get("use_argument_mining", False)

    @abstractmethod
    def build_context_graph(
        self,
        context: Context,
    ) -> Dict[str, Any]:
        """Build graph representation of context.

        Args:
            context: Input context

        Returns:
            Graph data structure
        """
        pass

    @abstractmethod
    def compute_coherence(
        self,
        context_hidden: Tensor,
        graph_structure: Dict[str, Any],
    ) -> Tensor:
        """Compute coherence scores for passages.

        Args:
            context_hidden: Hidden states for context
            graph_structure: Graph structure of context

        Returns:
            Coherence scores
        """
        pass

    @abstractmethod
    def extract_arguments(
        self,
        context: Context,
    ) -> List[Dict[str, Any]]:
        """Extract arguments from context.

        Args:
            context: Input context

        Returns:
            List of extracted arguments
        """
        pass


class DomainSpecificQABase(QABase[nn.Module]):
    """Abstract base class for domain-specific QA systems.

    Domain-specific QA systems are adapted for particular domains
    like medical, legal, or scientific texts.
    """

    def __init__(self, config: QAConfig, domain: str):
        """Initialize domain-specific QA base.

        Args:
            config: Configuration for the QA system
            domain: The target domain (e.g., 'medical', 'legal')
        """
        super().__init__(config)
        self.domain = domain
        self.entity_types: List[str] = []
        self.relation_types: List[str] = []

    @abstractmethod
    def preprocess_domain(
        self,
        example: QAExample,
    ) -> QAExample:
        """Preprocess example for domain.

        Args:
            example: Input example

        Returns:
            Preprocessed example
        """
        pass

    @abstractmethod
    def postprocess_answer(
        self,
        answer: Answer,
        context: Context,
    ) -> Answer:
        """Postprocess answer for domain.

        Args:
            answer: Raw answer
            context: Original context

        Returns:
            Postprocessed answer
        """
        pass

    @abstractmethod
    def extract_entities(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        """Extract domain-specific entities.

        Args:
            text: Input text

        Returns:
            List of extracted entities with types
        """
        pass


class RetrieverBase(ABC):
    """Abstract base class for retrieval systems.

    Retriever systems find relevant documents or passages
    for a given query.
    """

    def __init__(self, config: Optional[QAConfig] = None):
        """Initialize retriever.

        Args:
            config: Optional configuration
        """
        self.config = config
        self.device = torch.device(config.device if config else "cpu")

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieval results
        """
        pass

    @abstractmethod
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of documents to index
        """
        pass


class QAPipeline(ABC):
    """Abstract pipeline that combines retrieval and QA.

    This pipeline orchestrates retrieval followed by question answering,
    typical of open-domain QA systems.
    """

    def __init__(self, config: QAConfig):
        """Initialize QA pipeline.

        Args:
            config: Configuration for the pipeline
        """
        self.config = config
        self.device = torch.device(config.device)

    @abstractmethod
    def retrieve_and_answer(
        self,
        question: str,
        top_k_docs: int = 5,
    ) -> QAPrediction:
        """Retrieve documents and answer question.

        Args:
            question: Question to answer
            top_k_docs: Number of documents to retrieve

        Returns:
            Prediction with answer
        """
        pass

    @abstractmethod
    def batch_retrieve_and_answer(
        self,
        questions: List[str],
        top_k_docs: int = 5,
    ) -> List[QAPrediction]:
        """Batch retrieve and answer.

        Args:
            questions: List of questions
            top_k_docs: Number of documents per question

        Returns:
            List of predictions
        """
        pass


def create_qa_system(
    task_type: str,
    config: QAConfig,
) -> QABase:
    """Factory function to create QA systems.

    Args:
        task_type: Type of QA task ('extractive', 'generative', 'multi_hop', etc.)
        config: Configuration for the system

    Returns:
        QA system instance
    """
    from fishstick.question_answering.extractive import BERTExtractiveQA
    from fishstick.question_answering.generative import T5GenerativeQA
    from fishstick.question_answering.multi_hop import MultiHopReasoner

    task_type_lower = task_type.lower()

    if task_type_lower == "extractive":
        return BERTExtractiveQA(config)
    elif task_type_lower == "generative":
        return T5GenerativeQA(config)
    elif task_type_lower == "multi_hop":
        return MultiHopReasoner(config)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
