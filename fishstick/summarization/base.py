"""
Base Classes and Data Structures for Summarization
===================================================

Provides abstract base classes and common data structures used across
all summarization modules.

Author: Fishstick Team
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Union
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class SummarizationMethod(Enum):
    """Enumeration of available summarization methods."""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    LEAD_BASED = "lead_based"
    NEURAL = "neural"
    MULTI_DOCUMENT = "multi_document"
    HYBRID = "hybrid"


class SummaryStyle(Enum):
    """Style of summary to generate."""

    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    BULLET = "bullet"
    PARAGRAPH = "paragraph"


@dataclass
class Document:
    """Represents a document for summarization.

    Attributes:
        text: The raw text content of the document
        doc_id: Unique identifier for the document
        metadata: Additional metadata about the document
        sentences: Pre-split sentences
        embeddings: Pre-computed embeddings (optional)
    """

    text: str
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sentences: List[str] = field(default_factory=list)
    embeddings: Optional[NDArray[np.float32]] = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = f"doc_{hash(self.text) % 100000}"

    def get_sentences(self, force: bool = False) -> List[str]:
        """Get sentences from the document.

        Args:
            force: Force re-tokenization if sentences already exist

        Returns:
            List of sentences
        """
        if not self.sentences or force:
            from .utils import SentenceTokenizer

            tokenizer = SentenceTokenizer()
            self.sentences = tokenizer.tokenize(self.text)
        return self.sentences


@dataclass
class SummaryConfig:
    """Configuration for summarization.

    Attributes:
        method: The summarization method to use
        max_length: Maximum length of the summary in words
        min_length: Minimum length of the summary in words
        num_sentences: Number of sentences for extractive methods
        compression_ratio: Target compression ratio (0-1)
        style: Style of the summary
        language: Language of the input text
        preserve_keywords: Whether to preserve important keywords
        include_title: Whether to include the document title
        custom_stopwords: Additional stopwords to filter
    """

    method: SummarizationMethod = SummarizationMethod.EXTRACTIVE
    max_length: int = 200
    min_length: int = 30
    num_sentences: int = 3
    compression_ratio: float = 0.3
    style: SummaryStyle = SummaryStyle.STANDARD
    language: str = "en"
    preserve_keywords: bool = True
    include_title: bool = False
    custom_stopwords: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.compression_ratio <= 0 or self.compression_ratio > 1:
            self.compression_ratio = 0.3


@dataclass
class SummaryResult:
    """Container for summarization results.

    Attributes:
        summary: The generated summary text
        original_text: The original input text
        method: The method used to generate the summary
        sentences: Selected sentences (for extractive methods)
        sentence_scores: Scores for each sentence
        word_count: Number of words in the summary
        compression_ratio: Actual compression ratio achieved
        metadata: Additional metadata about the summarization
    """

    summary: str
    original_text: str
    method: SummarizationMethod
    sentences: List[str] = field(default_factory=list)
    sentence_scores: List[float] = field(default_factory=list)
    word_count: int = 0
    compression_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.summary.split())
        if self.compression_ratio == 0.0 and self.original_text:
            orig_len = len(self.original_text.split())
            if orig_len > 0:
                self.compression_ratio = self.word_count / orig_len


class Preprocessor(ABC):
    """Abstract base class for text preprocessing."""

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess text before summarization.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        pass

    def __call__(self, text: str) -> str:
        return self.preprocess(text)


class SummarizerBase(ABC):
    """Abstract base class for all summarizers.

    This class defines the interface that all summarization
    implementations must follow.
    """

    def __init__(self, config: Optional[SummaryConfig] = None):
        """Initialize the summarizer.

        Args:
            config: Summarization configuration
        """
        self.config = config or SummaryConfig()

    @abstractmethod
    def summarize(
        self,
        text: Union[str, Document, List[Union[str, Document]]],
        config: Optional[SummaryConfig] = None,
    ) -> SummaryResult:
        """Generate a summary from the input text.

        Args:
            text: Input text or Document to summarize
            config: Optional configuration override

        Returns:
            SummaryResult containing the generated summary

        Raises:
            ValueError: If input text is empty or invalid
        """
        pass

    def _validate_input(
        self, text: Union[str, Document, List[Union[str, Document]]]
    ) -> List[Document]:
        """Validate and convert input to list of Documents.

        Args:
            text: Input text(s)

        Returns:
            List of Document objects

        Raises:
            ValueError: If input is invalid
        """
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("Input text cannot be empty")
            return [Document(text=text)]
        elif isinstance(text, Document):
            return [text]
        elif isinstance(text, list):
            docs = []
            for item in text:
                if isinstance(item, str):
                    docs.append(Document(text=item))
                elif isinstance(item, Document):
                    docs.append(item)
                else:
                    raise ValueError(f"Invalid input type: {type(item)}")
            return docs
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def _create_result(
        self,
        summary: str,
        original_text: str,
        method: SummarizationMethod,
        sentences: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        **metadata: Any,
    ) -> SummaryResult:
        """Create a SummaryResult object.

        Args:
            summary: Generated summary text
            original_text: Original input text
            method: Method used
            sentences: Selected sentences
            scores: Sentence scores
            metadata: Additional metadata

        Returns:
            SummaryResult object
        """
        return SummaryResult(
            summary=summary,
            original_text=original_text,
            method=method,
            sentences=sentences or [],
            sentence_scores=scores or [],
            metadata=metadata,
        )


class BatchSummarizerBase(SummarizerBase):
    """Base class for batch summarization."""

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        batch_size: int = 8,
    ):
        """Initialize batch summarizer.

        Args:
            config: Summarization configuration
            batch_size: Number of documents to process at once
        """
        super().__init__(config)
        self.batch_size = batch_size

    @abstractmethod
    def summarize_batch(
        self,
        texts: List[str],
        config: Optional[SummaryConfig] = None,
    ) -> List[SummaryResult]:
        """Summarize a batch of documents.

        Args:
            texts: List of input texts
            config: Optional configuration override

        Returns:
            List of SummaryResult objects
        """
        pass

    def summarize(
        self,
        text: Union[str, Document, List[Union[str, Document]]],
        config: Optional[SummaryConfig] = None,
    ) -> Union[SummaryResult, List[SummaryResult]]:
        """Generate summaries from input text(s).

        Args:
            text: Input text(s) to summarize
            config: Optional configuration override

        Returns:
            SummaryResult or list of SummaryResults
        """
        docs = self._validate_input(text)
        texts = [doc.text for doc in docs]

        if len(texts) == 1:
            return self.summarize_batch(texts, config)[0]

        return self.summarize_batch(texts, config)


def create_summarizer(
    method: SummarizationMethod,
    config: Optional[SummaryConfig] = None,
    **kwargs: Any,
) -> SummarizerBase:
    """Factory function to create summarizers.

    Args:
        method: The summarization method to use
        config: Summarization configuration
        **kwargs: Additional arguments for specific summarizers

    Returns:
        SummarizerBase implementation

    Raises:
        ValueError: If method is unknown
    """
    if method == SummarizationMethod.EXTRACTIVE:
        from .extractive import TFIDFSummarizer

        return TFIDFSummarizer(config, **kwargs)
    elif method == SummarizationMethod.ABSTRACTIVE:
        from .abstractive import Seq2SeqSummarizer

        return Seq2SeqSummarizer(config, **kwargs)
    elif method == SummarizationMethod.LEAD_BASED:
        from .lead_based import LeadBasedSummarizer

        return LeadBasedSummarizer(config, **kwargs)
    elif method == SummarizationMethod.NEURAL:
        from .neural import TransformerSummarizer

        return TransformerSummarizer(config, **kwargs)
    elif method == SummarizationMethod.MULTI_DOCUMENT:
        from .multi_document import MultiDocSummarizer

        return MultiDocSummarizer(config, **kwargs)
    else:
        raise ValueError(f"Unknown summarization method: {method}")
