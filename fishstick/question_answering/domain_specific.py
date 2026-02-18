"""
Domain-specific Question Answering Implementation

This module provides implementations for domain-specific QA systems including
medical, legal, and scientific QA with domain adaptation.

Author: Fishstick AI Framework
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fishstick.question_answering.types import (
    QAExample,
    QAPrediction,
    Answer,
    AnswerType,
    Context,
    Question,
    QAConfig,
    QATaskType,
)
from fishstick.question_answering.base import DomainSpecificQABase


class DomainVocabulary:
    """Domain-specific vocabulary management.

    Manages domain-specific terminology and entities.
    """

    def __init__(self, domain: str) -> None:
        """Initialize DomainVocabulary.

        Args:
            domain: Domain name (e.g., 'medical', 'legal')
        """
        self.domain = domain
        self.terms: Dict[str, Dict[str, Any]] = {}

    def add_term(
        self,
        term: str,
        definition: str,
        category: str = "general",
    ) -> None:
        """Add a domain term.

        Args:
            term: Term text
            definition: Term definition
            category: Term category
        """
        self.terms[term.lower()] = {
            "definition": definition,
            "category": category,
        }

    def get_definition(self, term: str) -> Optional[str]:
        """Get term definition.

        Args:
            term: Term to look up

        Returns:
            Term definition or None
        """
        return self.terms.get(term.lower(), {}).get("definition")

    def contains_term(self, text: str) -> bool:
        """Check if text contains domain terms.

        Args:
            text: Text to check

        Returns:
            True if contains domain terms
        """
        text_lower = text.lower()
        return any(term in text_lower for term in self.terms.keys())

    def extract_terms(self, text: str) -> List[str]:
        """Extract domain terms from text.

        Args:
            text: Text to extract from

        Returns:
            List of found terms
        """
        text_lower = text.lower()
        found = []

        for term in self.terms.keys():
            if term in text_lower:
                found.append(term)

        return found


class MedicalVocabulary(DomainVocabulary):
    """Medical domain vocabulary."""

    def __init__(self) -> None:
        super().__init__("medical")
        self._initialize_medical_terms()

    def _initialize_medical_terms(self) -> None:
        """Initialize common medical terms."""
        medical_terms = {
            "myocardial infarction": "heart attack",
            "hypertension": "high blood pressure",
            "diabetes": "blood sugar disorder",
            "pneumonia": "lung infection",
            "carcinoma": "cancer",
            "benign": "non-cancerous",
            "malignant": "cancerous",
            "prognosis": "expected outcome",
            "etiology": "cause of disease",
            "pathogenesis": "development of disease",
            "syndrome": "set of symptoms",
            "acute": "sudden onset",
            "chronic": "long-lasting",
        }

        for term, definition in medical_terms.items():
            self.add_term(term, definition, "medical_condition")


class LegalVocabulary(DomainVocabulary):
    """Legal domain vocabulary."""

    def __init__(self) -> None:
        super().__init__("legal")
        self._initialize_legal_terms()

    def _initialize_legal_terms(self) -> None:
        """Initialize common legal terms."""
        legal_terms = {
            "plaintiff": "person who files lawsuit",
            "defendant": "person being sued",
            "jurisdiction": "court authority",
            "statute": "written law",
            "precedent": "prior court decision",
            "liability": "legal responsibility",
            "negligence": "failure to care",
            "contract": "legal agreement",
            "tort": "civil wrong",
            "appeal": "request for review",
            "plaintiff": "complaining party",
            "appellant": "party appealing",
            "appellee": "party responding to appeal",
        }

        for term, definition in legal_terms.items():
            self.add_term(term, definition, "legal_term")


class ScientificVocabulary(DomainVocabulary):
    """Scientific domain vocabulary."""

    def __init__(self) -> None:
        super().__init__("scientific")
        self._initialize_scientific_terms()

    def _initialize_scientific_terms(self) -> None:
        """Initialize common scientific terms."""
        scientific_terms = {
            "hypothesis": "testable explanation",
            "methodology": "research method",
            "correlation": "statistical relationship",
            "causation": "cause-effect relationship",
            "variable": "measurable factor",
            "control": "baseline condition",
            "replication": "repeating study",
            "peer review": "expert evaluation",
            "statistically significant": "unlikely due to chance",
            "sample size": "number of subjects",
            "confidence interval": "estimate range",
            "p-value": "statistical probability",
        }

        for term, definition in scientific_terms.items():
            self.add_term(term, definition, "scientific_term")


class DomainAdaptationQA(nn.Module):
    """Domain Adaptation for QA.

    Base class for adapting QA models to specific domains.
    """

    def __init__(
        self,
        domain: str,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ) -> None:
        """Initialize DomainAdaptationQA.

        Args:
            domain: Target domain
            hidden_size: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        self.domain = domain
        self.hidden_size = hidden_size

        self.domain_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10),
        )

        self.domain_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

    def encode_domain_features(
        self,
        text_hidden: Tensor,
    ) -> Tensor:
        """Encode domain-specific features.

        Args:
            text_hidden: [batch, seq_len, hidden]

        Returns:
            Domain features
        """
        output, (h_n, _) = self.domain_encoder(text_hidden)
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        """Forward pass for domain adaptation.

        Args:
            hidden_states: [batch, seq_len, hidden]

        Returns:
            Adapted features
        """
        domain_features = self.encode_domain_features(hidden_states)
        domain_logits = self.domain_classifier(domain_features)

        return domain_features


class MedicalQASystem(DomainSpecificQABase[nn.Module]):
    """Medical Domain QA System.

    Specialized QA for medical text.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize Medical QA System.

        Args:
            config: QA configuration
        """
        super().__init__(config, domain="medical")

        self.vocabulary = MedicalVocabulary()
        self.entity_types = [
            "disease",
            "symptom",
            "treatment",
            "medication",
            "anatomy",
            "procedure",
        ]

        self.hidden_size = config.metadata.get("hidden_size", 768)

        self.domain_adapter = DomainAdaptationQA(
            domain="medical",
            hidden_size=self.hidden_size,
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.medical_encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size // 2,
            num_layers=4,
            bidirectional=True,
            batch_first=True,
        )

        self.answer_extractor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.metadata.get("dropout", 0.1)),
            nn.Linear(self.hidden_size, 2),
        )

    def preprocess_domain(
        self,
        example: QAExample,
    ) -> QAExample:
        """Preprocess example for medical domain.

        Args:
            example: Input example

        Returns:
            Preprocessed example
        """
        if isinstance(example.context, Context):
            terms = self.vocabulary.extract_terms(example.context.text)
            example.metadata["domain_terms"] = terms

        return example

    def postprocess_answer(
        self,
        answer: Answer,
        context: Context,
    ) -> Answer:
        """Postprocess answer for medical domain.

        Args:
            answer: Raw answer
            context: Original context

        Returns:
            Postprocessed answer
        """
        answer.metadata["domain"] = "medical"

        if isinstance(context, Context):
            answer.metadata["confidence_modifiers"] = {
                "source_uncertainty": "not_specified",
                "evidence_quality": "moderate",
            }

        return answer

    def extract_entities(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        """Extract medical entities from text.

        Args:
            text: Input text

        Returns:
            List of extracted entities with types
        """
        entities = []

        medical_patterns = {
            "disease": r"\b(\w+itis|\w+osis|\w+emia)\b",
            "symptom": r"\b(pain|fever|cough|nausea)\b",
            "treatment": r"\b(therapy|treatment|surgery)\b",
            "medication": r"\b(\w+cin|\w+pril|\w+statin)\b",
        }

        for entity_type, pattern in medical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(
                    {
                        "text": match,
                        "type": entity_type,
                    }
                )

        return entities

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
        c_text = context.text if isinstance(context, Context) else context
        q_text = question.text if isinstance(question, Question) else question

        example = QAExample(
            id="temp",
            question=question,
            context=context,
        )
        example = self.preprocess_domain(example)

        context_tokens = c_text.split()
        context_hidden = torch.randn(1, len(context_tokens), self.hidden_size)

        output, (h_n, _) = self.medical_encoder(context_hidden)

        span_logits = self.answer_extractor(output)

        start_idx = span_logits[0, :, 0].argmax().item()
        end_idx = span_logits[0, :, 1].argmax().item()

        answer_tokens = context_tokens[
            start_idx : min(end_idx + 1, len(context_tokens))
        ]
        answer_text = " ".join(answer_tokens)

        answer = Answer(
            text=answer_text,
            type=AnswerType.SPAN,
            confidence=0.8,
        )

        answer = self.postprocess_answer(
            answer, context if isinstance(context, Context) else Context(text=c_text)
        )

        return answer

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
        predictions = []

        for example in examples:
            answer = self.forward(example.question, example.context)

            pred = QAPrediction(
                id=example.id,
                question=example.question.text
                if isinstance(example.question, Question)
                else example.question,
                answer=answer,
                context_used=example.context.text
                if isinstance(example.context, Context)
                else example.context,
            )
            predictions.append(pred)

        return predictions

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
        raise NotImplementedError("Training not implemented. Use QATrainer.")

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        torch.save(
            {
                "domain_adapter": self.domain_adapter.state_dict(),
                "medical_encoder": self.medical_encoder.state_dict(),
                "answer_extractor": self.answer_extractor.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.domain_adapter.load_state_dict(checkpoint["domain_adapter"])
        self.medical_encoder.load_state_dict(checkpoint["medical_encoder"])
        self.answer_extractor.load_state_dict(checkpoint["answer_extractor"])


class LegalQASystem(MedicalQASystem):
    """Legal Domain QA System.

    Specialized QA for legal text.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize Legal QA System.

        Args:
            config: QA configuration
        """
        super().__init__(config)
        self.domain = "legal"

        self.vocabulary = LegalVocabulary()
        self.entity_types = [
            "party",
            "court",
            "statute",
            "case",
            "document",
            "obligation",
        ]

    def preprocess_domain(
        self,
        example: QAExample,
    ) -> QAExample:
        """Preprocess example for legal domain.

        Args:
            example: Input example

        Returns:
            Preprocessed example
        """
        if isinstance(example.context, Context):
            terms = self.vocabulary.extract_terms(example.context.text)
            example.metadata["legal_terms"] = terms
            example.metadata["case_references"] = re.findall(
                r"\d+\s+\w+\s+\d+",
                example.context.text,
            )

        return example

    def postprocess_answer(
        self,
        answer: Answer,
        context: Context,
    ) -> Answer:
        """Postprocess answer for legal domain.

        Args:
            answer: Raw answer
            context: Original context

        Returns:
            Postprocessed answer
        """
        answer.metadata["domain"] = "legal"

        if isinstance(context, Context):
            citations = re.findall(r"\d+\s+\w+\s+\d+", context.text)
            answer.metadata["relevant_citations"] = citations

        return answer

    def extract_entities(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        """Extract legal entities from text.

        Args:
            text: Input text

        Returns:
            List of extracted entities with types
        """
        entities = []

        legal_patterns = {
            "party": r"\b(plaintiff|defendant|appellant|appellee|petitioner|respondent)\b",
            "court": r"\b(court|judge|justice|tribunal)\b",
            "statute": r"\b(statute|code|act|regulation)\b",
            "case": r"\bv\.\s+\w+\b",
        }

        for entity_type, pattern in legal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(
                    {
                        "text": match,
                        "type": entity_type,
                    }
                )

        return entities


class ScientificQASystem(MedicalQASystem):
    """Scientific Domain QA System.

    Specialized QA for scientific text.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize Scientific QA System.

        Args:
            config: QA configuration
        """
        super().__init__(config)
        self.domain = "scientific"

        self.vocabulary = ScientificVocabulary()
        self.entity_types = [
            "hypothesis",
            "method",
            "finding",
            "variable",
            "parameter",
        ]

    def preprocess_domain(
        self,
        example: QAExample,
    ) -> QAExample:
        """Preprocess example for scientific domain.

        Args:
            example: Input example

        Returns:
            Preprocessed example
        """
        if isinstance(example.context, Context):
            terms = self.vocabulary.extract_terms(example.context.text)
            example.metadata["scientific_terms"] = terms
            example.metadata["numeric_findings"] = re.findall(
                r"\d+\.?\d*%?",
                example.context.text,
            )

        return example

    def postprocess_answer(
        self,
        answer: Answer,
        context: Context,
    ) -> Answer:
        """Postprocess answer for scientific domain.

        Args:
            answer: Raw answer
            context: Original context

        Returns:
            Postprocessed answer
        """
        answer.metadata["domain"] = "scientific"

        if isinstance(context, Context):
            answer.metadata["evidence_level"] = "empirical"

        return answer

    def extract_entities(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        """Extract scientific entities from text.

        Args:
            text: Input text

        Returns:
            List of extracted entities with types
        """
        entities = []

        scientific_patterns = {
            "hypothesis": r"\bhypothesis\b",
            "method": r"\b(methodology|experiment|study|trial)\b",
            "finding": r"\bfinding|result|conclusion\b",
            "variable": r"\b(variable|factor|parameter)\b",
        }

        for entity_type, pattern in scientific_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(
                    {
                        "text": match,
                        "type": entity_type,
                    }
                )

        return entities
