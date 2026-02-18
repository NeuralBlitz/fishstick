"""
Model Cards Module for fishstick

Comprehensive model card management system for documenting machine learning models.
Provides structures, generation, validation, export, and visualization capabilities.
"""

from __future__ import annotations

import json
import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import inspect
import textwrap
import html


# =============================================================================
# Model Card Structure
# =============================================================================


@dataclass
class ModelDetails:
    """Core details about the model."""

    name: str
    version: str
    description: str = ""
    owners: List[str] = field(default_factory=list)
    license: str = ""
    citation: str = ""
    references: List[str] = field(default_factory=list)
    contact: str = ""
    developed_by: str = ""
    model_type: str = ""  # e.g., "Neural Network", "Decision Tree", etc.
    architecture: str = ""  # e.g., "Transformer", "CNN", etc.
    language: str = ""  # Programming language
    framework: str = ""  # e.g., "PyTorch", "TensorFlow", etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntendedUse:
    """Information about intended use cases and out-of-scope uses."""

    primary_uses: List[str] = field(default_factory=list)
    primary_users: List[str] = field(default_factory=list)
    out_of_scope_uses: List[str] = field(default_factory=list)
    out_of_scope_users: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Factor:
    """A single factor affecting model performance."""

    name: str
    description: str = ""
    categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Factors:
    """Factors affecting model performance."""

    evaluation_factors: List[Factor] = field(default_factory=list)
    demographic_factors: List[str] = field(default_factory=list)
    environmental_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_factors": [f.to_dict() for f in self.evaluation_factors],
            "demographic_factors": self.demographic_factors,
            "environmental_factors": self.environmental_factors,
        }


@dataclass
class Metric:
    """A single performance metric."""

    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    threshold: Optional[float] = None
    description: str = ""
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "unit": self.unit,
        }
        if self.confidence_interval:
            result["confidence_interval"] = list(self.confidence_interval)
        if self.threshold:
            result["threshold"] = self.threshold
        return result


@dataclass
class Metrics:
    """Collection of model performance metrics."""

    performance_metrics: List[Metric] = field(default_factory=list)
    fairness_metrics: List[Metric] = field(default_factory=list)
    robustness_metrics: List[Metric] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "performance_metrics": [m.to_dict() for m in self.performance_metrics],
            "fairness_metrics": [m.to_dict() for m in self.fairness_metrics],
            "robustness_metrics": [m.to_dict() for m in self.robustness_metrics],
        }


@dataclass
class EvaluationData:
    """Information about evaluation datasets."""

    dataset_name: str = ""
    dataset_description: str = ""
    dataset_size: int = 0
    preprocessing: str = ""
    split_strategy: str = ""  # e.g., "train/val/test", "k-fold"
    distribution: Dict[str, Any] = field(default_factory=dict)
    sensitive_attributes: List[str] = field(default_factory=list)
    data_collection_procedures: str = ""
    annotation_process: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingData:
    """Information about training datasets."""

    dataset_name: str = ""
    dataset_description: str = ""
    dataset_size: int = 0
    preprocessing: str = ""
    augmentation: str = ""
    sampling_strategy: str = ""
    class_distribution: Dict[str, int] = field(default_factory=dict)
    temporal_coverage: str = ""  # e.g., "2020-01 to 2023-12"
    geographic_coverage: str = ""
    sensitive_attributes: List[str] = field(default_factory=list)
    known_biases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuantitativeAnalysis:
    """Quantitative analysis results for a subgroup or condition."""

    group_name: str
    metrics: List[Metric] = field(default_factory=list)
    confusion_matrix: Optional[List[List[int]]] = None
    roc_curve: Optional[Dict[str, List[float]]] = None
    calibration_curve: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "group_name": self.group_name,
            "metrics": [m.to_dict() for m in self.metrics],
        }
        if self.confusion_matrix:
            result["confusion_matrix"] = self.confusion_matrix
        if self.roc_curve:
            result["roc_curve"] = self.roc_curve
        if self.calibration_curve:
            result["calibration_curve"] = self.calibration_curve
        return result


@dataclass
class QuantitativeAnalyses:
    """Collection of quantitative analyses across subgroups."""

    unitary_results: List[QuantitativeAnalysis] = field(default_factory=list)
    intersectional_results: List[QuantitativeAnalysis] = field(default_factory=list)
    comparison_baseline: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unitary_results": [r.to_dict() for r in self.unitary_results],
            "intersectional_results": [
                r.to_dict() for r in self.intersectional_results
            ],
            "comparison_baseline": self.comparison_baseline,
        }


@dataclass
class EthicalConsideration:
    """A single ethical consideration."""

    type: str  # e.g., "fairness", "privacy", "safety"
    description: str
    mitigation: str = ""
    risk_level: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EthicalConsiderations:
    """Ethical considerations and risks."""

    considerations: List[EthicalConsideration] = field(default_factory=list)
    fairness_analysis: str = ""
    privacy_considerations: str = ""
    safety_assessment: str = ""
    environmental_impact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "considerations": [c.to_dict() for c in self.considerations],
            "fairness_analysis": self.fairness_analysis,
            "privacy_considerations": self.privacy_considerations,
            "safety_assessment": self.safety_assessment,
            "environmental_impact": self.environmental_impact,
        }


@dataclass
class Caveat:
    """A single caveat or warning about the model."""

    type: str  # e.g., "performance", "usage", "generalization"
    description: str
    severity: str = "medium"  # low, medium, high, critical

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Caveats:
    """Collection of caveats and warnings."""

    caveats: List[Caveat] = field(default_factory=list)
    general_warnings: List[str] = field(default_factory=list)
    known_failures: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caveats": [c.to_dict() for c in self.caveats],
            "general_warnings": self.general_warnings,
            "known_failures": self.known_failures,
            "recommendations": self.recommendations,
        }


@dataclass
class ModelCard:
    """
    Complete model card containing all documentation for a machine learning model.

    Based on the Model Cards for Model Reporting paper (Mitchell et al., 2019).
    """

    # Core identifier
    card_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Model information
    model_details: ModelDetails = field(default_factory=ModelDetails)
    intended_use: IntendedUse = field(default_factory=IntendedUse)

    # Factors and metrics
    factors: Factors = field(default_factory=Factors)
    metrics: Metrics = field(default_factory=Metrics)

    # Data information
    evaluation_data: EvaluationData = field(default_factory=EvaluationData)
    training_data: TrainingData = field(default_factory=TrainingData)

    # Analysis
    quantitative_analyses: QuantitativeAnalyses = field(
        default_factory=QuantitativeAnalyses
    )
    ethical_considerations: EthicalConsiderations = field(
        default_factory=EthicalConsiderations
    )
    caveats: Caveats = field(default_factory=Caveats)

    # Versioning
    version_history: List[Dict[str, Any]] = field(default_factory=list)
    parent_card_id: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary."""
        return {
            "card_id": self.card_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "model_details": self.model_details.to_dict(),
            "intended_use": self.intended_use.to_dict(),
            "factors": self.factors.to_dict(),
            "metrics": self.metrics.to_dict(),
            "evaluation_data": self.evaluation_data.to_dict(),
            "training_data": self.training_data.to_dict(),
            "quantitative_analyses": self.quantitative_analyses.to_dict(),
            "ethical_considerations": self.ethical_considerations.to_dict(),
            "caveats": self.caveats.to_dict(),
            "version_history": self.version_history,
            "parent_card_id": self.parent_card_id,
            "tags": self.tags,
            "custom_fields": self.custom_fields,
        }

    def to_json(self) -> str:
        """Export model card to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()

    def compute_hash(self) -> str:
        """Compute hash of model card for versioning."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# Generation
# =============================================================================


class ModelCardGenerator(ABC):
    """Abstract base class for model card generators."""

    @abstractmethod
    def generate(self, **kwargs) -> ModelCard:
        """Generate a model card."""
        pass


class AutoGenerate(ModelCardGenerator):
    """
    Automatically generate model card from model metadata and artifacts.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        training_history: Optional[Dict] = None,
        dataset_info: Optional[Dict] = None,
        eval_results: Optional[Dict] = None,
    ):
        self.model = model
        self.training_history = training_history or {}
        self.dataset_info = dataset_info or {}
        self.eval_results = eval_results or {}

    def _extract_model_info(self) -> Dict[str, Any]:
        """Extract model information from model object."""
        info = {
            "name": "",
            "architecture": "",
            "framework": "",
            "parameters": 0,
        }

        if self.model is None:
            return info

        # Try to extract info based on framework
        model_module = type(self.model).__module__

        if "torch" in model_module:
            info["framework"] = "PyTorch"
            info["name"] = type(self.model).__name__
            try:
                import torch

                info["parameters"] = sum(p.numel() for p in self.model.parameters())
            except ImportError:
                pass
        elif "tensorflow" in model_module or "keras" in model_module:
            info["framework"] = "TensorFlow/Keras"
            info["name"] = type(self.model).__name__
            try:
                info["parameters"] = self.model.count_params()
            except:
                pass
        elif "sklearn" in model_module:
            info["framework"] = "scikit-learn"
            info["name"] = type(self.model).__name__
        else:
            info["name"] = type(self.model).__name__

        return info

    def _extract_metrics(self) -> Metrics:
        """Extract metrics from evaluation results."""
        metrics = Metrics()

        if not self.eval_results:
            return metrics

        for name, value in self.eval_results.items():
            if isinstance(value, (int, float)):
                metrics.performance_metrics.append(
                    Metric(name=name, value=float(value))
                )

        return metrics

    def generate(self, **kwargs) -> ModelCard:
        """Generate a model card automatically."""
        card = ModelCard()

        # Extract model info
        model_info = self._extract_model_info()

        card.model_details = ModelDetails(
            name=model_info.get("name", "Unknown Model"),
            version="1.0.0",
            description=kwargs.get("description", ""),
            architecture=model_info.get("architecture", ""),
            framework=model_info.get("framework", ""),
        )

        # Add metrics
        card.metrics = self._extract_metrics()

        # Add training data info
        if self.dataset_info:
            card.training_data = TrainingData(
                dataset_name=self.dataset_info.get("name", ""),
                dataset_size=self.dataset_info.get("size", 0),
                dataset_description=self.dataset_info.get("description", ""),
            )

        # Set intended use if provided
        if "intended_use" in kwargs:
            card.intended_use = kwargs["intended_use"]

        card.update_timestamp()
        return card


class TemplateFill(ModelCardGenerator):
    """
    Generate model card by filling a template.
    """

    def __init__(self, template: Dict[str, Any]):
        self.template = template

    def generate(self, **kwargs) -> ModelCard:
        """Generate model card from template."""
        card = ModelCard()

        # Fill in template values
        if "model_details" in self.template:
            details = self.template["model_details"]
            card.model_details = ModelDetails(**details)

        if "intended_use" in self.template:
            use = self.template["intended_use"]
            card.intended_use = IntendedUse(**use)

        if "factors" in self.template:
            factors = self.template["factors"]
            card.factors = Factors(**factors)

        if "ethical_considerations" in self.template:
            ethics = self.template["ethical_considerations"]
            card.ethical_considerations = EthicalConsiderations(**ethics)

        if "caveats" in self.template:
            caveats = self.template["caveats"]
            card.caveats = Caveats(**caveats)

        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(card, key):
                setattr(card, key, value)

        card.update_timestamp()
        return card


class FromCode(ModelCardGenerator):
    """
    Generate model card from Python code analysis.
    """

    def __init__(self, code: str, filename: str = "<string>"):
        self.code = code
        self.filename = filename

    def _extract_classes(self) -> List[str]:
        """Extract class definitions from code."""
        classes = []
        try:
            import ast

            tree = ast.parse(self.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except SyntaxError:
            pass
        return classes

    def _extract_imports(self) -> List[str]:
        """Extract imports from code."""
        imports = []
        try:
            import ast

            tree = ast.parse(self.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
        except SyntaxError:
            pass
        return imports

    def _detect_framework(self, imports: List[str]) -> str:
        """Detect ML framework from imports."""
        if any("torch" in imp for imp in imports):
            return "PyTorch"
        elif any("tensorflow" in imp or "keras" in imp for imp in imports):
            return "TensorFlow"
        elif any("sklearn" in imp for imp in imports):
            return "scikit-learn"
        elif any("jax" in imp for imp in imports):
            return "JAX"
        return "Unknown"

    def generate(self, **kwargs) -> ModelCard:
        """Generate model card from code analysis."""
        card = ModelCard()

        classes = self._extract_classes()
        imports = self._extract_imports()
        framework = self._detect_framework(imports)

        # Determine model name from main class or filename
        model_name = kwargs.get("model_name", classes[0] if classes else "Unknown")

        card.model_details = ModelDetails(
            name=model_name,
            version="1.0.0",
            description=f"Model implemented in {self.filename}",
            framework=framework,
            language="Python",
        )

        # Add framework references
        if framework == "PyTorch":
            card.model_details.references = ["https://pytorch.org/"]
        elif framework == "TensorFlow":
            card.model_details.references = ["https://tensorflow.org/"]

        # Add detected classes as custom field
        card.custom_fields["detected_classes"] = classes
        card.custom_fields["dependencies"] = list(
            set(imp.split(".")[0] for imp in imports)
        )

        card.update_timestamp()
        return card


# =============================================================================
# Templates
# =============================================================================


class ModelTemplate(ABC):
    """Abstract base class for model card templates."""

    @abstractmethod
    def get_template(self) -> Dict[str, Any]:
        """Return template dictionary."""
        pass

    def create_card(self, **overrides) -> ModelCard:
        """Create a model card from this template with optional overrides."""
        template = self.get_template()
        generator = TemplateFill(template)
        return generator.generate(**overrides)


class StandardTemplate(ModelTemplate):
    """Standard general-purpose model card template."""

    def get_template(self) -> Dict[str, Any]:
        return {
            "model_details": {
                "name": "",
                "version": "1.0.0",
                "description": "",
                "license": "",
                "model_type": "Machine Learning Model",
            },
            "intended_use": {
                "primary_uses": [],
                "primary_users": ["Data Scientists", "ML Engineers"],
                "out_of_scope_uses": [],
                "assumptions": ["Data is representative of the target distribution"],
                "limitations": [],
            },
            "factors": {
                "evaluation_factors": [
                    Factor(
                        name="General Performance",
                        description="Overall model performance",
                    ),
                ],
                "demographic_factors": [],
            },
            "ethical_considerations": {
                "considerations": [],
                "fairness_analysis": "Fairness evaluation should be performed before deployment.",
            },
            "caveats": {
                "general_warnings": [
                    "This model should be thoroughly tested before production use.",
                    "Performance may degrade on out-of-distribution data.",
                ],
                "recommendations": [
                    "Regularly retrain the model with new data.",
                    "Monitor model performance in production.",
                ],
            },
        }


class HealthcareTemplate(ModelTemplate):
    """Template for healthcare and medical AI models."""

    def get_template(self) -> Dict[str, Any]:
        return {
            "model_details": {
                "name": "",
                "version": "1.0.0",
                "description": "Healthcare AI model for clinical decision support.",
                "license": "Restricted - Clinical Use Only",
                "model_type": "Clinical Decision Support System",
                "contact": "clinical-team@example.com",
            },
            "intended_use": {
                "primary_uses": [
                    "Clinical decision support",
                    "Screening assistance",
                    "Risk stratification",
                ],
                "primary_users": [
                    "Licensed healthcare providers",
                    "Medical researchers",
                    "Hospital systems",
                ],
                "out_of_scope_uses": [
                    "Standalone diagnosis without clinician review",
                    "Emergency triage without supervision",
                    "Self-diagnosis by patients",
                    "Replacing clinical judgment",
                ],
                "assumptions": [
                    "Patient data is complete and accurate",
                    "Clinical context is available",
                    "Model used by qualified healthcare professionals",
                    "Regular model updates with latest clinical guidelines",
                ],
                "limitations": [
                    "Not a substitute for professional medical judgment",
                    "May not account for rare conditions",
                    "Performance varies across populations",
                    "Requires validation on local patient population",
                ],
            },
            "factors": {
                "evaluation_factors": [
                    Factor(
                        name="Age Groups",
                        description="Performance across age demographics",
                    ),
                    Factor(
                        name="Sex/Gender", description="Performance by biological sex"
                    ),
                    Factor(
                        name="Comorbidities",
                        description="Impact of pre-existing conditions",
                    ),
                    Factor(
                        name="Socioeconomic Status",
                        description="Healthcare access factors",
                    ),
                ],
                "demographic_factors": [
                    "Age",
                    "Sex",
                    "Ethnicity",
                    "Geographic location",
                ],
                "environmental_factors": [
                    "Healthcare facility type",
                    "Equipment used for data collection",
                    "Time of data collection",
                ],
            },
            "ethical_considerations": {
                "considerations": [
                    EthicalConsideration(
                        type="safety",
                        description="Incorrect predictions could lead to patient harm",
                        mitigation="Human-in-the-loop validation required",
                        risk_level="critical",
                    ),
                    EthicalConsideration(
                        type="fairness",
                        description="Potential bias in training data may affect different populations",
                        mitigation="Fairness metrics evaluated across subgroups",
                        risk_level="high",
                    ),
                    EthicalConsideration(
                        type="privacy",
                        description="Protected health information (PHI) must be safeguarded",
                        mitigation="HIPAA compliance and de-identification protocols",
                        risk_level="high",
                    ),
                ],
                "fairness_analysis": "Healthcare models require rigorous fairness evaluation across demographic groups, particularly for underrepresented populations. Disparities in healthcare access and data quality must be considered.",
                "privacy_considerations": "All patient data must be handled in compliance with HIPAA and local regulations. Data minimization principles should be applied.",
                "safety_assessment": "Clinical decision support systems must undergo rigorous validation and clinical trials before deployment. Adverse event monitoring is essential.",
            },
            "caveats": {
                "general_warnings": [
                    "This model is intended for clinical decision support only, not standalone diagnosis.",
                    "Performance metrics are based on retrospective studies and may not reflect real-world performance.",
                    "Model performance may vary across different healthcare settings and populations.",
                    "Regular recalibration may be necessary to maintain accuracy.",
                ],
                "known_failures": [
                    "Edge cases not represented in training data",
                    "Rare conditions with limited training examples",
                    "Data from outdated equipment or protocols",
                ],
                "recommendations": [
                    "Always validate model predictions with clinical expertise",
                    "Monitor for model drift and performance degradation",
                    "Report any adverse events or unexpected behaviors",
                    "Regularly update with diverse and representative data",
                    "Conduct periodic fairness audits",
                ],
            },
        }


class FinanceTemplate(ModelTemplate):
    """Template for financial services AI models."""

    def get_template(self) -> Dict[str, Any]:
        return {
            "model_details": {
                "name": "",
                "version": "1.0.0",
                "description": "Financial AI model for risk assessment and decision support.",
                "license": "Commercial - Financial Services",
                "model_type": "Financial Risk Model",
                "contact": "risk-team@example.com",
            },
            "intended_use": {
                "primary_uses": [
                    "Credit risk assessment",
                    "Fraud detection",
                    "Portfolio optimization",
                    "Market analysis",
                ],
                "primary_users": [
                    "Risk analysts",
                    "Portfolio managers",
                    "Compliance officers",
                    "Financial institutions",
                ],
                "out_of_scope_uses": [
                    "Individual investment advice without disclosure",
                    "Automated trading without human oversight",
                    "Regulatory compliance decisions without review",
                    "Discriminatory lending practices",
                ],
                "assumptions": [
                    "Historical patterns may predict future behavior",
                    "Market conditions remain relatively stable",
                    "Data quality is consistent and reliable",
                    "Regulatory environment is understood",
                ],
                "limitations": [
                    "Cannot predict black swan events",
                    "May perpetuate historical biases in lending",
                    "Performance degrades in volatile markets",
                    "Regulatory requirements vary by jurisdiction",
                ],
            },
            "factors": {
                "evaluation_factors": [
                    Factor(
                        name="Income Level",
                        description="Performance across income brackets",
                    ),
                    Factor(
                        name="Geographic Region",
                        description="Location-based performance",
                    ),
                    Factor(
                        name="Industry Sector", description="Sector-specific behavior"
                    ),
                    Factor(
                        name="Market Conditions",
                        description="Performance in different market regimes",
                    ),
                ],
                "demographic_factors": [
                    "Income level",
                    "Employment status",
                    "Geographic region",
                ],
                "environmental_factors": [
                    "Market volatility",
                    "Interest rate environment",
                    "Economic indicators",
                    "Regulatory changes",
                ],
            },
            "ethical_considerations": {
                "considerations": [
                    EthicalConsideration(
                        type="fairness",
                        description="Risk of discriminatory outcomes in lending decisions",
                        mitigation="Regular fairness audits and disparate impact analysis",
                        risk_level="high",
                    ),
                    EthicalConsideration(
                        type="privacy",
                        description="Sensitive financial data requires protection",
                        mitigation="Data encryption and access controls",
                        risk_level="high",
                    ),
                    EthicalConsideration(
                        type="safety",
                        description="Model errors could cause financial harm to consumers",
                        mitigation="Human review for adverse decisions",
                        risk_level="medium",
                    ),
                ],
                "fairness_analysis": "Financial models must comply with fair lending regulations (ECOA, FHA). Disparate impact analysis required across protected classes.",
                "privacy_considerations": "Financial data is subject to strict privacy regulations. Implement data minimization and purpose limitation.",
                "safety_assessment": "Model predictions affecting consumer financial outcomes require human oversight and appeal mechanisms.",
            },
            "caveats": {
                "general_warnings": [
                    "Past performance is not indicative of future results.",
                    "Model predictions should not be the sole basis for financial decisions.",
                    "Regulatory compliance requires additional human review.",
                    "Market conditions can change rapidly and unpredictably.",
                ],
                "known_failures": [
                    "Extreme market events (black swans)",
                    "Structural market changes",
                    "Regulatory regime shifts",
                ],
                "recommendations": [
                    "Regular model validation against current market conditions",
                    "Implement explainability for regulatory compliance",
                    "Monitor for discriminatory outcomes",
                    "Maintain human oversight for significant decisions",
                ],
            },
        }


class LegalTemplate(ModelTemplate):
    """Template for legal and compliance AI models."""

    def get_template(self) -> Dict[str, Any]:
        return {
            "model_details": {
                "name": "",
                "version": "1.0.0",
                "description": "Legal AI model for document analysis and research assistance.",
                "license": "Commercial - Legal Services",
                "model_type": "Legal Document Analysis",
                "contact": "legal-ai@example.com",
            },
            "intended_use": {
                "primary_uses": [
                    "Legal document review",
                    "Contract analysis",
                    "Legal research assistance",
                    "Regulatory compliance checking",
                ],
                "primary_users": [
                    "Licensed attorneys",
                    "Legal researchers",
                    "Compliance officers",
                    "Law firms",
                ],
                "out_of_scope_uses": [
                    "Legal advice to non-lawyers",
                    "Automated contract generation without review",
                    "Court filing preparation without attorney review",
                    "Replacing legal counsel",
                ],
                "assumptions": [
                    "Users have legal training and expertise",
                    "Documents are in supported formats and languages",
                    "Jurisdiction-specific requirements are understood",
                    "Model outputs are reviewed by qualified professionals",
                ],
                "limitations": [
                    "Not a substitute for qualified legal counsel",
                    "May not cover all jurisdictions or legal domains",
                    "Cannot provide legal advice or opinions",
                    "Accuracy depends on document quality and completeness",
                ],
            },
            "factors": {
                "evaluation_factors": [
                    Factor(
                        name="Document Type",
                        description="Performance by document category",
                    ),
                    Factor(name="Jurisdiction", description="Legal system variations"),
                    Factor(name="Language", description="Multilingual performance"),
                    Factor(name="Complexity", description="Document complexity levels"),
                ],
                "demographic_factors": [],
                "environmental_factors": [
                    "Jurisdiction",
                    "Legal domain (contracts, litigation, etc.)",
                    "Document format",
                    "Language",
                ],
            },
            "ethical_considerations": {
                "considerations": [
                    EthicalConsideration(
                        type="safety",
                        description="Incorrect legal analysis could harm clients",
                        mitigation="Attorney review required for all outputs",
                        risk_level="critical",
                    ),
                    EthicalConsideration(
                        type="privacy",
                        description="Legal documents contain privileged information",
                        mitigation="Attorney-client privilege protection protocols",
                        risk_level="high",
                    ),
                    EthicalConsideration(
                        type="fairness",
                        description="Unequal access to AI legal tools",
                        mitigation="Consider pro bono access programs",
                        risk_level="medium",
                    ),
                ],
                "fairness_analysis": "Legal AI should be evaluated for accessibility across different firm sizes and practice areas.",
                "privacy_considerations": "Legal documents are subject to attorney-client privilege and work product doctrine. Strict confidentiality required.",
                "safety_assessment": "Legal AI outputs must be reviewed by qualified attorneys. Unauthorized practice of law concerns must be addressed.",
            },
            "caveats": {
                "general_warnings": [
                    "This model does not provide legal advice.",
                    "All outputs must be reviewed by qualified legal professionals.",
                    "Laws and regulations change frequently and vary by jurisdiction.",
                    "Model cannot interpret or apply legal standards.",
                ],
                "known_failures": [
                    "Novel legal issues without precedent",
                    "Cross-jurisdictional complexities",
                    "Ambiguous contractual language",
                    "Rapidly evolving regulatory areas",
                ],
                "recommendations": [
                    "Always verify outputs against current law and regulations",
                    "Maintain attorney review for all legal decisions",
                    "Update training data with recent legal developments",
                    "Consider local counsel for jurisdiction-specific issues",
                ],
            },
        }


# =============================================================================
# Validation
# =============================================================================


class ValidationError(Exception):
    """Exception raised for model card validation errors."""

    pass


class ModelCardValidator:
    """
    Validator for model cards ensuring completeness and schema compliance.
    """

    REQUIRED_FIELDS = {
        "model_details": ["name", "version"],
        "intended_use": ["primary_uses"],
        "metrics": ["performance_metrics"],
    }

    VALID_SCHEMA_VERSIONS = ["1.0"]

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def check_completeness(self, card: ModelCard) -> Tuple[bool, List[str], List[str]]:
        """
        Check if model card has all required fields.

        Returns:
            Tuple of (is_complete, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Check model details
        details = card.model_details
        if not details.name:
            self.errors.append("Model name is required")
        if not details.version:
            self.errors.append("Model version is required")
        if not details.description:
            self.warnings.append("Model description is recommended")

        # Check intended use
        if not card.intended_use.primary_uses:
            self.warnings.append("Primary use cases should be specified")
        if not card.intended_use.limitations:
            self.warnings.append("Model limitations should be documented")

        # Check metrics
        if not card.metrics.performance_metrics:
            self.warnings.append("Performance metrics should be provided")

        # Check data information
        if not card.training_data.dataset_name:
            self.warnings.append("Training data information is recommended")
        if not card.evaluation_data.dataset_name:
            self.warnings.append("Evaluation data information is recommended")

        # Check ethical considerations
        if not card.ethical_considerations.considerations:
            self.warnings.append("Ethical considerations should be documented")

        # Check caveats
        if not card.caveats.caveats and not card.caveats.general_warnings:
            self.warnings.append("Caveats and warnings should be documented")

        is_complete = len(self.errors) == 0
        if self.strict and self.warnings:
            is_complete = False
            self.errors.extend(self.warnings)
            self.warnings = []

        return is_complete, self.errors, self.warnings

    def validate_schema(self, card: ModelCard) -> Tuple[bool, List[str]]:
        """
        Validate model card against schema version.

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        if card.schema_version not in self.VALID_SCHEMA_VERSIONS:
            errors.append(
                f"Invalid schema version: {card.schema_version}. "
                f"Valid versions: {self.VALID_SCHEMA_VERSIONS}"
            )

        # Validate card_id format
        try:
            uuid.UUID(card.card_id)
        except ValueError:
            errors.append(f"Invalid card_id format: {card.card_id}")

        # Validate timestamps
        if card.created_at > datetime.now():
            errors.append("Created timestamp is in the future")
        if card.updated_at < card.created_at:
            errors.append("Updated timestamp is before created timestamp")

        # Validate metrics values
        for metric in card.metrics.performance_metrics:
            if metric.value is None:
                errors.append(f"Metric '{metric.name}' has no value")

        return len(errors) == 0, errors

    def validate(self, card: ModelCard) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform full validation of model card.

        Returns:
            Tuple of (is_valid, validation_report)
        """
        is_complete, completeness_errors, warnings = self.check_completeness(card)
        is_schema_valid, schema_errors = self.validate_schema(card)

        report = {
            "is_complete": is_complete,
            "is_schema_valid": is_schema_valid,
            "completeness_errors": completeness_errors,
            "schema_errors": schema_errors,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat(),
        }

        is_valid = is_complete and is_schema_valid

        if not is_valid:
            raise ValidationError(
                f"Model card validation failed: "
                f"{len(completeness_errors)} completeness errors, "
                f"{len(schema_errors)} schema errors"
            )

        return is_valid, report


# =============================================================================
# Export
# =============================================================================


class ModelCardExporter:
    """
    Export model cards to various formats.
    """

    @staticmethod
    def to_markdown(card: ModelCard, include_all: bool = True) -> str:
        """
        Export model card to Markdown format.

        Args:
            card: Model card to export
            include_all: Whether to include all sections or only populated ones
        """
        lines = []

        # Header
        lines.append(f"# Model Card: {card.model_details.name}")
        lines.append("")
        lines.append(f"**Version:** {card.model_details.version}")
        lines.append(
            f"**Last Updated:** {card.updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("")

        # Model Details
        if card.model_details.description or include_all:
            lines.append("## Model Details")
            lines.append("")
            if card.model_details.description:
                lines.append(card.model_details.description)
                lines.append("")
            if card.model_details.owners:
                lines.append(f"**Owners:** {', '.join(card.model_details.owners)}")
            if card.model_details.license:
                lines.append(f"**License:** {card.model_details.license}")
            if card.model_details.framework:
                lines.append(f"**Framework:** {card.model_details.framework}")
            if card.model_details.architecture:
                lines.append(f"**Architecture:** {card.model_details.architecture}")
            lines.append("")

        # Intended Use
        if card.intended_use.primary_uses or include_all:
            lines.append("## Intended Use")
            lines.append("")
            if card.intended_use.primary_uses:
                lines.append("### Primary Uses")
                for use in card.intended_use.primary_uses:
                    lines.append(f"- {use}")
                lines.append("")
            if card.intended_use.primary_users:
                lines.append("### Primary Users")
                for user in card.intended_use.primary_users:
                    lines.append(f"- {user}")
                lines.append("")
            if card.intended_use.out_of_scope_uses:
                lines.append("### Out of Scope Uses")
                lines.append("*This model should NOT be used for:*")
                for use in card.intended_use.out_of_scope_uses:
                    lines.append(f"- {use}")
                lines.append("")
            if card.intended_use.assumptions:
                lines.append("### Assumptions")
                for assumption in card.intended_use.assumptions:
                    lines.append(f"- {assumption}")
                lines.append("")
            if card.intended_use.limitations:
                lines.append("### Limitations")
                for limitation in card.intended_use.limitations:
                    lines.append(f"- {limitation}")
                lines.append("")

        # Metrics
        if card.metrics.performance_metrics or include_all:
            lines.append("## Metrics")
            lines.append("")
            if card.metrics.performance_metrics:
                lines.append("| Metric | Value | Unit | Description |")
                lines.append("|--------|-------|------|-------------|")
                for metric in card.metrics.performance_metrics:
                    value_str = (
                        f"{metric.value:.4f}"
                        if isinstance(metric.value, float)
                        else str(metric.value)
                    )
                    lines.append(
                        f"| {metric.name} | {value_str} | {metric.unit} | {metric.description} |"
                    )
                lines.append("")

        # Training Data
        if card.training_data.dataset_name or include_all:
            lines.append("## Training Data")
            lines.append("")
            if card.training_data.dataset_name:
                lines.append(f"**Dataset:** {card.training_data.dataset_name}")
            if card.training_data.dataset_size:
                lines.append(f"**Size:** {card.training_data.dataset_size:,} samples")
            if card.training_data.dataset_description:
                lines.append(
                    f"**Description:** {card.training_data.dataset_description}"
                )
            if card.training_data.preprocessing:
                lines.append(f"**Preprocessing:** {card.training_data.preprocessing}")
            lines.append("")

        # Evaluation Data
        if card.evaluation_data.dataset_name or include_all:
            lines.append("## Evaluation Data")
            lines.append("")
            if card.evaluation_data.dataset_name:
                lines.append(f"**Dataset:** {card.evaluation_data.dataset_name}")
            if card.evaluation_data.dataset_size:
                lines.append(f"**Size:** {card.evaluation_data.dataset_size:,} samples")
            if card.evaluation_data.split_strategy:
                lines.append(
                    f"**Split Strategy:** {card.evaluation_data.split_strategy}"
                )
            lines.append("")

        # Ethical Considerations
        if card.ethical_considerations.considerations or include_all:
            lines.append("## Ethical Considerations")
            lines.append("")
            if card.ethical_considerations.considerations:
                for consideration in card.ethical_considerations.considerations:
                    lines.append(
                        f"### {consideration.type.title()} ({consideration.risk_level} risk)"
                    )
                    lines.append(consideration.description)
                    if consideration.mitigation:
                        lines.append(f"**Mitigation:** {consideration.mitigation}")
                    lines.append("")
            if card.ethical_considerations.fairness_analysis:
                lines.append("### Fairness Analysis")
                lines.append(card.ethical_considerations.fairness_analysis)
                lines.append("")

        # Caveats
        if card.caveats.caveats or card.caveats.general_warnings or include_all:
            lines.append("## Caveats and Warnings")
            lines.append("")
            if card.caveats.general_warnings:
                lines.append("### General Warnings")
                for warning in card.caveats.general_warnings:
                    lines.append(f"- ⚠️ {warning}")
                lines.append("")
            if card.caveats.known_failures:
                lines.append("### Known Failures")
                for failure in card.caveats.known_failures:
                    lines.append(f"- ❌ {failure}")
                lines.append("")
            if card.caveats.recommendations:
                lines.append("### Recommendations")
                for rec in card.caveats.recommendations:
                    lines.append(f"- ✅ {rec}")
                lines.append("")

        # Version History
        if card.version_history:
            lines.append("## Version History")
            lines.append("")
            lines.append("| Version | Date | Changes |")
            lines.append("|---------|------|---------|")
            for version in card.version_history:
                lines.append(
                    f"| {version.get('version', 'N/A')} | "
                    f"{version.get('date', 'N/A')} | "
                    f"{version.get('changes', 'N/A')} |"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_html(card: ModelCard, theme: str = "default") -> str:
        """
        Export model card to HTML format.

        Args:
            card: Model card to export
            theme: Visual theme (default, dark, minimal)
        """
        md_content = ModelCardExporter.to_markdown(card)

        # Convert markdown to HTML (simplified)
        html_content = md_content

        # Basic markdown to HTML conversion
        html_content = html.escape(html_content)
        html_content = html_content.replace("\n\n", "</p><p>")
        html_content = html_content.replace("\n", "<br>")

        # Headers
        for i in range(6, 0, -1):
            html_content = html_content.replace(f"{'#' * i} ", f"<h{i}>")
            html_content = html_content.replace(f"{'#' * i}", f"</h{i}>")

        # Bold
        html_content = html_content.replace("**", "<strong>", 1)
        html_content = html_content.replace("**", "</strong>", 1)

        # Lists
        html_content = html_content.replace("- ", "<li>")
        html_content = html_content.replace("</li><br>", "</li>")

        # Wrap in HTML
        css = ModelCardExporter._get_css(theme)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Card: {html.escape(card.model_details.name)}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="model-card">
        {html_content}
    </div>
</body>
</html>"""

    @staticmethod
    def _get_css(theme: str) -> str:
        """Get CSS styles for HTML export."""
        themes = {
            "default": """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f5f5f5; font-weight: bold; }
        li { margin: 5px 0; }
        strong { color: #2c3e50; }
            """,
            "dark": """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #1a1a1a; color: #e0e0e0; }
        h1 { color: #61dafb; border-bottom: 2px solid #61dafb; padding-bottom: 10px; }
        h2 { color: #bbbbbb; margin-top: 30px; }
        h3 { color: #888888; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #444; padding: 12px; text-align: left; }
        th { background-color: #2d2d2d; font-weight: bold; }
        li { margin: 5px 0; }
        strong { color: #61dafb; }
            """,
            "minimal": """
        body { font-family: Georgia, serif; line-height: 1.8; max-width: 700px; margin: 0 auto; padding: 40px 20px; color: #222; }
        h1 { font-size: 2em; font-weight: normal; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        h2 { font-size: 1.3em; font-weight: normal; margin-top: 40px; color: #555; }
        h3 { font-size: 1.1em; font-weight: normal; color: #777; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }
        th, td { border-bottom: 1px solid #ddd; padding: 10px; text-align: left; }
        th { font-weight: normal; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px; }
            """,
        }
        return themes.get(theme, themes["default"])

    @staticmethod
    def to_pdf(card: ModelCard, output_path: Optional[str] = None) -> bytes:
        """
        Export model card to PDF format.

        Note: This is a placeholder implementation. Full PDF support would require
        libraries like reportlab or weasyprint.

        Args:
            card: Model card to export
            output_path: Optional path to save PDF

        Returns:
            PDF content as bytes
        """
        # Placeholder implementation
        # In practice, you would use reportlab, fpdf2, or weasyprint
        content = ModelCardExporter.to_markdown(card).encode("utf-8")

        if output_path:
            with open(output_path, "wb") as f:
                f.write(content)

        return content

    @staticmethod
    def to_json(card: ModelCard, pretty: bool = True) -> str:
        """
        Export model card to JSON format.

        Args:
            card: Model card to export
            pretty: Whether to format JSON with indentation

        Returns:
            JSON string
        """
        return card.to_json()


# =============================================================================
# Visualization
# =============================================================================


class ModelCardRenderer:
    """
    Render model cards in various visual formats.
    """

    def __init__(self, card: ModelCard):
        self.card = card

    def render_summary(self) -> str:
        """Render a compact summary of the model card."""
        lines = [
            f"Model: {self.card.model_details.name} v{self.card.model_details.version}",
            f"Type: {self.card.model_details.model_type or 'Unknown'}",
            f"Framework: {self.card.model_details.framework or 'Unknown'}",
            f"Updated: {self.card.updated_at.strftime('%Y-%m-%d')}",
        ]

        if self.card.metrics.performance_metrics:
            lines.append("\nKey Metrics:")
            for metric in self.card.metrics.performance_metrics[:3]:
                lines.append(f"  - {metric.name}: {metric.value:.4f}")

        return "\n".join(lines)

    def render_metrics_table(self) -> str:
        """Render metrics as a formatted table."""
        if not self.card.metrics.performance_metrics:
            return "No metrics available"

        # Calculate column widths
        name_width = max(len(m.name) for m in self.card.metrics.performance_metrics) + 2
        value_width = 12
        unit_width = max(len(m.unit) for m in self.card.metrics.performance_metrics) + 2

        lines = []
        lines.append(
            "┌"
            + "─" * name_width
            + "┬"
            + "─" * value_width
            + "┬"
            + "─" * unit_width
            + "┐"
        )
        lines.append(
            "│"
            + "Metric".center(name_width)
            + "│"
            + "Value".center(value_width)
            + "│"
            + "Unit".center(unit_width)
            + "│"
        )
        lines.append(
            "├"
            + "─" * name_width
            + "┼"
            + "─" * value_width
            + "┼"
            + "─" * unit_width
            + "┤"
        )

        for metric in self.card.metrics.performance_metrics:
            value_str = (
                f"{metric.value:.4f}"
                if isinstance(metric.value, float)
                else str(metric.value)
            )
            lines.append(
                "│"
                + metric.name.ljust(name_width)
                + "│"
                + value_str.center(value_width)
                + "│"
                + metric.unit.ljust(unit_width)
                + "│"
            )

        lines.append(
            "└"
            + "─" * name_width
            + "┴"
            + "─" * value_width
            + "┴"
            + "─" * unit_width
            + "┘"
        )

        return "\n".join(lines)

    def render_risk_assessment(self) -> str:
        """Render risk assessment from ethical considerations."""
        if not self.card.ethical_considerations.considerations:
            return "No risk assessment available"

        risk_levels = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for consideration in self.card.ethical_considerations.considerations:
            risk_levels[consideration.risk_level] += 1

        lines = ["Risk Assessment:", ""]

        for level, count in risk_levels.items():
            if count > 0:
                symbol = (
                    "🔴"
                    if level == "critical"
                    else "🟠"
                    if level == "high"
                    else "🟡"
                    if level == "medium"
                    else "🟢"
                )
                lines.append(f"{symbol} {level.title()}: {count} consideration(s)")

        return "\n".join(lines)


class InteractiveCard:
    """
    Create interactive model card visualizations.

    Note: This creates HTML/JS-based interactive visualizations.
    """

    def __init__(self, card: ModelCard):
        self.card = card

    def generate_html(self) -> str:
        """Generate interactive HTML model card."""
        data = self.card.to_dict()

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Model Card - {html.escape(self.card.model_details.name)}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .meta {{ opacity: 0.9; font-size: 1.1em; }}
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        .tab {{
            flex: 1;
            padding: 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
            color: #666;
        }}
        .tab:hover {{ background: #e9ecef; }}
        .tab.active {{ 
            background: white; 
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
        }}
        .content {{ padding: 40px; min-height: 400px; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin: 5px;
        }}
        .risk-critical {{ background: #e74c3c; color: white; }}
        .risk-high {{ background: #e67e22; color: white; }}
        .risk-medium {{ background: #f1c40f; color: #333; }}
        .risk-low {{ background: #2ecc71; color: white; }}
        .progress-bar {{
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 1s ease;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .alert {{
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .alert-warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .alert-danger {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .alert-info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{html.escape(self.card.model_details.name)}</h1>
            <div class="meta">
                Version {self.card.model_details.version} | 
                Updated {self.card.updated_at.strftime("%Y-%m-%d")}
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">Overview</button>
            <button class="tab" onclick="showTab('metrics')">Metrics</button>
            <button class="tab" onclick="showTab('data')">Data</button>
            <button class="tab" onclick="showTab('ethics')">Ethics & Risks</button>
            <button class="tab" onclick="showTab('caveats')">Caveats</button>
        </div>
        
        <div class="content">
            <div id="overview" class="tab-content active">
                <h2>Model Overview</h2>
                <p>{html.escape(self.card.model_details.description or "No description available.")}</p>
                
                <h3>Intended Use</h3>
                <ul>
                    {"".join(f"<li>{html.escape(use)}</li>" for use in self.card.intended_use.primary_uses) if self.card.intended_use.primary_uses else "<li>No use cases specified</li>"}
                </ul>
                
                <h3>Limitations</h3>
                <div class="alert alert-warning">
                    {"".join(f"<p>⚠️ {html.escape(limit)}</p>" for limit in self.card.intended_use.limitations) if self.card.intended_use.limitations else "<p>No limitations specified</p>"}
                </div>
            </div>
            
            <div id="metrics" class="tab-content">
                <h2>Performance Metrics</h2>
                {self._generate_metrics_html()}
            </div>
            
            <div id="data" class="tab-content">
                <h2>Training Data</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Dataset Name</td><td>{html.escape(self.card.training_data.dataset_name or "N/A")}</td></tr>
                    <tr><td>Dataset Size</td><td>{self.card.training_data.dataset_size:,} samples</td></tr>
                    <tr><td>Preprocessing</td><td>{html.escape(self.card.training_data.preprocessing or "None specified")}</td></tr>
                </table>
                
                <h2>Evaluation Data</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Dataset Name</td><td>{html.escape(self.card.evaluation_data.dataset_name or "N/A")}</td></tr>
                    <tr><td>Dataset Size</td><td>{self.card.evaluation_data.dataset_size:,} samples</td></tr>
                    <tr><td>Split Strategy</td><td>{html.escape(self.card.evaluation_data.split_strategy or "None specified")}</td></tr>
                </table>
            </div>
            
            <div id="ethics" class="tab-content">
                <h2>Ethical Considerations</h2>
                {self._generate_ethics_html()}
            </div>
            
            <div id="caveats" class="tab-content">
                <h2>Caveats and Warnings</h2>
                {self._generate_caveats_html()}
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>"""

        return html_template

    def _generate_metrics_html(self) -> str:
        """Generate HTML for metrics section."""
        if not self.card.metrics.performance_metrics:
            return "<p>No metrics available</p>"

        html_parts = []
        for metric in self.card.metrics.performance_metrics:
            html_parts.append(f"""
            <div class="metric-card">
                <h4>{html.escape(metric.name)}</h4>
                <div class="metric-value">{metric.value:.4f if isinstance(metric.value, float) else metric.value}</div>
                <p>{html.escape(metric.description)}</p>
                {f'<div class="progress-bar"><div class="progress-fill" style="width: {min(metric.value * 100, 100)}%"></div></div>' if isinstance(metric.value, float) and 0 <= metric.value <= 1 else ""}
            </div>
            """)
        return "".join(html_parts)

    def _generate_ethics_html(self) -> str:
        """Generate HTML for ethics section."""
        if not self.card.ethical_considerations.considerations:
            return "<p>No ethical considerations documented</p>"

        html_parts = ['<div class="alert alert-info">']

        for consideration in self.card.ethical_considerations.considerations:
            html_parts.append(f"""
            <h4>{html.escape(consideration.type.title())} 
                <span class="risk-badge risk-{consideration.risk_level}">{consideration.risk_level.upper()}</span>
            </h4>
            <p><strong>Description:</strong> {html.escape(consideration.description)}</p>
            {f"<p><strong>Mitigation:</strong> {html.escape(consideration.mitigation)}</p>" if consideration.mitigation else ""}
            <hr>
            """)

        html_parts.append("</div>")
        return "".join(html_parts)

    def _generate_caveats_html(self) -> str:
        """Generate HTML for caveats section."""
        html_parts = []

        if self.card.caveats.general_warnings:
            html_parts.append(
                '<div class="alert alert-warning"><h4>⚠️ General Warnings</h4>'
            )
            for warning in self.card.caveats.general_warnings:
                html_parts.append(f"<p>{html.escape(warning)}</p>")
            html_parts.append("</div>")

        if self.card.caveats.known_failures:
            html_parts.append(
                '<div class="alert alert-danger"><h4>❌ Known Failures</h4>'
            )
            for failure in self.card.caveats.known_failures:
                html_parts.append(f"<p>{html.escape(failure)}</p>")
            html_parts.append("</div>")

        if self.card.caveats.recommendations:
            html_parts.append(
                '<div class="alert alert-info"><h4>✅ Recommendations</h4>'
            )
            for rec in self.card.caveats.recommendations:
                html_parts.append(f"<p>{html.escape(rec)}</p>")
            html_parts.append("</div>")

        return "".join(html_parts) if html_parts else "<p>No caveats documented</p>"

    def save(self, output_path: str) -> None:
        """Save interactive HTML to file."""
        html_content = self.generate_html()
        with open(output_path, "w") as f:
            f.write(html_content)


# =============================================================================
# Versioning
# =============================================================================


def version_card(
    card: ModelCard,
    new_version: str,
    changes: str,
    author: str = "",
) -> ModelCard:
    """
    Create a new version of a model card.

    Args:
        card: Original model card
        new_version: New version string
        changes: Description of changes
        author: Author of the changes

    Returns:
        New model card with updated version and history
    """
    # Create deep copy of card
    import copy

    new_card = copy.deepcopy(card)

    # Add current version to history
    new_card.version_history.append(
        {
            "version": card.model_details.version,
            "date": card.updated_at.isoformat(),
            "changes": "Previous version",
            "hash": card.compute_hash(),
        }
    )

    # Update version and timestamps
    new_card.model_details.version = new_version
    new_card.card_id = str(uuid.uuid4())
    new_card.parent_card_id = card.card_id
    new_card.created_at = datetime.now()
    new_card.update_timestamp()

    # Add version entry
    new_card.version_history.append(
        {
            "version": new_version,
            "date": new_card.updated_at.isoformat(),
            "changes": changes,
            "author": author,
            "hash": new_card.compute_hash(),
        }
    )

    return new_card


def compare_cards(card1: ModelCard, card2: ModelCard) -> Dict[str, Any]:
    """
    Compare two model cards and identify differences.

    Args:
        card1: First model card
        card2: Second model card

    Returns:
        Dictionary containing differences
    """
    differences = {
        "added": [],
        "removed": [],
        "modified": [],
        "unchanged": [],
    }

    def compare_dicts(d1: Dict, d2: Dict, path: str = "") -> None:
        """Recursively compare dictionaries."""
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in d1:
                differences["added"].append(
                    {
                        "path": current_path,
                        "value": d2[key],
                    }
                )
            elif key not in d2:
                differences["removed"].append(
                    {
                        "path": current_path,
                        "value": d1[key],
                    }
                )
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                compare_dicts(d1[key], d2[key], current_path)
            elif d1[key] != d2[key]:
                differences["modified"].append(
                    {
                        "path": current_path,
                        "old_value": d1[key],
                        "new_value": d2[key],
                    }
                )
            else:
                differences["unchanged"].append(current_path)

    dict1 = card1.to_dict()
    dict2 = card2.to_dict()

    # Remove timestamps for comparison
    for d in [dict1, dict2]:
        d.pop("created_at", None)
        d.pop("updated_at", None)
        d.pop("card_id", None)

    compare_dicts(dict1, dict2)

    return differences


# =============================================================================
# Utilities
# =============================================================================


class ModelCardRegistry:
    """
    Registry for managing multiple model cards.
    """

    def __init__(self):
        self._cards: Dict[str, ModelCard] = {}
        self._name_index: Dict[str, List[str]] = {}  # name -> card_ids
        self._tags_index: Dict[str, Set[str]] = {}  # tag -> card_ids

    def register(self, card: ModelCard) -> None:
        """Register a model card."""
        self._cards[card.card_id] = card

        # Index by name
        name = card.model_details.name
        if name not in self._name_index:
            self._name_index[name] = []
        if card.card_id not in self._name_index[name]:
            self._name_index[name].append(card.card_id)

        # Index by tags
        for tag in card.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(card.card_id)

    def get(self, card_id: str) -> Optional[ModelCard]:
        """Get a model card by ID."""
        return self._cards.get(card_id)

    def get_by_name(self, name: str) -> List[ModelCard]:
        """Get all model cards with a given name."""
        card_ids = self._name_index.get(name, [])
        return [self._cards[card_id] for card_id in card_ids if card_id in self._cards]

    def get_by_tag(self, tag: str) -> List[ModelCard]:
        """Get all model cards with a given tag."""
        card_ids = self._tags_index.get(tag, set())
        return [self._cards[card_id] for card_id in card_ids if card_id in self._cards]

    def list_all(self) -> List[ModelCard]:
        """List all registered model cards."""
        return list(self._cards.values())

    def search(self, query: str) -> List[ModelCard]:
        """Search model cards by name or description."""
        results = []
        query_lower = query.lower()

        for card in self._cards.values():
            if (
                query_lower in card.model_details.name.lower()
                or query_lower in card.model_details.description.lower()
                or any(query_lower in tag.lower() for tag in card.tags)
            ):
                results.append(card)

        return results

    def remove(self, card_id: str) -> bool:
        """Remove a model card from the registry."""
        if card_id not in self._cards:
            return False

        card = self._cards[card_id]

        # Remove from name index
        name = card.model_details.name
        if name in self._name_index:
            self._name_index[name].remove(card_id)
            if not self._name_index[name]:
                del self._name_index[name]

        # Remove from tags index
        for tag in card.tags:
            if tag in self._tags_index:
                self._tags_index[tag].discard(card_id)
                if not self._tags_index[tag]:
                    del self._tags_index[tag]

        # Remove card
        del self._cards[card_id]
        return True

    def get_version_history(self, name: str) -> List[ModelCard]:
        """Get all versions of a model by name, sorted by version."""
        cards = self.get_by_name(name)
        return sorted(cards, key=lambda c: c.model_details.version, reverse=True)

    def export_registry(self, output_path: str) -> None:
        """Export all cards to a JSON file."""
        data = {
            "cards": [card.to_dict() for card in self._cards.values()],
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_cards": len(self._cards),
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def import_registry(self, input_path: str) -> int:
        """Import cards from a JSON file."""
        with open(input_path, "r") as f:
            data = json.load(f)

        count = 0
        for card_dict in data.get("cards", []):
            # Convert timestamps back to datetime
            if "created_at" in card_dict:
                card_dict["created_at"] = datetime.fromisoformat(
                    card_dict["created_at"]
                )
            if "updated_at" in card_dict:
                card_dict["updated_at"] = datetime.fromisoformat(
                    card_dict["updated_at"]
                )

            card = ModelCard(**card_dict)
            self.register(card)
            count += 1

        return count


def publish_card(
    card: ModelCard, destination: str, format: str = "markdown", **kwargs
) -> str:
    """
    Publish a model card to a destination.

    Args:
        card: Model card to publish
        destination: Output path or URL
        format: Export format (markdown, html, json, pdf)
        **kwargs: Additional options for export

    Returns:
        Path or URL where card was published
    """
    exporter = ModelCardExporter()

    if format == "markdown":
        content = exporter.to_markdown(card, **kwargs)
        with open(destination, "w") as f:
            f.write(content)
    elif format == "html":
        content = exporter.to_html(card, **kwargs)
        with open(destination, "w") as f:
            f.write(content)
    elif format == "json":
        content = exporter.to_json(card)
        with open(destination, "w") as f:
            f.write(content)
    elif format == "pdf":
        exporter.to_pdf(card, destination)
    elif format == "interactive":
        interactive = InteractiveCard(card)
        interactive.save(destination)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return destination


# =============================================================================
# Convenience Functions
# =============================================================================


def create_model_card(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    template: Optional[str] = None,
    **kwargs,
) -> ModelCard:
    """
    Create a new model card with optional template.

    Args:
        name: Model name
        version: Model version
        description: Model description
        template: Template type (standard, healthcare, finance, legal)
        **kwargs: Additional fields to populate

    Returns:
        New ModelCard instance
    """
    if template:
        templates = {
            "standard": StandardTemplate(),
            "healthcare": HealthcareTemplate(),
            "finance": FinanceTemplate(),
            "legal": LegalTemplate(),
        }

        if template not in templates:
            raise ValueError(f"Unknown template: {template}")

        card = templates[template].create_card(
            model_details={
                "name": name,
                "version": version,
                "description": description,
            },
            **kwargs,
        )
    else:
        card = ModelCard()
        card.model_details = ModelDetails(
            name=name,
            version=version,
            description=description,
        )

    card.update_timestamp()
    return card


def load_model_card(path: str) -> ModelCard:
    """Load a model card from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Convert timestamps
    if "created_at" in data:
        data["created_at"] = datetime.fromisoformat(data["created_at"])
    if "updated_at" in data:
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])

    # Reconstruct nested objects
    if "model_details" in data:
        data["model_details"] = ModelDetails(**data["model_details"])
    if "intended_use" in data:
        data["intended_use"] = IntendedUse(**data["intended_use"])
    if "factors" in data:
        factors_data = data["factors"]
        if "evaluation_factors" in factors_data:
            factors_data["evaluation_factors"] = [
                Factor(**f) for f in factors_data["evaluation_factors"]
            ]
        data["factors"] = Factors(**factors_data)
    if "metrics" in data:
        metrics_data = data["metrics"]
        for key in ["performance_metrics", "fairness_metrics", "robustness_metrics"]:
            if key in metrics_data:
                metrics_data[key] = [Metric(**m) for m in metrics_data[key]]
        data["metrics"] = Metrics(**metrics_data)
    if "training_data" in data:
        data["training_data"] = TrainingData(**data["training_data"])
    if "evaluation_data" in data:
        data["evaluation_data"] = EvaluationData(**data["evaluation_data"])
    if "quantitative_analyses" in data:
        analyses_data = data["quantitative_analyses"]
        for key in ["unitary_results", "intersectional_results"]:
            if key in analyses_data:
                analyses_data[key] = [
                    QuantitativeAnalysis(**a) for a in analyses_data[key]
                ]
        data["quantitative_analyses"] = QuantitativeAnalyses(**analyses_data)
    if "ethical_considerations" in data:
        ethics_data = data["ethical_considerations"]
        if "considerations" in ethics_data:
            ethics_data["considerations"] = [
                EthicalConsideration(**c) for c in ethics_data["considerations"]
            ]
        data["ethical_considerations"] = EthicalConsiderations(**ethics_data)
    if "caveats" in data:
        caveats_data = data["caveats"]
        if "caveats" in caveats_data:
            caveats_data["caveats"] = [Caveat(**c) for c in caveats_data["caveats"]]
        data["caveats"] = Caveats(**caveats_data)

    return ModelCard(**data)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Model Card Structure
    "ModelCard",
    "ModelDetails",
    "IntendedUse",
    "Factors",
    "Factor",
    "Metrics",
    "Metric",
    "EvaluationData",
    "TrainingData",
    "QuantitativeAnalyses",
    "QuantitativeAnalysis",
    "EthicalConsiderations",
    "EthicalConsideration",
    "Caveats",
    "Caveat",
    # Generation
    "ModelCardGenerator",
    "AutoGenerate",
    "TemplateFill",
    "FromCode",
    # Templates
    "ModelTemplate",
    "StandardTemplate",
    "HealthcareTemplate",
    "FinanceTemplate",
    "LegalTemplate",
    # Validation
    "ModelCardValidator",
    "ValidationError",
    # Export
    "ModelCardExporter",
    # Visualization
    "ModelCardRenderer",
    "InteractiveCard",
    # Versioning
    "version_card",
    "compare_cards",
    # Utilities
    "ModelCardRegistry",
    "publish_card",
    "create_model_card",
    "load_model_card",
]
