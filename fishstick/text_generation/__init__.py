"""
Text Generation Module
======================

Comprehensive text generation tools for the fishstick AI framework.

Modules:
- decoding: Text decoding algorithms (beam search, nucleus, top-k, etc.)
- prompt_engineering: Prompt templates, few-shot learning, chain-of-thought
- text_quality: Evaluation metrics (perplexity, BLEU, ROUGE, diversity)
- generation_strategies: Advanced generation strategies
- lm_wrapper: Unified language model interface
- token_utils: Token utilities for generation
"""

from .decoding import (
    GreedyDecoder,
    BeamSearchDecoder,
    TopKDecoder,
    NucleusDecoder,
    TemperatureDecoder,
    DecoderBase,
    DecodingResult,
)
from .prompt_engineering import (
    PromptTemplate,
    FewShotTemplate,
    ChainOfThoughtTemplate,
    PromptFormatter,
    TemplateVariable,
)
from .text_quality import (
    PerplexityMetric,
    BleuScore,
    RougeScore,
    DiversityMetric,
    TextQualityEvaluator,
)
from .generation_strategies import (
    GenerationStrategy,
    GreedyStrategy,
    BeamSearchStrategy,
    SamplingStrategy,
    ContrastiveDecoding,
    GuidedGeneration,
)
from .lm_wrapper import (
    LanguageModel,
    GenerationConfig,
    GenerationResult,
    ModelRegistry,
)
from .token_utils import (
    Tokenizer,
    TokenStream,
    RepetitionPenalty,
    LengthPenalty,
)

__all__ = [
    "DecoderBase",
    "DecodingResult",
    "GreedyDecoder",
    "BeamSearchDecoder",
    "TopKDecoder",
    "NucleusDecoder",
    "TemperatureDecoder",
    "PromptTemplate",
    "FewShotTemplate",
    "ChainOfThoughtTemplate",
    "PromptFormatter",
    "TemplateVariable",
    "PerplexityMetric",
    "BleuScore",
    "RougeScore",
    "DiversityMetric",
    "TextQualityEvaluator",
    "GenerationStrategy",
    "GreedyStrategy",
    "BeamSearchStrategy",
    "SamplingStrategy",
    "ContrastiveDecoding",
    "GuidedGeneration",
    "LanguageModel",
    "GenerationConfig",
    "GenerationResult",
    "ModelRegistry",
    "Tokenizer",
    "TokenStream",
    "RepetitionPenalty",
    "LengthPenalty",
]
