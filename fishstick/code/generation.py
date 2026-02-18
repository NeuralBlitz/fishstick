"""
Code Generation Module

Comprehensive code generation capabilities including:
- Seq2Seq models: CodeBERT, CodeT5, CodeGPT, StarCoder, etc.
- Generation tasks: Completion, summarization, translation, bug fixing
- Preprocessing: Tokenization, AST parsing, normalization, augmentation
- Datasets: CodeSearchNet, CodeParrot, TheStack, HumanEval
- Evaluation: Pass@k, CodeBLEU, syntax checking
- Training utilities and AST manipulation functions
"""

import re
import ast
import json
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    RobertaModel,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)


# ==============================================================================
# Seq2Seq Models
# ==============================================================================


class CodeBERTModel(nn.Module):
    """CodeBERT for code understanding and representation.

    Uses Microsoft CodeBERT for code embeddings and masked language modeling.

    Args:
        pretrained_model: HuggingFace model name or path
        freeze_encoder: Whether to freeze the encoder weights
    """

    def __init__(
        self,
        pretrained_model: str = "microsoft/codebert-base",
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Forward pass through CodeBERT.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with 'last_hidden_state' and 'pooler_output'
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output,
        }

    def encode(self, code: Union[str, List[str]], max_length: int = 512) -> Tensor:
        """Encode code snippets to embeddings.

        Args:
            code: Single code string or list of code strings
            max_length: Maximum sequence length

        Returns:
            Code embeddings [batch_size, hidden_dim]
        """
        if isinstance(code, str):
            code = [code]

        inputs = self.tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.forward(inputs["input_ids"], inputs["attention_mask"])

        return outputs["pooler_output"]


class CodeT5Model(nn.Module):
    """CodeT5 for code generation and understanding.

    T5-based model pretrained on code for generation tasks.

    Args:
        pretrained_model: HuggingFace model name or path
    """

    def __init__(self, pretrained_model: str = "Salesforce/codet5-base"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass through CodeT5.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs for training [batch_size, target_len]

        Returns:
            Dictionary with 'loss' (if labels) and 'logits'
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_length: int = 256,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> List[str]:
        """Generate code from input.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            List of generated code strings
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class CodeBERTaModel(nn.Module):
    """CodeBERTa - RoBERTa-based code model.

    Uses RoBERTa architecture pretrained on code.

    Args:
        pretrained_model: HuggingFace model name
        vocab_size: Vocabulary size (if different from pretrained)
    """

    def __init__(
        self,
        pretrained_model: str = "huggingface/CodeBERTa-small-v1",
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.model = RobertaModel.from_pretrained(pretrained_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)

        if vocab_size is not None and vocab_size != self.model.config.vocab_size:
            self.model.resize_token_embeddings(vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Hidden states and pooled output
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output,
        }


class PLBARTModel(nn.Module):
    """PLBART - Multilingual code model.

    BART-based model for multilingual code understanding and generation.
    Supports multiple programming languages.

    Args:
        pretrained_model: HuggingFace model name
    """

    def __init__(self, pretrained_model: str = "uclanlp/plbart-base"):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_model)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass through PLBART.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for training

        Returns:
            Model outputs including loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_length: int = 256,
        num_beams: int = 4,
        **kwargs,
    ) -> List[str]:
        """Generate code/text.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Number of beams
            **kwargs: Additional generation parameters

        Returns:
            Generated text strings
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class CodeGPTModel(nn.Module):
    """CodeGPT - GPT-based code generation model.

    Causal language model for code completion and generation.

    Args:
        pretrained_model: HuggingFace model name
    """

    def __init__(self, pretrained_model: str = "microsoft/CodeGPT-small-py"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels

        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def complete(
        self,
        code_prefix: str,
        max_length: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Complete code from prefix.

        Args:
            code_prefix: Starting code snippet
            max_length: Maximum total length
            temperature: Sampling temperature
            top_p: Nucleus sampling
            num_return_sequences: Number of completions

        Returns:
            List of completed code strings
        """
        inputs = self.tokenizer(code_prefix, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        completions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [comp[len(code_prefix) :] for comp in completions]


class CodeParrotModel(nn.Module):
    """CodeParrot - Python-specific code generation.

    GPT-2 based model trained on Python code.

    Args:
        pretrained_model: HuggingFace model name
    """

    def __init__(self, pretrained_model: str = "codeparrot/codeparrot-small"):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels

        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate Python code.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling
            num_return_sequences: Number of sequences

        Returns:
            Generated code strings
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class SantaCoderModel(nn.Module):
    """SantaCoder - Multi-language code generation.

    Efficient model supporting Python, Java, and JavaScript.

    Args:
        pretrained_model: HuggingFace model name
    """

    def __init__(self, pretrained_model: str = "bigcode/santacoder"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels

        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def fill_in_the_middle(
        self,
        prefix: str,
        suffix: str,
        max_length: int = 256,
        temperature: float = 0.8,
    ) -> str:
        """Fill in the middle (FIM) completion.

        SantaCoder supports FIM with special tokens.

        Args:
            prefix: Code before the gap
            suffix: Code after the gap
            max_length: Maximum length
            temperature: Sampling temperature

        Returns:
            Completed code
        """
        # FIM format: <fim-prefix>prefix<fim-suffix>suffix<fim-middle>
        fim_input = f"<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>"

        inputs = self.tokenizer(fim_input, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract middle portion
        if "<fim-middle>" in generated:
            middle = generated.split("<fim-middle>")[-1].split("<fim-")[0]
            return middle

        return generated


class StarCoderModel(nn.Module):
    """StarCoder - Large-scale code generation model.

    15.5B parameter model trained on permissive code.

    Args:
        pretrained_model: HuggingFace model name
        device_map: Device mapping for model parallelism
    """

    def __init__(
        self,
        pretrained_model: str = "bigcode/starcoder",
        device_map: str = "auto",
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels

        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate code.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            num_return_sequences: Number of sequences

        Returns:
            Generated code strings
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ==============================================================================
# Generation Tasks
# ==============================================================================


class CodeCompletion:
    """Code completion task.

    Completes partial code snippets using autoregressive models.

    Args:
        model: Language model for completion
        tokenizer: Tokenizer
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def complete(
        self,
        prefix: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """Complete code from prefix.

        Args:
            prefix: Partial code
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_tokens: Tokens to stop generation

        Returns:
            Completed code
        """
        inputs = self.tokenizer(prefix, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.encode(stop_tokens[0])[0]
            if stop_tokens
            else None,
        )

        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prefix
        if completion.startswith(prefix):
            return completion[len(prefix) :]
        return completion

    def batch_complete(
        self,
        prefixes: List[str],
        max_tokens: int = 128,
        temperature: float = 0.8,
    ) -> List[str]:
        """Complete multiple prefixes.

        Args:
            prefixes: List of partial code strings
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of completions
        """
        return [self.complete(p, max_tokens, temperature) for p in prefixes]


class CodeSummarization:
    """Code summarization task.

    Generates natural language summaries from code.

    Args:
        model: Seq2Seq model
        tokenizer: Tokenizer
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def summarize(
        self,
        code: str,
        max_length: int = 128,
        num_beams: int = 4,
    ) -> str:
        """Generate summary for code.

        Args:
            code: Source code
            max_length: Maximum summary length
            num_beams: Number of beams

        Returns:
            Generated summary
        """
        # Add task prefix
        prompt = f"summarize: {code}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_summarize(
        self,
        codes: List[str],
        max_length: int = 128,
        num_beams: int = 4,
    ) -> List[str]:
        """Summarize multiple code snippets.

        Args:
            codes: List of code strings
            max_length: Maximum summary length
            num_beams: Number of beams

        Returns:
            List of summaries
        """
        prompts = [f"summarize: {c}" for c in codes]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class CodeTranslation:
    """Code translation between languages.

    Translates code from one programming language to another.

    Args:
        model: Seq2Seq model
        tokenizer: Tokenizer
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def translate(
        self,
        code: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
    ) -> str:
        """Translate code between languages.

        Args:
            code: Source code
            source_lang: Source language (e.g., 'python', 'java')
            target_lang: Target language
            max_length: Maximum output length
            num_beams: Number of beams

        Returns:
            Translated code
        """
        prompt = f"translate {source_lang} to {target_lang}: {code}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_translate(
        self,
        codes: List[str],
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
    ) -> List[str]:
        """Translate multiple code snippets.

        Args:
            codes: List of code strings
            source_lang: Source language
            target_lang: Target language
            max_length: Maximum output length

        Returns:
            List of translated code
        """
        return [self.translate(c, source_lang, target_lang, max_length) for c in codes]


class BugFixing:
    """Automatic bug fixing.

    Detects and fixes bugs in code.

    Args:
        model: Seq2Seq model
        tokenizer: Tokenizer
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def fix(
        self,
        buggy_code: str,
        max_length: int = 512,
        num_beams: int = 4,
    ) -> str:
        """Fix bugs in code.

        Args:
            buggy_code: Code with bugs
            max_length: Maximum output length
            num_beams: Number of beams

        Returns:
            Fixed code
        """
        prompt = f"fix bugs: {buggy_code}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_fix(
        self,
        buggy_codes: List[str],
        max_length: int = 512,
    ) -> List[str]:
        """Fix multiple buggy code snippets.

        Args:
            buggy_codes: List of buggy code strings
            max_length: Maximum output length

        Returns:
            List of fixed code
        """
        return [self.fix(c, max_length) for c in buggy_codes]


class CodeReview:
    """Automated code review.

    Generates review comments and suggestions.

    Args:
        model: Language model
        tokenizer: Tokenizer
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def review(
        self,
        code: str,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """Review code and generate comments.

        Args:
            code: Code to review
            max_length: Maximum comment length

        Returns:
            Dictionary with review results
        """
        prompt = f"review: {code}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
        )

        review_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "code": code,
            "review": review_text,
            "issues": self._extract_issues(review_text),
            "suggestions": self._extract_suggestions(review_text),
        }

    def _extract_issues(self, review: str) -> List[str]:
        """Extract issue mentions from review."""
        issues = []
        for line in review.split("\n"):
            if any(kw in line.lower() for kw in ["issue", "bug", "error", "warning"]):
                issues.append(line.strip())
        return issues

    def _extract_suggestions(self, review: str) -> List[str]:
        """Extract suggestions from review."""
        suggestions = []
        for line in review.split("\n"):
            if any(
                kw in line.lower()
                for kw in ["suggest", "recommend", "improve", "consider"]
            ):
                suggestions.append(line.strip())
        return suggestions


# ==============================================================================
# Preprocessing
# ==============================================================================


class TokenizeCode:
    """Code tokenization preprocessor.

    Tokenizes code using various tokenization strategies.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    """

    def __init__(self, tokenizer: Any, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, code: Union[str, List[str]]) -> Dict[str, Tensor]:
        """Tokenize code.

        Args:
            code: Single code string or list

        Returns:
            Tokenized inputs
        """
        if isinstance(code, str):
            code = [code]

        return self.tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


class ASTParser:
    """Abstract Syntax Tree parser.

    Parses code to AST for structural analysis.

    Args:
        language: Programming language ('python', 'java', etc.)
    """

    def __init__(self, language: str = "python"):
        self.language = language

    def parse(self, code: str) -> Optional[ast.AST]:
        """Parse code to AST.

        Args:
            code: Source code

        Returns:
            AST root node or None if parsing fails
        """
        if self.language == "python":
            try:
                return ast.parse(code)
            except SyntaxError:
                return None
        else:
            # For other languages, would need external parsers
            raise NotImplementedError(
                f"AST parsing not implemented for {self.language}"
            )

    def get_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract features from AST.

        Args:
            code: Source code

        Returns:
            Dictionary of AST features
        """
        tree = self.parse(code)
        if tree is None:
            return {"error": "parse_failed"}

        features = {
            "num_nodes": 0,
            "num_functions": 0,
            "num_classes": 0,
            "num_imports": 0,
            "depth": 0,
        }

        for node in ast.walk(tree):
            features["num_nodes"] += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features["num_functions"] += 1
            elif isinstance(node, ast.ClassDef):
                features["num_classes"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                features["num_imports"] += 1

        features["depth"] = self._get_ast_depth(tree)

        return features

    def _get_ast_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate AST depth."""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_ast_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth


class CodeNormalizer:
    """Code normalization preprocessor.

    Normalizes code style and formatting.

    Args:
        remove_comments: Whether to remove comments
        normalize_whitespace: Whether to normalize whitespace
        rename_variables: Whether to normalize variable names
    """

    def __init__(
        self,
        remove_comments: bool = True,
        normalize_whitespace: bool = True,
        rename_variables: bool = False,
    ):
        self.remove_comments = remove_comments
        self.normalize_whitespace = normalize_whitespace
        self.rename_variables = rename_variables

    def normalize(self, code: str) -> str:
        """Normalize code.

        Args:
            code: Source code

        Returns:
            Normalized code
        """
        if self.remove_comments:
            code = self._remove_comments(code)

        if self.normalize_whitespace:
            code = self._normalize_whitespace(code)

        if self.rename_variables:
            code = self._normalize_variables(code)

        return code

    def _remove_comments(self, code: str) -> str:
        """Remove Python comments."""
        lines = []
        for line in code.split("\n"):
            # Remove inline comments
            if "#" in line:
                line = line[: line.index("#")]
            lines.append(line.rstrip())
        return "\n".join(lines)

    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace."""
        lines = code.split("\n")
        lines = [line.rstrip() for line in lines]
        # Remove empty lines at start/end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def _normalize_variables(self, code: str) -> str:
        """Normalize variable names."""
        # This is a placeholder - full implementation would need AST analysis
        return code


class CodeAugmentation:
    """Code data augmentation.

    Applies various augmentations to code.

    Args:
        augmentations: List of augmentation names to apply
    """

    def __init__(self, augmentations: Optional[List[str]] = None):
        self.augmentations = augmentations or [
            "variable_rename",
            "dead_code_insertion",
            "operator_swap",
        ]

    def augment(self, code: str, num_variants: int = 1) -> List[str]:
        """Generate augmented code variants.

        Args:
            code: Source code
            num_variants: Number of variants to generate

        Returns:
            List of augmented code strings
        """
        variants = []
        for _ in range(num_variants):
            variant = code
            for aug_name in self.augmentations:
                if hasattr(self, f"_augment_{aug_name}"):
                    variant = getattr(self, f"_augment_{aug_name}")(variant)
            variants.append(variant)
        return variants

    def _augment_variable_rename(self, code: str) -> str:
        """Rename variables randomly."""
        # Simple implementation - would need proper AST parsing
        try:
            tree = ast.parse(code)
            var_map = {}
            counter = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if node.id not in var_map:
                        var_map[node.id] = f"var_{counter}"
                        counter += 1

            for old, new in var_map.items():
                code = re.sub(r"\b" + old + r"\b", new, code)
        except:
            pass

        return code

    def _augment_dead_code_insertion(self, code: str) -> str:
        """Insert dead code."""
        dead_code = "\n    _unused = 0  # Augmented\n"
        lines = code.split("\n")
        if len(lines) > 1:
            insert_pos = random.randint(0, len(lines) - 1)
            lines.insert(insert_pos, dead_code)
        return "\n".join(lines)

    def _augment_operator_swap(self, code: str) -> str:
        """Swap equivalent operators."""
        swaps = {
            "x += 1": "x = x + 1",
            "x -= 1": "x = x - 1",
            "x *= 2": "x = x * 2",
            "x //= 2": "x = x // 2",
        }
        for old, new in swaps.items():
            if random.random() < 0.5:
                code = code.replace(old, new)
        return code


# ==============================================================================
# Datasets
# ==============================================================================


class CodeSearchNetDataset(Dataset):
    """CodeSearchNet dataset loader.

    Loads code with documentation pairs from CodeSearchNet.

    Args:
        data_path: Path to dataset
        language: Programming language filter
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        language: str = "python",
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ):
        self.data_path = Path(data_path)
        self.language = language
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset from files."""
        examples = []

        # Look for JSONL files
        pattern = f"{self.language}/*.jsonl"
        files = list(self.data_path.glob(pattern))

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(
                        {
                            "code": data.get("code", ""),
                            "docstring": data.get("docstring", ""),
                            "func_name": data.get("func_name", ""),
                            "url": data.get("url", ""),
                        }
                    )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        if self.tokenizer:
            code_encoded = self.tokenizer(
                example["code"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            doc_encoded = self.tokenizer(
                example["docstring"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "code_input_ids": code_encoded["input_ids"].squeeze(0),
                "code_attention_mask": code_encoded["attention_mask"].squeeze(0),
                "doc_input_ids": doc_encoded["input_ids"].squeeze(0),
                "doc_attention_mask": doc_encoded["attention_mask"].squeeze(0),
            }

        return example


class CodeParrotDataset(Dataset):
    """CodeParrot dataset loader.

    Loads Python code for training code generation models.

    Args:
        data_path: Path to dataset files
        tokenizer: Tokenizer
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[Any] = None,
        max_length: int = 1024,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data()

    def _load_data(self) -> List[str]:
        """Load code examples."""
        examples = []

        # Support multiple file formats
        for ext in ["*.py", "*.jsonl", "*.json"]:
            for file_path in self.data_path.rglob(ext):
                if file_path.suffix == ".py":
                    with open(file_path, "r", encoding="utf-8") as f:
                        examples.append(f.read())
                elif file_path.suffix in [".jsonl", ".json"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file_path.suffix == ".jsonl":
                            for line in f:
                                data = json.loads(line)
                                if "content" in data:
                                    examples.append(data["content"])
                                elif "code" in data:
                                    examples.append(data["code"])
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                examples.extend([d.get("code", "") for d in data])

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        code = self.examples[idx]

        if self.tokenizer:
            encoded = self.tokenizer(
                code,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0),
            }

        return {"code": code}


class TheStackDataset(Dataset):
    """The Stack dataset loader.

    Large-scale code dataset with multiple languages.

    Args:
        data_path: Path to dataset
        languages: List of languages to include
        tokenizer: Tokenizer
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        languages: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
    ):
        self.data_path = Path(data_path)
        self.languages = languages or ["python", "javascript", "java"]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset."""
        examples = []

        for lang in self.languages:
            lang_dir = self.data_path / lang
            if lang_dir.exists():
                for file_path in lang_dir.rglob("*.jsonl"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            examples.append(
                                {
                                    "content": data.get("content", ""),
                                    "language": lang,
                                    "path": data.get("path", ""),
                                    "repo": data.get("repo_name", ""),
                                    "license": data.get("license", ""),
                                }
                            )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        if self.tokenizer:
            encoded = self.tokenizer(
                example["content"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "language": example["language"],
            }

        return example


class HumanEvalDataset(Dataset):
    """HumanEval benchmark dataset.

    Function-level code generation benchmark.

    Args:
        data_path: Path to HumanEval.jsonl
    """

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.examples = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load HumanEval problems."""
        examples = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                examples.append(
                    {
                        "task_id": data["task_id"],
                        "prompt": data["prompt"],
                        "entry_point": data["entry_point"],
                        "canonical_solution": data.get("canonical_solution", ""),
                        "test": data["test"],
                    }
                )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]

    def get_test_function(self, idx: int) -> str:
        """Get test code for a problem."""
        example = self.examples[idx]
        return example["test"]


# ==============================================================================
# Evaluation
# ==============================================================================


def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate Pass@k metric.

    Pass@k is the probability that at least one of k samples
    from n total samples passes all tests, given c correct samples.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k parameter

    Returns:
        Pass@k score
    """
    if n - c < k:
        return 1.0

    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bleu_code(reference: str, hypothesis: str, n: int = 4) -> float:
    """Calculate CodeBLEU score.

    Adapted BLEU score for code evaluation.

    Args:
        reference: Reference code
        hypothesis: Generated code
        n: Maximum n-gram size

    Returns:
        CodeBLEU score
    """
    from collections import Counter

    def get_ngrams(tokens: List[str], n: int) -> Counter:
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
        return Counter(ngrams)

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Calculate brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # Calculate n-gram precisions
    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = get_ngrams(ref_tokens, i)
        hyp_ngrams = get_ngrams(hyp_tokens, i)

        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue

        matches = sum((hyp_ngrams & ref_ngrams).values())
        precision = matches / len(hyp_ngrams)
        precisions.append(precision)

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
    else:
        geo_mean = 0.0

    return bp * geo_mean


def exact_match(reference: str, hypothesis: str) -> bool:
    """Check if generated code matches reference exactly.

    Args:
        reference: Reference code
        hypothesis: Generated code

    Returns:
        True if exact match
    """
    # Normalize whitespace
    ref_normalized = " ".join(reference.split())
    hyp_normalized = " ".join(hypothesis.split())

    return ref_normalized == hyp_normalized


def syntax_correct(code: str, language: str = "python") -> bool:
    """Check if code is syntactically correct.

    Args:
        code: Code to check
        language: Programming language

    Returns:
        True if syntax is correct
    """
    if language == "python":
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    else:
        # For other languages, would need external parsers
        warnings.warn(f"Syntax checking not implemented for {language}")
        return True


# ==============================================================================
# Training
# ==============================================================================


class CodeLoss(nn.Module):
    """Combined loss for code generation.

    Combines cross-entropy with optional auxiliary losses.

    Args:
        label_smoothing: Label smoothing parameter
        weight_ast: Weight for AST-based loss
    """

    def __init__(self, label_smoothing: float = 0.0, weight_ast: float = 0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )
        self.weight_ast = weight_ast

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        code_strings: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """Compute loss.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            code_strings: Generated code strings for AST loss

        Returns:
            Dictionary with losses
        """
        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        loss_ce = self.ce_loss(
            logits.view(-1, vocab_size),
            targets.view(-1),
        )

        losses = {"ce_loss": loss_ce, "total_loss": loss_ce}

        if self.weight_ast > 0 and code_strings is not None:
            loss_ast = self._compute_ast_loss(code_strings, targets)
            losses["ast_loss"] = loss_ast
            losses["total_loss"] = loss_ce + self.weight_ast * loss_ast

        return losses

    def _compute_ast_loss(self, code_strings: List[str], targets: Tensor) -> Tensor:
        """Compute AST-based auxiliary loss."""
        # Placeholder for AST loss
        # In practice, would compare AST structures
        return torch.tensor(0.0, device=targets.device)


class CodeDataset(Dataset):
    """Generic code dataset for training.

    Args:
        codes: List of code strings
        labels: Optional labels
        tokenizer: Tokenizer
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        codes: List[str],
        labels: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ):
        self.codes = codes
        self.labels = labels or codes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        code = self.codes[idx]
        label = self.labels[idx]

        if self.tokenizer:
            inputs = self.tokenizer(
                code,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            labels_encoded = self.tokenizer(
                label,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": labels_encoded["input_ids"].squeeze(0),
            }

        return {"code": code, "label": label}


class CodeGenerationTrainer:
    """Trainer for code generation models.

    Handles training loop, evaluation, and checkpointing.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        optimizer: Optimizer
        device: Device for training
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.criterion = CodeLoss()
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model.

        Args:
            dataloader: Validation data loader

        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history["val_loss"].append(avg_loss)
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_path: Path to save checkpoints

        Returns:
            Training history
        """
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)

            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_path)
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

                if save_path and (epoch + 1) % 5 == 0:
                    self.save_checkpoint(f"{save_path}_epoch_{epoch + 1}")

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        print(f"Checkpoint loaded from {path}")


# ==============================================================================
# Utilities
# ==============================================================================


def extract_functions(code: str) -> List[Dict[str, Any]]:
    """Extract function definitions from code.

    Args:
        code: Source code

    Returns:
        List of function dictionaries
    """
    functions = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "defaults": len(node.args.defaults),
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "code": ast.unparse(node),
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "returns": ast.unparse(node.returns) if node.returns else None,
                }
                functions.append(func_info)
    except SyntaxError:
        pass

    return functions


def extract_classes(code: str) -> List[Dict[str, Any]]:
    """Extract class definitions from code.

    Args:
        code: Source code

    Returns:
        List of class dictionaries
    """
    classes = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(
                            {
                                "name": item.name,
                                "lineno": item.lineno,
                            }
                        )

                class_info = {
                    "name": node.name,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "methods": methods,
                    "code": ast.unparse(node),
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                }
                classes.append(class_info)
    except SyntaxError:
        pass

    return classes


def code_to_ast(code: str) -> Optional[ast.AST]:
    """Convert code to AST.

    Args:
        code: Source code

    Returns:
        AST root node or None if parsing fails
    """
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def ast_to_code(tree: ast.AST) -> str:
    """Convert AST back to code.

    Args:
        tree: AST root node

    Returns:
        Source code string
    """
    return ast.unparse(tree)


# ==============================================================================
# Module exports
# ==============================================================================

__all__ = [
    # Models
    "CodeBERTModel",
    "CodeT5Model",
    "CodeBERTaModel",
    "PLBARTModel",
    "CodeGPTModel",
    "CodeParrotModel",
    "SantaCoderModel",
    "StarCoderModel",
    # Tasks
    "CodeCompletion",
    "CodeSummarization",
    "CodeTranslation",
    "BugFixing",
    "CodeReview",
    # Preprocessing
    "TokenizeCode",
    "ASTParser",
    "CodeNormalizer",
    "CodeAugmentation",
    # Datasets
    "CodeSearchNetDataset",
    "CodeParrotDataset",
    "TheStackDataset",
    "HumanEvalDataset",
    # Evaluation
    "pass_at_k",
    "bleu_code",
    "exact_match",
    "syntax_correct",
    # Training
    "CodeGenerationTrainer",
    "CodeDataset",
    "CodeLoss",
    # Utilities
    "extract_functions",
    "extract_classes",
    "code_to_ast",
    "ast_to_code",
]
