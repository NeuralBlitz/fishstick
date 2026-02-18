"""
Language Model Wrapper
=====================

Unified interface for language models with generation support.
Provides a clean API for text generation across different model types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Callable

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GenerationResult:
    """Container for generation results."""

    sequences: list[str]
    tokens: Optional[Tensor] = None
    log_probs: Optional[Tensor] = None
    num_generated_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LanguageModel(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 100,
        **kwargs,
    ) -> Tensor:
        """Generate tokens given input IDs."""
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        """Get model device."""
        pass


class TorchLanguageModel(LanguageModel):
    """PyTorch-based language model wrapper."""

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        device: Optional[torch.device] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        self.model = model
        self._vocab_size = vocab_size
        self.device = device or next(model.parameters()).device
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.model.to(self.device)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through the model."""
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs

    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """Generate tokens using greedy decoding by default."""
        input_ids = input_ids.to(self.device)
        self.model.eval()

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                logits = self.model(generated)[:, -1, :]

                if repetition_penalty != 1.0:
                    for b in range(generated.size(0)):
                        for token_id in generated[b]:
                            if logits[b, token_id] < 0:
                                logits[b, token_id] *= repetition_penalty
                            else:
                                logits[b, token_id] /= repetition_penalty

                if temperature != 1.0:
                    logits = logits / temperature

                if top_k > 0:
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumsum_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_mask = cumsum_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    indices_to_remove = sorted_mask.scatter(
                        1, sorted_indices, sorted_mask
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_tokens], dim=1)

                if (
                    self.eos_token_id is not None
                    and (next_tokens == self.eos_token_id).all()
                ):
                    break

        return generated

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_device(self) -> torch.device:
        return self.device


class ModelRegistry:
    """Registry for managing language models."""

    def __init__(self):
        self._models: dict[str, LanguageModel] = {}
        self._factories: dict[str, Callable] = {}

    def register(
        self,
        name: str,
        model: LanguageModel,
    ) -> None:
        """Register a model."""
        self._models[name] = model

    def register_factory(
        self,
        name: str,
        factory: Callable[[dict], LanguageModel],
    ) -> None:
        """Register a model factory."""
        self._factories[name] = factory

    def get(self, name: str) -> Optional[LanguageModel]:
        """Get a registered model."""
        return self._models.get(name)

    def create(self, name: str, config: dict) -> LanguageModel:
        """Create a model using a registered factory."""
        if name not in self._factories:
            raise KeyError(f"No factory registered for: {name}")
        return self._factories[name](config)

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self._models.keys())

    def unregister(self, name: str) -> None:
        """Unregister a model."""
        self._models.pop(name, None)


_global_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _global_registry


def register_model(name: str) -> Callable:
    """Decorator to register a model factory."""

    def decorator(func: Callable[[dict], LanguageModel]) -> Callable:
        _global_registry.register_factory(name, func)
        return func

    return decorator


class GenerationPipeline:
    """High-level generation pipeline."""

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        return_tokens: bool = False,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        generated_tensor = self.model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        generated_text = self.tokenizer.decode(generated_tensor[0].tolist())

        return GenerationResult(
            sequences=[generated_text],
            tokens=generated_tensor if return_tokens else None,
            num_generated_tokens=generated_tensor.size(1) - len(input_ids),
        )

    def batch_generate(
        self,
        prompts: list[str],
        **kwargs,
    ) -> GenerationResult:
        """Generate text for multiple prompts."""
        input_ids_list = [self.tokenizer.encode(p) for p in prompts]
        max_len = max(len(ids) for ids in input_ids_list)

        padded = []
        for ids in input_ids_list:
            padded.append(ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)))

        input_tensor = torch.tensor(padded, dtype=torch.long)

        generated_tensor = self.model.generate(
            input_tensor,
            max_length=kwargs.get("max_length", 100),
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", 0),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )

        generated_texts = [
            self.tokenizer.decode(ids.tolist()) for ids in generated_tensor
        ]

        return GenerationResult(
            sequences=generated_texts,
            tokens=generated_tensor,
            num_generated_tokens=generated_tensor.size(1) - max_len,
        )


class HFModelWrapper(LanguageModel):
    """Wrapper for HuggingFace-style models."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")

        config = getattr(model, "config", None)
        self._vocab_size = getattr(config, "vocab_size", tokenizer.vocab_size)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through the model."""
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[0]

    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """Generate tokens using the model."""
        input_ids = input_ids.to(self.device)

        generation_kwargs = {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k if top_k > 0 else None,
            "top_p": top_p if top_p < 1.0 else None,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature != 1.0 or top_k > 0 or top_p < 1.0,
        }

        with torch.no_grad():
            outputs = self.model.generate(input_ids, **generation_kwargs)

        return outputs

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_device(self) -> torch.device:
        return self.device


def create_pipeline(
    model: Any,
    tokenizer: Any,
    device: Optional[torch.device] = None,
) -> GenerationPipeline:
    """Create a generation pipeline from a model and tokenizer."""
    if hasattr(model, "generate"):
        wrapped_model = HFModelWrapper(model, tokenizer, device)
    else:
        wrapped_model = TorchLanguageModel(
            model,
            vocab_size=tokenizer.vocab_size,
            device=device,
        )

    return GenerationPipeline(wrapped_model, tokenizer)
