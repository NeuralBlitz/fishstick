"""
GPT-Style Language Model for Generative Tasks.

Implements GPT-2 style language model generation as described in:
Radford et al. (2019) "Language Models are Unsupervised Multitask Learners"

Includes generation utilities and sampling methods.
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class GPT2Config:
    """Configuration for GPT-2 model."""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_ctx: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu",
        resid_dropout: float = 0.1,
        embd_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner or 4 * n_embd
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embd_dropout = embd_dropout
        self.attn_dropout = attn_dropout
        self.layer_norm_epsilon = layer_norm_epsilon


class GPT2Attention(nn.Module):
    """GPT-2 attention module."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.n_ctx = config.n_ctx
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
        )

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(
        self,
        x: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass with optional kv cache."""
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.config.n_embd, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn_weights = attn_weights.masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.config.n_embd)
        )
        out = self.resid_dropout(self.proj(out))

        return out, present


class GPT2MLP(nn.Module):
    """GPT-2 feedforward network."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    """GPT-2 transformer block."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        x: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through block."""
        attn_out, present = self.attn(self.ln_1(x), layer_past)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))

        return x, present


class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 Language Model Head.

    Complete GPT-2 model for language generation.

    Args:
        config: GPT-2 configuration
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_dropout)

        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tie_weights()

    def tie_weights(self) -> None:
        """Tie weights between embeddings and output layer."""
        self.lm_head.weight = self.wte.weight

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            position_ids: Position IDs
            token_type_ids: Token type IDs
            past_key_values: Past key values for efficient generation

        Returns:
            logits, present_key_values
        """
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids.shape[-1] < seq_len + past_length:
            position_ids = torch.arange(
                past_length, past_length + seq_len, device=input_ids.device
            ).unsqueeze(0)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        presents = []
        for i, block in enumerate(self.h):
            hidden_states, present = block(hidden_states, past_key_values[i])
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)

        lm_logits = self.lm_head(hidden_states)

        return lm_logits, presents

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample

        Returns:
            Generated token IDs
        """
        self.eval()

        past_key_values = None

        for _ in range(max_new_tokens):
            logits, past_key_values = self.forward(
                input_ids
                if input_ids.shape[1] <= self.config.n_positions
                else input_ids[:, -self.config.n_positions :],
                past_key_values=past_key_values,
            )

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)

                cumulative_probs = probs.cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., :-1] = sorted_indices_to_remove[..., 1:]
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class GPTGenerator:
    """
    Generation utilities for GPT models.

    Provides various sampling strategies and beam search.
    """

    def __init__(self, model: GPT2LMHeadModel, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text or token IDs
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of sequences to generate

        Returns:
            Generated text strings
        """
        self.model.eval()

        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string input")
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        else:
            input_ids = torch.tensor(prompt).unsqueeze(0)

        input_ids = input_ids.to(self.model.parameters().__next__().device)

        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)

        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=temperature > 0,
        )

        if self.tokenizer is not None:
            outputs = self.tokenizer.batch_decode(generated)
            return outputs

        return generated.tolist()

    @torch.no_grad()
    def beam_search(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 100,
        num_beams: int = 5,
        length_penalty: float = 1.0,
    ) -> str:
        """
        Beam search generation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens
            num_beams: Number of beams
            length_penalty: Length penalty

        Returns:
            Generated text
        """
        self.model.eval()

        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string input")
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        else:
            input_ids = torch.tensor(prompt).unsqueeze(0)

        input_ids = input_ids.to(self.model.parameters().__next__().device)

        beam_scores = torch.zeros((1,), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view(-1)

        beam_scores = beam_scores.unsqueeze(0).expand(num_beams, -1)

        beam_tokens = input_ids.repeat(num_beams, 1)

        done = [False] * num_beams

        for _ in range(max_new_tokens):
            logits, _ = self.model.forward(beam_tokens)
            next_token_logits = logits[:, -1, :]

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)

            vocab_size = next_token_scores.shape[-1]

            next_token_scores = next_token_scores.view(num_beams, -1)

            top_scores, top_tokens = torch.topk(next_token_scores, num_beams, dim=1)

            beam_idx = top_tokens // vocab_size
            next_token = top_tokens % vocab_size

            beam_scores = top_scores.view(-1)

            beam_tokens = torch.cat(
                [beam_tokens[beam_idx], next_token.unsqueeze(-1)], dim=-1
            )

        best_beam = torch.argmax(beam_scores)

        if self.tokenizer is not None:
            return self.tokenizer.decode(beam_tokens[best_beam])

        return beam_tokens[best_beam].tolist()
