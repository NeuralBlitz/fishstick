"""
Video Captioning Module for fishstick

Comprehensive video captioning models including:
- Video encoder (CNN + Transformer)
- Caption decoder (LSTM/Transformer)
- Beam search decoding
- Attention mechanisms (soft, hierarchical)

Author: Fishstick Team
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


@dataclass
class CaptioningConfig:
    """Configuration for video captioning."""

    vocab_size: int = 50000
    embed_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.3
    max_length: int = 30
    beam_size: int = 5


class VideoCaptionEncoder(nn.Module):
    """
    Video Encoder for Caption Generation.

    Encodes video features using CNN backbone and temporal transformer.

    Args:
        feature_dim: Dimension of input video features
        embed_dim: Output embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.context_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode video features.

        Args:
            features: Video features (B, T, D) or (B, D, T, H, W)

        Returns:
            Tuple of (encoded_features, context_vector)
        """
        if features.dim() == 5:
            B, D, T, H, W = features.shape
            features = features.permute(0, 2, 3, 4, 1)
            features = features.reshape(B, T, H * W, D)
            features = features.mean(dim=2)

        x = self.feature_proj(features)

        encoded = self.transformer(x)

        context = torch.tanh(self.context_proj(encoded.mean(dim=1)))

        return encoded, context


class CaptionDecoder(nn.Module):
    """
    Caption Decoder with LSTM.

    Generates captions word-by-word using LSTM with attention.

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Word embedding dimension
        hidden_dim: LSTM hidden dimension
        encoder_dim: Encoder output dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        encoder_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim + encoder_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = Attention(
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_out: Tensor,
        context: Tensor,
        captions: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """
        Forward pass during training.

        Args:
            encoder_out: Encoded video features (B, T, E)
            context: Global video context (B, E)
            captions: Target captions (B, max_len)
            lengths: Actual lengths of captions

        Returns:
            Prediction logits (B, vocab_size)
        """
        batch_size = encoder_out.size(0)
        max_len = captions.size(1)

        embeddings = self.embedding(captions)

        outputs = []

        hidden = None

        for t in range(max_len):
            current_embed = embeddings[:, t]

            context_t, attention_weights = self.attention(hidden, encoder_out)

            lstm_input = torch.cat([current_embed, context_t], dim=1)
            lstm_input = lstm_input.unsqueeze(1)

            output, hidden = self.lstm(lstm_input, hidden)

            output = output.squeeze(1)
            output = self.dropout(output)

            pred = self.fc(output)

            outputs.append(pred)

        outputs = torch.stack(outputs, dim=1)

        return outputs

    def decode_step(
        self,
        encoder_out: Tensor,
        context: Tensor,
        word: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """
        Single decoding step.

        Args:
            encoder_out: Encoded video features
            context: Global context
            word: Current word embedding
            hidden: LSTM hidden state

        Returns:
            Tuple of (prediction, hidden_state, attention_weights)
        """
        embed = self.embedding(word)

        context_t, attention_weights = self.attention(hidden, encoder_out)

        lstm_input = torch.cat([embed, context_t], dim=1).unsqueeze(1)

        output, hidden = self.lstm(lstm_input, hidden)

        output = output.squeeze(1)
        output = self.dropout(output)

        pred = self.fc(output)

        return pred, hidden, attention_weights

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize hidden states."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


class Attention(nn.Module):
    """
    Attention Mechanism for Caption Decoder.

    Computes attention over video features.

    Args:
        encoder_dim: Encoder feature dimension
        hidden_dim: Decoder hidden dimension
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.encoder_attn = nn.Linear(encoder_dim, hidden_dim)
        self.decoder_attn = nn.Linear(hidden_dim, hidden_dim)
        self.full_attn = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden: Optional[Tuple[Tensor, Tensor]],
        encoder_out: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention.

        Args:
            hidden: Decoder hidden state
            encoder_out: Encoder outputs (B, T, E)

        Returns:
            Tuple of (context, attention_weights)
        """
        B, T, E = encoder_out.shape

        if hidden is None:
            h = torch.zeros(B, encoder_out.size(-1), device=encoder_out.device)
        else:
            h = hidden[0][-1]

        encoder_feat = self.encoder_attn(encoder_out)

        decoder_feat = self.decoder_attn(h)

        attention = encoder_feat + decoder_feat.unsqueeze(1)
        attention = torch.tanh(attention)
        attention = self.full_attn(attention).squeeze(-1)

        alpha = F.softmax(attention, dim=-1)

        context = torch.bmm(alpha.unsqueeze(1), encoder_out).squeeze(1)

        return context, alpha


class TransformerDecoder(nn.Module):
    """
    Transformer-based Caption Decoder.

    Uses multi-head attention for caption generation.

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Word embedding dimension
        hidden_dim: Hidden dimension
        encoder_dim: Encoder output dimension
        num_heads: Number of attention heads
        num_layers: Number of decoder layers
        dropout: Dropout probability
        max_length: Maximum caption length
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        encoder_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_length: int = 30,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))

        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        encoder_out: Tensor,
        captions: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass during training.

        Args:
            encoder_out: Encoded video features (B, T, E)
            captions: Target captions (B, L)
            mask: Attention mask

        Returns:
            Prediction logits
        """
        B, L = captions.shape

        encoder_out = self.encoder_proj(encoder_out)

        embeddings = self.embedding(captions)

        positions = torch.arange(L, device=captions.device).unsqueeze(0)
        embeddings = embeddings + self.pos_embedding[:, :L, :]

        dec_input = torch.zeros(B, L, encoder_out.size(-1), device=encoder_out.device)

        for i in range(L):
            if i == 0:
                dec_input[:, i : i + 1, :] = embeddings[:, i : i + 1, :]
            else:
                dec_input[:, i, :] = embeddings[:, i, :] + encoder_out.mean(
                    1, keepdim=True
                )

        if mask is not None:
            key_padding_mask = mask
        else:
            key_padding_mask = torch.zeros(
                B, L, dtype=torch.bool, device=captions.device
            )

        output = self.transformer(
            dec_input.permute(1, 0, 2),
            src_key_padding_mask=key_padding_mask,
        )
        output = output.permute(1, 0, 2)

        output = self.fc(output)

        return output

    def generate(
        self,
        encoder_out: Tensor,
        start_token: int,
        end_token: int,
        max_length: Optional[int] = None,
    ) -> Tensor:
        """
        Generate captions using greedy decoding.

        Args:
            encoder_out: Encoded video features
            start_token: Start token ID
            end_token: End token ID
            max_length: Maximum generation length

        Returns:
            Generated caption token IDs
        """
        if max_length is None:
            max_length = self.max_length

        B = encoder_out.size(0)

        encoder_out = self.encoder_proj(encoder_out)

        output = torch.full(
            (B, 1), start_token, dtype=torch.long, device=encoder_out.device
        )

        finished = torch.zeros(B, dtype=torch.bool, device=encoder_out.device)

        for _ in range(max_length - 1):
            embeddings = self.embedding(output)

            L = embeddings.size(1)
            embeddings = embeddings + self.pos_embedding[:, :L, :]

            output_transposed = embeddings.permute(1, 0, 2)

            output_encoded = self.transformer(output_transposed)
            output_encoded = output_encoded.permute(1, 0, 2)

            logits = self.fc(output_encoded)

            next_token = logits[:, -1, :].argmax(dim=-1)

            next_token = next_token.masked_fill(finished, end_token)

            output = torch.cat([output, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == end_token)

            if finished.all():
                break

        return output


class SoftAttention(nn.Module):
    """
    Soft Attention Mechanism.

    Standard attention mechanism for video captioning.

    Args:
        encoder_dim: Encoder feature dimension
        decoder_dim: Decoder hidden dimension
        attention_dim: Attention hidden dimension
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        attention_dim: int = 512,
    ):
        super().__init__()

        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.full_attn = nn.Linear(attention_dim, 1)

    def forward(
        self,
        encoder_out: Tensor,
        decoder_hidden: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute soft attention.

        Args:
            encoder_out: Video features (B, T, E)
            decoder_hidden: Decoder hidden state (B, D)

        Returns:
            Tuple of (context, weights)
        """
        attn1 = self.encoder_attn(encoder_out)
        attn2 = self.decoder_attn(decoder_hidden)

        attn = attn1 + attn2.unsqueeze(1)
        attn = torch.tanh(attn)
        attn = self.full_attn(attn).squeeze(-1)

        alpha = F.softmax(attn, dim=-1)

        context = torch.bmm(alpha.unsqueeze(1), encoder_out).squeeze(1)

        return context, alpha


class HierarchicalAttention(nn.Module):
    """
    Hierarchical Attention for Video Captioning.

    Applies attention at both frame-level and video-level.

    Args:
        encoder_dim: Encoder feature dimension
        decoder_dim: Decoder hidden dimension
        attention_dim: Attention hidden dimension
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        attention_dim: int = 512,
    ):
        super().__init__()

        self.frame_attention = SoftAttention(encoder_dim, decoder_dim, attention_dim)

        self.video_attention = nn.Sequential(
            nn.Linear(encoder_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self,
        encoder_out: Tensor,
        decoder_hidden: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute hierarchical attention.

        Args:
            encoder_out: Video features (B, T, E)
            decoder_hidden: Decoder hidden state (B, D)

        Returns:
            Tuple of (frame_context, video_context, frame_weights)
        """
        frame_context, frame_weights = self.frame_attention(encoder_out, decoder_hidden)

        video_weights = self.video_attention(encoder_out)
        video_weights = F.softmax(video_weights, dim=1)

        video_context = torch.bmm(video_weights.permute(0, 2, 1), encoder_out).squeeze(
            1
        )

        combined = torch.cat([frame_context, video_context], dim=-1)

        return combined, video_context, frame_weights


class BeamSearchDecoder:
    """
    Beam Search Decoder for Caption Generation.

    Implements beam search for improved caption generation.

    Args:
        decoder: Caption decoder model
        beam_size: Beam size for search
        max_length: Maximum caption length
        length_penalty: Length penalty factor
    """

    def __init__(
        self,
        decoder: nn.Module,
        beam_size: int = 5,
        max_length: int = 30,
        length_penalty: float = 1.0,
    ):
        self.decoder = decoder
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty

    def decode(
        self,
        encoder_out: Tensor,
        context: Tensor,
        start_token: int,
        end_token: int,
    ) -> List[Tuple[List[int], float]]:
        """
        Decode using beam search.

        Args:
            encoder_out: Encoded video features
            context: Global video context
            start_token: Start token ID
            end_token: End token ID

        Returns:
            List of (caption, score) tuples
        """
        device = encoder_out.device
        B = encoder_out.size(0)

        encoder_out = encoder_out.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
        encoder_out = encoder_out.view(B * self.beam_size, -1, encoder_out.size(-1))

        context = context.unsqueeze(1).repeat(1, self.beam_size, 1)
        context = context.view(B * self.beam_size, -1)

        beams = [([start_token], 0.0, None)]
        complete = []

        for step in range(self.max_length - 1):
            all_candidates = []

            for seq, score, hidden in beams:
                if seq[-1] == end_token:
                    complete.append((seq, score))
                    continue

                word = torch.tensor([seq[-1]], dtype=torch.long, device=device)

                pred, hidden, _ = self.decoder.decode_step(
                    encoder_out, context, word, hidden
                )

                log_probs = F.log_softmax(pred, dim=-1)

                topk_log_probs, topk_indices = log_probs.topk(self.beam_size)

                for i in range(self.beam_size):
                    new_seq = seq + [topk_indices[0, i].item()]
                    new_score = score + topk_log_probs[0, i].item()

                    penalty = ((len(new_seq) + 5) ** self.length_penalty) / (
                        6**self.length_penalty
                    )
                    new_score = new_score / penalty

                    all_candidates.append((new_seq, new_score, hidden))

            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            beams = all_candidates[: self.beam_size]

            if len(complete) >= self.beam_size:
                break

        all_candidates = beams + complete
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)

        return [(seq, score) for seq, score, _ in all_candidates[: self.beam_size]]


class VideoCaptionModel(nn.Module):
    """
    Complete Video Captioning Model.

    Combines video encoder and caption decoder.

    Args:
        encoder: Video encoder
        decoder: Caption decoder (LSTM or Transformer)
        config: Captioning configuration
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        config: Optional[CaptioningConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = CaptioningConfig()

        if encoder is None:
            encoder = VideoCaptionEncoder(
                feature_dim=2048,
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )

        if decoder is None:
            decoder = CaptionDecoder(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                encoder_dim=config.embed_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        video_features: Tensor,
        captions: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass during training.

        Args:
            video_features: Input video features
            captions: Target captions
            lengths: Caption lengths

        Returns:
            Prediction logits
        """
        encoder_out, context = self.encoder(video_features)

        outputs = self.decoder(encoder_out, context, captions, lengths)

        return outputs

    def generate(
        self,
        video_features: Tensor,
        start_token: int = 1,
        end_token: int = 2,
        max_length: Optional[int] = None,
        use_beam_search: bool = False,
        beam_size: int = 5,
    ) -> List[List[int]]:
        """
        Generate captions for videos.

        Args:
            video_features: Input video features
            start_token: Start token ID
            end_token: End token ID
            max_length: Maximum caption length
            use_beam_search: Whether to use beam search
            beam_size: Beam size for beam search

        Returns:
            List of generated caption token IDs
        """
        if max_length is None:
            max_length = (
                self.decoder.vocab_size if hasattr(self.decoder, "vocab_size") else 30
            )

        self.eval()

        with torch.no_grad():
            encoder_out, context = self.encoder(video_features)

            if use_beam_search:
                beam_decoder = BeamSearchDecoder(self.decoder, beam_size, max_length)
                results = beam_decoder.decode(
                    encoder_out, context, start_token, end_token
                )
                return [seq for seq, _ in results]
            else:
                B = video_features.size(0)
                device = video_features.device

                output = torch.full(
                    (B, 1), start_token, dtype=torch.long, device=device
                )
                hidden = self.decoder.init_hidden(B, device)

                finished = torch.zeros(B, dtype=torch.bool, device=device)

                for _ in range(max_length - 1):
                    word = output[:, -1]

                    pred, hidden, _ = self.decoder.decode_step(
                        encoder_out, context, word, hidden
                    )

                    next_token = pred.argmax(dim=-1)
                    next_token = next_token.masked_fill(finished, end_token)

                    output = torch.cat([output, next_token.unsqueeze(1)], dim=1)

                    finished = finished | (next_token == end_token)

                    if finished.all():
                        break

                return output.tolist()


def create_captions_model(
    vocab_size: int = 50000,
    encoder_type: str = "transformer",
    decoder_type: str = "lstm",
    **kwargs,
) -> VideoCaptionModel:
    """
    Create video captioning model.

    Args:
        vocab_size: Size of vocabulary
        encoder_type: Type of encoder ('transformer')
        decoder_type: Type of decoder ('lstm' or 'transformer')
        **kwargs: Additional arguments

    Returns:
        Video captioning model
    """
    embed_dim = kwargs.get("embed_dim", 512)
    hidden_dim = kwargs.get("hidden_dim", 512)
    num_layers = kwargs.get("num_layers", 2)
    num_heads = kwargs.get("num_heads", 8)
    dropout = kwargs.get("dropout", 0.3)
    max_length = kwargs.get("max_length", 30)

    encoder = VideoCaptionEncoder(
        feature_dim=kwargs.get("feature_dim", 2048),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    if decoder_type == "transformer":
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_length=max_length,
        )
    else:
        decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    return VideoCaptionModel(encoder=encoder, decoder=decoder)
