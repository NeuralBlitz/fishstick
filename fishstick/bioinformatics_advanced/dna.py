import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class GenomicVariant:
    chrom: str
    position: int
    ref: str
    alt: str
    quality: float
    genotype: Optional[torch.Tensor] = None


class DNASequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 8192, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = tokens.shape

        x = self.token_embedding(tokens)

        if seq_len <= self.position_embedding.size(1):
            x = x + self.position_embedding[:, :seq_len, :]

        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)

        return x

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        dna_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

        tokens = torch.tensor(
            [[dna_to_idx.get(bp.upper(), 4) for bp in sequence]],
            device=next(self.parameters()).device,
        )

        with torch.no_grad():
            embeddings = self.forward(tokens)

        return embeddings


class VariantEffectPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = DNASequenceEncoder(embed_dim=embed_dim)

        self.variant_encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        ref_tokens: torch.Tensor,
        alt_tokens: torch.Tensor,
        pos_tokens: torch.Tensor,
    ) -> torch.Tensor:
        ref_emb = self.encoder(ref_tokens)
        alt_emb = self.encoder(alt_tokens)
        pos_emb = self.encoder(pos_tokens)

        variant_features = torch.cat([ref_emb, alt_emb, pos_emb], dim=-1)
        variant_emb = self.variant_encoder(variant_features)

        logits = self.classifier(variant_emb.mean(dim=1))

        return logits

    def predict_effect(
        self,
        ref_sequence: str,
        alt_sequence: str,
        variant_pos: int,
        context_size: int = 100,
    ) -> Tuple[int, torch.Tensor]:
        dna_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

        start = max(0, variant_pos - context_size)
        end = variant_pos + context_size

        ref_context = ref_sequence[start:end]
        alt_context = alt_sequence[start:end]

        ref_tokens = torch.tensor(
            [[dna_to_idx.get(bp.upper(), 4) for bp in ref_context]],
            device=next(self.parameters()).device,
        )
        alt_tokens = torch.tensor(
            [[dna_to_idx.get(bp.upper(), 4) for bp in alt_context]],
            device=next(self.parameters()).device,
        )

        pos_tokens = torch.zeros_like(ref_tokens)
        pos_tokens[0, context_size] = 1

        with torch.no_grad():
            logits = self.forward(ref_tokens, alt_tokens, pos_tokens)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

        return pred.item(), probs[0]


class GenomicVariantCaller(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_features: int = 64,
    ):
        super().__init__()
        self.encoder = DNASequenceEncoder(embed_dim=embed_dim)

        self.read_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_features),
        )

        self.variant_mlp = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_features, 3),
        )

        self.quality_predictor = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, 1),
        )

    def forward(
        self,
        ref_tokens: torch.Tensor,
        read_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_emb = self.encoder(ref_tokens)

        batch_size, num_reads, seq_len = read_tokens.shape
        read_flat = read_tokens.view(batch_size * num_reads, seq_len)
        read_emb = self.encoder(read_flat)
        read_emb = read_emb.view(batch_size, num_reads, seq_len, -1)

        read_features = self.read_encoder(read_emb)

        ref_features = self.read_encoder(ref_emb).unsqueeze(1)

        combined = torch.cat(
            [
                ref_features.expand(-1, num_reads, -1, -1),
                read_features,
            ],
            dim=-1,
        )

        combined_flat = combined.view(batch_size * num_reads, seq_len, -1)
        variant_logits = self.variant_mlp(combined_flat.mean(dim=1))
        variant_logits = variant_logits.view(batch_size, num_reads, -1)

        quality = self.quality_predictor(read_features).squeeze(-1)

        return variant_logits, quality

    def call_variants(
        self,
        ref_sequence: str,
        reads: List[str],
    ) -> List[GenomicVariant]:
        dna_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

        ref_tokens = torch.tensor(
            [[dna_to_idx.get(bp.upper(), 4) for bp in ref_sequence]],
            device=next(self.parameters()).device,
        )

        read_tokens_list = []
        for read in reads:
            read_tokens_list.append([dna_to_idx.get(bp.upper(), 4) for bp in read])

        read_tokens = torch.tensor(
            [read_tokens_list], device=next(self.parameters()).device
        )

        with torch.no_grad():
            variant_logits, quality = self.forward(ref_tokens, read_tokens)

        variant_probs = F.softmax(variant_logits, dim=-1)
        variant_calls = torch.argmax(variant_probs, dim=-1)

        variants = []
        for i in range(len(reads)):
            for j in range(variant_calls.size(1)):
                if variant_calls[0, i, j].item() != 0:
                    qual = quality[0, i, j].item()
                    variants.append(
                        GenomicVariant(
                            chrom="chr1",
                            position=j,
                            ref=ref_sequence[j],
                            alt="N",
                            quality=qual,
                            genotype=variant_probs[0, i, j],
                        )
                    )

        return variants


class ReadAligner(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
    ):
        super().__init__()
        self.encoder = DNASequenceEncoder(embed_dim=embed_dim)

        self.alignment_head = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )

        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        ref_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_emb = self.encoder(query_tokens)
        ref_emb = self.encoder(ref_tokens)

        alignment_scores, _ = self.alignment_head(
            query_emb,
            ref_emb,
            ref_emb,
        )

        alignment_scores = alignment_scores + query_emb

        scores = self.score_predictor(alignment_scores).squeeze(-1)

        return scores, alignment_scores

    def align_read(
        self,
        read_sequence: str,
        reference: str,
    ) -> Tuple[torch.Tensor, int]:
        dna_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

        read_tokens = torch.tensor(
            [[dna_to_idx.get(bp.upper(), 4) for bp in read_sequence]],
            device=next(self.parameters()).device,
        )

        ref_tokens = torch.tensor(
            [[dna_to_idx.get(bp.upper(), 4) for bp in reference]],
            device=next(self.parameters()).device,
        )

        with torch.no_grad():
            scores, _ = self.forward(read_tokens, ref_tokens)

        best_pos = torch.argmax(scores.mean(dim=0)).item()

        return scores, best_pos
