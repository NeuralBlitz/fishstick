import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ProteinStructure:
    sequence: str
    coordinates: torch.Tensor
    confidence: Optional[torch.Tensor] = None


class ProteinGraph(nn.Module):
    def __init__(
        self,
        num_amino_acids: int = 21,
        node_hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_amino_acids = num_amino_acids
        self.node_hidden_dim = node_hidden_dim

        self.aa_embedding = nn.Embedding(num_amino_acids, node_hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_hidden_dim * 2 + 1, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, 1),
        )

        self.node_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=node_hidden_dim,
                    nhead=4,
                    dim_feedforward=node_hidden_dim * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        aa_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.aa_embedding(aa_indices)

        for layer in self.node_layers:
            x = layer(x)

        return x


class ESMEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 33,
        embed_dim: int = 480,
        num_layers: int = 6,
        num_heads: int = 6,
        ff_dim: int = 1920,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 1024, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
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
        else:
            x = x + self.position_embedding[:, :1, :].expand(-1, seq_len, -1)

        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)

        return x

    def extract_embeddings(
        self,
        tokens: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        x = self.forward(tokens)
        return x


class ProteinStructurePredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 480,
        num_angles: int = 3,
        num_iterations: int = 48,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_angles = num_angles
        self.num_iterations = num_iterations

        self.esm_embedder = ESMEmbedder(embed_dim=embed_dim)

        self.angle_prediction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_angles),
        )

        self.frame_transition = nn.Sequential(
            nn.Linear(embed_dim + 6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 6),
        )

        self.structure_module = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim + 3,
                    nhead=4,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                )
                for _ in range(3)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.esm_embedder(tokens, attention_mask)

        angles = self.angle_prediction(embeddings)
        angles = torch.tanh(angles)

        batch_size, seq_len, _ = embeddings.shape
        coordinates = torch.zeros(batch_size, seq_len, 3, device=tokens.device)

        for i in range(self.num_iterations):
            frame_input = torch.cat(
                [
                    embeddings,
                    coordinates.flatten(2),
                ],
                dim=-1,
            )

            transition = self.frame_transition(frame_input)
            t = transition[..., :3]

            coordinates = coordinates + t.mean(dim=1, keepdim=True)

            for layer in self.structure_module:
                node_input = torch.cat([embeddings, coordinates], dim=-1)
                node_input = layer(node_input)

        confidence = torch.sigmoid(angles.abs().mean(dim=-1))

        return coordinates, confidence

    def predict_structure(
        self,
        sequence: str,
    ) -> ProteinStructure:
        aa_to_idx = {
            "A": 0,
            "R": 1,
            "N": 2,
            "D": 3,
            "C": 4,
            "Q": 5,
            "E": 6,
            "G": 7,
            "H": 8,
            "I": 9,
            "L": 10,
            "K": 11,
            "M": 12,
            "F": 13,
            "P": 14,
            "S": 15,
            "T": 16,
            "W": 17,
            "Y": 18,
            "V": 19,
            "X": 20,
        }

        tokens = torch.tensor(
            [[aa_to_idx.get(aa, 20) for aa in sequence]],
            device=next(self.parameters()).device,
        )

        with torch.no_grad():
            coordinates, confidence = self.forward(tokens)

        return ProteinStructure(
            sequence=sequence,
            coordinates=coordinates[0],
            confidence=confidence[0] if confidence is not None else None,
        )
