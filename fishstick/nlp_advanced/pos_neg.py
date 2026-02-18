import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


POS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

DEP_LABELS = [
    "ROOT",
    "acl",
    "acomp",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "attr",
    "agent",
    "aux",
    "auxpass",
    "case",
    "cc",
    "ccomp",
    "compound",
    "conj",
    "cop",
    "csubj",
    "csubjpass",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "dobj",
    "expl",
    "iobj",
    "mark",
    "mwe",
    "neg",
    "nn",
    "nmod",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "num",
    "oprd",
    "parataxis",
    "pcomp",
    "pobj",
    "poss",
    "preconj",
    "predet",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "relcl",
    "root",
    "xcomp",
    "xsubj",
]


class BiAffineAttention(nn.Module):
    def __init__(self, dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dim = dim

        self.W = nn.Bilinear(dim, dim, num_labels, bias=False)
        self.u_head = nn.Linear(dim, num_labels, bias=False)
        self.u_dep = nn.Linear(dim, num_labels, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        head_repr: torch.Tensor,
        dep_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        head_repr = self.dropout(head_repr)
        dep_repr = self.dropout(dep_repr)

        bilinear = self.W(head_repr, dep_repr)
        u_h = self.u_head(head_repr)
        u_d = self.u_dep(dep_repr)

        scores = bilinear + u_h + u_d

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        return scores


class BiaffineParser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 400,
        num_layers: int = 3,
        num_pos_tags: int = 18,
        num_dep_labels: int = 47,
        dropout: float = 0.33,
        use_char_rnn: bool = True,
        char_embedding_dim: int = 50,
        char_hidden_dim: int = 50,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_pos_tags = num_pos_tags
        self.num_dep_labels = num_dep_labels

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(num_pos_tags, embedding_dim, padding_idx=0)

        self.use_char_rnn = use_char_rnn
        if use_char_rnn:
            self.char_embedding = nn.Embedding(
                vocab_size, char_embedding_dim, padding_idx=0
            )
            self.char_rnn = nn.LSTM(
                char_embedding_dim,
                char_hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.char_projection = nn.Linear(char_hidden_dim * 2, embedding_dim)
            self.input_dim = embedding_dim * 3
        else:
            self.input_dim = embedding_dim * 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dep_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.biaffine = BiAffineAttention(hidden_dim, num_dep_labels, dropout)

        self.pos_classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pos_tags),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        pos_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        word_embeds = self.word_embedding(input_ids)

        pos_embeds = torch.zeros_like(word_embeds)

        word_repr = torch.cat([word_embeds, pos_embeds], dim=-1)

        if self.use_char_rnn and char_ids is not None:
            char_embeds = self.char_embedding(char_ids)

            char_lengths = (char_ids != 0).sum(dim=-1).clamp(min=1)
            char_max_len = char_ids.size(-1)

            char_embeds_packed = nn.utils.rnn.pack_padded_sequence(
                char_embeds.view(-1, char_max_len, char_embeds.size(-1)),
                char_lengths.view(-1),
                batch_first=True,
                enforce_sorted=False,
            )

            char_output, _ = self.char_rnn(char_embeds_packed)
            char_output, _ = nn.utils.rnn.pad_packed_sequence(
                char_output, batch_first=True
            )

            char_repr = char_output.mean(dim=1)
            char_repr = self.char_projection(char_repr)
            char_repr = char_repr.view(batch_size, seq_len, -1)

            word_repr = torch.cat([word_repr, char_repr], dim=-1)

        encoded = self.encoder(word_repr)

        head_repr = self.head_mlp(encoded)
        dep_repr = self.dep_mlp(encoded)

        dep_scores = self.biaffine(head_repr, dep_repr)

        pos_logits = self.pos_classifier(encoded)

        return pos_logits, dep_scores

    def predict(
        self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_logits, dep_scores = self.forward(input_ids, char_ids)

        pos_preds = pos_logits.argmax(dim=-1)

        batch_size, seq_len, num_labels = dep_scores.shape

        head_preds = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=input_ids.device
        )

        for b in range(batch_size):
            for i in range(seq_len):
                head_scores = dep_scores[b, i, :]
                head_preds[b, i] = head_scores.argmax()

            root_idx = torch.argmax(dep_scores[b, :, 0])
            head_preds[b, 0] = root_idx

        return pos_preds, head_preds


class NeuralPOSTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_tags: int = 18,
        dropout: float = 0.2,
        use_crf: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.use_crf = use_crf

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_tags)

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        out = F.relu(self.fc(lstm_out))
        out = self.dropout(out)

        emissions = self.classifier(out)

        if self.training and tags is not None and self.use_crf:
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss

        return emissions

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        emissions = self.forward(input_ids)

        if self.use_crf:
            predictions = self.crf.decode(emissions, mask=mask)
            return torch.tensor(predictions, device=input_ids.device)

        return emissions.argmax(dim=-1)


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == "mean":
            return -llh.mean()
        elif reduction == "sum":
            return -llh.sum()
        else:
            return -llh

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def decode(
        self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> List[List[int]]:
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class BiLSTMPOSTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_tags: int = 18,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x != 0

        embeds = self.embedding(x)
        embeds = self.dropout(embeds)

        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)

        logits = self.fc(lstm_out)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits.argmax(dim=-1)


class DependencyParser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 400,
        num_tags: int = 18,
        num_labels: int = 47,
        num_layers: int = 3,
        dropout: float = 0.33,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(num_tags, embedding_dim, padding_idx=0)

        total_dim = embedding_dim * 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_mlp = nn.Linear(total_dim, hidden_dim)
        self.dep_mlp = nn.Linear(total_dim, hidden_dim)

        self.biaffine = BiAffineAttention(hidden_dim, num_labels, dropout)

    def forward(
        self,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
        heads: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        word_embeds = self.embedding(words)
        pos_embeds = self.pos_embedding(pos_tags)

        combined = torch.cat([word_embeds, pos_embeds], dim=-1)

        encoded = self.encoder(combined)

        head_repr = F.relu(self.head_mlp(encoded))
        dep_repr = F.relu(self.dep_mlp(encoded))

        scores = self.biaffine(head_repr, dep_repr)

        return scores

    def parse(
        self, words: torch.Tensor, pos_tags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.forward(words, pos_tags)

        batch_size, seq_len, _ = scores.shape

        pred_heads = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=words.device
        )

        for b in range(batch_size):
            for i in range(seq_len):
                pred_heads[b, i] = scores[b, i].argmax()

        return pred_heads


class ConstituencyParser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_tags: int = 18,
        num_constituents: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.scorer = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, num_constituents)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)

        encoded, _ = self.encoder(embeds)
        encoded = self.dropout(encoded)

        span_scores = self.scorer(encoded).squeeze(-1)

        label_logits = self.classifier(encoded)

        return span_scores, label_logits

    def predict(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        span_scores, label_logits = self.forward(x)

        constituents = (span_scores > threshold).long()
        labels = label_logits.argmax(dim=-1)

        return constituents, labels


def build_pos_tagger(
    vocab_size: int, num_tags: int = 18, model_type: str = "bilstm"
) -> nn.Module:
    if model_type == "bilstm":
        return BiLSTMPOSTagger(vocab_size, num_tags=num_tags)
    elif model_type == "transformer":
        return NeuralPOSTagger(vocab_size, num_tags=num_tags)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_dependency_parser(
    vocab_size: int, num_tags: int = 18, num_labels: int = 47
) -> nn.Module:
    return DependencyParser(vocab_size, num_tags=num_tags, num_labels=num_labels)


class NLTKPOSMapper:
    def __init__(self):
        self._nltk_to_universal = {
            "NN": "NOUN",
            "NNS": "NOUN",
            "NNP": "PROPN",
            "NNPS": "PROPN",
            "VB": "VERB",
            "VBD": "VERB",
            "VBG": "VERB",
            "VBN": "VERB",
            "VBP": "VERB",
            "VBZ": "VERB",
            "JJ": "ADJ",
            "JJR": "ADJ",
            "JJS": "ADJ",
            "RB": "ADV",
            "RBR": "ADV",
            "RBS": "ADV",
            "PRP": "PRON",
            "PRP$": "PRON",
            "WP": "PRON",
            "WP$": "PRON",
            "DT": "DET",
            "PDT": "DET",
            "WDT": "DET",
            "IN": "ADP",
            "TO": "PART",
            "CC": "CCONJ",
            "CD": "NUM",
            "MD": "AUX",
            "CCONJ": "CCONJ",
            "SCONJ": "SCONJ",
            "POS": "PART",
            "RP": "PART",
            "EX": "PRON",
            "FW": "X",
            "LS": "X",
            "UH": "INTJ",
            "SYM": "SYM",
            ":": "PUNCT",
            ".": "PUNCT",
        }

    def to_universal(self, tag: str) -> str:
        return self._nltk_to_universal.get(tag, "X")

    def to_nltk(self, tag: str) -> str:
        reverse_map = {v: k for k, v in self._nltk_to_universal.items()}
        return reverse_map.get(tag, "X")
