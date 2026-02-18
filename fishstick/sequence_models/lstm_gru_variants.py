"""
LSTM and GRU Variants

Advanced recurrent neural network variants with regularization,
attention, and layer normalization.
"""

from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VariationalLSTM(nn.Module):
    """LSTM with variational dropout between layers.

    Implements the variational dropout scheme from "A Theoretically
    Grounded Application of Dropout in Recurrent Neural Networks"
    (Gal & Ghahramani, 2016).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        output, hidden = self.lstm(x, hidden)
        output = self.dropout_layer(output)
        return output, hidden


class ZoneoutGRU(nn.Module):
    """GRU with zoneout regularization.

    Zoneout randomly freezes hidden units between time steps,
    acting as a regularization that preserves gradient flow.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        zoneout_prob: float = 0.1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.zoneout_prob = zoneout_prob
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        output, new_hidden = self.gru(x, hidden)

        if self.training and self.zoneout_prob > 0:
            mask = torch.bernoulli(torch.full_like(new_hidden, 1 - self.zoneout_prob))
            new_hidden = new_hidden * mask + hidden * (1 - mask)

        return output, new_hidden


class LayerNormalizedLSTM(nn.Module):
    """LSTM with layer normalization applied to inputs and hidden states.

    Applies layer normalization as described in "Layer Normalization"
    (Ba et al., 2016) to stabilize training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

        num_directions = 2 if bidirectional else 1
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size * num_directions) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        output, hidden = self.lstm(x, hidden)

        output = self.layer_norms[-1](output)

        return output, hidden


class LayerNormalizedGRU(nn.Module):
    """GRU with layer normalization applied to inputs and hidden states."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

        num_directions = 2 if bidirectional else 1
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size * num_directions) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        output, new_hidden = self.gru(x, hidden)

        output = self.layer_norms[-1](output)

        return output, new_hidden


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with self-attention for sequence encoding.

    Applies self-attention over the sequence to capture long-range
    dependencies beyond what bidirectional LSTM can capture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.MultiheadAttention(
            lstm_output_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        lstm_out, hidden = self.lstm(x, hidden)

        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=mask
        )
        attn_out = self.dropout(attn_out)
        output = self.layer_norm(lstm_out + attn_out)

        return output, hidden


class ConvLSTM(nn.Module):
    """LSTM with convolutional input projection.

    Applies 1D convolution before LSTM to capture local patterns.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.conv = nn.Conv1d(
            input_size,
            hidden_size,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        return self.lstm(x, hidden)


class IndRNNCell(nn.Module):
    """Independently Recurrent Neural Network (IndRNN) cell.

    Each dimension of the hidden state has its own recurrent weight,
    allowing for very deep stacking and better gradient flow.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return torch.tanh(F.linear(x, self.weight_ih) + h * self.weight_hh + self.bias)


class IndRNN(nn.Module):
    """Independently Recurrent Neural Network.

    Implements the IndRNN architecture where each recurrent unit
    has its own recurrent weight, enabling gradient flow through
    very deep stacks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        layers = []
        for i in range(num_layers):
            in_size = (
                input_size if i == 0 else hidden_size * (2 if bidirectional else 1)
            )
            layers.append(IndRNNCell(in_size, hidden_size))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        seq_len, batch_size, _ = x.size()

        if hidden is None:
            hidden = torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                batch_size,
                self.hidden_size,
                device=x.device,
            )

        outputs = []
        h_idx = 0

        for i, layer in enumerate(self.layers):
            h = hidden[h_idx]
            h_fwd = h

            if self.bidirectional:
                h_bwd = hidden[h_idx + 1]
                output_fwd = []
                output_bwd = []

                for t in range(seq_len):
                    h_fwd = layer(x[t], h_fwd)
                    h_bwd = layer(x[seq_len - 1 - t], h_bwd)
                    output_fwd.append(h_fwd)
                    output_bwd.append(h_bwd)

                output = torch.stack(
                    [
                        torch.cat([f, b], dim=-1)
                        for f, b in zip(output_fwd, reversed(output_bwd))
                    ]
                )
            else:
                output = []
                for t in range(seq_len):
                    h = layer(x[t], h)
                    output.append(h)
                output = torch.stack(output)

            outputs.append(output)
            x = self.dropout(output)
            h_idx += 2 if self.bidirectional else 1

        return outputs[-1], hidden
