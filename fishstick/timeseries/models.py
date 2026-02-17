"""
Time Series Forecasting Models
"""

from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TimeSeriesModel(nn.Module):
    """Base class for time series models."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class TemporalConvolutionalNetwork(TimeSeriesModel):
    """Temporal Convolutional Network for time series forecasting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_base**i
            padding = (kernel_size - 1) * dilation

            layer = nn.Sequential(
                nn.Conv1d(
                    hidden_dim if i > 0 else input_dim,
                    hidden_dim,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.layers.append(layer)

        self.final = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)[:, :, -x.size(2) :]

        x = self.final(x)
        return x.transpose(1, 2)


class TransformerTimeSeries(TimeSeriesModel):
    """Transformer-based time series model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return self.output_projection(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LSTMForecaster(TimeSeriesModel):
    """LSTM-based time series forecaster."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        _, (h, _) = self.lstm(x)
        h = h[-1]
        return self.fc(h).unsqueeze(1)


class GRUForecaster(TimeSeriesModel):
    """GRU-based time series forecaster."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        _, h = self.gru(x)
        h = h[-1]
        return self.fc(h).unsqueeze(1)


class WaveNet(TimeSeriesModel):
    """WaveNet model for time series."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 8,
        kernel_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            layer = nn.Conv1d(
                hidden_dim if i > 0 else input_dim,
                hidden_dim,
                kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation,
            )
            self.layers.append(
                nn.ModuleDict(
                    {
                        "conv": layer,
                        "gate": nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
                        "output": nn.Conv1d(hidden_dim, hidden_dim, 1),
                    }
                )
            )

        self.final = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)

        for layer_dict in self.layers:
            x = layer_dict["conv"](x)[:, :, -x.size(2) :]

            gates = layer_dict["gate"](x)
            gate_input, gate_transform = gates.chunk(2, dim=1)
            x = torch.tanh(gate_input) * torch.sigmoid(gate_transform)

            x = x + layer_dict["output"](x)[:, :, -x.size(2) :]
            x = F.relu(x)

        return self.final(x).transpose(1, 2)


class Informer(TimeSeriesModel):
    """Informer model for long-term time series forecasting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, output_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * factor,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = self.encoder(x)
        return self.output_projection(x[:, -1:, :])


class NBeats(TimeSeriesModel):
    """N-BEATS model for time series forecasting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stack_types: list = ["trend", "seasonality"],
        nb_blocks_per_stack: int = 3,
        forecast_length: int = 1,
        backcast_length: int = 10,
        thetas_dim: list = [4, 8],
        share_weights_in_stack: bool = False,
    ):
        super().__init__(input_dim, output_dim)

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length

        self.stacks = nn.ModuleList()
        for stack_type in stack_types:
            for _ in range(nb_blocks_per_stack):
                self.stacks.append(
                    NBeatsBlock(
                        units=128,
                        thetas_dim=thetas_dim[0]
                        if stack_type == "trend"
                        else thetas_dim[1],
                        backcast_length=backcast_length,
                        forecast_length=forecast_length,
                    )
                )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        backcast = x
        for block in self.stacks:
            b, f = block(backcast)
            backcast = backcast - b

        return f


class NBeatsBlock(nn.Module):
    def __init__(
        self, units: int, thetas_dim: int, backcast_length: int, forecast_length: int
    ):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)

        self.backcast_fc = nn.Linear(units, backcast_length)
        self.forecast_fc = nn.Linear(units, forecast_length)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)

        return backcast, forecast
