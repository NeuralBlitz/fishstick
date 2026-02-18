import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Callable
import copy


class ProgressiveColumn(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        column_id: int,
        lateral_dim: Optional[int] = None,
    ):
        super().__init__()
        self.column_id = column_id
        self.hidden_dims = hidden_dims
        self.lateral_dim = lateral_dim

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)

        if lateral_dim is not None:
            self.lateral_connections = nn.ModuleDict()
        else:
            self.lateral_connections = nn.ModuleDict()

    def forward(
        self,
        x: torch.Tensor,
        lateral_inputs: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))

            if (
                lateral_inputs is not None
                and self.column_id in self.lateral_connections
            ):
                lateral_layer = self.lateral_connections[self.column_id]
                if (
                    isinstance(lateral_inputs, dict)
                    and self.column_id in lateral_inputs
                ):
                    h = h + lateral_layer(lateral_inputs[self.column_id])

        return self.output_layer(h)

    def add_lateral_input(
        self,
        from_column: int,
        lateral_module: nn.Module,
    ):
        self.lateral_connections[str(from_column)] = lateral_module


class ProgressiveNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        n_tasks: int,
        lateral_scale: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_tasks = n_tasks
        self.lateral_scale = lateral_scale

        self.columns: nn.ModuleList = nn.ModuleList()
        self.task_outputs: nn.ModuleList = nn.ModuleList()
        self.lateral_layers: nn.ModuleDict = nn.ModuleDict()

        self.current_task = 0
        self.column_dims: List[int] = []

    def add_task(
        self,
        output_dim: Optional[int] = None,
        freeze_previous: bool = False,
    ) -> int:
        if self.current_task >= self.n_tasks:
            raise RuntimeError(f"Cannot add more than {self.n_tasks} tasks")

        task_output_dim = output_dim if output_dim is not None else self.output_dim

        if self.current_task == 0:
            input_dim = self.input_dim
        else:
            input_dim = self.hidden_dims[-1] + sum(
                self.column_dims[: self.current_task]
            )

        column = ProgressiveColumn(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=task_output_dim,
            column_id=self.current_task,
            lateral_dim=self.hidden_dims[-1] if self.current_task > 0 else None,
        )

        self.columns.append(column)
        self.column_dims.append(task_output_dim)

        task_output = nn.Linear(task_output_dim, task_output_dim)
        self.task_outputs.append(task_output)

        if self.current_task > 0:
            self._create_lateral_connections()

        if freeze_previous and self.current_task > 0:
            for col in self.columns[:-1]:
                for param in col.parameters():
                    param.requires_grad = False

        task_id = self.current_task
        self.current_task += 1

        return task_id

    def _create_lateral_connections(self):
        new_column = self.columns[-1]
        prev_output_dim = self.hidden_dims[-1]

        for i in range(len(self.columns) - 1):
            lateral = nn.Linear(prev_output_dim, self.hidden_dims[-1])
            self.lateral_layers[f"{i}_to_{len(self.columns) - 1}"] = lateral
            new_column.add_lateral_input(i, lateral)

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        if task_id is None:
            task_id = self.current_task - 1

        if task_id >= len(self.columns):
            raise ValueError(f"Task {task_id} not found")

        column_outputs = []
        lateral_inputs: Dict[int, torch.Tensor] = {}

        for i in range(task_id + 1):
            if i == 0:
                col_input = x
            else:
                col_input = torch.cat([x] + column_outputs[:i], dim=-1)

            if i > 0 and i in [c.column_id for c in self.columns[:i]]:
                col_output = self.columns[i](col_input, lateral_inputs)
            else:
                col_output = self.columns[i](col_input)

            column_outputs.append(col_output)

        if return_all:
            return torch.stack(column_outputs, dim=0)

        return column_outputs[task_id]

    def get_task_output(self, features: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.task_outputs[task_id](features)

    def freeze_task(self, task_id: int):
        if task_id < len(self.columns):
            for param in self.columns[task_id].parameters():
                param.requires_grad = False
            for param in self.task_outputs[task_id].parameters():
                param.requires_grad = False

    def unfreeze_task(self, task_id: int):
        if task_id < len(self.columns):
            for param in self.columns[task_id].parameters():
                param.requires_grad = True
            for param in self.task_outputs[task_id].parameters():
                param.requires_grad = True

    def get_task_params(self, task_id: int) -> Dict[str, torch.Tensor]:
        params = {}
        if task_id < len(self.columns):
            for name, param in self.columns[task_id].named_parameters():
                params[f"column_{task_id}.{name}"] = param
            for name, param in self.task_outputs[task_id].named_parameters():
                params[f"task_output_{task_id}.{name}"] = param
        return params


def add_lateral_connections(
    model: nn.Module,
    column_indices: List[int],
    lateral_dim: int,
) -> nn.Module:
    for i, j in zip(column_indices[:-1], column_indices[1:]):
        lateral_layer = nn.Linear(lateral_dim, lateral_dim)
        model.lateral_layers[f"{i}_to_{j}"] = lateral_layer
    return model


def adapt_to_task(
    model: ProgressiveNetwork,
    task_id: int,
    x: torch.Tensor,
    use_all_columns: bool = True,
) -> torch.Tensor:
    if use_all_columns:
        return model(x, task_id=task_id)
    else:
        if task_id < len(model.columns):
            col_input = x
            if task_id > 0:
                col_input = torch.cat(
                    [x]
                    + [
                        torch.zeros(x.size(0), model.column_dims[i], device=x.device)
                        for i in range(task_id)
                    ],
                    dim=-1,
                )
            return model.columns[task_id](col_input)
        else:
            raise ValueError(f"Task {task_id} not found")


class ProgressiveWithAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        n_tasks: int,
        adapter_dim: int = 64,
    ):
        super().__init__()
        self.progressive = ProgressiveNetwork(
            input_dim, hidden_dims, output_dim, n_tasks
        )
        self.adapters = nn.ModuleDict()

        for task_id in range(n_tasks):
            adapter = nn.Sequential(
                nn.Linear(input_dim, adapter_dim),
                nn.ReLU(),
                nn.Linear(adapter_dim, input_dim),
            )
            self.adapters[str(task_id)] = adapter

    def forward(
        self,
        x: torch.Tensor,
        task_id: int,
        use_adapter: bool = True,
    ) -> torch.Tensor:
        if use_adapter and str(task_id) in self.adapters:
            adapted_x = x + self.adapters[str(task_id)](x)
        else:
            adapted_x = x

        return self.progressive(adapted_x, task_id=task_id)


class LateralConnection(nn.Module):
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale
        self.lateral = nn.Linear(source_dim, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.lateral(x)


def create_task_specific_head(
    input_dim: int,
    output_dim: int,
    bias: bool = True,
) -> nn.Linear:
    return nn.Linear(input_dim, output_dim, bias=bias)


class ProgressiveEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        n_tasks: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_tasks = n_tasks

        self.task_encoders = nn.ModuleList()

        for _ in range(n_tasks):
            encoder = nn.ModuleList()
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                encoder.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim

            self.task_encoders.append(encoder)

        self.lateral_connections = nn.ModuleList()

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        h = x

        for i in range(task_id + 1):
            encoder = self.task_encoders[i]
            for layer in encoder:
                h = F.relu(layer(h))

                if i > 0 and i < len(self.lateral_connections):
                    h = h + self.lateral_connections[i - 1](x)

        return h
