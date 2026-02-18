import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any


class DartsOperation(nn.Module):
    def __init__(self, C_in: int, C_out: int, stride: int = 1):
        super().__init__()
        self.ops = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_out, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(C_out),
                ),
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_out, 5, stride, 2, bias=False),
                    nn.BatchNorm2d(C_out),
                ),
                nn.AvgPool2d(3, stride, 1, count_include_pad=False),
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C_in, C_out, 1, stride, 0, bias=False),
                    nn.BatchNorm2d(C_out),
                ),
            ]
        )
        self.alpha = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class DartsCell(nn.Module):
    def __init__(self, C_in: int, C_out: int, num_nodes: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.preprocess = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
        )
        self.ops = nn.ModuleList(
            [DartsOperation(C_out, C_out) for _ in range(num_nodes)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        states = [x]
        for i in range(self.num_nodes):
            new_state = self.ops[i](states[-1])
            states.append(new_state)
        return states[1:]


class DartsNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_layers: int = 8,
        C: int = 36,
        num_nodes: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.cells = nn.ModuleList(
            [
                DartsCell(C, C, num_nodes) if i % 2 == 0 else DartsCell(C, C, num_nodes)
                for i in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(C, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)[-1]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


class NasNetCell(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        num_steps: int = 4,
        reduction: bool = False,
    ):
        super().__init__()
        self.num_steps = num_steps
        if reduction:
            self.ops = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AvgPool2d(2, 2),
                        nn.ReLU(),
                        nn.Conv2d(C_in, C_out, 1, bias=False),
                        nn.BatchNorm2d(C_out),
                    ),
                    nn.Sequential(
                        nn.MaxPool2d(2, 2),
                        nn.ReLU(),
                        nn.Conv2d(C_in, C_out, 1, bias=False),
                        nn.BatchNorm2d(C_out),
                    ),
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(C_in, C_out, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(C_out),
                    ),
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(C_in, C_out, 5, 2, 2, bias=False),
                        nn.BatchNorm2d(C_out),
                    ),
                ]
            )
        else:
            self.ops = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(C_in, C_out, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(C_out),
                    ),
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(C_in, C_out, 5, 1, 2, bias=False),
                        nn.BatchNorm2d(C_out),
                    ),
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(C_in, C_out, 7, 1, 3, bias=False),
                        nn.BatchNorm2d(C_out),
                    ),
                    nn.AvgPool2d(3, 1, 1),
                ]
            )
        self.alphas = nn.Parameter(torch.randn(num_steps, len(self.ops)))

    def forward(self, states: List[torch.Tensor]) -> torch.Tensor:
        new_states = []
        for step in range(self.num_steps):
            weights = F.softmax(self.alphas[step], dim=0)
            new_state = sum(w * op(s) for w, op in zip(weights, self.ops))
            new_states.append(new_state)
        return sum(new_states)


class NasNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_cells: int = 18,
        stem_filters: int = 32,
        filters: int = 44,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_filters),
        )
        self.cells = nn.ModuleList()
        for i in range(num_cells):
            is_reduction = i % 2 == 1
            C_in = stem_filters if i == 0 else filters
            cell = NasNetCell(C_in, filters, reduction=is_reduction)
            self.cells.append(cell)
        self.classifier = nn.Linear(filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        states = [x]
        for cell in self.cells:
            x = cell(states)
            states.append(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


class FBNetBlock(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        hidden_dim = C_in * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(C_in, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, C_out, 1, bias=False),
                nn.BatchNorm2d(C_out),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FBNetStage(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        num_blocks: int,
        stride: int = 1,
    ):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1
            blocks.append(FBNetBlock(C_in, C_out, block_stride))
            C_in = C_out
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class FBNetSupernet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        stem_filters: int = 16,
        stages: List[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        if stages is None:
            stages = [
                (16, 16, 1, 1),
                (16, 24, 2, 2),
                (24, 32, 3, 2),
                (32, 96, 4, 2),
                (96, 256, 3, 1),
            ]
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_filters),
        )
        self.stages = nn.ModuleList()
        for stage_cfg in stages:
            C_in, C_out, num_blocks, stride = stage_cfg
            self.stages.append(FBNetStage(C_in, C_out, num_blocks, stride))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


def create_darts_search_space(num_classes: int = 10) -> DartsNetwork:
    return DartsNetwork(num_classes=num_classes)


def create_nasnet_search_space(num_classes: int = 10) -> NasNet:
    return NasNet(num_classes=num_classes)


def create_fbnet_search_space(num_classes: int = 10) -> FBNetSupernet:
    return FBNetSupernet(num_classes=num_classes)
