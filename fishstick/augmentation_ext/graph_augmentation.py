"""
Graph Augmentation Module

Augmentation techniques for graph-structured data including:
- Node and edge dropping
- Attribute masking
- Subgraph extraction
- Graph mixing
"""

from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import torch
import numpy as np
from numpy.typing import NDArray

from fishstick.augmentation_ext.base import AugmentationBase


@dataclass
class GraphData:
    """Data structure for graph data."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    num_nodes: Optional[int] = None


class NodeDrop(AugmentationBase):
    """
    Randomly drop nodes from the graph.

    Reference: Rong et al., "DropEdge: Towards Deep Graph Convolutional Networks", 2020
    """

    def __init__(
        self,
        drop_ratio: float = 0.1,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.drop_ratio = drop_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, data: Union[GraphData, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[GraphData, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply node dropping to graph.

        Args:
            data: GraphData object or tuple of (node_features, edge_index)

        Returns:
            Augmented graph
        """
        if not self._should_apply():
            return data

        if isinstance(data, GraphData):
            return self._apply_to_graphdata(data)
        else:
            x, edge_index = data
            return self._apply_to_tensors(x, edge_index)

    def _apply_to_graphdata(self, data: GraphData) -> GraphData:
        num_nodes = data.x.size(0)
        keep_nodes = int(num_nodes * (1 - self.drop_ratio))

        if keep_nodes >= num_nodes or keep_nodes <= 0:
            return data

        node_idx = self.rng.permutation(num_nodes)[:keep_nodes]
        node_idx_set = set(node_idx.tolist())

        new_x = data.x[node_idx]

        edge_mask = torch.tensor(
            [n in node_idx_set for n in range(num_nodes)], device=data.edge_index.device
        )

        edge_index = data.edge_index[:, edge_mask[data.edge_index]]
        new_edge_index = edge_index.clone()

        old_to_new = torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)
        old_to_new[node_idx] = torch.arange(keep_nodes, device=data.x.device)

        new_edge_index = old_to_new[new_edge_index]

        edge_attr = data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr[edge_mask[data.edge_index]]

        return GraphData(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=data.y,
            num_nodes=keep_nodes,
        )

    def _apply_to_tensors(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        keep_nodes = int(num_nodes * (1 - self.drop_ratio))

        if keep_nodes >= num_nodes or keep_nodes <= 0:
            return x, edge_index

        node_idx = self.rng.permutation(num_nodes)[:keep_nodes]
        node_idx_set = set(node_idx.tolist())

        new_x = x[node_idx]

        edge_mask = torch.tensor(
            [n in node_idx_set for n in range(num_nodes)], device=edge_index.device
        )

        edge_index = edge_index[:, edge_mask[edge_index]]
        old_to_new = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        old_to_new[node_idx] = torch.arange(keep_nodes, device=edge_index.device)
        new_edge_index = old_to_new[edge_index]

        return new_x, new_edge_index


class EdgeDrop(AugmentationBase):
    """
    Randomly drop edges from the graph.

    Reference: Rong et al., "DropEdge", 2020
    """

    def __init__(
        self,
        drop_ratio: float = 0.1,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.drop_ratio = drop_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, data: Union[GraphData, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[GraphData, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply edge dropping to graph.

        Args:
            data: GraphData object or tuple of (node_features, edge_index)

        Returns:
            Augmented graph
        """
        if not self._should_apply():
            return data

        if isinstance(data, GraphData):
            return self._apply_to_graphdata(data)
        else:
            x, edge_index = data
            return self._apply_to_tensors(x, edge_index)

    def _apply_to_graphdata(self, data: GraphData) -> GraphData:
        num_edges = data.edge_index.size(1)
        keep_edges = int(num_edges * (1 - self.drop_ratio))

        if keep_edges >= num_edges or keep_edges <= 0:
            return data

        edge_idx = self.rng.permutation(num_edges)[:keep_edges]
        new_edge_index = data.edge_index[:, edge_idx]

        edge_attr = data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr[edge_idx]

        return GraphData(
            x=data.x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=data.y,
            num_nodes=data.num_nodes,
        )

    def _apply_to_tensors(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_edges = edge_index.size(1)
        keep_edges = int(num_edges * (1 - self.drop_ratio))

        if keep_edges >= num_edges or keep_edges <= 0:
            return x, edge_index

        edge_idx = self.rng.permutation(num_edges)[:keep_edges]
        return x, edge_index[:, edge_idx]


class AttributeMasking(AugmentationBase):
    """Randomly mask node or edge attributes."""

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_type: str = "node",
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.rng = np.random.RandomState(seed)

    def __call__(self, data: GraphData) -> GraphData:
        """
        Apply attribute masking to graph.

        Args:
            data: GraphData object

        Returns:
            Augmented graph with masked attributes
        """
        if not self._should_apply():
            return data

        new_x = data.x.clone()
        new_edge_attr = data.edge_attr.clone() if data.edge_attr is not None else None

        if self.mask_type in ["node", "both"]:
            num_nodes = data.x.size(0)
            mask_nodes = int(num_nodes * self.mask_ratio)
            node_idx = self.rng.permutation(num_nodes)[:mask_nodes]
            new_x[node_idx] = 0

        if self.mask_type in ["edge", "both"] and new_edge_attr is not None:
            num_edges = new_edge_attr.size(0)
            mask_edges = int(num_edges * self.mask_ratio)
            edge_idx = self.rng.permutation(num_edges)[:mask_edges]
            new_edge_attr[edge_idx] = 0

        return GraphData(
            x=new_x,
            edge_index=data.edge_index,
            edge_attr=new_edge_attr,
            y=data.y,
            num_nodes=data.num_nodes,
        )


class SubgraphExtraction(AugmentationBase):
    """Extract random subgraphs from the graph."""

    def __init__(
        self,
        ratio: float = 0.5,
        num_samples: Optional[int] = None,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.ratio = ratio
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)

    def __call__(self, data: GraphData) -> GraphData:
        """
        Extract subgraph from the graph.

        Args:
            data: GraphData object

        Returns:
            Subgraph
        """
        if not self._should_apply():
            return data

        num_nodes = data.x.size(0)
        if self.num_samples is not None:
            num_keep = min(self.num_samples, num_nodes)
        else:
            num_keep = max(1, int(num_nodes * self.ratio))

        if num_keep >= num_nodes:
            return data

        start_nodes = self.rng.choice(num_nodes, size=min(5, num_nodes), replace=False)

        visited = set()
        queue = list(start_nodes)

        while queue and len(visited) < num_keep:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            edges_from_node = torch.where(data.edge_index[0] == node)[0]
            edges_to_node = torch.where(data.edge_index[1] == node)[0]

            neighbors = (
                data.edge_index[0, edges_from_node].tolist()
                + data.edge_index[1, edges_to_node].tolist()
            )

            for n in neighbors:
                if n not in visited:
                    queue.append(n)

        if len(visited) < 2:
            return data

        node_list = list(visited)
        node_set = set(node_list)
        node_to_new = {old: new for new, old in enumerate(node_list)}

        new_x = data.x[node_list]

        edge_mask = torch.tensor(
            [n in node_set for n in data.edge_index[0].tolist()]
            and [n in node_set for n in data.edge_index[1].tolist()],
            device=data.edge_index.device,
        )

        edge_index = data.edge_index[:, edge_mask]
        new_edge_index = torch.tensor(
            [
                [node_to_new[n] for n in edge_index[0].tolist()],
                [node_to_new[n] for n in edge_index[1].tolist()],
            ],
            device=data.edge_index.device,
            dtype=torch.long,
        )

        edge_attr = data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr[edge_mask]

        return GraphData(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=edge_attr,
            y=data.y,
            num_nodes=len(node_list),
        )


class NodeFeatureNoise(AugmentationBase):
    """Add noise to node features."""

    def __init__(
        self,
        noise_level: float = 0.1,
        noise_type: str = "gaussian",
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.rng = np.random.RandomState(seed)

    def __call__(self, data: GraphData) -> GraphData:
        """
        Add noise to node features.

        Args:
            data: GraphData object

        Returns:
            Graph with noisy node features
        """
        if not self._should_apply():
            return data

        x = data.x.clone()

        if self.noise_type == "gaussian":
            noise = torch.randn_like(x) * self.noise_level
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(x) - 0.5) * 2 * self.noise_level
        else:
            noise = torch.randn_like(x) * self.noise_level

        x = x + noise

        return GraphData(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y,
            num_nodes=data.num_nodes,
        )


class EdgeWeightPerturbation(AugmentationBase):
    """Perturb edge weights."""

    def __init__(
        self,
        perturbation_scale: float = 0.1,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.perturbation_scale = perturbation_scale
        self.rng = np.random.RandomState(seed)

    def __call__(self, data: GraphData) -> GraphData:
        """
        Perturb edge weights.

        Args:
            data: GraphData object

        Returns:
            Graph with perturbed edge weights
        """
        if not self._should_apply():
            return data

        if data.edge_attr is None:
            num_edges = data.edge_index.size(1)
            edge_attr = torch.ones(num_edges, device=data.edge_index.device)
        else:
            edge_attr = data.edge_attr.clone()

        noise = torch.randn_like(edge_attr) * self.perturbation_scale
        edge_attr = edge_attr + noise

        edge_attr = torch.clamp(edge_attr, min=0.0)

        return GraphData(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            y=data.y,
            num_nodes=data.num_nodes,
        )


class GraphMixup(AugmentationBase):
    """Mixup for graphs."""

    def __init__(
        self,
        alpha: float = 0.2,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def __call__(self, data_list: List[GraphData]) -> List[GraphData]:
        """
        Apply graph mixup.

        Args:
            data_list: List of GraphData objects

        Returns:
            List of mixed graphs
        """
        if not self._should_apply() or len(data_list) < 2:
            return data_list

        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        idx = self.rng.choice(len(data_list), size=2, replace=False)
        data1, data2 = data_list[idx[0]], data_list[idx[1]]

        if data1.x.size(0) != data2.x.size(0):
            min_nodes = min(data1.x.size(0), data2.x.size(0))
            x1, x2 = data1.x[:min_nodes], data2.x[:min_nodes]
            edge_index1 = data1.edge_index[:, data1.edge_index[0] < min_nodes]
            edge_index2 = data2.edge_index[:, data2.edge_index[0] < min_nodes]
        else:
            x1, x2 = data1.x, data2.x
            edge_index1, edge_index2 = data1.edge_index, data2.edge_index

        new_x = lam * x1 + (1 - lam) * x2
        new_edge_index = edge_index1

        return [
            GraphData(
                x=new_x,
                edge_index=new_edge_index,
                edge_attr=data1.edge_attr,
                y=data1.y,
                num_nodes=new_x.size(0),
            )
        ]


class PersonalizedPageRank(AugmentationBase):
    """Generate augmented graph using personalized PageRank."""

    def __init__(
        self,
        alpha: float = 0.85,
        iterations: int = 30,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.alpha = alpha
        self.iterations = iterations
        self.rng = np.random.RandomState(seed)

    def __call__(self, data: GraphData) -> GraphData:
        """
        Generate augmented graph using personalized PageRank.

        Args:
            data: GraphData object

        Returns:
            Augmented graph
        """
        if not self._should_apply():
            return data

        num_nodes = data.x.size(0)
        edge_index = data.edge_index

        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1
        P = adj / degree

        ppr = torch.ones(num_nodes, num_nodes, device=edge_index.device) / num_nodes

        for _ in range(self.iterations):
            ppr = (1 - self.alpha) / num_nodes + self.alpha * P @ ppr.T
            ppr = ppr.T

        edge_weights = ppr[edge_index[0], edge_index[1]]
        threshold = torch.quantile(edge_weights, 0.75)

        new_edge_mask = edge_weights > threshold
        new_edge_index = edge_index[:, new_edge_mask]

        return GraphData(
            x=data.x,
            edge_index=new_edge_index,
            edge_attr=edge_weights[new_edge_mask].unsqueeze(-1)
            if data.edge_attr is not None
            else None,
            y=data.y,
            num_nodes=num_nodes,
        )


class GraphDataAugmentation:
    """Complete graph data augmentation pipeline."""

    def __init__(
        self,
        node_drop_ratio: float = 0.1,
        edge_drop_ratio: float = 0.1,
        mask_ratio: float = 0.15,
        noise_level: float = 0.1,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.node_drop = NodeDrop(drop_ratio=node_drop_ratio, p=p, seed=seed)
        self.edge_drop = EdgeDrop(drop_ratio=edge_drop_ratio, p=p, seed=seed)
        self.attribute_mask = AttributeMasking(mask_ratio=mask_ratio, p=p, seed=seed)
        self.feature_noise = NodeFeatureNoise(noise_level=noise_level, p=p, seed=seed)

    def __call__(self, data: GraphData) -> GraphData:
        """
        Apply full augmentation pipeline.

        Args:
            data: GraphData object

        Returns:
            Augmented graph
        """
        data = self.node_drop(data)
        data = self.edge_drop(data)
        data = self.attribute_mask(data)
        data = self.feature_noise(data)
        return data


def get_graph_augmentation_pipeline(
    task: str = "node_classification",
    intensity: float = 1.0,
) -> List[Any]:
    """
    Get a pre-configured graph augmentation pipeline.

    Args:
        task: Task type (node_classification, graph_classification, link_prediction)
        intensity: Overall augmentation intensity

    Returns:
        List of augmentation operations
    """
    return [
        NodeDrop(drop_ratio=0.1 * intensity),
        EdgeDrop(drop_ratio=0.1 * intensity),
        AttributeMasking(mask_ratio=0.15 * intensity),
        NodeFeatureNoise(noise_level=0.1 * intensity),
    ]
