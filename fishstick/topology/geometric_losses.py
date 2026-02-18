"""
Geometric Deep Learning TDA Losses.

Provides loss functions for graph neural networks and
geometric deep learning with topological regularization.
"""

from typing import List, Optional, Dict, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphTopologicalRegularization(nn.Module):
    """
    Graph Topological Regularization.

    Regularizes graph neural network outputs to maintain
    desirable topological properties.
    """

    def __init__(
        self,
        target_betti: Optional[Dict[int, float]] = None,
        weight: float = 1.0,
        dimensions: List[int] = [0, 1],
    ):
        """
        Initialize graph topological regularization.

        Args:
            target_betti: Target Betti numbers per dimension
            weight: Loss weight
            dimensions: Dimensions to regularize
        """
        super().__init__()
        self.target_betti = target_betti or {}
        self.weight = weight
        self.dimensions = dimensions

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Compute graph topological regularization.

        Args:
            node_features: Node features [n_nodes, feature_dim]
            edge_index: Edge indices [2, n_edges]

        Returns:
            Regularization loss
        """
        loss = 0.0

        adj = self._build_adjacency(edge_index, node_features.shape[0])

        for dim in self.dimensions:
            betti = self._estimate_betti(adj, dim)

            if dim in self.target_betti:
                target = self.target_betti[dim]
                loss += (betti - target) ** 2
            else:
                loss += torch.abs(betti)

        return self.weight * loss

    def _build_adjacency(
        self,
        edge_index: Tensor,
        n_nodes: int,
    ) -> Tensor:
        """Build adjacency matrix."""
        adj = torch.zeros(n_nodes, n_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        return adj

    def _estimate_betti(
        self,
        adj: Tensor,
        dimension: int,
    ) -> float:
        """Estimate Betti number at given dimension."""
        if dimension == 0:
            n, _ = adj.shape
            visited = torch.zeros(n, dtype=torch.bool)
            components = 0

            for i in range(n):
                if not visited[i]:
                    components += 1
                    self._dfs(i, adj, visited)

            return float(components)

        return 0.0

    def _dfs(
        self,
        node: int,
        adj: Tensor,
        visited: Tensor,
    ) -> None:
        """Depth-first search for component counting."""
        stack = [node]

        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True

            neighbors = torch.where(adj[current] > 0)[0]
            for n in neighbors:
                if not visited[n]:
                    stack.append(n)


class PersistentGraphAlignmentLoss(nn.Module):
    """
    Persistent Graph Alignment Loss.

    Aligns topological features between graphs using
    persistent homology.
    """

    def __init__(
        self,
        weight: float = 1.0,
        use_wasserstein: bool = True,
    ):
        """
        Initialize graph alignment loss.

        Args:
            weight: Loss weight
            use_wasserstein: Use Wasserstein distance
        """
        super().__init__()
        self.weight = weight
        self.use_wasserstein = use_wasserstein

    def forward(
        self,
        graph1_features: Tensor,
        graph1_edges: Tensor,
        graph2_features: Tensor,
        graph2_edges: Tensor,
    ) -> Tensor:
        """
        Compute alignment loss between graphs.

        Args:
            graph1_features: First graph node features
            graph1_edges: First graph edge indices
            graph2_features: Second graph node features
            graph2_edges: Second graph edge indices

        Returns:
            Alignment loss
        """
        from .persistence import PersistentHomology
        from .vietoris_rips import VietorisRipsComplex

        ph = PersistentHomology(max_dimension=1)

        vr1 = VietorisRipsComplex(max_dimension=1)
        simplices1, filtrations1 = vr1.build_from_points(graph1_features)
        boundary_op1 = BoundaryOperator(simplices1)
        diagrams1 = ph.compute(filtrations1, boundary_op1.get_matrices())

        vr2 = VietorisRipsComplex(max_dimension=1)
        simplices2, filtrations2 = vr2.build_from_points(graph2_features)
        boundary_op2 = BoundaryOperator(simplices2)
        diagrams2 = ph.compute(filtrations2, boundary_op2.get_matrices())

        if len(diagrams1) < 2 or len(diagrams2) < 2:
            return torch.tensor(0.0, device=graph1_features.device)

        return self._compute_diagram_distance(diagrams1[1], diagrams2[1])

    def _compute_diagram_distance(
        self,
        diagram1,
        diagram2,
    ) -> Tensor:
        """Compute diagram distance."""
        if self.use_wasserstein:
            from .persistence import wasserstein_distance

            return wasserstein_distance(diagram1, diagram2)
        else:
            from .persistence import bottleneck_distance

            return bottleneck_distance(diagram1, diagram2)


class TopologicalGraphDistillationLoss(nn.Module):
    """
    Topological Graph Distillation Loss.

    Distills topological knowledge from teacher to student
    graph neural networks.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 0.5,
    ):
        """
        Initialize distillation loss.

        Args:
            temperature: Softmax temperature
            alpha: Distillation weight
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_repr: Tensor,
        teacher_repr: Tensor,
        student_edges: Tensor,
        teacher_edges: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute distillation loss.

        Args:
            student_repr: Student representations
            teacher_repr: Teacher representations
            student_edges: Student graph edges
            teacher_edges: Teacher graph edges

        Returns:
            Tuple of (loss, loss_dict)
        """
        repr_loss = F.mse_loss(student_repr, teacher_repr)

        topo_loss = self._compute_topological_distillation(
            student_repr,
            student_edges,
            teacher_repr,
            teacher_edges,
        )

        total_loss = self.alpha * repr_loss + (1 - self.alpha) * topo_loss

        loss_dict = {
            "repr_loss": repr_loss,
            "topo_loss": topo_loss,
            "total": total_loss,
        }

        return total_loss, loss_dict

    def _compute_topological_distillation(
        self,
        student_repr: Tensor,
        student_edges: Tensor,
        teacher_repr: Tensor,
        teacher_edges: Tensor,
    ) -> Tensor:
        """Compute topological knowledge distillation."""
        student_diag = self._extract_topology(student_repr, student_edges)
        teacher_diag = self._extract_topology(teacher_repr, teacher_edges)

        if len(student_diag) == 0 or len(teacher_diag) == 0:
            return torch.tensor(0.0, device=student_repr.device)

        return F.mse_loss(student_diag, teacher_diag)

    def _extract_topology(
        self,
        features: Tensor,
        edges: Tensor,
    ) -> Tensor:
        """Extract topological features."""
        from .persistence import PersistentHomology

        ph = PersistentHomology()
        diagrams = ph.compute_from_distance(features)

        if len(diagrams) > 0:
            return diagrams[0].to_tensor()

        return torch.zeros(1, 2)


class HomologicalConnectivityLoss(nn.Module):
    """
    Homological Connectivity Loss.

    Encourages specific connectivity patterns in
    learned graph structures.
    """

    def __init__(
        self,
        target_components: int = 1,
        target_cycles: int = 0,
        weight: float = 1.0,
    ):
        """
        Initialize connectivity loss.

        Args:
            target_components: Target number of connected components
            target_cycles: Target number of cycles
            weight: Loss weight
        """
        super().__init__()
        self.target_components = target_components
        self.target_cycles = target_cycles
        self.weight = weight

    def forward(
        self,
        adj: Tensor,
    ) -> Tensor:
        """
        Compute connectivity loss.

        Args:
            adj: Adjacency matrix [n_nodes, n_nodes]

        Returns:
            Connectivity loss
        """
        n_nodes = adj.shape[0]

        n_components = self._count_components(adj)
        n_cycles = self._count_cycles(adj)

        component_loss = (n_components - self.target_components) ** 2
        cycle_loss = (n_cycles - self.target_cycles) ** 2

        return self.weight * (component_loss + cycle_loss)

    def _count_components(self, adj: Tensor) -> int:
        """Count connected components."""
        n = adj.shape[0]
        visited = torch.zeros(n, dtype=torch.bool)
        components = 0

        for i in range(n):
            if not visited[i]:
                components += 1
                stack = [i]

                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True

                    neighbors = torch.where(adj[node] > 0)[0]
                    for n in neighbors:
                        if not visited[n]:
                            stack.append(n)

        return components

    def _count_cycles(self, adj: Tensor) -> int:
        """Count cycles using Betti number estimation."""
        n = adj.shape[0]
        n_edges = (adj > 0).sum() // 2

        n_components = self._count_components(adj)

        betti_1 = n_edges - n + n_components

        return max(0, betti_1)


class SimplicialNeuralNetworkLoss(nn.Module):
    """
    Simplicial Neural Network Loss.

    Loss functions for simplicial complex neural networks
    that operate on cells of different dimensions.
    """

    def __init__(
        self,
        cochain_levels: List[int] = [0, 1, 2],
        weight: float = 1.0,
    ):
        """
        Initialize simplicial loss.

        Args:
            cochain_levels: Levels of cochains to regularize
            weight: Loss weight
        """
        super().__init__()
        self.cochain_levels = cochain_levels
        self.weight = weight

    def forward(
        self,
        cochains: Dict[int, Tensor],
        target_cochains: Optional[Dict[int, Tensor]] = None,
    ) -> Tensor:
        """
        Compute simplicial loss.

        Args:
            cochains: Dictionary of cochain values by level
            target_cochains: Optional target values

        Returns:
            Loss value
        """
        loss = 0.0

        for level in self.cochain_levels:
            if level not in cochains:
                continue

            cochain = cochains[level]

            if target_cochains is not None and level in target_cochains:
                loss += F.mse_loss(cochain, target_cochains[level])
            else:
                loss += torch.mean(torch.abs(cochain))

            if level > 0:
                boundary_consistency = self._check_boundary_consistency(cochain, level)
                loss += boundary_consistency

        return self.weight * loss

    def _check_boundary_consistency(
        self,
        cochain: Tensor,
        level: int,
    ) -> Tensor:
        """Check coboundary consistency."""
        if cochain.ndim < 2:
            return torch.tensor(0.0)

        diff = torch.diff(cochain, dim=0)

        return torch.mean(torch.abs(diff))


class TopologicalGraphMatchingLoss(nn.Module):
    """
    Topological Graph Matching Loss.

    Matches topological structure between graphs using
    persistence-based matching.
    """

    def __init__(
        self,
        weight: float = 1.0,
        n_samples: int = 10,
    ):
        """
        Initialize matching loss.

        Args:
            weight: Loss weight
            n_samples: Number of persistence samples
        """
        super().__init__()
        self.weight = weight
        self.n_samples = n_samples

    def forward(
        self,
        source_repr: Tensor,
        target_repr: Tensor,
        source_edges: Tensor,
        target_edges: Tensor,
    ) -> Tensor:
        """
        Compute topological matching loss.

        Args:
            source_repr: Source graph representations
            target_repr: Target graph representations
            source_edges: Source graph edges
            target_edges: Target graph edges

        Returns:
            Matching loss
        """
        source_diag = self._get_persistence_features(source_repr)
        target_diag = self._get_persistence_features(target_repr)

        if len(source_diag) == 0 or len(target_diag) == 0:
            return torch.tensor(0.0, device=source_repr.device)

        n = min(self.n_samples, len(source_diag), len(target_diag))

        source_sample = source_diag[:n]
        target_sample = target_diag[:n]

        cost_matrix = torch.cdist(source_sample, target_sample)

        matched_cost = torch.min(cost_matrix, dim=1)[0].mean()

        return self.weight * matched_cost

    def _get_persistence_features(
        self,
        features: Tensor,
    ) -> Tensor:
        """Extract persistence features from graph."""
        from .persistence import PersistentHomology

        ph = PersistentHomology(max_dimension=1)

        try:
            diagrams = ph.compute_from_distance(features)

            if len(diagrams) > 0:
                return diagrams[0].persistences.unsqueeze(-1)

        except:
            pass

        return torch.zeros(1, 1, device=features.device)


class BoundaryOperatorLoss(nn.Module):
    """
    Boundary Operator Consistency Loss.

    Ensures consistency of boundary operators in
    simplicial complex representations.
    """

    def __init__(
        self,
        weight: float = 1.0,
    ):
        """
        Initialize boundary loss.

        Args:
            weight: Loss weight
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        simplices: List,
        boundary_matrices: List[Tensor],
    ) -> Tensor:
        """
        Compute boundary consistency loss.

        Args:
            simplices: List of simplices
            boundary_matrices: List of boundary matrices

        Returns:
            Loss value
        """
        if len(boundary_matrices) < 2:
            return torch.tensor(0.0)

        loss = 0.0

        for i in range(len(boundary_matrices) - 1):
            b1 = boundary_matrices[i]
            b2 = boundary_matrices[i + 1]

            if b1.shape[1] > 0 and b2.shape[0] > 0:
                product = torch.matmul(b1, b2)

                loss += torch.mean(torch.abs(product))

        return self.weight * loss


class LaplacianSmoothingLoss(nn.Module):
    """
    Topological Laplacian Smoothing Loss.

    Encourages smooth features that respect the
    underlying graph topology.
    """

    def __init__(
        self,
        weight: float = 0.1,
    ):
        """
        Initialize smoothing loss.

        Args:
            weight: Loss weight
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Compute topological smoothing loss.

        Args:
            features: Node features [n_nodes, feature_dim]
            edge_index: Edge indices [2, n_edges]

        Returns:
            Smoothing loss
        """
        n_nodes = features.shape[0]

        row, col = edge_index

        diff = features[row] - features[col]

        laplacian_loss = torch.mean(diff**2)

        return self.weight * laplacian_loss
